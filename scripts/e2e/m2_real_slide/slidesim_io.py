"""Reading the outputs of a slidesim run on a slideviz mesh: VTU files, wall tags, wall pieces, the direction map.

Shared by the analysis and figure scripts of scripts/e2e/m2_real_slide.sh. slidesim writes one VTU file with a
piece per process; `read_vtu` merges the pieces, and `Run` merges the points they share, so that every cell and
every node is there once.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import zarr
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

TYPES = {"Float64": "<f8", "Float32": "<f4", "Int32": "<i4", "Int64": "<i8", "UInt8": "u1"}
ARRAY = re.compile(r'type="(\w+)" Name="([^"]+)"\s+NumberOfComponents="(\d+)" format="appended" offset="(\d+)"')
# The Gauss points of an edge from its first end (slidesim's EDGE_GAUSS).
EDGE_GAUSS = np.array([0.5 - 0.5 / np.sqrt(3.0), 0.5 + 0.5 / np.sqrt(3.0)])


@dataclass
class Vtu:
    """The cells (triangles), points ([x, y] um) and point data of a VTU file, pieces concatenated."""

    points: np.ndarray
    cells: np.ndarray
    data: dict[str, np.ndarray]


def read_vtu(path: Path) -> Vtu:
    """A VTU file of slidesim (PETSc's writer: raw appended data with 64 bit headers), all pieces."""
    raw = path.read_bytes()
    start = raw.index(b"<AppendedData")
    head = raw[:start].decode()
    base = raw.index(b"_", start) + 1
    points, cells, data = [], [], {}
    offset = 0
    for piece in re.findall(r"<Piece .*?</Piece>", head, re.DOTALL):
        arrays = {}
        for kind, name, components, at in ARRAY.findall(piece):
            o = base + int(at)
            n = int.from_bytes(raw[o:o + 8], "little")
            values = np.frombuffer(raw[o + 8:o + 8 + n], dtype=TYPES[kind])
            arrays[name] = values.reshape(-1, int(components)) if int(components) > 1 else values
        points.append(arrays["Position"][:, :2])
        cells.append(arrays["connectivity"].reshape(-1, 3) + offset)
        offset += len(arrays["Position"])
        for name, values in arrays.items():
            if name not in ("Position", "connectivity", "offsets", "types", "Rank"):
                data.setdefault(name, []).append(values)
    return Vtu(np.concatenate(points), np.concatenate(cells), {k: np.concatenate(v) for k, v in data.items()})


def cell_gradient(points: np.ndarray, cells: np.ndarray, f: np.ndarray) -> np.ndarray:
    """The gradient of the P1 field f (one value per point) on each cell."""
    x = points[cells]
    d1, d2 = x[:, 1] - x[:, 0], x[:, 2] - x[:, 0]
    det = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    v = f[cells]
    f1, f2 = v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]
    return np.c_[(f1 * d2[:, 1] - f2 * d1[:, 1]) / det, (-f1 * d2[:, 0] + f2 * d1[:, 0]) / det]


def weighted_quantiles(values: np.ndarray, weights: np.ndarray, qs: list[float]) -> np.ndarray:
    """Quantiles of values weighed by weights (midpoint rule on the sorted cumulative weights)."""
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cw = (np.cumsum(w) - 0.5 * w) / w.sum()
    return np.interp(qs, cw, v)


class DirectionMap:
    """The map of the scene that slidesim takes the direction of the permeability from, with its gradient as
    slidesim computes it: central differences at the pixel centres (one-sided at the array border), interpolated
    bilinearly."""

    def __init__(self, scene_path: Path, map_id: str) -> None:
        scene = json.loads(scene_path.read_text())
        (spec,) = [m for m in scene["maps"] if m["id"] == map_id]
        values = zarr.open_array(scene_path.parent / spec["path"], mode="r")[:].astype(float)
        if not np.isfinite(values).all():
            raise ValueError(f"map '{map_id}' has pixels without a value; the analysis needs them all")
        h = float(spec["um_per_px"])
        self.h, self.origin = h, np.asarray(spec["origin_um"], float)
        gx, gy = np.empty_like(values), np.empty_like(values)
        gx[:, 1:-1] = (values[:, 2:] - values[:, :-2]) / (2 * h)
        gx[:, 0], gx[:, -1] = (values[:, 1] - values[:, 0]) / h, (values[:, -1] - values[:, -2]) / h
        gy[1:-1] = (values[2:] - values[:-2]) / (2 * h)
        gy[0], gy[-1] = (values[1] - values[0]) / h, (values[-1] - values[-2]) / h
        self.pixel_gradient = np.stack([gx, gy], axis=-1)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """The gradient [d/dx, d/dy] at the points x (..., 2), per um."""
        rows, cols = self.pixel_gradient.shape[:2]
        p = (x - self.origin) / self.h
        if (p < 0).any() or (p[..., 0] > cols - 1).any() or (p[..., 1] > rows - 1).any():
            raise ValueError("a point outside the direction map")
        c = np.minimum(np.floor(p[..., 0]).astype(int), cols - 2)
        r = np.minimum(np.floor(p[..., 1]).astype(int), rows - 2)
        tx, ty = (p[..., 0] - c)[..., None], (p[..., 1] - r)[..., None]
        g = self.pixel_gradient
        return ((1 - ty) * ((1 - tx) * g[r, c] + tx * g[r, c + 1])
                + ty * ((1 - tx) * g[r + 1, c] + tx * g[r + 1, c + 1]))


def permeability(perfusion: dict, direction: DirectionMap, x: np.ndarray) -> np.ndarray:
    """K at the points x (..., 2) as (..., 2, 2): k_across I + (k_along - k_across) g g^T / (|g|^2 + g0^2)."""
    k_along, k_across = perfusion["k_along"], perfusion["k_across"]
    if not all(isinstance(k, (int, float)) for k in (k_along, k_across)):
        raise ValueError("the analysis takes constant permeabilities only")
    g = direction.gradient(x)
    b = (k_along - k_across) / ((g**2).sum(-1) + perfusion["direction"]["g0"] ** 2)
    return k_across * np.eye(2) + b[..., None, None] * g[..., :, None] * g[..., None, :]


class Run:
    """A slidesim run on a slideviz mesh: the merged mesh of its pressure.vtu, the tags of its boundary edges (from
    the slideviz mesh record and _view.npz) and its study."""

    def __init__(self, run_dir: Path, study_path: Path, record_path: Path) -> None:
        self.dir = run_dir
        self.study = json.loads(study_path.read_text())
        self.summary = json.loads((run_dir / "summary.json").read_text())
        pressure = read_vtu(run_dir / "pressure.vtu")
        velocity = read_vtu(run_dir / "velocity.vtu")
        if not (np.array_equal(pressure.points, velocity.points) and np.array_equal(pressure.cells, velocity.cells)):
            raise ValueError(f"{run_dir}: pressure.vtu and velocity.vtu have different meshes")
        # Merge the points the pieces share.
        self.unique, self.first, inverse = np.unique(np.round(pressure.points, 6), axis=0, return_index=True,
                                                     return_inverse=True)
        self.inverse = inverse.reshape(-1)
        self.points = self.unique
        self.cells = self.inverse[pressure.cells]
        self.pressure = pressure.data["pressure.Pa"][self.first]
        self.velocity = velocity.data["velocity"][self.first, :2]
        x = self.points[self.cells]
        d1, d2 = x[:, 1] - x[:, 0], x[:, 2] - x[:, 0]
        self.areas = 0.5 * np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0])
        self.centroids = x.mean(axis=1)
        self._boundary()
        self._tags(record_path)

    def field(self, field: str, index: int) -> np.ndarray:
        """The field `field` at the output `index`, one value per merged point."""
        vtu = read_vtu(self.dir / f"{field}_{index}.vtu")
        if len(vtu.points) != len(self.inverse):
            raise ValueError(f"{field}_{index}.vtu has another mesh than pressure.vtu")
        return vtu.data[f"{field}.mM"][self.first]

    def _boundary(self) -> None:
        """The edges with one cell: their ends (in the direction that keeps the cell on the left), the cell, and the
        outward normal with the length of the edge."""
        c = self.cells
        ends = np.concatenate([c[:, [0, 1]], c[:, [1, 2]], c[:, [2, 0]]])
        cell = np.tile(np.arange(len(c)), 3)
        key = np.sort(ends, axis=1)
        _, index, counts = np.unique(key, axis=0, return_index=True, return_counts=True)
        once = index[counts == 1]
        self.edge_ends, self.edge_cell = ends[once], cell[once]
        a, b = self.points[self.edge_ends[:, 0]], self.points[self.edge_ends[:, 1]]
        normal = np.c_[b[:, 1] - a[:, 1], a[:, 0] - b[:, 0]]
        inside = self.centroids[self.edge_cell] - 0.5 * (a + b)
        normal[(normal * inside).sum(axis=1) > 0] *= -1
        self.edge_normal = normal

    def _tags(self, record_path: Path) -> None:
        """The tag of each boundary edge: that of the edge of the slideviz mesh (level 0) which holds its midpoint."""
        record = json.loads(record_path.read_text())
        names = {number: name for name, number in record["tags"]["edges"].items()}
        view_name = next(f for f in record["files"] if f.endswith("_view.npz"))
        with np.load(record_path.with_name(view_name)) as view:
            p, edges, tag = view["points_um"][:, :2], view["edges"], view["edge_tag"]
        a, b = p[edges[:, 0]], p[edges[:, 1]]
        mid = 0.5 * (self.points[self.edge_ends[:, 0]] + self.points[self.edge_ends[:, 1]])
        _, candidates = cKDTree(0.5 * (a + b)).query(mid, k=min(8, len(edges)))
        ab = b[candidates] - a[candidates]
        t = np.clip(((mid[:, None] - a[candidates]) * ab).sum(-1) / (ab**2).sum(-1), 0, 1)
        distance = np.linalg.norm(a[candidates] + t[..., None] * ab - mid[:, None], axis=-1)
        best = candidates[np.arange(len(mid)), distance.argmin(axis=1)]
        if distance.min(axis=1).max() > 1e-6:
            raise ValueError("a boundary edge of the run lies on no edge of the slideviz mesh")
        self.edge_tag = np.array([names[int(t)] for t in tag[best]])

    def nodes(self, tag: str) -> np.ndarray:
        """The points on the edges of the tag."""
        return np.unique(self.edge_ends[self.edge_tag == tag])

    def pieces(self, tag: str) -> list[np.ndarray]:
        """The connected pieces of the edges of the tag: the indices of their boundary edges."""
        selected = np.flatnonzero(self.edge_tag == tag)
        e = self.edge_ends[selected]
        n = len(self.points)
        graph = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
        _, label = connected_components(graph, directed=False)
        piece = label[e[:, 0]]
        return [selected[piece == k] for k in np.unique(piece)]

    def edge_velocity(self, direction: DirectionMap) -> np.ndarray:
        """The velocity at the two Gauss points of each boundary edge (edges, 2, 2) as slidesim's mass balance takes
        it: the pressure gradient of the edge's cell with the permeability at the points, over the viscosity."""
        perfusion = self.study["perfusion"]
        grad = cell_gradient(self.points, self.cells, self.pressure)[self.edge_cell]
        a, b = self.points[self.edge_ends[:, 0]], self.points[self.edge_ends[:, 1]]
        x = a[:, None] + EDGE_GAUSS[None, :, None] * (b - a)[:, None]
        k = permeability(perfusion, direction, x)
        return -np.einsum("eqij,ej->eqi", k, grad) / perfusion["viscosity"]

    def consistent_inflow(self, direction: DirectionMap) -> np.ndarray:
        """The consistent inflow of the pressure equation at each point, R_i = int K grad p . grad v_i / mu, by a
        3-point rule of degree 2 on each cell (slidesim integrates with PETSc's degree 2 rule; the sums over the
        nodes of a tag agree with its tag_flows to the error of the rules). Zero at free points up to the solver
        tolerance; at the points of a pressure tag the flow into the domain there."""
        perfusion = self.study["perfusion"]
        x = self.points[self.cells]
        bary = np.array([[2 / 3, 1 / 6, 1 / 6], [1 / 6, 2 / 3, 1 / 6], [1 / 6, 1 / 6, 2 / 3]])
        k = permeability(perfusion, direction, np.einsum("qv,cvd->cqd", bary, x)).mean(axis=1)
        grad_p = cell_gradient(self.points, self.cells, self.pressure)
        flux = np.einsum("cij,cj->ci", k, grad_p) / perfusion["viscosity"]
        # gradients of the three shape functions of each cell
        d1, d2 = x[:, 1] - x[:, 0], x[:, 2] - x[:, 0]
        det = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
        g1 = np.c_[d2[:, 1], -d2[:, 0]] / det[:, None]
        g2 = np.c_[-d1[:, 1], d1[:, 0]] / det[:, None]
        shape = np.stack([-g1 - g2, g1, g2], axis=1)
        local = self.areas[:, None] * np.einsum("cvd,cd->cv", shape, flux)
        return np.bincount(self.cells.ravel(), weights=local.ravel(), minlength=len(self.points))
