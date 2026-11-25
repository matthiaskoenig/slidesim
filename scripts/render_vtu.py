# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib", "numpy"]
# ///
"""Render the `.vtu` output of `slidesim run` as PNG images, for checking results by eye
without ParaView.

Usage (from the repository root):

    uv run scripts/render_vtu.py OUT_DIR PNG_DIR [--field oxygen] [--reference REF_DIR]

For every `<field>_<index>.vtu` of OUT_DIR it writes `PNG_DIR/<field>_<index>.png`: the field
(Gouraud shaded, one color range for all output times, the time from `OUT_DIR/summary.json`).
It also writes `<field>_mesh.png` (the mesh of the last output, whole and zoomed in at the
largest value) and `<field>_ranks.png` (the process of each cell, from the `Rank` cell data).
With `--reference REF_DIR` (the output of another run of the same study, for example on one
process), it writes `<field>_<index>_difference.png` (this run minus the reference, matched by
point coordinates) and prints the largest difference.

The reader handles what PETSc's VTK viewer writes: XML `UnstructuredGrid`, one piece per
process, raw appended data with UInt64 headers, triangles.
"""

import argparse
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

DTYPES = {
    "Float64": "<f8",
    "Float32": "<f4",
    "Int32": "<i4",
    "Int64": "<i8",
    "UInt8": "u1",
}
VTK_TRIANGLE = 5


def read_vtu(path: Path) -> dict:
    """Points (n, 2), triangles (m, 3), point data and cell data of all pieces, merged."""
    raw = path.read_bytes()
    marker = raw.index(b"<AppendedData")
    start = raw.index(b"_", marker) + 1
    header = raw[:marker].decode() + "</VTKFile>"
    root = ET.fromstring(header)
    assert root.get("header_type") == "UInt64", root.attrib

    def array(element) -> np.ndarray:
        offset = start + int(element.get("offset"))
        size = int(np.frombuffer(raw, "<u8", 1, offset)[0])
        values = np.frombuffer(
            raw,
            DTYPES[element.get("type")],
            offset=offset + 8,
            count=size // np.dtype(DTYPES[element.get("type")]).itemsize,
        )
        components = int(element.get("NumberOfComponents", "1"))
        return values.reshape(-1, components) if components > 1 else values

    points, triangles, point_data, cell_data = [], [], {}, {}
    base = 0
    for piece in root.iter("Piece"):
        xyz = array(piece.find("Points/DataArray"))
        cells = {a.get("Name"): array(a) for a in piece.find("Cells")}
        assert np.all(cells["types"] == VTK_TRIANGLE), "triangles only"
        assert np.all(np.diff(cells["offsets"]) == 3)
        points.append(xyz[:, :2])
        triangles.append(cells["connectivity"].reshape(-1, 3) + base)
        base += len(xyz)
        for kind, data in (("PointData", point_data), ("CellData", cell_data)):
            element = piece.find(kind)
            for a in element if element is not None else []:
                data.setdefault(a.get("Name"), []).append(array(a))
    return {
        "points": np.concatenate(points),
        "triangles": np.concatenate(triangles),
        "point_data": {k: np.concatenate(v) for k, v in point_data.items()},
        "cell_data": {k: np.concatenate(v) for k, v in cell_data.items()},
    }


def values(mesh: dict, field: str) -> tuple[str, np.ndarray]:
    """The point data of `field` (named `<field>.<unit>` by slidesim) and its unit."""
    for name, data in mesh["point_data"].items():
        if name.split(".")[0] == field:
            return name.partition(".")[2], data
    raise KeyError(f"no point data of {field!r}: {list(mesh['point_data'])}")


def triangulation(mesh: dict) -> mtri.Triangulation:
    return mtri.Triangulation(
        mesh["points"][:, 0], mesh["points"][:, 1], mesh["triangles"]
    )


def frame(ax, title: str) -> None:
    ax.set_aspect("equal")
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    ax.set_title(title)


def render_field(
    mesh: dict, field: str, time: float | None, vmax: float, png: Path
) -> None:
    unit, u = values(mesh, field)
    fig, ax = plt.subplots(figsize=(6.4, 5.4), layout="constrained")
    shaded = ax.tripcolor(
        triangulation(mesh), u, shading="gouraud", cmap="viridis", vmin=0.0, vmax=vmax
    )
    fig.colorbar(shaded, ax=ax, label=f"{field} ({unit})")
    when = "" if time is None else f" at t = {time:g} s"
    frame(ax, f"{field}{when}: min {u.min():.4g}, max {u.max():.4g} {unit}")
    fig.savefig(png, dpi=150)
    plt.close(fig)


def render_mesh(mesh: dict, field: str, png: Path) -> None:
    unit, u = values(mesh, field)
    tri = triangulation(mesh)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), layout="constrained")
    x, y = mesh["points"][np.argmax(u)]
    for ax, zoom in zip(axes, (None, 30.0)):
        shaded = ax.tripcolor(tri, u, shading="gouraud", cmap="viridis", vmin=0.0)
        ax.triplot(
            tri, color="white", linewidth=0.2 if zoom is None else 0.6, alpha=0.7
        )
        if zoom is not None:
            ax.set_xlim(x - zoom, x + zoom)
            ax.set_ylim(y - zoom, y + zoom)
        frame(
            ax,
            f"mesh ({len(mesh['triangles'])} cells)"
            if zoom is None
            else f"zoomed in at ({x:.0f}, {y:.0f}) um",
        )
    fig.colorbar(shaded, ax=axes, label=f"{field} ({unit})")
    fig.savefig(png, dpi=150)
    plt.close(fig)


def render_ranks(mesh: dict, png: Path) -> None:
    ranks = mesh["cell_data"]["Rank"]
    fig, ax = plt.subplots(figsize=(6.4, 5.4), layout="constrained")
    count = int(ranks.max()) + 1
    shaded = ax.tripcolor(
        triangulation(mesh),
        facecolors=ranks.astype(float),
        cmap=plt.get_cmap("tab10", count),
        vmin=-0.5,
        vmax=count - 0.5,
    )
    fig.colorbar(shaded, ax=ax, label="process", ticks=range(count))
    frame(ax, f"partition over {count} process{'' if count == 1 else 'es'}")
    fig.savefig(png, dpi=150)
    plt.close(fig)


def render_difference(mesh: dict, reference: dict, field: str, png: Path) -> float:
    unit, u = values(mesh, field)
    _, r = values(reference, field)
    # Points shared by pieces appear once per piece: match by coordinates.
    lookup = {tuple(np.round(p, 9)): v for p, v in zip(reference["points"], r)}
    difference = np.array(
        [v - lookup[tuple(np.round(p, 9))] for p, v in zip(mesh["points"], u)]
    )
    largest = float(np.abs(difference).max())
    fig, ax = plt.subplots(figsize=(6.4, 5.4), layout="constrained")
    bound = max(largest, 1e-300)
    shaded = ax.tripcolor(
        triangulation(mesh),
        difference,
        shading="gouraud",
        cmap="RdBu_r",
        vmin=-bound,
        vmax=bound,
    )
    fig.colorbar(shaded, ax=ax, label=f"difference ({unit})")
    frame(ax, f"{field} minus reference: max |difference| {largest:.2e} {unit}")
    fig.savefig(png, dpi=150)
    plt.close(fig)
    return largest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("png_dir", type=Path)
    parser.add_argument("--field", default="oxygen")
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()

    files = sorted(
        args.out_dir.glob(f"{args.field}_*.vtu"),
        key=lambda p: int(re.search(r"_(\d+)\.vtu$", p.name).group(1)),
    )
    if not files:
        raise SystemExit(f"no {args.field}_<index>.vtu in {args.out_dir}")
    summary = args.out_dir / "summary.json"
    times = (
        json.loads(summary.read_text())["fields"][args.field]["times"]
        if summary.exists()
        else [None] * len(files)
    )
    meshes = [read_vtu(f) for f in files]
    vmax = max(values(m, args.field)[1].max() for m in meshes)
    args.png_dir.mkdir(parents=True, exist_ok=True)
    for path, mesh, time in zip(files, meshes, times):
        render_field(mesh, args.field, time, vmax, args.png_dir / f"{path.stem}.png")
        if args.reference is not None:
            reference = read_vtu(args.reference / path.name)
            png = args.png_dir / f"{path.stem}_difference.png"
            print(
                f"{path.name}: max |difference| {render_difference(mesh, reference, args.field, png):.3e}"
            )
    render_mesh(meshes[-1], args.field, args.png_dir / f"{args.field}_mesh.png")
    render_ranks(meshes[-1], args.png_dir / f"{args.field}_ranks.png")
    print(f"wrote {args.png_dir}")


if __name__ == "__main__":
    main()
