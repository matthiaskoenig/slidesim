"""Figures of one slidesim run of scripts/e2e/m2_real_slide.sh: pressure, flow over the H&E, tracer front.

Writes <out>/pressure.png, <out>/velocity.png (or velocity_<label>.png with --suffix) and <out>/sucrose.png. Runs in
the uv environment of slideviz (numpy, scipy, matplotlib, zarr):

    uv run --project ~/git/slideviz python render.py --run OUT/runs/baseline_r2 \
        --study OUT/baseline_r2.study.json --mesh OUT/mesh/<mesh>.json --he OUT/analysis/he.npz --out FIGURES
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

from slidesim_io import Run

PORTAL, CENTRAL, OTHER = "#1f5fd6", "#d62728", "0.45"
# The color scale of the velocity ends at this quantile of the area-weighted interstitial velocity.
VELOCITY_CAP_QUANTILE = 0.98
TIMES = (1.0, 3.0, 10.0)
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11})


def walls(ax: plt.Axes, run: Run, lw: float = 1.6, others: bool = True) -> None:
    """The portal and central walls, and the other boundary edges (no flow) in gray."""
    xy = run.points[run.edge_ends]
    for tag, color in (("portal_wall", PORTAL), ("central_wall", CENTRAL)):
        ax.add_collection(LineCollection(xy[run.edge_tag == tag], colors=color, linewidths=lw, zorder=5))
    if others:
        other = ~np.isin(run.edge_tag, ["portal_wall", "central_wall"])
        ax.add_collection(LineCollection(xy[other], colors=OTHER, linewidths=0.9, zorder=5))


def legend(fig: plt.Figure, pressures: bool) -> None:
    """The legend of the walls below the axes, outside the field (the figure has a constrained layout)."""
    suffix = (" (390 Pa)", " (0 Pa)") if pressures else ("", "")
    handles = [Line2D([], [], color=PORTAL, lw=2, label=f"portal wall{suffix[0]}"),
               Line2D([], [], color=CENTRAL, lw=2, label=f"central wall{suffix[1]}"),
               Line2D([], [], color=OTHER, lw=1.2, label="other edge (no flow)")]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, fontsize=9, frameon=False)


def frame(ax: plt.Axes, run: Run) -> None:
    """The field, y pointing down as the rows of the slide."""
    (x0, y0), (x1, y1) = run.points.min(axis=0), run.points.max(axis=0)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.set_aspect("equal")
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")


def pressure(run: Run, triangulation: mtri.Triangulation, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 6.2), layout="constrained")
    mappable = ax.tripcolor(triangulation, run.pressure, shading="gouraud", cmap="viridis", vmin=0, vmax=390,
                            rasterized=True)
    ax.tricontour(triangulation, run.pressure, levels=np.arange(39, 390, 39), colors="w", linewidths=0.5,
                  alpha=0.7)
    walls(ax, run)
    frame(ax, run)
    fig.colorbar(mappable, ax=ax, label="pressure (Pa)", shrink=0.92, pad=0.03)
    ax.set_title(f"Pressure{title}; contours every 39 Pa")
    legend(fig, pressures=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def velocity(run: Run, triangulation: mtri.Triangulation, he_path: Path, title: str, path: Path) -> float:
    """Streamlines of the Darcy velocity over the H&E; returns the cap of the color scale (um/s)."""
    phi = run.study["fields"][0]["porosity"]
    (x0, y0), (x1, y1) = run.points.min(axis=0), run.points.max(axis=0)
    gx, gy = np.linspace(x0, x1, 300), np.linspace(y0, y1, 300)
    grid = np.meshgrid(gx, gy)
    u = mtri.LinearTriInterpolator(triangulation, run.velocity[:, 0])(*grid)
    w = mtri.LinearTriInterpolator(triangulation, run.velocity[:, 1])(*grid)
    speed = np.ma.hypot(u, w) / phi
    cell_speed = np.hypot(*run.velocity[run.cells].mean(axis=1).T) / phi
    order = np.argsort(cell_speed)
    weights = np.cumsum(run.areas[order]) / run.areas.sum()
    cap = float(cell_speed[order][np.searchsorted(weights, VELOCITY_CAP_QUANTILE)])

    with np.load(he_path) as he:
        rgb, extent = he["rgb"], he["extent"]
    # The H&E in gray, contrast stretched to its 1-99 % range, then lightened, so that the colored streamlines stay
    # readable over it.
    gray = rgb[..., :3].astype(float) @ [0.299, 0.587, 0.114]
    lo, hi = np.percentile(gray, [1, 99])
    gray = np.clip((gray - lo) / (hi - lo), 0, 1)
    fig, ax = plt.subplots(figsize=(8.0, 7.0), layout="constrained")
    ax.imshow(0.4 + 0.6 * gray, extent=extent, cmap="gray", vmin=0, vmax=1, interpolation="antialiased")
    lines = ax.streamplot(gx, gy, u.filled(np.nan), w.filled(np.nan), color=speed.filled(np.nan), cmap="viridis",
                          norm=Normalize(0, cap), density=1.6, linewidth=1.1, arrowsize=0.9,
                          broken_streamlines=True)
    walls(ax, run, lw=2.0)
    frame(ax, run)
    fig.colorbar(lines.lines, ax=ax, shrink=0.92, pad=0.03, extend="max",
                 label=f"interstitial velocity |u|/phi (um/s), scale capped at {cap:.0f} um/s (98th percentile)")
    ax.set_title(f"Flow over the H&E (gray){title}")
    legend(fig, pressures=False)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return cap


def sucrose(run: Run, triangulation: mtri.Triangulation, field: str, title: str, path: Path) -> None:
    times = run.summary["fields"][field]["times"]
    fig, axes = plt.subplots(1, len(TIMES), figsize=(12.6, 5.0), sharey=True, layout="constrained")
    for ax, t in zip(axes, TIMES, strict=True):
        index = int(np.argmin(np.abs(np.array(times) - t)))
        c = run.field(field, index)
        mappable = ax.tripcolor(triangulation, np.clip(c, 0, 1), shading="gouraud", cmap="magma", vmin=0, vmax=1,
                                rasterized=True)
        ax.tricontour(triangulation, c, levels=[0.5], colors="w", linewidths=0.8)
        walls(ax, run, lw=1.4)
        frame(ax, run)
        if ax is not axes[0]:
            ax.set_ylabel("")
        ax.set_title(f"t = {times[index]:g} s")
    fig.colorbar(mappable, ax=axes, label=f"{field} (mM)", shrink=0.8)
    handles = [Line2D([], [], color=PORTAL, lw=2, label="portal wall (1 mM)"),
               Line2D([], [], color=CENTRAL, lw=2, label="central wall (free outflow)"),
               Line2D([], [], color=OTHER, lw=1.2, label="other edge"),
               Line2D([], [], color="w", lw=1.2, label="0.5 mM")]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, fontsize=9, frameon=True, facecolor="0.85",
               edgecolor="none")
    fig.suptitle(f"{field.capitalize()}{title}; 1 mM at the portal walls from t = 0")
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main() -> None:
    """Write the figures of one run."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, required=True, help="the output folder of the run")
    parser.add_argument("--study", type=Path, required=True, help="the study file of the run")
    parser.add_argument("--mesh", type=Path, required=True, help="the slideviz mesh record (.json)")
    parser.add_argument("--he", type=Path, required=True, help="the H&E crop of he_crop.py")
    parser.add_argument("--field", default="sucrose", help="the tracer field [sucrose]")
    parser.add_argument("--label", default="", help="added to the titles, e.g. baseline")
    parser.add_argument("--only-velocity", action="store_true", help="write the flow figure only")
    parser.add_argument("--suffix", default="", help="added to the file names, e.g. _radial10")
    parser.add_argument("--out", type=Path, required=True, help="the folder of the figures")
    args = parser.parse_args()
    plt.switch_backend("Agg")
    run = Run(args.run, args.study, args.mesh)
    triangulation = mtri.Triangulation(run.points[:, 0], run.points[:, 1], run.cells)
    title = f", {args.label}" if args.label else ""
    args.out.mkdir(parents=True, exist_ok=True)
    cap = velocity(run, triangulation, args.he, title, args.out / f"velocity{args.suffix}.png")
    print(f"{args.run.name}: velocity scale capped at {cap:.1f} um/s")
    if not args.only_velocity:
        pressure(run, triangulation, title, args.out / f"pressure{args.suffix}.png")
        sucrose(run, triangulation, args.field, title, args.out / f"{args.field}{args.suffix}.png")


if __name__ == "__main__":
    main()
