"""The tracer at one output time in several runs side by side (scripts/e2e/m2_real_slide.sh).

Each run is given as LABEL=RUN_DIR:STUDY. Runs in the uv environment of slideviz:

    uv run --project ~/git/slideviz python cases.py --mesh OUT/mesh/<mesh>.json --index 3 \
        --out FIGURES/sucrose_cases_3s.png baseline=OUT/runs/baseline_r2:OUT/baseline_r2.study.json ...
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib.lines import Line2D

from render import CENTRAL, OTHER, PORTAL, frame, walls
from slidesim_io import Run


def main() -> None:
    """Write the figure."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", metavar="LABEL=RUN_DIR:STUDY")
    parser.add_argument("--mesh", type=Path, required=True, help="the slideviz mesh record (.json)")
    parser.add_argument("--index", type=int, required=True, help="the output index (of time.outputs)")
    parser.add_argument("--field", default="sucrose", help="the tracer field [sucrose]")
    parser.add_argument("--out", type=Path, required=True, help="the PNG file to write")
    args = parser.parse_args()
    plt.switch_backend("Agg")
    fig, axes = plt.subplots(1, len(args.runs), figsize=(3.3 * len(args.runs) + 1.0, 4.4), sharey=True,
                             layout="constrained")
    for ax, spec in zip(np.atleast_1d(axes), args.runs, strict=True):
        label, rest = spec.split("=", 1)
        run_dir, study = rest.split(":")
        run = Run(Path(run_dir), Path(study), args.mesh)
        t = run.summary["fields"][args.field]["times"][args.index]
        triangulation = mtri.Triangulation(run.points[:, 0], run.points[:, 1], run.cells)
        c = run.field(args.field, args.index)
        mappable = ax.tripcolor(triangulation, np.clip(c, 0, 1), shading="gouraud", cmap="magma", vmin=0, vmax=1,
                                rasterized=True)
        ax.tricontour(triangulation, c, levels=[0.5], colors="w", linewidths=0.8)
        walls(ax, run, lw=1.3)
        frame(ax, run)
        if ax is not np.atleast_1d(axes)[0]:
            ax.set_ylabel("")
        ax.set_title(label, fontsize=10)
    fig.colorbar(mappable, ax=axes, label=f"{args.field} (mM)", shrink=0.85)
    handles = [Line2D([], [], color=PORTAL, lw=2, label="portal wall (1 mM)"),
               Line2D([], [], color=CENTRAL, lw=2, label="central wall (free outflow)"),
               Line2D([], [], color=OTHER, lw=1.2, label="other edge"),
               Line2D([], [], color="w", lw=1.2, label="0.5 mM")]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, fontsize=9, frameon=True, facecolor="0.85",
               edgecolor="none")
    fig.suptitle(f"{args.field.capitalize()} at t = {t:g} s", fontsize=11)
    fig.savefig(args.out, dpi=95)


if __name__ == "__main__":
    main()
