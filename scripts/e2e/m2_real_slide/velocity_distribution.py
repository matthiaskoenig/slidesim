"""Interstitial velocity against the local portal-central distance, and its distribution over the tissue, for
several runs (scripts/e2e/m2_real_slide.sh).

Each run is given as LABEL=RUN_DIR:STUDY. Left: median and 10-90 % band of |u|/phi per class of the distance to
the nearest portal wall plus that to the nearest central wall, with the bound of a straight one-dimensional path,
k_along/mu * dp / distance / phi (dp the pressure drop), dashed: one line per distinct bound, gray where runs
share it. Right: the area-weighted histogram with the velocities measured in mice. Runs in the uv environment of
slideviz:

    uv run --project ~/git/slideviz python velocity_distribution.py --mesh OUT/mesh/<mesh>.json \
        --out FIGURES/velocity_distribution.png baseline=OUT/runs/baseline_r2:OUT/baseline_r2.study.json ...
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
from matplotlib.lines import Line2D
from scipy.spatial import cKDTree

from slidesim_io import Run, weighted_quantiles

COLORS = ["#1b6ca8", "#c2410c", "#4d7c0f", "#7e22ce"]
SHARED = "0.35"
DISTANCES = np.geomspace(150, 2000, 22)
# tissue area (um^2) a distance class needs to be drawn
MIN_AREA = 2000.0
MEASURED = (69.0, 180.0)
L_UM = 230.0


def ticks(ax: plt.Axes, axis: str, values: list[float]) -> None:
    getattr(ax, f"set_{axis}scale")("log")
    getattr(ax, f"set_{axis}ticks")(values, [f"{v:g}" for v in values])
    getattr(ax, f"{axis}axis").set_minor_formatter(matplotlib.ticker.NullFormatter())


def main() -> None:
    """Write the figure."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", metavar="LABEL=RUN_DIR:STUDY")
    parser.add_argument("--mesh", type=Path, required=True, help="the slideviz mesh record (.json)")
    parser.add_argument("--out", type=Path, required=True, help="the PNG file to write")
    args = parser.parse_args()
    plt.switch_backend("Agg")
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.0), layout="constrained")
    bounds: dict[float, list] = {}
    lowest, highest_bin = np.inf, 0.0
    for spec, color in zip(args.runs, COLORS, strict=False):
        label, rest = spec.split("=", 1)
        run_dir, study = rest.split(":")
        run = Run(Path(run_dir), Path(study), args.mesh)
        phi = run.study["fields"][0]["porosity"]
        perfusion = run.study["perfusion"]
        drop = max(b["pressure"] for b in perfusion["boundaries"]) - min(b["pressure"] for b in perfusion["boundaries"])
        a = run.areas
        speed = np.hypot(*run.velocity[run.cells].mean(axis=1).T) / phi
        d = (cKDTree(run.points[run.nodes("portal_wall")]).query(run.centroids)[0]
             + cKDTree(run.points[run.nodes("central_wall")]).query(run.centroids)[0])
        mids, low, median, high = [], [], [], []
        for d0, d1 in itertools.pairwise(DISTANCES):
            m = (d >= d0) & (d < d1)
            if a[m].sum() < MIN_AREA:
                continue
            q = weighted_quantiles(speed[m], a[m], [0.1, 0.5, 0.9])
            mids.append(np.sqrt(d0 * d1))
            low.append(q[0])
            median.append(q[1])
            high.append(q[2])
        lowest = min(lowest, min(low))
        axes[0].plot(mids, median, "o-", color=color, ms=3.5, lw=1.4, label=f"{label}: median")
        axes[0].fill_between(mids, low, high, color=color, alpha=0.15, lw=0)
        scale = perfusion["k_along"] / perfusion["viscosity"] * drop / phi
        bounds.setdefault(round(scale, 9), []).append((label, color))
        h, edges = np.histogram(speed, bins=np.geomspace(0.3, 600, 70), weights=a)
        highest_bin = max(highest_bin, 100 * h.max() / a.sum())
        axes[1].stairs(100 * h / a.sum(), edges, color=color, lw=1.4,
                       label=f"{label}: mean {(speed * a).sum() / a.sum():.0f} um/s")
    x = np.geomspace(150, 2000, 50)
    handles = []
    for scale, runs in bounds.items():
        color = SHARED if len(runs) > 1 else runs[0][1]
        names = " = ".join(name for name, _ in runs)
        axes[0].plot(x, scale / x, "--", color=color, lw=1.1)
        handles.append(Line2D([], [], color=color, ls="--", lw=1.1, label=f"1D bound: {names}"))
    axes[0].axvline(L_UM, color="0.4", lw=0.8, ls=":")
    axes[0].text(L_UM * 1.03, 0.6 * max(lowest, 0.02) + 1.5, f"L = {L_UM:g} um", fontsize=8, color="0.3")
    ticks(axes[0], "x", [200, 300, 500, 1000, 2000])
    ticks(axes[0], "y", [v for v in (0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300) if v >= 0.5 * lowest])
    axes[0].set_ylim(0.7 * lowest, 500)
    axes[0].set_xlabel("distance to the nearest portal wall + to the nearest central wall (um)")
    axes[0].set_ylabel("interstitial velocity |u|/phi (um/s)")
    axes[0].set_title("Median and 10-90 % band per distance class;\ndashed: 1D bound k_along/mu x pressure drop / "
                      "distance / phi", fontsize=10)
    medians = axes[0].get_legend_handles_labels()[0]
    # below both axes, centred on the figure
    fig.legend(handles=medians + handles, fontsize=8, loc="outside lower center", ncol=4, frameon=False)
    axes[1].axvspan(*MEASURED, color="0.88", lw=0, zorder=0)
    axes[1].fill_between([], [], color="0.88", label=f"measured in mice, {MEASURED[0]:g}-{MEASURED[1]:g} um/s")
    axes[1].set_ylim(0, highest_bin * 1.45)
    ticks(axes[1], "x", [1, 3, 10, 30, 100, 300])
    axes[1].set_xlabel("interstitial velocity |u|/phi (um/s)")
    axes[1].set_ylabel("tissue area per bin (%)")
    axes[1].set_title("Distribution over the tissue (area weighted)", fontsize=10)
    axes[1].legend(fontsize=8, loc="upper center", ncol=2)
    fig.savefig(args.out, dpi=105)


if __name__ == "__main__":
    main()
