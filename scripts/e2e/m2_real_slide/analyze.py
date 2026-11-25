"""The numbers of one slidesim run of scripts/e2e/m2_real_slide.sh, as JSON.

Flows per tag and per connected wall piece (the consistent flux of the pressure equation), the mass balance of
the tracer per output time (closure, advective and diffusive inflow per tag, the outflow through walls without
flow per tag and per piece), the interstitial velocity, the pressure gradient, and the tracer front. Runs in the
uv environment of slideviz (numpy, scipy, zarr):

    uv run --project ~/git/slideviz python analyze.py --run OUT/runs/baseline_r2 \
        --study OUT/baseline_r2.study.json --mesh OUT/mesh/<mesh>.json --scene OUT/scene/scene.json --out a.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from slidesim_io import EDGE_GAUSS, DirectionMap, Run, cell_gradient, weighted_quantiles

PORTAL, CENTRAL = "portal_wall", "central_wall"
# The distance to the nearest portal wall plus that to the nearest central wall, below which tissue is "near"
# (um); L = 230 um in docs/parameters/m2-perfusion-mouse.md.
NEAR_UM = 300.0


def flows(run: Run, direction: DirectionMap) -> dict:
    """tag_flows of slidesim, and the consistent flow into the domain through each connected piece of each pressure
    tag. A point on pieces of two pressure tags counts for the tag listed last, as in slidesim."""
    inflow = run.consistent_inflow(direction)
    tags = [b["tag"] for b in run.study["perfusion"]["boundaries"]]
    owner = {}
    for tag in tags:
        for node in run.nodes(tag):
            owner[node] = tag
    tag_flows = run.summary["perfusion"]["tag_flows"]
    out = {"tag_flows": tag_flows,
           "sum_over_largest": sum(tag_flows.values()) / max(abs(v) for v in tag_flows.values()), "pieces": {}}
    for tag in tags:
        pieces = []
        for edges in run.pieces(tag):
            nodes = [n for n in np.unique(run.edge_ends[edges]) if owner[n] == tag]
            centre = run.points[np.unique(run.edge_ends[edges])].mean(axis=0)
            pieces.append({"centre_um": [round(float(c), 1) for c in centre], "edges": len(edges),
                           "inflow_um2_s": float(inflow[nodes].sum())})
        total = sum(p["inflow_um2_s"] for p in pieces)
        out["pieces"][tag] = {"count": len(pieces), "sum_um2_s": total,
                              "sum_vs_tag_flow": total / tag_flows[tag] - 1, "pieces": pieces}
    portal = out["pieces"].get(PORTAL, {"pieces": []})["pieces"]
    central = out["pieces"].get(CENTRAL, {"pieces": []})["pieces"]
    out["every_portal_piece_inflow"] = all(p["inflow_um2_s"] > 0 for p in portal)
    out["every_central_piece_outflow"] = all(p["inflow_um2_s"] < 0 for p in central)
    return out


def balances(run: Run, direction: DirectionMap, field: str) -> list:
    """The mass balance of the field per output time (from summary.json): its closure, the inflows per tag, and
    the outflow through the boundary without condition per connected piece of each tag, recomputed from the
    pressure (validated against the advective part of the summary)."""
    summary = run.summary["fields"][field]
    conditions = {b["tag"] for f in run.study["fields"] if f["id"] == field for b in f["boundaries"]}
    portal_flow = run.summary["perfusion"]["tag_flows"][PORTAL]
    u = run.edge_velocity(direction)
    u_n = np.einsum("eqi,ei->eq", u, run.edge_normal)
    out = []
    for index, (t, balance) in enumerate(zip(summary["times"], summary["mass_balance"], strict=True)):
        if balance is None:
            continue
        inflows = balance["boundary"]
        total = sum(b["advective"] + b["diffusive"] for b in inflows.values())
        terms = [balance["storage_rate"], balance["source"], balance["advection_defect"]]
        terms += [x for b in inflows.values() for x in (b["advective"], b["diffusive"])]
        residual = balance["storage_rate"] - balance["source"] - total + balance["advection_defect"]
        c = run.field(field, index)
        ends = run.edge_ends
        c_q = (1 - EDGE_GAUSS)[None, :] * c[ends[:, :1]] + EDGE_GAUSS[None, :] * c[ends[:, 1:]]
        outflow = 0.5 * (np.clip(u_n, 0, None) * c_q).sum(axis=1)
        free = {}
        for tag in sorted(set(run.edge_tag) - conditions):
            mid = 0.5 * (run.points[ends[:, 0]] + run.points[ends[:, 1]])
            pieces = []
            for e in run.pieces(tag):
                # where the piece lets the tracer out: the outflow-weighted centre of its edges
                weight = outflow[e]
                centre = (weight @ mid[e]) / weight.sum() if weight.sum() > 0 else mid[e].mean(axis=0)
                pieces.append({"outflow_centre_um": [round(float(x), 1) for x in centre],
                               "outflow": float(weight.sum())})
            pieces.sort(key=lambda p: -p["outflow"])
            recomputed = float(outflow[run.edge_tag == tag].sum())
            free[tag] = {"outflow": recomputed, "summary_advective": inflows[tag]["advective"],
                         "pct_of_portal_flow": 100 * recomputed / portal_flow, "pieces": pieces[:3]}
        out.append({
            "t": t, "storage_rate": balance["storage_rate"], "source": balance["source"],
            "advection_defect": balance["advection_defect"],
            "defect_pct_of_portal_flow": 100 * balance["advection_defect"] / portal_flow,
            "closure_over_largest_term": residual / max(abs(x) for x in terms),
            "inflow": inflows, "portal_advective_over_flow": inflows[PORTAL]["advective"] / portal_flow,
            "free_outflow": free,
        })
    return out


def velocity(run: Run) -> dict:
    """The interstitial velocity |u|/phi (cell mean of the nodal velocity), area weighted, and the pressure gradient,
    over the tissue and by the distance to the walls."""
    phi = run.study["fields"][0]["porosity"]
    a = run.areas
    speed = np.hypot(*run.velocity[run.cells].mean(axis=1).T) / phi
    q = weighted_quantiles(speed, a, [0.1, 0.5, 0.9])
    gradient = np.hypot(*cell_gradient(run.points, run.cells, run.pressure).T)
    dp = cKDTree(run.points[run.nodes(PORTAL)]).query(run.centroids)[0]
    dc = cKDTree(run.points[run.nodes(CENTRAL)]).query(run.centroids)[0]
    d = dp + dc

    def mean(values: np.ndarray, where: np.ndarray) -> float:
        return float((values[where] * a[where]).sum() / a[where].sum())

    near = d < NEAR_UM
    qn = weighted_quantiles(speed[near], a[near], [0.5])
    qd = weighted_quantiles(d, a, [0.1, 0.5, 0.9])
    everywhere = np.ones_like(near)
    bands = {"portal_20um": dp < 20, "central_20um": dc < 20, "elsewhere": (dp >= 20) & (dc >= 20)}
    return {
        "area_um2": float(a.sum()), "cells": len(a),
        "interstitial_um_s": {"mean": mean(speed, everywhere), "p10": float(q[0]), "median": float(q[1]),
                              "p90": float(q[2]), "p98": float(weighted_quantiles(speed, a, [0.98])[0])},
        "near": {"area_pct": float(100 * a[near].sum() / a.sum()), "mean": mean(speed, near),
                 "median": float(qn[0])},
        "within_20um": {"portal": mean(speed, bands["portal_20um"]), "central": mean(speed, bands["central_20um"])},
        "below_10_um_s_area_pct": float(100 * a[speed < 10].sum() / a.sum()),
        "in_69_180_um_s_area_pct": float(100 * a[(speed >= 69) & (speed <= 180)].sum() / a.sum()),
        "portal_plus_central_um": {"p10": float(qd[0]), "median": float(qd[1]), "p90": float(qd[2])},
        "grad_p_Pa_um": {"mean": mean(gradient, everywhere), **{k: mean(gradient, m) for k, m in bands.items()}},
        "pressure_Pa": {"min": float(run.pressure.min()), "max": float(run.pressure.max()),
                        **dict(zip(("p10", "median", "p90"), map(float, weighted_quantiles(
                            run.pressure[run.cells].mean(axis=1), a, [0.1, 0.5, 0.9])), strict=True))},
    }


def front(run: Run, field: str) -> list:
    """Per output time: the tissue above 0.5 mM (cell means), the mean at the central wall nodes, the extremes and
    the total mass."""
    summary = run.summary["fields"][field]
    central = run.nodes(CENTRAL)
    out = []
    for index, t in enumerate(summary["times"]):
        c = run.field(field, index)
        above = c[run.cells].mean(axis=1) > 0.5
        out.append({"t": t, "area_above_half_pct": float(100 * run.areas[above].sum() / run.areas.sum()),
                    "central_wall_mean": float(c[central].mean()), "min": float(c.min()), "max": float(c.max()),
                    "total_mass": summary["total_mass"][index]})
    return out


def main() -> None:
    """Print or write the numbers of one run."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, required=True, help="the output folder of the run")
    parser.add_argument("--study", type=Path, required=True, help="the study file of the run")
    parser.add_argument("--mesh", type=Path, required=True, help="the slideviz mesh record (.json)")
    parser.add_argument("--scene", type=Path, required=True, help="the scene of the run (for the direction map)")
    parser.add_argument("--field", default="sucrose", help="the tracer field [sucrose]")
    parser.add_argument("--out", type=Path, required=True, help="the JSON file to write")
    args = parser.parse_args()
    run = Run(args.run, args.study, args.mesh)
    direction = DirectionMap(args.scene, run.study["perfusion"]["direction"]["map"])
    seconds = args.run / "seconds"
    numbers = {"run": args.run.name, "seconds": float(seconds.read_text()) if seconds.exists() else None,
               "flows": flows(run, direction), "balance": balances(run, direction, args.field),
               "velocity": velocity(run), "front": front(run, args.field)}
    args.out.write_text(json.dumps(numbers, indent=1) + "\n")


if __name__ == "__main__":
    main()
