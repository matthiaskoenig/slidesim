"""The tables of docs/results/2026-11-m2-real-slide.md from the analysis JSON files of analyze.py, as Markdown.

    python report.py OUT/analysis/baseline_r2.json OUT/analysis/calibrated_r2.json ... > report.md

Needs the standard library only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# The tags without condition for the tracer whose outflow is the leak through walls without flow (the central walls
# are the outlet).
OUTLET = "central_wall"


def whole(x: float) -> str:
    """x rounded to a whole number, without the sign of a value that rounds to zero ("0", not "-0")."""
    return str(round(x) + 0)


def flows(runs: dict) -> list[str]:
    lines = [("| Run | portal_wall | central_wall | sum / largest | portal pieces (all inflow) | central pieces "
             "(all outflow) | pieces vs tag_flows |"), "|---|---|---|---|---|---|---|"]
    for name, r in runs.items():
        f = r["flows"]
        p, c = f["pieces"]["portal_wall"], f["pieces"]["central_wall"]
        lines.append(f"| {name} | {f['tag_flows']['portal_wall']:.1f} | {f['tag_flows']['central_wall']:.1f} | "
                     f"{f['sum_over_largest']:.1e} | {p['count']} ({'yes' if f['every_portal_piece_inflow'] else 'NO'})"
                     f" | {c['count']} ({'yes' if f['every_central_piece_outflow'] else 'NO'}) | "
                     f"{max(abs(p['sum_vs_tag_flow']), abs(c['sum_vs_tag_flow'])):.1e} |")
    return lines


def pieces(runs: dict) -> list[str]:
    lines = []
    for tag in ("portal_wall", "central_wall"):
        names = list(runs)
        first = runs[names[0]]["flows"]["pieces"][tag]["pieces"]
        lines += [f"| {tag} piece (centre, um) | edges | " + " | ".join(names) + " |",
                  "|---|---|" + "---|" * len(names)]
        for i, piece in enumerate(first):
            values = [runs[n]["flows"]["pieces"][tag]["pieces"][i]["inflow_um2_s"] for n in names]
            centre = ", ".join(f"{x:.0f}" for x in piece["centre_um"])
            lines.append(f"| ({centre}) | {piece['edges']} | " + " | ".join(f"{v:.1f}" for v in values) + " |")
        lines.append("")
    return lines


def balance(runs: dict) -> list[str]:
    lines = [("| Run | t (s) | storage_rate | portal_wall adv. + diff. | central_wall adv. | leak: "
             "vessel_wall / field_cut / section_edge (% of portal flow) | advection_defect (%) | closure |"),
             "|---|---|---|---|---|---|---|---|"]
    for name, r in runs.items():
        for b in r["balance"]:
            portal = b["inflow"]["portal_wall"]
            free = b["free_outflow"]
            leak = " / ".join(f"{free[t]['pct_of_portal_flow']:.2f}" if t in free else "-"
                              for t in ("vessel_wall", "field_cut", "section_edge"))
            lines.append(f"| {name} | {b['t']:g} | {whole(b['storage_rate'])} | {whole(portal['advective'])} + "
                         f"{whole(portal['diffusive'])} | {whole(b['inflow'][OUTLET]['advective'])} | {leak} | "
                         f"{b['defect_pct_of_portal_flow']:.1f} | {b['closure_over_largest_term']:.0e} |")
    return lines


def balance_terms(runs: dict) -> list[str]:
    """The terms of the balance in mM*um^2/s, per run and output time."""
    lines = [("| Run | t (s) | storage_rate | portal_wall advective + diffusive | central_wall advective | "
              "vessel_wall / field_cut / section_edge advective | advection_defect |"),
             "|---|---|---|---|---|---|---|"]
    for name, r in runs.items():
        for b in r["balance"]:
            inflow = b["inflow"]
            portal = inflow["portal_wall"]
            walls = " / ".join(whole(inflow[t]["advective"]) if t in inflow else "-"
                               for t in ("vessel_wall", "field_cut", "section_edge"))
            lines.append(f"| {name} | {b['t']:g} | {whole(b['storage_rate'])} | {whole(portal['advective'])} + "
                         f"{whole(portal['diffusive'])} | {whole(inflow[OUTLET]['advective'])} | {walls} | "
                         f"{whole(b['advection_defect'])} |")
    return lines


def leak_pieces(runs: dict) -> list[str]:
    lines = [("| Run | t (s) | tag | largest pieces: outflow-weighted centre (um), outflow (mM*um^2/s) | sum: recomputed / "
             "summary |"),
             "|---|---|---|---|---|"]
    for name, r in runs.items():
        for b in r["balance"]:
            for tag, free in b["free_outflow"].items():
                if tag == OUTLET or free["outflow"] <= 0.005 * abs(b["inflow"]["portal_wall"]["advective"]):
                    continue
                top = "; ".join(f"({p['outflow_centre_um'][0]:.0f}, {p['outflow_centre_um'][1]:.0f}) {p['outflow']:.1f}"
                                for p in free["pieces"][:2])
                lines.append(f"| {name} | {b['t']:g} | {tag} | {top} | {free['outflow']:.2f} / "
                             f"{-free['summary_advective']:.2f} |")
    return lines


def velocity(runs: dict) -> list[str]:
    lines = [("| Run | mean | p10 | median | p90 | p98 | near: mean (median) | within 20 um of portal / central | "
             "area < 10 um/s | area in 69-180 um/s | mean grad p |"), "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in runs.items():
        v = r["velocity"]
        u = v["interstitial_um_s"]
        lines.append(f"| {name} | {u['mean']:.1f} | {u['p10']:.1f} | {u['median']:.1f} | {u['p90']:.1f} | "
                     f"{u['p98']:.1f} | {v['near']['mean']:.1f} ({v['near']['median']:.1f}) | "
                     f"{v['within_20um']['portal']:.0f} / {v['within_20um']['central']:.0f} | "
                     f"{v['below_10_um_s_area_pct']:.0f} % | {v['in_69_180_um_s_area_pct']:.1f} % | "
                     f"{v['grad_p_Pa_um']['mean']:.2f} |")
    return lines


def front(runs: dict) -> list[str]:
    lines = ["| Run | t (s) | tissue > 0.5 mM | mean at central wall nodes | min | max | total mass |",
             "|---|---|---|---|---|---|---|"]
    for name, r in runs.items():
        for f in r["front"]:
            lines.append(f"| {name} | {f['t']:g} | {f['area_above_half_pct']:.1f} % | {f['central_wall_mean']:.3f} | "
                         f"{f['min']:.1e} | {f['max']:.6f} | {whole(f['total_mass'])} |")
    return lines


def main() -> None:
    """Print the tables."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("analyses", nargs="+", type=Path, help="JSON files of analyze.py")
    args = parser.parse_args()
    runs = {p.stem: json.loads(p.read_text()) for p in args.analyses}
    for title, table in (("Flows", flows), ("Flow per wall piece (um^2/s, positive into the domain)", pieces),
                         ("Mass balance of the tracer (mM*um^2/s; leak and defect in % of the portal flow)",
                          balance),
                         ("Terms of the mass balance of the tracer (mM*um^2/s)", balance_terms),
                         ("Outflow through walls without flow, by piece", leak_pieces),
                         ("Interstitial velocity (um/s)", velocity), ("Tracer front", front)):
        print(f"## {title}\n")
        print("\n".join(table(runs)) + "\n")
    print("## Pressure (Pa)\n")
    for name, r in runs.items():
        p = r["velocity"]["pressure_Pa"]
        d = r["velocity"]["portal_plus_central_um"]
        print(f"- {name}: min {p['min']:.4f}, max {p['max']:.4f}, median over the tissue {p['median']:.0f} "
              f"(p10 {p['p10']:.0f}, p90 {p['p90']:.0f}); portal + central distance median {d['median']:.0f} um "
              f"(p10 {d['p10']:.0f}, p90 {d['p90']:.0f}); run {r['seconds']} s")


if __name__ == "__main__":
    main()
