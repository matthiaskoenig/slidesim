"""Compare coupled runs at several split steps with a reference run (scripts/bench/split_step.sh).

Usage: python3 scripts/bench/split_step.py REFERENCE RUN...

Each argument is the output folder of a run on one process, `<name>.study.json` beside it. All runs
are of the same study and mesh, and their cells are in the same order (checked by the centroids).
Prints Markdown tables:

- uptake: the amount each field received from the cells since t = 0 (`cells.exchanged` of
  summary.json) at the end, and its relative error against the reference, at the end and the
  largest over the output times;
- cell states: per state the difference to the reference at every output time after 0, scaled by
  the largest magnitude of the reference's state over the cells at that time; the largest and
  the mean over the cells and times;
- necrosis: the necrotic share (the area of the cells with `necrosis > 0.5` over the area of all
  cells) at the end, its difference to the reference, and the Dice coefficient of the necrotic
  cells against the reference (area weighted) at the end and the smallest over the output times
  at which either run has necrotic cells;
- observed orders between successive split steps h1 > h2: from the errors against the reference,
  ln(e1 / e2) / ln(h1 / h2), and from the differences of successive runs where the ratio of
  successive split steps is constant;
- cost: the phases of summary.json, the steps of the cells and the wall time (wall.txt).
"""

from __future__ import annotations

import itertools
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

TYPES = {"Float64": "<f8", "Float32": "<f4", "Int32": "<i4", "Int64": "<i8", "UInt8": "u1"}
ARRAY = re.compile(r'type="(\w+)" Name="([^"]+)"\s+NumberOfComponents="(\d+)" format="appended" offset="(\d+)"')
STATES = ["S_ext", "P_ext", "S", "P", "T", "necrosis"]
NECROTIC = 0.5


def read_vtu(path: Path) -> dict[str, np.ndarray]:
    """The arrays of a VTU file of slidesim with one piece (PETSc's raw appended data, 64 bit headers)."""
    raw = path.read_bytes()
    start = raw.index(b"<AppendedData")
    head = raw[:start].decode()
    if head.count("<Piece ") != 1:
        sys.exit(f"{path}: one piece expected (a run on one process)")
    base = raw.index(b"_", start) + 1
    arrays = {}
    for kind, name, components, at in ARRAY.findall(head):
        o = base + int(at)
        n = int.from_bytes(raw[o : o + 8], "little")
        values = np.frombuffer(raw[o + 8 : o + 8 + n], dtype=TYPES[kind])
        arrays[name] = values.reshape(-1, int(components)) if int(components) > 1 else values
    return arrays


class Run:
    def __init__(self, folder: Path) -> None:
        self.name = folder.name
        self.summary = json.loads((folder / "summary.json").read_text())
        study = json.loads(folder.with_name(folder.name + ".study.json").read_text())
        self.split = float(study["cells"]["split_step"])
        self.dt = float(study["time"]["dt"])
        self.times = [float(t) for t in study["time"]["outputs"]]
        self.cells = [read_vtu(folder / f"cells_{i}.vtu") for i in range(len(self.times))]
        wall = (folder / "wall.txt").read_text().split() if (folder / "wall.txt").exists() else ["nan"]
        self.wall = float(wall[0])
        mesh = self.cells[0]
        p = mesh["Position"][:, :2]
        tri = mesh["connectivity"].reshape(-1, 3)
        a, b, c = p[tri[:, 0]], p[tri[:, 1]], p[tri[:, 2]]
        self.area = 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
        self.centroid = (a + b + c) / 3.0

    def exchanged(self, field: str) -> np.ndarray:
        return np.array(self.summary["cells"]["exchanged"][field])

    def necrotic(self, i: int) -> np.ndarray:
        return self.cells[i]["necrosis"] > NECROTIC

    def share(self, i: int) -> float:
        return float(self.area[self.necrotic(i)].sum() / self.area.sum())


def dice(area: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    total = area[a].sum() + area[b].sum()
    return 1.0 if total == 0 else float(2.0 * area[a & b].sum() / total)


def order(e1: float, e2: float, h1: float, h2: float) -> str:
    if e1 <= 0 or e2 <= 0:
        return "-"
    return f"{math.log(e1 / e2) / math.log(h1 / h2):.2f}"


def main() -> None:
    ref = Run(Path(sys.argv[1]))
    runs = [Run(Path(a)) for a in sys.argv[2:] if Path(a).resolve() != Path(sys.argv[1]).resolve()]
    runs.sort(key=lambda r: -r.split)
    for r in runs:
        if r.times != ref.times or not np.allclose(r.centroid, ref.centroid):
            sys.exit(f"{r.name}: other output times or cells than {ref.name}")
    fields = sorted(ref.summary["cells"]["exchanged"])
    last = len(ref.times) - 1
    print(
        f"Reference {ref.name}: split step {ref.split:g} s, transport step {ref.dt:g} s, "
        f"end {ref.times[-1]:g} s, {len(ref.area)} cells, outputs {len(ref.times)}\n"
    )

    print(
        "Uptake: exchanged amount at the end (mM*um^2), relative error against the reference at the end "
        "and the largest over the output times\n"
    )
    print("| run | split (s) | dt (s) | " + " | ".join(f"{f} end | {f} err end | {f} err max" for f in fields) + " |")
    print("|---" * (3 + 3 * len(fields)) + "|")
    uptake_err = {}
    for r in [*runs, ref]:
        cols = []
        for f in fields:
            x, xr = r.exchanged(f), ref.exchanged(f)
            err = np.abs(x[1:] - xr[1:]) / np.abs(xr[1:])
            uptake_err[(r.name, f)] = float(err[-1])
            cols.append(f"{x[-1]:.6g} | {err[-1]:.2e} | {err.max():.2e}")
        print(f"| {r.name} | {r.split:g} | {r.dt:g} | " + " | ".join(cols) + " |")

    print(
        "\nCell states: difference to the reference scaled by the reference's largest magnitude over the "
        "cells at each output time, the largest / the mean over the cells and output times after 0\n"
    )
    print("| run | " + " | ".join(STATES) + " |")
    print("|---" * (1 + len(STATES)) + "|")
    for r in runs:
        cols = []
        for s in STATES:
            diffs = []
            for i in range(1, len(ref.times)):
                xr = ref.cells[i][s]
                scale = np.abs(xr).max()
                d = np.abs(r.cells[i][s] - xr)
                diffs.append(d / scale if scale > 0 else d)
            d = np.concatenate(diffs)
            cols.append(f"{d.max():.1e} / {d.mean():.1e}")
        print(f"| {r.name} | " + " | ".join(cols) + " |")

    print(
        f"\nNecrosis (cells with necrosis > {NECROTIC}, area weighted): share at the end, its difference to "
        "the reference, Dice against the reference at the end and the smallest over the output times\n"
    )
    print("| run | share | share - ref | Dice end | Dice min |")
    print("|---|---|---|---|---|")
    share_err = {}
    for r in [*runs, ref]:
        dices = [
            dice(ref.area, r.necrotic(i), ref.necrotic(i))
            for i in range(1, len(ref.times))
            if r.necrotic(i).any() or ref.necrotic(i).any()
        ]
        share_err[r.name] = abs(r.share(last) - ref.share(last))
        dice_min = f"{min(dices):.4f}" if dices else "-"
        print(
            f"| {r.name} | {r.share(last):.4f} | {r.share(last) - ref.share(last):+.4f} | "
            f"{dice(ref.area, r.necrotic(last), ref.necrotic(last)):.4f} | {dice_min} |"
        )
    print(
        "\nNecrotic share of the reference over time: "
        + ", ".join(f"{t / 3600:g} h {ref.share(i):.4f}" for i, t in enumerate(ref.times))
    )

    print(
        "\nObserved orders between successive split steps, from the errors against the reference "
        "(uptake at the end per field, necrotic share)\n"
    )
    print("| h1 -> h2 (s) | " + " | ".join(f"{f} uptake" for f in fields) + " | necrotic share |")
    print("|---" * (2 + len(fields)) + "|")
    for r1, r2 in itertools.pairwise(runs):
        cols = [order(uptake_err[(r1.name, f)], uptake_err[(r2.name, f)], r1.split, r2.split) for f in fields]
        cols.append(order(share_err[r1.name], share_err[r2.name], r1.split, r2.split))
        print(f"| {r1.split:g} -> {r2.split:g} | " + " | ".join(cols) + " |")
    chain = [*runs, ref]
    rows = []
    for r1, r2, r3 in zip(chain, chain[1:], chain[2:]):
        if math.isclose(r1.split / r2.split, r2.split / r3.split):
            cols = []
            for f in fields:
                d1 = abs(r1.exchanged(f)[-1] - r2.exchanged(f)[-1])
                d2 = abs(r2.exchanged(f)[-1] - r3.exchanged(f)[-1])
                cols.append(order(d1, d2, r1.split, r2.split))
            rows.append(f"| {r1.split:g}, {r2.split:g}, {r3.split:g} | " + " | ".join(cols) + " |")
    if rows:
        print(
            "\nObserved orders from the differences of successive runs at a constant ratio of the split steps "
            "(uptake at the end)\n"
        )
        print("| h1, h2, h3 (s) | " + " | ".join(f"{f} uptake" for f in fields) + " |")
        print("|---" * (1 + len(fields)) + "|")
        print("\n".join(rows))

    print("\nCost: phases of summary.json (s), steps of the cells, wall time (s), against the run at split step 60 s\n")
    print("| run | setup | pressure | transport | exchange | cells | output | cell steps | wall | wall / wall(60 s) |")
    print("|---" * 10 + "|")
    base = next((r.wall for r in [*runs, ref] if r.split == 60), float("nan"))
    for r in [*runs, ref]:
        p = r.summary["phases"]
        print(
            f"| {r.name} | {p['setup']:.1f} | {p['pressure']:.1f} | {p['transport']:.1f} | {p['exchange']:.1f} | "
            f"{p['cells']:.1f} | {p['output']:.1f} | {r.summary['cells']['steps']} | {r.wall:.1f} | "
            f"{r.wall / base:.2f} |"
        )


if __name__ == "__main__":
    main()
