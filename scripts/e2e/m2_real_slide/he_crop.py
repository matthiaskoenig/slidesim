"""Crop the H&E slide over a mesh field, at the pyramid level nearest to 1 um/px, for the flow figure.

Writes an npz with `rgb` (rows, cols, 3) and `extent` ([left, right, bottom, top] um, matplotlib's imshow
extent for y pointing down) and `um_per_px`. The field is the bounding box of the slideviz mesh (x along the
columns, y along the rows, um on the level 0 frame). Runs in the uv environment of slideviz (it reads the slide
with slideviz.io.reader):

    uv run --project ~/git/slideviz python he_crop.py --slides SLIDEVIZ_DATA --slide mouse_apap_000mg_m1_he \
        --mesh OUT/mesh/<mesh>.json --out OUT/analysis/he.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from slideviz.io.reader import find_slide, open_slide

# The finest level at or above this size (um per pixel) is used.
TARGET_UM_PER_PX = 0.85


def main() -> None:
    """Write the crop."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--slides", type=Path, required=True, help="the folder of the slides (read only)")
    parser.add_argument("--slide", required=True, help="the stem of the H&E slide")
    parser.add_argument("--mesh", type=Path, required=True, help="the slideviz mesh record (.json)")
    parser.add_argument("--out", type=Path, required=True, help="the npz file to write")
    args = parser.parse_args()

    record = json.loads(args.mesh.read_text())
    view_name = next(f for f in record["files"] if f.endswith("_view.npz"))
    with np.load(args.mesh.with_name(view_name)) as view:
        points = view["points_um"][:, :2]
    (x0, y0), (x1, y1) = points.min(axis=0), points.max(axis=0)

    info, levels = open_slide(find_slide(args.slides, args.slide))
    width = levels[0].shape[1]
    for level in levels:
        um = info.pixel_size_um * width / level.shape[1]
        if um >= TARGET_UM_PER_PX:
            break
    r0, r1 = int(np.floor(y0 / um)), int(np.ceil(y1 / um)) + 1
    c0, c1 = int(np.floor(x0 / um)), int(np.ceil(x1 / um)) + 1
    rgb = np.asarray(level[r0:r1, c0:c1])[..., :3]
    # pixel (r, c) is centred at (c um, r um)
    extent = [(c0 - 0.5) * um, (c1 - 0.5) * um, (r1 - 0.5) * um, (r0 - 0.5) * um]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, rgb=rgb, extent=np.array(extent), um_per_px=um)
    print(f"{args.out}: {rgb.shape[1]} x {rgb.shape[0]} px at {um:.3f} um/px")


if __name__ == "__main__":
    main()
