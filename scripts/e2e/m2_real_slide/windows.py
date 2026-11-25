"""Tagged vessels per square window of a slide: the search behind the field of scripts/e2e/m2_real_slide.sh.

For every window of FIELD um (step 50 um) with at least 97 % tissue on the zonation grid, counts the lumina by
class in two layers: the connected pieces of `lumen_class` of zonation_layers.py (1 portal, 2 central, 3 other),
and the lumina of the lumen mask the mesh is built from (the predictions), each classed as mesh/build_mesh.py
classes it (majority of class 1 or 2 pixels within 10 um, else other). Also the CYP2E1 peaks and valleys
(lobules) per window. Writes JSON: slide totals, the best windows by the lumen_class count (at least 500 um
apart), and the counts of the windows at --centre. Runs in the uv environment of slideviz:

    uv run --project ~/git/slideviz python windows.py --slides SLIDEVIZ_DATA --slide mouse_apap_000mg_m1_he \
        --lobules OUT/zonation/<slide>_cyp2e1_lobules.npz --lumen SLIDEVIZ_DATA/predictions/<animal>_lumen.npz \
        --centre 10114 16388 --centre 14000 12000 --out OUT/analysis/windows.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from slideviz.io.reader import find_slide, slide_info

STEP_UM = 50.0
MIN_TISSUE = 0.97
RING_UM = 10.0  # build_mesh.py: the lumen class beside a lumen within this distance
LUMEN_MIN_UM2 = 500.0  # slideviz tissue.py: the smallest lumen of the mask
APART_UM = 500.0
BEST = 6


def lumen_class_pieces(lumen_class: np.ndarray, um: float, origin: np.ndarray) -> dict[int, np.ndarray]:
    """The centres ([row, col] um) of the connected pieces of each class of lumen_class."""
    centres = {}
    for k in (1, 2, 3):
        label, n = ndi.label(lumen_class == k)
        cm = np.array(ndi.center_of_mass(np.ones_like(label), label, range(1, n + 1))).reshape(-1, 2)
        centres[k] = cm * um + origin
    return centres


def mask_lumina(lumen: np.ndarray, lumen_class: np.ndarray, um: float, origin: np.ndarray) -> tuple:
    """The lumina of the mask (same grid as lumen_class): their centres ([row, col] um), classes and areas."""
    label, n = ndi.label(lumen)
    ring = int(np.ceil(RING_UM / um))
    classes, centres = [], []
    for i, box in enumerate(ndi.find_objects(label), 1):
        grown = tuple(slice(max(s.start - ring, 0), s.stop + ring) for s in box)
        mask = label[grown] == i
        near = lumen_class[grown][ndi.binary_dilation(mask, iterations=ring)]
        portal, central = (near == 1).sum(), (near == 2).sum()
        classes.append(1 if portal > central else 2 if central > portal else 3)
        cm = ndi.center_of_mass(mask)
        centres.append(((grown[0].start + cm[0]) * um + origin[0], (grown[1].start + cm[1]) * um + origin[1]))
    areas = ndi.sum(np.ones_like(label), label, range(1, n + 1)) * um**2
    return np.array(centres).reshape(-1, 2), np.array(classes), np.asarray(areas)


def count(points: np.ndarray, centre: np.ndarray, size: float) -> int:
    inside = (np.abs(points - centre) < size / 2).all(axis=1) if len(points) else np.zeros(0, bool)
    return int(inside.sum())


def main() -> None:
    """Write the window counts."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--slides", type=Path, required=True, help="the folder of the slides (read only)")
    parser.add_argument("--slide", required=True, help="the stem of the H&E slide (for its level 0 pixel size)")
    parser.add_argument("--lobules", type=Path, required=True, help="the _lobules.npz of zonation_layers.py")
    parser.add_argument("--lumen", type=Path, required=True, help="the lumen prediction (npz with 'lumen')")
    parser.add_argument("--field-um", type=float, default=1500.0, help="the window edge in um [1500]")
    parser.add_argument("--centre", type=float, nargs=2, action="append", default=[], metavar=("ROW", "COL"),
                        help="a window centre in H&E level 0 pixels to report (repeatable)")
    parser.add_argument("--out", type=Path, required=True, help="the JSON file to write")
    args = parser.parse_args()

    um0 = slide_info(find_slide(args.slides, args.slide)).pixel_size_um
    with np.load(args.lobules) as stored:
        lumen_class, position = stored["lumen_class"], stored["position"]
        um, origin = float(stored["working_um_per_px"]), np.asarray(stored["working_origin_um"], float)
        lobule_um, lobule_offset = float(stored["um_per_px"]), float(stored["offset_um"])
        peaks = stored["peaks"] * lobule_um + lobule_offset
        valleys = stored["valleys"] * lobule_um + lobule_offset
    with np.load(args.lumen) as stored:
        lumen = stored["lumen"]
    if lumen.shape != lumen_class.shape:
        raise ValueError(f"lumen mask {lumen.shape} and lumen_class {lumen_class.shape} are on different grids")
    pieces = lumen_class_pieces(lumen_class, um, origin)
    centres, classes, areas = mask_lumina(lumen, lumen_class, um, origin)
    size = args.field_um

    def counts(centre: np.ndarray) -> dict:
        return {"lumen_class": {name: count(pieces[k], centre, size)
                                for name, k in (("portal", 1), ("central", 2), ("other", 3))},
                "lumen_mask": {name: count(centres[classes == k], centre, size)
                               for name, k in (("portal", 1), ("central", 2), ("other", 3))},
                "cyp2e1_peaks": count(peaks, centre, size), "cyp2e1_valleys": count(valleys, centre, size)}

    tissue = np.isfinite(position).astype(float)
    n = round(size / um)
    fraction = ndi.uniform_filter(tissue, size=n, mode="constant")
    step = round(STEP_UM / um)
    rows = []
    for r0 in range(0, tissue.shape[0] - n, step):
        for c0 in range(0, tissue.shape[1] - n, step):
            f = fraction[r0 + n // 2, c0 + n // 2]
            if f < MIN_TISSUE:
                continue
            centre = origin + (np.array([r0, c0]) + n / 2) * um
            p, c = count(pieces[1], centre, size), count(pieces[2], centre, size)
            rows.append((p + c, min(p, c), centre, float(f)))
    rows.sort(key=lambda r: (r[0], r[1]), reverse=True)
    best = []
    for _, _, centre, f in rows:
        if all(np.abs(centre - b["centre_um"]).max() > APART_UM for b in best):
            best.append({"centre_um": centre, "centre_px": (centre / um0).round().astype(int).tolist(),
                         "tissue": round(f, 3), **counts(centre)})
        if len(best) == BEST:
            break
    for b in best:
        b["centre_um"] = [round(float(x), 1) for x in b["centre_um"]]
    out = {
        "windows": len(rows), "field_um": size, "level0_um_per_px": um0,
        "slide": {"lumen_class": {name: len(pieces[k]) for name, k in (("portal", 1), ("central", 2), ("other", 3))},
                  "lumen_mask": {name: int((classes == k).sum()) for name, k in (("portal", 1), ("central", 2),
                                                                                  ("other", 3))},
                  "lumen_mask_at_least_500um2": {name: int(((classes == k) & (areas >= LUMEN_MIN_UM2)).sum())
                                                 for name, k in (("portal", 1), ("central", 2), ("other", 3))},
                  "cyp2e1_peaks": len(peaks), "cyp2e1_valleys": len(valleys)},
        "best_by_lumen_class": best,
        "centres": [{"centre_px": [round(r), round(c)], **counts(np.array([r, c]) * um0)} for r, c in args.centre],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
