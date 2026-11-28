"""The comparison of a gmsh mesh with slidesim's mesh of the same field, as a Markdown table (scripts/e2e/m4_meshing.sh).

usage: compare.py NAME GMSH_OUT RUST_OUT POLYGONS GMSH_RECORD GMSH_MINSICN

GMSH_OUT and RUST_OUT are `slidesim mesh` outputs of the gmsh scene and of the polygons scene, POLYGONS the polygons/1
file of the latter, GMSH_RECORD the build_mesh.py record and GMSH_MINSICN gmsh's own minSICN of the .msh, unrounded
(gmsh_sicn.py; the record rounds it to 3 digits). Criteria: the same tag names, tissue area within 0.5 %, the length of
each edge tag within 1 %, triangles within 25 %, the min SICN of slidesim's mesh at least gmsh's less 0.1, and the min
SICN slidesim computes on the gmsh mesh equal to gmsh's within 1e-6 (and to the record's within its rounding).
"""

import json
import sys
from pathlib import Path

import numpy as np
from shapely.geometry import shape


def load(out: Path) -> dict:
    mesh = json.loads((out / "mesh.json").read_text())
    quality = json.loads((out / "quality.json").read_text())
    v, t = np.array(mesh["vertices"]), np.array(mesh["triangles"])
    p = v[t]
    u, w = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]
    area = 0.5 * np.abs(u[:, 0] * w[:, 1] - u[:, 1] * w[:, 0])
    e = np.array(mesh["edges"])
    length = np.linalg.norm(v[e[:, 0]] - v[e[:, 1]], axis=1)
    tags = mesh["tags"]
    edge_len = {tags[k]["name"]: float(length[np.array(mesh["edge_tags"]) == k].sum())
                for k in range(len(tags)) if tags[k]["dim"] == 1}
    return {"names": sorted(t["name"] for t in tags), "area": float(area.sum()), "edges": edge_len,
            "triangles": len(t), "min_sicn": quality["min_sicn"], "min_angle": quality["min_angle_deg"],
            "mean_sicn": quality["mean_sicn"]}


name, gmsh_out, rust_out, polygons, record, gmsh_minsicn = sys.argv[1:7]
g, r = load(Path(gmsh_out)), load(Path(rust_out))
collection = json.loads(Path(polygons).read_text())
poly_area = sum(shape(f["geometry"]).area for f in collection["features"] if f["properties"]["kind"] == "area")
rec = json.loads(Path(record).read_text())["checks"]
gmsh_minsicn = float(gmsh_minsicn)
rows, ok = [], True


def check(what, gv, rv, diff, passed):
    global ok
    ok &= passed
    rows.append((what, gv, rv, diff, "pass" if passed else "FAIL"))


check("tag names", ", ".join(g["names"]), ", ".join(r["names"]), "", g["names"] == r["names"])
d = r["area"] / g["area"] - 1
check("tissue area (um2)", f"{g['area']:.1f}", f"{r['area']:.1f}", f"{100 * d:+.3f} %", abs(d) <= 0.005)
for tag in sorted(g["edges"]):
    gl, rl = g["edges"][tag], r["edges"].get(tag, 0.0)
    d = rl / gl - 1 if gl else float("inf")
    check(f"{tag} length (um)", f"{gl:.1f}", f"{rl:.1f}", f"{100 * d:+.3f} %", abs(d) <= 0.01)
d = r["triangles"] / g["triangles"] - 1
check("triangles", g["triangles"], r["triangles"], f"{100 * d:+.1f} %", abs(d) <= 0.25)
check("min SICN (Rust >= gmsh - 0.1)", f"{g['min_sicn']:.4f}", f"{r['min_sicn']:.4f}",
      f"{r['min_sicn'] - g['min_sicn']:+.4f}", r["min_sicn"] >= g["min_sicn"] - 0.1)
check("min SICN of the gmsh mesh = gmsh's", f"{gmsh_minsicn:.7f} (record {rec['quality_min']})",
      f"{g['min_sicn']:.7f} (slidesim)", f"{g['min_sicn'] - gmsh_minsicn:+.1e}",
      abs(g["min_sicn"] - gmsh_minsicn) <= 1e-6 and abs(rec["quality_min"] - gmsh_minsicn) <= 5e-4)
print(f"### {name}\n")
print(f"Polygon area {poly_area:.1f} um2 (Rust mesh {100 * (r['area'] / poly_area - 1):+.2e} %). "
      f"Min angle gmsh {g['min_angle']:.2f} deg, Rust {r['min_angle']:.2f} deg; mean SICN gmsh "
      f"{g['mean_sicn']:.3f}, Rust {r['mean_sicn']:.3f}.\n")
print("| Criterion | gmsh | Rust | Difference | |\n| --- | --- | --- | --- | --- |")
for row in rows:
    print("| " + " | ".join(str(x) for x in row) + " |")
print(f"\n{name}: {'PASS' if ok else 'FAIL'}\n")
