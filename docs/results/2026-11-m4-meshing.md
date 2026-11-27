# Milestone 4: meshing real slides

Status: 2026-11-27. slidesim branch `feat/m4-lobules`, slideviz branch `feat/tagged-polygons`
(task 8 of the milestone 4 plan and its fix round). Verdict: slidesim's meshes of the real fields
pass all six criteria against gmsh's, after the size field was capped by the length of short
boundary segments. slideviz retires `build_mesh.py` (gmsh); slidesim meshes the polygons.

## Question

slideviz writes tagged polygons (`polygons/1`: areas and tagged edge chains), and slidesim meshes
them itself ([Meshing](../methods/meshing.md)). Can slidesim's mesher replace slideviz's gmsh
step (`mesh/build_mesh.py`)? The plan's criteria, on the field of milestone 2 (`000mg_m1`) and of
milestone 3 (`281mg_m1`), 1500 um, element sizes 5:20 um, with `slidesim mesh` on the gmsh scene
and on the polygons scene:

1. the same tag names;
2. the tissue area within 0.5 %;
3. the length of each edge tag within 1 %;
4. the number of triangles within 25 %;
5. the min SICN of slidesim's mesh at least gmsh's less 0.1;
6. the min SICN of the gmsh mesh (as slidesim computes it) equal to slideviz's record within
   1e-6, which pins the SICN definition.

## Method

`scripts/e2e/m4_meshing.sh` (step `compare`) per field:

- slideviz `zonation_layers.py`, `mesh_inputs.py` (the 1500 um square of the earlier milestones:
  `000mg_m1` centred at H&E level-0 pixel (10114, 16388), `281mg_m1` at (9336, 30608)) and
  `mesh/polygons.py --tolerance-um 2` in both variants: `follow` (areas `surviving` and
  `necrotic`) and `plain` (one area `tissue`);
- gmsh: `build_mesh.py --variant plain --size-um 5:20` on the `follow` polygons (it joins them
  into one area, as milestone 3 did), and `write_scene.py --mesh` for its scene;
- slidesim: `write_scene.py --polygons <plain polygons> --size-um 5:20`, the scene slidesim meshes;
- `slidesim mesh` (release build, one process) of both scenes, and `m4_meshing/compare.py` on the
  two outputs.

Criterion 6 deviates in one point: the record of `build_mesh.py` rounds the min SICN to 3 digits
(`checks.quality_min`), so 1e-6 cannot be checked against it. `m4_meshing/gmsh_sicn.py` reads the
`.msh` with gmsh and computes gmsh's own `minSICN` of its triangles unrounded; slidesim's value on
the gmsh mesh is checked against that within 1e-6, and the record against it within its rounding
(5e-4).

## Results

The tables are those of the mesher with the cap by the segment lengths (below); the first run,
without it, is under [Short boundary segments](#short-boundary-segments).

### 000mg_m1

Polygon area 2207823.9 um^2 (slidesim's mesh: -2.2e-16 relative). Min angle gmsh 21.62 deg,
slidesim 23.00 deg; mean SICN gmsh 0.980, slidesim 0.957; edge length / size of slidesim's mesh
0.40 to 1.66.

| Criterion | gmsh | slidesim | Difference | |
| --- | --- | --- | --- | --- |
| tag names | central_wall, field_cut, portal_wall, section_edge, tissue, vessel_wall | the same | | pass |
| tissue area (um^2) | 2207823.9 | 2207823.9 | -0.000 % | pass |
| central_wall length (um) | 2579.6 | 2579.6 | -0.000 % | pass |
| field_cut length (um) | 5917.5 | 5917.5 | -0.000 % | pass |
| portal_wall length (um) | 1084.3 | 1084.3 | -0.000 % | pass |
| section_edge length (um) | 27.2 | 27.2 | -0.000 % | pass |
| vessel_wall length (um) | 200.3 | 200.3 | -0.000 % | pass |
| triangles | 42561 | 38676 | -9.1 % | pass |
| min SICN (slidesim >= gmsh - 0.1) | 0.5534 | 0.5366 | -0.0169 | pass |
| min SICN of the gmsh mesh = gmsh's | 0.5534131 (record 0.553) | 0.5534131 | -1.0e-13 | pass |

### 281mg_m1

Polygon area 2164493.7 um^2 (slidesim's mesh: +8.9e-16 relative). Min angle gmsh 21.80 deg,
slidesim 21.01 deg; mean SICN gmsh 0.979, slidesim 0.958; edge length / size of slidesim's mesh
0.46 to 1.58.

| Criterion | gmsh | slidesim | Difference | |
| --- | --- | --- | --- | --- |
| tag names | central_wall, field_cut, portal_wall, section_edge, tissue, vessel_wall | the same | | pass |
| tissue area (um^2) | 2164493.7 | 2164493.7 | +0.000 % | pass |
| central_wall length (um) | 3334.9 | 3334.9 | -0.000 % | pass |
| field_cut length (um) | 5910.5 | 5910.5 | +0.000 % | pass |
| portal_wall length (um) | 828.6 | 828.6 | -0.000 % | pass |
| section_edge length (um) | 12.4 | 12.4 | +0.000 % | pass |
| vessel_wall length (um) | 202.0 | 202.0 | -0.000 % | pass |
| triangles | 41252 | 37250 | -9.7 % | pass |
| min SICN (slidesim >= gmsh - 0.1) | 0.5570 | 0.5416 | -0.0154 | pass |
| min SICN of the gmsh mesh = gmsh's | 0.5569532 (record 0.557) | 0.5569532 | +9.6e-14 | pass |

The gmsh mesh of `281mg_m1` is the one of milestone 3 (41252 triangles, as its record then).
`slidesim mesh` took 1.2 s (the whole command, 63 MB), gmsh's `generate` 0.43 s.

### Verdict

All six criteria pass on both fields. Both meshers keep the same polygons: the areas and the
lengths of every tag agree to rounding, slidesim's SICN is gmsh's to 1e-13, slidesim makes 9 to
10 % fewer triangles and its worst triangle is 0.015 to 0.017 below gmsh's in SICN, and its
meshes meet the mesher's own bounds (min angle 20 deg, SICN 0.4, edge ratios within [0.4, 2.0]).
**slideviz retires `build_mesh.py`** (and with it `mesh/fields.py`, `write_scene.py --mesh`
and gmsh and DOLFINx in its mesh environment) in commit `a5d287b`: slidesim meshes the polygons.

### Short boundary segments

The polygons follow the lumen and tissue masks of H&E level 3, 1.77 um per pixel, and the 2 um
simplification keeps some single-pixel steps: 172 (`000mg_m1`) and 166 (`281mg_m1`) of about
1400 boundary edges are shorter than 2 um, the shortest 1.25 um. The mesher keeps every input
vertex and only splits segments, and the gap cap of the size field skips neighbouring segments
(which share a point), so a 1.77 um step kept the full 5 um size: a frozen edge of metric length
0.35, with a triangle whose third vertex sat 5 to 7 um away. The first run, without a cap by the
segment lengths:

| Field | Min SICN slidesim | Min SICN gmsh | Min angle slidesim | Edge length / size |
| --- | --- | --- | --- | --- |
| `000mg_m1` | 0.4062 | 0.5534 | 14.04 deg | 0.35 to 1.65 |
| `281mg_m1` | 0.4075 | 0.5570 | 14.04 deg | 0.27 to 1.74 |

failing criterion 5 by about 0.05. Its worst triangles, all on a 1.77 um boundary edge
(coordinates in um on the H&E level-0 frame, x along the columns, y along the rows):

| Field | Centroid (x, y) | Edges (um) | SICN slidesim | Min SICN of gmsh within 10 um |
| --- | --- | --- | --- | --- |
| `000mg_m1` | (4055.8, 2395.7) | 5.97, 4.84, 1.77 | 0.406 | 0.784 |
| `000mg_m1` | (4120.4, 2725.6) | 7.07, 1.77, 7.29 | 0.408 | 0.813 |
| `000mg_m1` | (4053.4, 2340.5) | 1.77, 4.98, 3.75 | 0.446 | 0.627 |
| `281mg_m1` | (6697.2, 2113.7) | 7.07, 1.77, 7.29 | 0.408 | 0.693 |
| `281mg_m1` | (6431.9, 2528.1) | 2.47, 3.95, 1.77 | 0.414 | 0.693 |
| `281mg_m1` | (6606.9, 1366.4) | 4.40, 5.59, 1.77 | 0.416 | 0.799 |

gmsh's size here is the Threshold field alone: `build_mesh.py` turns off
`Mesh.MeshSizeExtendFromBoundary`, `Mesh.MeshSizeFromPoints` and `Mesh.MeshSizeFromCurvature`, so
gmsh does not grade the size at short edges either. Its better triangles there come from its 2D
algorithm (Frontal-Delaunay), which places each new vertex relative to the length of the front
edge it grows from, so the first layer on a 1.77 um edge is about that size.

The fix is a rule of slidesim's size field ([Meshing](../methods/meshing.md)): at both ends of a
boundary segment of length \(l\) the size is capped at \(1.5\,l\) (not below \(h_\min / 4\)),
growing by 0.3 per um away from them, the grading of the gap cap. The triangles beside a short
frozen segment then take its length (an isosceles triangle on it has SICN 0.89), and the boundary
splitting grades the neighbouring segments down towards it. Segments of nearly the size (a
polygonized circle of \(\lceil 2 \pi r / h \rceil\) segments) get no cap, so the lobule meshes
and the stored fixture mesh are unchanged bit for bit. The mesher test `pixel_steps_are_resolved`
(a wall of 1.25 um pixel steps under 5 um triangles) gave a min angle of 14.04 deg without the cap
and 25.3 deg, SICN 0.63, edge ratios 0.48 to 1.45 with it. Input segments shorter than
\(h_\min / 4\) stay outside the bounds (reported, not bounded).

## Example fields

Three fields of 500 um around a central vein with its portal fields are in the repository,
`testdata/studies/slide_fields` (CC BY 4.0, Matthias König lab, data from the APAP mouse liver
study): the polygons of variant `follow`, the scene (5:20 um) and the `zonation`, `cyp2e1` and
`necrosis` maps cropped to the field, with the SPT studies of milestone 3 on both paths. 1.5 MB
on disk together. The fields were chosen by a search over each section: squares fully in the
tissue with a central lumen of at least 500 um^2 near the centre and as many portal lumina of the
lumen mask as found. Portal lumina are rare in the masks (20 to 31 per section), so each field
has one to three.

| Field | Slide | Centre (row, col) | Necrotic | Tags | Size on disk |
| --- | --- | --- | --- | --- | --- |
| `000mg_m1` | `mouse_apap_000mg_m1_he` | (14120, 15016) | 0 % | central_wall, field_cut, portal_wall, surviving | 436 KB |
| `281mg_m1` | `mouse_apap_281mg_m1_he` | (11656, 28904) | 52 % | central_wall, field_cut, necrosis_border, necrotic, portal_wall, section_edge, surviving, vessel_wall | 516 KB |
| `500mg_m1` | `mouse_apap_500mg_m1_he` | (8072, 15016) | 43 % | central_wall, field_cut, necrosis_border, necrotic, portal_wall, surviving | 528 KB |

`slidesim mesh` of the scenes (the first run without the cap by the segment lengths in brackets):

| Field | Triangles | Vertices | Min angle | Min SICN | Mean SICN | Edge length / size |
| --- | --- | --- | --- | --- | --- | --- |
| `000mg_m1` | 6106 | 3229 | 23.24 deg (15.95) | 0.547 (0.458) | 0.955 | 0.46 to 1.53 (0.35 to 1.56) |
| `281mg_m1` | 13791 | 7208 | 23.03 deg (16.25) | 0.498 (0.419) | 0.948 | 0.43 to 1.63 (0.35 to 1.63) |
| `500mg_m1` | 17175 | 8961 | 22.44 deg (16.35) | 0.541 (0.447) | 0.944 | 0.41 to 1.67 (0.35 to 1.70) |

All three meet the bounds, including the necrosis borders of variant `follow` with corners down
to 26.6 deg (pixel steps of the necrosis map). The tests of `crates/slidesim` mesh all three
(`slide_field_examples_mesh`: portal and central walls, the area of the polygons to 1e-9, min
angle 20 deg, SICN 0.4, edge ratios within [0.4, 2.0]) and run the batched SPT study of
`281mg_m1` for 10 min (`slide_field_spt_runs`: the flows through the portal and central walls
equal and opposite, every mass balance closing).

## Reproduce

```bash
export SLIDEVIZ_DATA=/data/koeniglab/image.analysis/APAP/APAP_zarr  # read only
cd ~/git/slidesim
scripts/e2e/m4_meshing.sh  # OUT=target/e2e/m4_meshing; STEPS="fields compare"
# OUT/slide_fields/<animal>: the example fields; OUT/compare.md: the tables above
cargo nextest run -p slidesim --test cli -E 'test(slide_field)'
```

The example fields in the repository are the output of the `fields` step (their polygons and maps
are identical across runs). Their provenance attributes name the files they were made from with
their SHA-256, as `$SLIDEVIZ_DATA/...` and `$OUT/...`. The `compare` step needs the gmsh mesher
of slideviz, so a checkout before its retirement in `a5d287b` (`98b30a7`, `SLIDEVIZ_REPO`).
