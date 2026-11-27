# Milestone 4: meshing real slides

Status: 2026-11-27. slidesim branch `feat/m4-lobules`, slideviz branch `feat/tagged-polygons`
(commit `98b30a7`, task 8 of the milestone 4 plan). Verdict: slidesim's meshes of the real fields
match gmsh's in tags, area, wall lengths and size, but their worst triangles are worse by more
than the margin. slideviz keeps `build_mesh.py` (gmsh) for now.

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

### 000mg_m1

Polygon area 2207823.9 um^2 (slidesim's mesh: -2.2e-16 relative). Min angle gmsh 21.62 deg,
slidesim 14.04 deg; mean SICN gmsh 0.980, slidesim 0.958.

| Criterion | gmsh | slidesim | Difference | |
| --- | --- | --- | --- | --- |
| tag names | central_wall, field_cut, portal_wall, section_edge, tissue, vessel_wall | the same | | pass |
| tissue area (um^2) | 2207823.9 | 2207823.9 | -0.000 % | pass |
| central_wall length (um) | 2579.6 | 2579.6 | -0.000 % | pass |
| field_cut length (um) | 5917.5 | 5917.5 | -0.000 % | pass |
| portal_wall length (um) | 1084.3 | 1084.3 | -0.000 % | pass |
| section_edge length (um) | 27.2 | 27.2 | -0.000 % | pass |
| vessel_wall length (um) | 200.3 | 200.3 | -0.000 % | pass |
| triangles | 42561 | 38159 | -10.3 % | pass |
| min SICN (slidesim >= gmsh - 0.1) | 0.5534 | 0.4062 | -0.1472 | **fail** |
| min SICN of the gmsh mesh = gmsh's | 0.5534131 (record 0.553) | 0.5534131 | -1.0e-13 | pass |

### 281mg_m1

Polygon area 2164493.7 um^2 (slidesim's mesh: +8.9e-16 relative). Min angle gmsh 21.80 deg,
slidesim 14.04 deg; mean SICN gmsh 0.979, slidesim 0.958.

| Criterion | gmsh | slidesim | Difference | |
| --- | --- | --- | --- | --- |
| tag names | central_wall, field_cut, portal_wall, section_edge, tissue, vessel_wall | the same | | pass |
| tissue area (um^2) | 2164493.7 | 2164493.7 | +0.000 % | pass |
| central_wall length (um) | 3334.9 | 3334.9 | -0.000 % | pass |
| field_cut length (um) | 5910.5 | 5910.5 | +0.000 % | pass |
| portal_wall length (um) | 828.6 | 828.6 | -0.000 % | pass |
| section_edge length (um) | 12.4 | 12.4 | +0.000 % | pass |
| vessel_wall length (um) | 202.0 | 202.0 | -0.000 % | pass |
| triangles | 41252 | 36723 | -11.0 % | pass |
| min SICN (slidesim >= gmsh - 0.1) | 0.5570 | 0.4075 | -0.1494 | **fail** |
| min SICN of the gmsh mesh = gmsh's | 0.5569532 (record 0.557) | 0.5569532 | +9.6e-14 | pass |

The gmsh mesh of `281mg_m1` is the one of milestone 3 (41252 triangles, as its record then).
Meshing took 1.1 to 1.2 s in slidesim (the whole `slidesim mesh`, 62 MB) and 1.8 to 2.2 s in gmsh
(`generate` alone).

### Verdict

Five of six criteria pass on both fields, criterion 5 fails on both by about 0.05. Both meshers
keep the same polygons: the areas and the lengths of every tag agree to rounding, and slidesim's
SICN is gmsh's to 1e-13. slidesim makes 10 to 11 % fewer triangles. **slideviz keeps
`build_mesh.py`**; it is retired when criterion 5 holds.

### Why the worst triangles are worse

The polygons follow the lumen and tissue masks of H&E level 3, 1.77 um per pixel, and the 2 um
simplification keeps some single-pixel steps: 172 (`000mg_m1`) and 166 (`281mg_m1`) of about
1400 boundary edges are shorter than 2 um, the shortest 1.25 um. slidesim keeps every input vertex
and splits the boundary to the size field (5 um at the walls), so a 1.77 um edge gets a triangle
whose third vertex sits 5 to 7 um away. gmsh grades the size down at such an edge. The worst
triangles of slidesim's meshes, all on a 1.77 um boundary edge (coordinates in um on the H&E
level-0 frame, x along the columns, y along the rows):

| Field | Centroid (x, y) | Edges (um) | SICN slidesim | Min SICN of gmsh within 10 um |
| --- | --- | --- | --- | --- |
| `000mg_m1` | (4055.8, 2395.7) | 5.97, 4.84, 1.77 | 0.406 | 0.784 |
| `000mg_m1` | (4120.4, 2725.6) | 7.07, 1.77, 7.29 | 0.408 | 0.813 |
| `000mg_m1` | (4053.4, 2340.5) | 1.77, 4.98, 3.75 | 0.446 | 0.627 |
| `000mg_m1` | (4192.5, 1505.2) | 3.71, 4.94, 1.77 | 0.450 | 0.768 |
| `281mg_m1` | (6697.2, 2113.7) | 7.07, 1.77, 7.29 | 0.408 | 0.693 |
| `281mg_m1` | (6431.9, 2528.1) | 2.47, 3.95, 1.77 | 0.414 | 0.693 |
| `281mg_m1` | (6606.9, 1366.4) | 4.40, 5.59, 1.77 | 0.416 | 0.799 |
| `281mg_m1` | (7454.3, 2688.9) | 5.11, 3.84, 1.77 | 0.425 | 0.802 |
| `281mg_m1` | (6808.2, 1580.9) | 5.06, 3.81, 1.77 | 0.435 | 0.693 |
| `281mg_m1` | (6670.1, 2150.3) | 3.96, 2.52, 1.77 | 0.439 | 0.761 |

These are 4 (`000mg_m1`) and 6 (`281mg_m1`) triangles below gmsh's min SICN less 0.1, and 6 and
13 below 0.5, where gmsh has none. The edge ratios go down to 0.35 and 0.27 (1.77 and 1.35 um
edges at 5 um) and the smallest angles to 14 deg: the polygons are outside the range for which
the mesher's bounds hold ([Meshing](../methods/meshing.md#the-steps): a size field which the
boundary splitting resolves). Two ways to close the gap: a size field which also takes the
lengths of the input edges (the local feature size, as gmsh does at the points of the geometry),
or polygons without edges below about 0.4 of the smallest size (a simplification in slideviz
which removes single-pixel steps while keeping the shared vertices of neighbouring areas).

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

`slidesim mesh` of the scenes:

| Field | Triangles | Vertices | Min angle | Min SICN | Mean SICN | Edge length / size |
| --- | --- | --- | --- | --- | --- | --- |
| `000mg_m1` | 6002 | 3160 | 15.95 deg | 0.458 | 0.956 | 0.35 to 1.56 |
| `281mg_m1` | 13598 | 7102 | 16.25 deg | 0.419 | 0.948 | 0.35 to 1.63 |
| `500mg_m1` | 16782 | 8755 | 16.35 deg | 0.447 | 0.944 | 0.35 to 1.70 |

The min SICN stays above 0.4, the smallest angles (16 to 16.4 deg) and edge ratios (0.35) miss
the bounds of the generated meshes for the reason above; the necrosis borders of variant `follow`
add corners down to 26.6 deg (pixel steps of the necrosis map). The tests of `crates/slidesim`
mesh all three (`slide_field_examples_mesh`: portal and central walls, the area of the polygons
to 1e-9, min SICN at least 0.2) and run the batched SPT study of `281mg_m1` for 10 min
(`slide_field_spt_runs`: 13598 cells, the flows through the portal and central walls equal and
opposite, every mass balance closing).

## Reproduce

```bash
export SLIDEVIZ_DATA=/data/koeniglab/image.analysis/APAP/APAP_zarr  # read only
cd ~/git/slidesim
scripts/e2e/m4_meshing.sh  # OUT=target/e2e/m4_meshing; STEPS="fields compare"
# OUT/slide_fields/<animal>: the example fields; OUT/compare.md: the tables above
cargo nextest run -p slidesim --test cli -E 'test(slide_field)'
```

The example fields in the repository are the output of the `fields` step (their polygons and maps
are identical to a second run; the provenance attributes name the files of the run which made
them).
