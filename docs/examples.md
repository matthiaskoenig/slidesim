# Example studies

The studies under `testdata/studies/` run end to end in the tests of `crates/slidesim`, and
show the parts of `study/1` one at a time. The gmsh meshes are written by
`testdata/meshes/make_fixtures.py`, named as slideviz names its tags; the lobule studies run on
meshes slidesim generates, the real-slide fields on meshes slidesim makes of slideviz's tagged
polygons.

| Study | Scene | What it shows |
| --- | --- | --- |
| `tissue_oxygen/study.json` | `tissue_small.msh`: a 200 um square with two vessel holes | diffusion-reaction of one field, `oxygen`, held at 0.1 mM on `vessel_wall`, with decay |
| `lobule_toy/study.json` | `lobule_toy.msh`: a 400 um square with a portal and a central hole, and a `zonation` map | Darcy perfusion from `portal_wall` (390 Pa) to `central_wall` (0 Pa), sucrose carried by the flow from the portal wall |
| `cells_toy/cells.study.json` | the scene of `lobule_toy` | the linear cell model `linear_uptake.xml`: uptake from the blood at a rate from the zonation; the amount in blood and cells changes by the wall flows only |
| `cells_toy/spt.study.json` | the scene of `lobule_toy` | the SPT liver cell model `spt_liver_cell.xml`, `S_ext` and `P_ext` coupled to the fields `S` (5 mM at the portal wall) and `P`, for 60 min, with a lowered necrosis threshold so that necrosis starts within the hour |
| `lobules/spt.study.json`, `lobules/spt_batched.study.json` | `lobules/scene.json`: one ideal lobule, generated | the SPT model on one lobule, 24 h, hourly outputs, engine and batched path |
| `lobules/spt_quick.study.json` | `lobules/scene.json` | the batched SPT study on the unrefined mesh for 1 h: the quick case of the benchmarks |
| `lobules/ring1_spt.study.json`, `lobules/ring1_spt_batched.study.json` | `lobules/ring1.scene.json`: 7 ideal lobules, generated | the same on a ring of lobules |
| `lobules/linear.study.json` | `lobules/scene.json` | the linear cell model on the generated lobule for 1 h: conservation, agreement across process counts, order of the splitting |
| `slide_fields/<field>/spt.study.json`, `slide_fields/<field>/spt_batched.study.json` | `slide_fields/<field>/scene.json`: 500 um of a real slide, tagged polygons of slideviz, meshed by slidesim | the SPT model on three real fields (control, 281 and 500 mg/kg), engine and batched path, with the slide's `zonation`, `cyp2e1` and `necrosis` maps |
| `cells_manufactured/study.json` | the unit square | a linear cell model in a closed domain whose coupled solution is known: the order of the Strang splitting |
| `m2_real_slide/*.study.json` | a real slide (not in the repository) | the four perfusion studies of the [milestone 2 results](results/2026-11-m2-real-slide.md): baseline, calibrated, whole blood, radial10 |
| `m3_real_slide/spt.study.json` | a dosed real slide (not in the repository) | the SPT model for 24 h on the 281 mg/kg slide, the coupled baseline of the [performance results](results/2026-11-m3-performance.md) |

The real-slide studies refer to `scene/scene.json` next to them, which the end-to-end scripts
write from the slides (`scripts/e2e/`).

## Diffusion-reaction

`testdata/studies/tissue_oxygen/study.json`, the smallest study: one field, one condition, no
maps.

```json
{
  "version": "study/1",
  "scene": "scene.json",
  "fields": [
    {
      "id": "oxygen",
      "diffusion": 2000.0,
      "decay": 0.1,
      "source": 0.0,
      "initial": 0.0,
      "boundaries": [
        {"tag": "vessel_wall", "dirichlet": 0.1}
      ]
    }
  ],
  "time": {"end": 60.0, "dt": 0.5, "adaptive": false,
           "outputs": [0.0, 10.0, 30.0, 60.0]},
  "mesh": {"refinements": 2}
}
```

## Perfusion and advection

`testdata/studies/lobule_toy/study.json`: the parameters of the
[perfusion parameter document](parameters/m2-perfusion-mouse.md) on a toy lobule. Its scene has
the map `zonation`, written by `make_maps.py` (0 at the portal hole, 1 at the central hole),
whose gradient gives the direction of the permeability.

```json
{
  "version": "study/1",
  "scene": "scene.json",
  "fields": [
    {
      "id": "sucrose",
      "diffusion": 700.0,
      "decay": 0.0,
      "source": 0.0,
      "porosity": 0.15,
      "advection": true,
      "initial": 0.0,
      "boundaries": [
        {"tag": "portal_wall", "dirichlet": 1.0}
      ]
    }
  ],
  "perfusion": {
    "viscosity": 0.0024,
    "k_along": 0.0156,
    "k_across": 0.0175,
    "direction": {"map": "zonation", "g0": 0.00043},
    "boundaries": [
      {"tag": "portal_wall", "pressure": 390.0},
      {"tag": "central_wall", "pressure": 0.0}
    ]
  },
  "time": {"end": 10.0, "dt": 0.5, "adaptive": false, "outputs": [0.0, 5.0, 10.0]},
  "mesh": {"refinements": 1}
}
```

## Cell models on a real slide

`testdata/studies/m3_real_slide/spt.study.json` couples the SPT model to the flow of a dosed
slide: CYP2E1 (`protein_ext`) from the slide's `cyp2e1` map scaled by 1/0.7405, the zonation
position from `zonation`, and pattern 6 (the CYP2E1 profile of control mice by position) instead
of the measured CYP2E1 where the measured `necrosis` map is above 0.5. Its `cells` section is the
example of [Contracts](contracts.md#cells) (with `"solver": "engine"`); the fields `S` (5 mM at
the portal walls) and `P` are carried by the flow, for 24 h with hourly outputs:

```json
"time": {
  "end": 86400.0,
  "dt": 10.0,
  "adaptive": false,
  "outputs": [0, 3600, 7200, 10800, 14400, 18000, 21600, 25200, 28800, 32400, 36000, 39600, 43200,
              46800, 50400, 54000, 57600, 61200, 64800, 68400, 72000, 75600, 79200, 82800, 86400]
},
"mesh": {"refinements": 1}
```

## Ideal lobules

`testdata/studies/lobules` runs on meshes slidesim generates from parameters
([Meshing](methods/meshing.md#ideal-lobules)): `scene.json` is one hexagonal lobule of 300 um
circumradius with a central vein of 30 um radius and six portal fields of 25 um radius at the
corners, `ring1.scene.json` the lobule with its ring of 6 neighbours (7 lobules, 24 portal
fields). The edge tags are `central_wall`, `portal_wall` and `field_cut` (the outer border, the
cut through the tissue), the area tag `tissue`, and the zonation map `zonation` is computed from
the geometry (0 at the portal walls, 1 at the central walls):

```json
{
  "version": "scene/1",
  "dimension": 2,
  "units": {"length": "um"},
  "mesh": {
    "format": "lobules",
    "lobule_radius_um": 300.0,
    "rings": 0,
    "jitter": 0.0,
    "seed": 0,
    "centre": {"radius_um": 30.0, "tag": "central_wall"},
    "corner": {"radius_um": 25.0, "tag": "portal_wall"},
    "outer_tag": "field_cut",
    "area_tag": "tissue",
    "size": {"wall_um": 5.0, "interior_um": 20.0},
    "zonation": {"map": "zonation", "um_per_px": 2.0}
  }
}
```

| Scene | Coarse mesh | Quality | Meshing (debug build) |
| --- | --- | --- | --- |
| `scene.json` | 5,902 triangles, 3,093 vertices | min angle 25.5 deg, SICN min 0.64, edge length / size 0.51 to 1.58 | 0.2 s |
| `ring1.scene.json` | 41,094 triangles, 21,166 vertices | min angle 24.3 deg, SICN min 0.61, edge length / size 0.52 to 1.53 | 12 s |

The SPT studies are those of `cells_toy/spt.study.json` for 24 h with hourly outputs: `S` held
at 5 mM at the portal walls, the perfusion of the toy lobule, CYP2E1 (`protein_ext`) and the
position from the zonation, one refinement (the cells on the coarse mesh, the fields on its
refinement), the model's own necrosis threshold, and the linear solves of the transport to
`solver.linear_rtol` 1e-6 (the production setting: its error in the fields is below a
thousandth of that of the time steps, [performance](results/2026-11-m4-performance.md)).
`spt.study.json` and `ring1_spt.study.json` take the engine path, the `_batched` studies the
batched one. `spt_quick.study.json` is `spt_batched.study.json` on the unrefined mesh for 1 h,
a run of seconds, the default case of `scripts/bench/coupled.sh`.

```bash
target/debug/slidesim mesh testdata/studies/lobules/scene.json --out mesh
target/release/slidesim run testdata/studies/lobules/spt_batched.study.json --out spt
```

## Real-slide fields

`testdata/studies/slide_fields` holds three fields of 500 um of real mouse liver slides (CC BY
4.0, see its README): `000mg_m1` (control), `281mg_m1` and `500mg_m1` (dosed), each a central vein
with portal fields. Their scenes are `polygons` scenes: the tagged polygons of slideviz
(`polygons.geojson`, areas `surviving` and `necrotic`, edge chains such as `portal_wall`,
`central_wall` and `field_cut`), meshed by slidesim with 5 um triangles at the walls and 20 um far
from them ([Meshing](methods/meshing.md)), and the maps `zonation`, `cyp2e1` and `necrosis`
cropped to the field:

```json
"mesh": {
  "format": "polygons",
  "path": "polygons.geojson",
  "size": {
    "min_um": 5.0,
    "max_um": 20.0,
    "dist_min_um": 10.0,
    "dist_max_um": 200.0,
    "walls": ["portal_wall", "central_wall", "necrosis_border"]
  }
}
```

| Field | Coarse mesh | Quality |
| --- | --- | --- |
| `000mg_m1` | 6,106 triangles, 3,229 vertices | min angle 23.2 deg, SICN min 0.55, edge length / size 0.46 to 1.53 |
| `281mg_m1` | 13,791 triangles, 7,208 vertices | min angle 23.0 deg, SICN min 0.50, edge length / size 0.43 to 1.63 |
| `500mg_m1` | 17,175 triangles, 8,961 vertices | min angle 22.4 deg, SICN min 0.54, edge length / size 0.41 to 1.67 |

The polygons keep edges of one pixel (1.77 um) of the masks; the size is capped at 1.5 times the
length of such a segment around it ([Meshing](methods/meshing.md)), so the fields meet the
quality bounds of the generated lobules
([milestone 4 meshing results](results/2026-11-m4-meshing.md)). The studies are
`m3_real_slide/spt.study.json` on each field, adapted by hand: 24 h, CYP2E1 from the `cyp2e1`
map, pattern 6 where the `necrosis` map is above 0.5, the linear solves to 1e-6 as in the
lobule studies; `spt.study.json` on the engine path, `spt_batched.study.json` on the batched
one.

```bash
target/release/slidesim mesh testdata/studies/slide_fields/281mg_m1/scene.json --out mesh
target/release/slidesim run testdata/studies/slide_fields/281mg_m1/spt_batched.study.json --out spt
```
