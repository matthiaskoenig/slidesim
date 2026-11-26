# Example studies

The studies under `testdata/studies/` run end to end in the tests of `crates/slidesim`, and
show the parts of `study/1` one at a time. The meshes are written by
`testdata/meshes/make_fixtures.py` (gmsh), named as slideviz names its tags.

| Study | Scene | What it shows |
| --- | --- | --- |
| `tissue_oxygen/study.json` | `tissue_small.msh`: a 200 um square with two vessel holes | diffusion-reaction of one field, `oxygen`, held at 0.1 mM on `vessel_wall`, with decay |
| `lobule_toy/study.json` | `lobule_toy.msh`: a 400 um square with a portal and a central hole, and a `zonation` map | Darcy perfusion from `portal_wall` (390 Pa) to `central_wall` (0 Pa), sucrose carried by the flow from the portal wall |
| `cells_toy/cells.study.json` | the scene of `lobule_toy` | the linear cell model `linear_uptake.xml`: uptake from the blood at a rate from the zonation; the amount in blood and cells changes by the wall flows only |
| `cells_toy/spt.study.json` | the scene of `lobule_toy` | the SPT liver cell model `spt_liver_cell.xml`, `S_ext` and `P_ext` coupled to the fields `S` (5 mM at the portal wall) and `P`, for 60 min, with a lowered necrosis threshold so that necrosis starts within the hour |
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
