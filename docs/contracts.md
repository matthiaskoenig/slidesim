# Contracts

slidesim reads two versioned JSON contracts. Each has one Rust definition (serde, crate
`contracts`) and a JSON Schema generated from it: `schemas/scene-1.schema.json` and
`schemas/study-1.schema.json`. slideviz writes `scene/1` against a copy of the schema, which its
tests validate the scenes against. Unknown keys are errors in both.

Units throughout: length um, time s, concentration mM, pressure Pa, permeability um^2,
viscosity Pa*s, velocity um/s; porosity is dimensionless.

## scene/1

The geometry, written by slideviz next to the mesh: the mesh with its tags, and the maps over
the tissue.

```json
{
  "version": "scene/1",
  "dimension": 2,
  "units": {"length": "um"},
  "mesh": {"format": "gmsh", "path": "../../meshes/lobule_toy.msh"},
  "maps": [
    {
      "id": "zonation",
      "path": "zonation.zarr",
      "units": "1",
      "um_per_px": 4.0,
      "origin_um": [-8.0, -8.0]
    }
  ]
}
```

| Key | Meaning |
| --- | --- |
| `version` | exactly `scene/1` |
| `dimension` | 2 or 3; the solvers take 2D triangle meshes, 3D meshes come in a later milestone |
| `units.length` | `um`, the only unit of the mesh coordinates |
| `mesh` | `format` `gmsh` and `path` of the `.msh` file, relative to the scene file; its physical groups are the area tags (`surviving`, `necrotic`, `tissue`) and the edge tags (`portal_wall`, `central_wall`, `vessel_wall`, `section_edge`, `field_cut`, ...) the study refers to |
| `maps` | raster maps over the tissue (optional), see below |

A **map** is a 2D Zarr v3 array (rows along y, columns along x, float32 or float64), sampled
bilinearly by slidesim at the quadrature points of every mesh level that needs it. Each process
reads only the chunks over its part of the mesh.

| Key | Meaning |
| --- | --- |
| `id` | the name a study uses: an ASCII letter, then ASCII letters, digits and `_` |
| `path` | the Zarr v3 array, relative to the scene file |
| `units` | units of the values, free text (`1` for dimensionless) |
| `um_per_px` | pixel size in um, the same on both axes |
| `origin_um` | `[x, y]` in um of the centre of pixel (row 0, column 0) |
| `no_value` | optional: pixels with this value (and NaN, always) carry no data |

A quadrature point outside the raster, or next to a pixel without data, is an error naming the
map, never a silent default. The slideviz scenes of the liver carry `zonation` (0 periportal to 1
pericentral), and from milestone 3 on `cyp2e1` (the relative CYP2E1 readout) and `necrosis` (the
measured necrosis calls).

## study/1

What to simulate on a scene: the fields, the perfusion, the cell models, the time and the
solvers.

| Key | Meaning |
| --- | --- |
| `version` | exactly `study/1` |
| `scene` | the `scene/1` file, relative to the study file |
| `fields` | the diffusing fields, at least one, with unique ids |
| `perfusion` | optional: the Darcy flow; required for fields with `advection` |
| `cells` | optional: a cell model on the elements, coupled to the fields |
| `time` | `end` (s), `dt` (s, the first step if adaptive), `adaptive` (default false), `outputs` (s, sorted, within `[0, end]`) |
| `mesh` | `refinements`: the number of uniform refinements of the scene's mesh (default 0) |
| `solver` | `petsc_options`: a PETSc options string, for example `-ksp_type cg -pc_type jacobi` |

### Coefficients

A coefficient is a number, or a table over a map: piecewise linear in the map value, held
constant beyond the first and the last point.

```json
{"map": "zonation", "points": [[0, 500], [1, 900]]}
```

The `points` are `[value, output]` pairs, at least two, the values strictly increasing. A table
naming a map which the scene does not have is an error before any solve, listing the maps.

### Fields

| Key | Meaning |
| --- | --- |
| `id` | an ASCII letter, then ASCII letters, digits and `_`; also the prefix of the PETSc options of the field, `<id>_` |
| `diffusion` | diffusion coefficient (um^2/s, > 0), a coefficient |
| `decay` | first-order decay rate (1/s, >= 0), a coefficient, default 0 |
| `source` | source (mM/s), a coefficient, default 0 |
| `porosity` | the fraction of the volume the field lives in (> 0), a coefficient, default 1 |
| `advection` | carried by the Darcy velocity of `perfusion` (default false) |
| `initial` | uniform initial concentration (mM), default 0 |
| `boundaries` | conditions by tag: `{"tag": ..., "dirichlet": <mM>}` or `{"tag": ..., "flux": <mM*um/s>}` (inward); tags not listed have zero diffusive flux |

With advection, a tag with net inflow needs a `dirichlet` condition for the field (an error
before the first step otherwise), and outflow tags without a condition are free outflow.

### Perfusion

```json
"perfusion": {
  "viscosity": 0.0024,
  "k_along": 0.0156,
  "k_across": 0.0175,
  "direction": {"map": "zonation", "g0": 0.00043},
  "boundaries": [
    {"tag": "portal_wall", "pressure": 390.0},
    {"tag": "central_wall", "pressure": 0.0}
  ]
}
```

| Key | Meaning |
| --- | --- |
| `viscosity` | dynamic viscosity (Pa*s, > 0) |
| `k_along`, `k_across` | permeability along and across the direction (um^2, > 0), coefficients |
| `direction` | the gradient of the map `map` gives the direction; below the gradient magnitude `g0` (1/um) the permeability fades to isotropic |
| `boundaries` | fixed pressures (Pa) by tag, at least one, unique tags; every other boundary has no flow |

### Cells

```json
"cells": {
  "model": "../../models/spt_liver_cell.xml",
  "solver": "batched",
  "time_unit_s": 60.0,
  "species": [
    {"species": "S_ext", "field": "S"},
    {"species": "P_ext", "field": "P"}
  ],
  "volumes": {"blood": "Vext", "cells": "Vli"},
  "parameters": [
    {"id": "pattern", "value": 7,
     "replace": {"map": "necrosis", "above": 0.5, "value": 6}},
    {"id": "protein_ext",
     "value": {"map": "cyp2e1", "points": [[0, 0], [0.7405, 1]]}},
    {"id": "position", "value": {"map": "zonation", "points": [[0, 0], [1, 1]]}}
  ],
  "split_step": 60.0,
  "outputs": ["protein"]
}
```

| Key | Meaning |
| --- | --- |
| `model` | the SBML file, relative to the study file |
| `solver` | `batched` (default) or `engine` |
| `time_unit_s` | seconds per time unit of the model (> 0) |
| `areas` | the area tags of the mesh which hold cells (default all) |
| `species` | the model species exchanged with fields: `{"species": <model id>, "field": <field id>}` |
| `volumes` | the compartments of the model set per element: `blood` to the porosity, `cells` to one minus the porosity (only their ratio matters) |
| `parameters` | model parameters set per element: `id` and `value` (a coefficient, averaged over the element's tissue); `replace` sets `value` of the replacement where the element mean of `map` is above `above` |
| `split_step` | the Strang splitting step (s, > 0, at least `time.dt`) |
| `rtol`, `atol` | tolerances of the cell solver (default 1e-6 and 1e-9) |
| `outputs` | extra targets of assignment rules written per cell to `cells_<index>.vtu` |

The linked fields and the maps named by `parameters` must exist (checked with the study); the
`outputs` are checked against the compiled model at the start of the run.
