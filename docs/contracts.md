# Contracts

slidesim reads three versioned contracts. Each has one Rust definition (serde, crate
`contracts`) and a JSON Schema generated from it: `schemas/scene-1.schema.json`,
`schemas/polygons-1.schema.json` and `schemas/study-1.schema.json`. slideviz writes `scene/1` and
`polygons/1` against a copy of the schema, which its tests validate the files against. Unknown
keys are errors in all of them.

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
| `mesh` | where the mesh comes from, by `format`: `gmsh`, `polygons` or `lobules`, see below |
| `maps` | raster maps over the tissue (optional), see below |

### Mesh sources

`mesh.format` selects the source. The tags of the mesh (area and edge tags) are what a study
refers to; slidesim has no built-in tag names.

**`gmsh`**: `{"format": "gmsh", "path": "mesh.msh"}`. `path` is the `.msh` file, relative to the
scene file; its physical groups are the area tags (`surviving`, `necrotic`, `tissue`) and the edge
tags (`portal_wall`, `central_wall`, `vessel_wall`, `section_edge`, `field_cut`, ...).

**`polygons`**: `{"format": "polygons", "path": "tissue.geojson", "size": {...}}`. `path` is a
`polygons/1` file (below), relative to the scene file, which slidesim meshes with the size field
`size`:

| Key | Meaning |
| --- | --- |
| `min_um` | element size near the walls, > 0 and <= `max_um` |
| `max_um` | element size far from the walls |
| `dist_min_um` | distance up to which the size is `min_um`, >= 0 and < `dist_max_um` |
| `dist_max_um` | distance from which the size is `max_um`, linear in between |
| `walls` | edge tags the distance is measured from (default none: `max_um` everywhere) |

**`lobules`**: parametric hexagonal liver lobules, generated and meshed by slidesim, with a
central vein and a portal vein at every corner. All lengths in um.

| Key | Meaning |
| --- | --- |
| `lobule_radius_um` | circumradius of the hexagonal lobule, > 0 (default 300) |
| `rings` | rings of neighbouring lobules around the central one (default 0: one lobule) |
| `jitter` | largest displacement of a vein as a fraction of `lobule_radius_um`, `0 <= jitter <= 0.25` (default 0) |
| `seed` | seed of the displacements (default 0) |
| `centre` | the central vein: `radius_um` (> 0, default 30) and `tag`, the edge tag of its wall |
| `corner` | the portal veins at the corners: `radius_um` (> 0, default 25) and `tag` |
| `outer_tag` | edge tag of the outer boundary |
| `area_tag` | area tag of the tissue |
| `size` | `wall_um` (default 5) and `interior_um` (default 20), `0 < wall_um <= interior_um` |
| `zonation` | `map`, the id of the generated zonation map (0 periportal to 1 pericentral), and `um_per_px` (default 2) |

The tags (`outer_tag`, `area_tag`, `centre.tag`, `corner.tag`) and `zonation.map` have no
defaults; the four tags must differ and `zonation.map` must not be the id of an entry of `maps`.
The veins must not touch even in the worst displacement: `centre.radius_um + corner.radius_um`
and `2 corner.radius_um` are both below `(1 - 2 jitter) lobule_radius_um / 2`. The size field of
the mesh is `wall_um` up to `2 wall_um` from the vein walls (the centre and corner tags), linear
to `interior_um` at `10 interior_um`. Studies refer to the zonation map like to an entry of
`maps`.

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

## polygons/1

Tagged polygons for the `polygons` mesh source: a GeoJSON FeatureCollection with the foreign
members `version` and `meta` (free JSON, ignored). Coordinates are `[x, y]` in the um of the scene.

```json
{
  "type": "FeatureCollection",
  "version": "polygons/1",
  "meta": {"source": "slideviz"},
  "features": [
    {"type": "Feature",
     "properties": {"kind": "area", "tag": "tissue"},
     "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [100, 0], [100, 100], [0, 100], [0, 0]]]}},
    {"type": "Feature",
     "properties": {"kind": "edge", "tag": "central_wall"},
     "geometry": {"type": "LineString", "coordinates": [[10, 10], [20, 10], [20, 20], [10, 10]]}}
  ]
}
```

| Property | Meaning |
| --- | --- |
| `kind` | `area` (geometry `Polygon` or `MultiPolygon`) or `edge` (`LineString` or `MultiLineString`) |
| `tag` | the tag name; a tag is either an area tag or an edge tag |
| `value` | integer tag value > 0 (default 1) |

The first ring of a polygon is its outer boundary, the others are holes. `Polygons::load` checks
the kinds, the geometry types, the values and that no tag is both an area and an edge tag; the
geometry (closed rings, no self-intersections, edges on area boundaries) is checked by the mesher.

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

## result/1

What a viewer reads from a finished run: the Zarr v3 store `result.zarr` (zstd compression). Its
root attributes are the `ResultMeta` JSON (schema `schemas/result-1.schema.json`, checked by
`ResultMeta::check`). Units: length um, time s, concentration mM.

| Array | dtype | shape | chunks | Meaning |
| --- | --- | --- | --- | --- |
| `time` | f64 | `[T]` | all | output times in s |
| `mesh/vertices` | f64 | `[V, 2]` | all | vertex coordinates in um |
| `mesh/triangles` | u32 | `[N, 3]` | all | vertex indices, counterclockwise |
| `mesh/element` | u64 | `[N]` | all | coarse element id of each finest triangle |
| `mesh/tags/<name>` | u32 | `[k]` or `[k, 2]` | all | area tag: triangle indices; edge tag: vertex index pairs |
| `fields/<id>` | f64 | `[T, V]` | `[1, V]` | P1 nodal values on the finest mesh |
| `cells/element` | u64 | `[C]` | all | coarse element ids that hold cells |
| `cells/<value>` | f64 | `[T, C]` | `[1, C]` | per-element cell value |
| `raster/<field>/<level>` | f32 | `[T, rows, cols]` | `[1, 256, 256]` | optional; NaN outside the mesh; level k+1 is the mean of the valid pixels of 2 x 2 of level k |

Element ids are partition independent: `Mesh::element_ids` numbers the coarse cells of the
undistributed mesh, the id of a finest triangle is `(coarse << (2 * level)) | child bits`. The
canonical mesh holds the finest triangles sorted by that fine id, the vertices numbered by first
appearance in that order, and `mesh/element` the coarse id of each triangle. A run on any number of
processes writes the same arrays.

Attributes (`ResultMeta`):

| Key | Meaning |
| --- | --- |
| `version` | `result/1` |
| `study`, `scene` | the files the run was made from |
| `refinements` | uniform refinements beyond the coarse mesh |
| `times` | output times in s, strictly increasing (the `time` array) |
| `written` | time indices written so far; an index `t < written` is complete and readable alone |
| `complete` | the run finished; implies `written == times.len()` |
| `fields` | `{id, units}` of every `fields/<id>`, ids unique, units `mM` |
| `cell_values` | the names of the `cells/<value>` arrays |
| `tags` | `{name, dim}` of every `mesh/tags/<name>`: `dim` 2 area, 1 edge |
| `raster` | optional `{um_per_px, origin_um, levels, shapes}`: pixel size of level 0 (> 0), the um of the corner of pixel (0, 0), the number of levels (>= 1) and `[rows, cols]` per level, level 0 first |

### Live API

The browser build exposes the same arrays for a running simulation; the declarations are in
`wasm/js/slidesim.d.ts` (type check: `npx --yes -p typescript@5.6.3 tsc --noEmit -p
wasm/js/tsconfig.json`). `createSimulation(options)` resolves to a `Simulation` whose `mesh`
holds `mesh/vertices`, `mesh/triangles`, `mesh/element` and `mesh/tags/*`, `times` is the `time`
array and `meta` the `ResultMeta`. `step()` advances to the next output time and returns its index
(`null` when done); `field(id)` is `fields/<id>` at the current time index, `cellIds` is
`cells/element` and `cellValues(name)` is `cells/<name>` at the current time index. The arrays
are copies the caller owns.

Live semantics: after `createSimulation` resolves the simulation is at output index 0, the
initial state: `time == times[0]`, `field()` and `cellValues()` return the values at index 0, and
`meta.written == 1`. Each `step()` advances to the next output index and returns it (the first
`step()` returns 1) and `null` when there is none left; then `meta.complete` is true. `written`
is the number of output times reached so far, as in `result/1` where an index `t < written` is
complete. `meshQuality()` is the quality report of a generated mesh, `null` for a loaded mesh. `lobuleScene(params, tags)` builds a lobule `scene/1`; the tag names
are supplied by the caller (the lobule demo uses the area `tissue`, the edges `central_wall`, `portal_wall` and `field_cut`, and the map `zonation`); slidesim fixes no names.
`CreateOptions.memory.maximumPages` caps the wasm memory, exceeding it is an `Error`.
