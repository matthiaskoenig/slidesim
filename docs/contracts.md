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
`size` ([Meshing](methods/meshing.md)); the area and edge tags of the file are the tags of the
mesh:

| Key | Meaning |
| --- | --- |
| `min_um` | element size near the walls, > 0 and <= `max_um` |
| `max_um` | element size far from the walls |
| `dist_min_um` | distance up to which the size is `min_um`, >= 0 and < `dist_max_um` |
| `dist_max_um` | distance from which the size is `max_um`, linear in between |
| `walls` | edge tags the distance is measured from (default none: `max_um` everywhere) |

**`lobules`**: parametric hexagonal liver lobules, generated and meshed by slidesim, with a
central vein and a portal vein at every corner ([Meshing](methods/meshing.md#ideal-lobules)). All
lengths in um.

| Key | Meaning |
| --- | --- |
| `lobule_radius_um` | circumradius of the hexagonal lobule, > 0 (default 300) |
| `rings` | rings of neighbouring lobules around the central one, at most 4 (default 0: one lobule; 4: 61 lobules) |
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
`maps`. It is computed, not read: the relative position `d_p / (d_p + d_c)` between the walls
(`d_p` the distance to the nearest corner wall, `d_c` to the nearest centre wall), at the pixel
centres of a raster of `um_per_px` over the bounding box of the lobules.

```json
"mesh": {
  "format": "lobules",
  "lobule_radius_um": 300.0,
  "rings": 0,
  "centre": {"radius_um": 30.0, "tag": "central_wall"},
  "corner": {"radius_um": 25.0, "tag": "portal_wall"},
  "outer_tag": "field_cut",
  "area_tag": "tissue",
  "size": {"wall_um": 5.0, "interior_um": 20.0},
  "zonation": {"map": "zonation", "um_per_px": 2.0}
}
```

Both generated sources are 2D (`dimension` 2), meshed by process 0 of a run. An input the mesher
cannot mesh is an error naming the polygons file and the place in um.

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
geometry is checked by the mesher, whose errors name the file and the place in um: rings or
chains which cross, areas which overlap, a boundary segment without an edge tag (a gap between
areas: borders shared by areas must match exactly), an edge tag off the boundaries of the areas.

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
| `time` | `end` (s), `dt` (s, the first step if adaptive), `adaptive` (default false), `outputs` (s, required: at least one time, strictly increasing, the first >= 0 and the last equal to `end`; output index 0 is the state at the first time, the initial state when it is 0) |
| `mesh` | `refinements`: the number of uniform refinements of the scene's mesh (default 0) |
| `result` | optional: `raster` `{um_per_px, origin_um, levels}` also writes the fields as the raster pyramid of [`result/1`](#result1): `um_per_px` > 0 and finite, `origin_um` (finite, default the lower corner of the mesh's bounding box), `levels` >= 1 |
| `solver` | `linear_rtol`: the relative tolerance of the linear solves of the transport, in (0, 1), default 1e-10 (the residual of each solve reduced by it; with fixed steps the time integration has no error estimate to derive it from, so the study sets it; `-ksp_rtol` in `petsc_options` overrides it); `petsc_options`: a PETSc options string, for example `-ksp_type cg -pc_type jacobi` |

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

Each triangle starts at its vertex with the smallest x (of two with the same x, the smaller y)
and runs counterclockwise. An area tag holds the indices of its triangles ascending; an edge tag
holds its edges as vertex index pairs `[min, max]`, the pairs ascending. A tag of dimension 0
(points) is not written: no array and no entry of `tags`. `cells/element` is ascending; a study
without cells has an empty `cells/element` (shape `[0]`) and no `cells/<value>`. The rows of
`fields/<id>` and `cells/<value>` hold the fill value NaN until their output time is written:
`slidesim run` writes each output time as it reaches it (the chunks first, then `written`, whose
root attributes replace the old ones by a rename), so a viewer may read a store while the run goes
on. That order holds against a crash of the process: a store left by a run which ended early is
complete below `written`. It is not made durable against a power loss (no `fsync`): after one,
the file system may hold a `written` newer than the chunks it counts.

The raster pyramid (`result.raster` of the study) frames the results for a slide image. Level 0
covers the bounding box of the mesh from `origin_um` (default: the lower corner of the box):
`cols = ceil((x_max - x0) / um_per_px)` and `rows = ceil((y_max - y0) / um_per_px)`, at least 1
each. Row 0 is at the smallest y (y grows with the row index). Pixel `(r, c)` has its centre at
`(x0 + (c + 0.5) h, y0 + (r + 0.5) h)` with `h = um_per_px`, and holds the P1 value of the field
there, as f32; a centre outside the mesh is NaN, a centre on a shared edge or vertex takes any
containing triangle (the field is continuous). With an `origin_um` right of or above the mesh's
lower corner the pixels cover only from the origin on: nothing left of or below it is stored.
Level k+1 has `ceil(rows / 2)` x `ceil(cols / 2)` pixels (`shapes[k]` is `[rows, cols]`), each the
mean of the valid (not NaN) pixels of its block of 2 x 2 of level k, a block cut by the edge
averaging the pixels it has, a block without a valid pixel NaN; levels past 1 x 1 stay 1 x 1.
Each output time is one chunk per 256 x 256 block, written after the fields of that time and
before `written` counts it. A raster of more than 2^28 pixels in level 0 is refused when the run
starts, the message naming `result.raster.um_per_px`. Process 0 makes the raster and holds, per
pixel of level 0, the triangle it lies in (4 bytes) and each output time the pyramid of a field
(f32, about 5.3 bytes with the coarser levels) and the chunks it encodes: at the cap of 2^28
pixels more than 3 GB, so a raster near the cap needs a machine with that much memory on
process 0.

The mesh is limited to about 40 million finest triangles: the writer gathers the canonical mesh
onto process 0 (`Mesh::canonical`, about 44 bytes per triangle with its vertices and tags), and
MPI counts the bytes of a gather in a 32-bit integer (2 GiB). A larger mesh is an error.

Attributes (`ResultMeta`):

| Key | Meaning |
| --- | --- |
| `version` | `result/1` |
| `study`, `scene` | the files the run was made from: their file names, without directories (no path of the machine) |
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
`wasm/js/slidesim.d.ts` (type check: `npm run typecheck --prefix wasm/js`, TypeScript 5.6.3 of
its lock), implemented by `wasm/js/slidesim.mjs`. `createSimulation(options)` resolves to a
`Simulation` whose `mesh` holds `mesh/vertices`, `mesh/triangles`, `mesh/element` and
`mesh/tags/*`, `times` is the `time` array and `meta` the `ResultMeta`. `step()` advances to the
next output time and returns its index (`null` when done); `field(id)` is `fields/<id>` at the
current time index, `cellIds` is `cells/element` and `cellValues(name)` is `cells/<name>` at the
current time index. The arrays are copies the caller owns.

Live semantics: after `createSimulation` resolves the simulation is at output index 0, the state
at `times[0]` (the initial state when `times[0] == 0`; a later first output time is reached by a
step within `createSimulation`): `time == times[0]`, `field()` and `cellValues()` return the
values at index 0, and `meta.written == 1`. Each `step()` advances to the next output index and
returns it (the first `step()` returns 1) and `null` when there is none left; then
`meta.complete` is true, and the simulation stays at the last output time, which `study/1` makes
`time.end`: `time` is the last of `times`, and `field()` and `cellValues()` return the values of
the last index. `written` is the number of output times reached so far, as in `result/1` where an
index `t < written` is complete. `meshQuality()` is the quality report of a generated mesh, `null`
for a loaded mesh, and `log()` the array `log` (the quality line of a generated mesh).
`lobuleScene(params, tags)` builds a lobule `scene/1`; the tag names are supplied by the caller
(the lobule demo uses the area `tissue`, the edges `central_wall`, `portal_wall` and `field_cut`,
and the map `zonation`); slidesim fixes no names.

`CreateOptions.memory.maximumPages` caps the wasm memory: the maximum of the memory the host
creates for the simulation, for a compiled module and bytes alike. Exceeding it is an `Error`
naming the cap: an error of slidesim when PETSc's allocation failed (the simulation stays
usable), a trap when Rust's or libCEED's did (the simulation is unusable). Mmg's own cap of 2 GiB
in wasm, below, holds above it. Each simulation has its own instance of `slidesim.wasm` and its
own memory, so a trap ends only the simulation it happened in. The getters of a `Simulation`
throw after a trap and after `free()` like its methods; `mesh` is the simulation's own arrays,
the same on every access (not copies).

Under the live API lies the C ABI of `slidesim.wasm` (`crates/slidesim-wasm`, a WASI reactor,
`_initialize` first), over its memory, pointers and lengths `u32` byte offsets and counts. The
module imports its memory, `env.memory`, which the host creates at the minimum of the import (the
custom section `slidesim.memory` holds it as `{"minimum_pages": n}`, for hosts which cannot read
the limits of an import, as JavaScript cannot), with the host's cap as its maximum, and gives to
the kernel as well; the module exports it again as `memory` for WASI hosts which look for it
there.

| Export | Meaning |
| --- | --- |
| `slidesim_alloc(len) -> ptr`, `slidesim_free(ptr, len)` | memory for the caller's bytes |
| `slidesim_create(ptr, len) -> i32` | the simulation of the JSON `{"scene", "study", "kernel", "mesh"?}`: its handle, -1 with the error |
| `slidesim_next_output(h) -> i32` | steps to the next output time: its index (the first at `times[0]`, without a step), -1 when none is left (no step: the simulation stays at the last output time), -2 with the error |
| `slidesim_time(h) -> f64` | the time reached (s) |
| `slidesim_array(h, name_ptr, name_len) -> ptr`, `slidesim_array_len() -> u32` | the bytes of an array now, valid until the next `slidesim_array`; 0 with the error |
| `slidesim_error_ptr() -> ptr`, `slidesim_error_len() -> u32` | the message of the last error (UTF-8) |
| `slidesim_destroy(h)` | frees the simulation |

`scene` and `study` are the documents, `kernel` the `kernel.json` of the batched kernel of the
cell model (required with `cells`; the host compiles `kernel.wasm` from the SBML with
sbmlsim-core's `compiler.wasm` and instantiates it with the same memory and `mathlib.wasm`),
`mesh` a `mesh.json` of `slidesim mesh` in place of the scene's mesh. The build reads no files:
a scene whose mesh (without `mesh`) or maps are files is an error naming the file.
`slidesim.wasm` imports `slidesim_kernel.init`, `slidesim_kernel.advance` and
`slidesim_kernel.assign`, which the host binds to the kernel's `sbmlsim_kernel_init`,
`sbmlsim_kernel_advance` and `sbmlsim_kernel_assign` once the kernel is instantiated (`assign`
evaluates the targets of the assignment rules in `cells.outputs`; with a kernel which has none,
whose `kernel.json` has no `assigned_ids`, such an output is refused and `assign` never called).
The arrays are the names of `result/1`, little endian (`fields/<id>`, `cells/<value>` and
`raster/<field>/<level>` the row now), and `meta` (the `ResultMeta`, `written` and `complete` as
above), `summary` (`summary.json`) and `quality` (`null` for a mesh of the inputs) as JSON, and
`log`: what the simulation reported, UTF-8, one line each (the quality line of a generated mesh,
which `slidesim run` prints); slidesim writes nothing to stdout or stderr.
A memory growth past the host's cap fails the call (a trap or an error); Mmg, the remesher of the
mesher, caps itself at 2 GiB in wasm (a constant of
`scripts/deps/patches/mmg-0001-memory-size-on-wasm.patch`; PETSc's adaptor sets no
`MMG2D_IPARAM_mem`), so a lower cap of the host is the one that holds.

After a trap of `slidesim.wasm` (a panic, an allocation past the host's cap) the instance must be
discarded with every simulation in it: the trap leaves slidesim in the middle of a call (its state
borrowed, PETSc's objects half updated), so every later call would trap again or worse. The host
instantiates `slidesim.wasm`, mathlib and the kernel again for the next simulation; the live API
rejects the `createSimulation` or `step()` that trapped with an `Error` and marks the
`Simulation` unusable. An error (-1, -2, a null array) is not a trap: the instance and its other
simulations stay usable. A pointer with a length of 0 is never read, so it may be anything,
0 included.
