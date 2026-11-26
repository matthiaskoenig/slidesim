# slidesim milestone 4: meshing in Rust, lobules, speed, `result/1`, the browser build

Status: draft for approval (2026-11-26).

## Intent

slidesim has to become a fast, self-contained simulator that the next pieces of work can build on:
a viewer that shows precalculated results or simulations running live in the browser, and, later,
the calibration of the biological problem on larger hardware. Milestone 3 delivered the coupled
cell models (SPT on slide fields) and met its performance target thinly (24 h of a 281 mg/kg
slide field in 9.5 min on 8 processes). The calibration against measured necrosis is deferred by
the user to later hardware and later speedups.

The user wants, in their words: a working simulator sped up as much as possible; the core pieces
finished so that work can go on with the visualization and the simulation in the browser;
the SPT model computed on lobulus geometries; all the meshing in Rust ("probably also linking an
open meshing library"); ideal lobule geometries and, in addition, some examples from real slides;
a viewer that works with precalculated results or with live simulations run directly in the
browser.

## Goals

1. **Meshing in Rust.** slidesim builds its 2D meshes itself, natively and in wasm: tagged
   polygons in, a tagged DMPlex out. No gmsh step in the chain. slideviz keeps the image analysis
   and hands over polygons.
2. **Parametric lobule geometries** (ideal lobules), generated in Rust from a few parameters, with
   a zonation computed from the geometry, as scenes that every study runs on unchanged.
3. **Lobule examples from real slides**: a few fields of real slides meshed by the Rust mesher from
   slideviz polygons, committed as small example scenes beside the ideal ones.
4. **More speed** on the coupled runs (transport linear solves, cell work balance, exchange),
   measured on the slide field of milestone 3 and on the lobules.
5. **`result/1`**: the result contract the viewer reads, written by native runs.
6. **The browser build**: slidesim compiled to `wasm32-wasip1` (sequential PETSc and libCEED, as
   the milestone 0 spike proved), generating a lobule, running SPT on it, and exposing fields and
   cell values in memory to a host page, the same data as `result/1`.

## Non-goals

- The viewer itself (milestone 4 of the master design names "viewer v1"; here it stays out: this
  milestone delivers what the viewer reads, natively and live).
- Calibration and comparison with measured necrosis (parked on `wip/m3-dosed-slides-calibration`).
- 3D meshing (milestone 7: octree from label volumes), GPU backends (milestone 6).
- Python bindings.

## Anti-goals

- Two meshing paths that disagree: slides and lobules go through the same Rust mesher.
- A browser build whose numbers differ from the native run beyond solver tolerances, or that needs
  a second solver stack (the master design's "one stack everywhere").
- GPL code in the default builds (gmsh, CGAL); non-free code (Triangle); AGPL (TetGen).
- Speed bought with silent accuracy loss: every speedup keeps the parity (1, 2, 3 processes within
  1e-10 with fixed transport steps), the exact mass balances and the Strang order tests.

## Constraints

All Global Constraints of the milestone 3 plan carry over (MIT core, pinned PETSc and libCEED,
fmt/clippy/tests at every commit, collective errors, no liver-specific code in slidesim's crates,
units um/s/mM, no em dash, no agent attribution). Our own libraries at their latest release
(sbmlsim-core: currently pinned by `rev` to `develop` until a release contains #119).

## Approaches

### Meshing (decided by the user)

| Option | Summary | Verdict |
|---|---|---|
| A. Pure Rust: `spade` plus own grading | CDT and Ruppert refinement (min angle, max area); trivially wasm; no size field, own grading and quality code; 2D only. | Fallback if B fails in wasm. |
| **B. `spade` triangulates the polygons, PETSc remeshes with MMG** | `spade` (MIT/Apache) gives a boundary-conforming triangulation with the tags; PETSc's `DMAdaptMetric` with Mmg 5.8 (LGPL, C, in the pinned PETSc, needs PT-Scotch, which slidesim builds) remeshes it to a metric from a size field: graded meshes, good quality, a path to 3D (mmg3d) and parallel remeshing (ParMmg) later. The mesh lands as a DMPlex. | **Chosen.** Risk: Mmg in `wasm32-wasip1` is untested; a spike comes first. LGPL in the wasm bundle is the same situation as SuiteSparse KLU in sbmlsim-core's engine today. |
| C. gmsh as a library | GPL, very large for wasm. | Rejected. |

### Cell models in the browser (controller's call)

Native runs compile the generated C with the system compiler and `dlopen` it; a browser has
neither. Options:

| Option | Summary | Verdict |
|---|---|---|
| **1. A batched wasm kernel from sbmlsim-core's compiler** | The master design's "batched SIMD128 emitter": sbmlsim-core's codegen (also as `compiler.wasm` in the page) emits the model together with the per-point RODAS4 integrator as one wasm module with one entry point that advances a block of points (states, parameters, controls in linear memory, the same fields and step control as the QFunction). slidesim's wasm build calls it once per cell step for all cells (one cross-module call per step, not per right hand side). Any SBML model runs in the browser. Work in sbmlsim-core, released within this milestone. | **Chosen** (controller's call, following the master design). The algorithm and step control are the QFunction's; agreement with the native batched path within the integrator tolerances, not bitwise. |
| 2. The C header with sizes known at run time, importing `model.wasm` functions | One integrator source; but `sbmlsim_rosenbrock.h` needs compile-time sizes and a dense Jacobian while `model.wasm` (model/2) has runtime sizes, a sparse Jacobian and imports its memory and math, and calls cross modules per right hand side. Risks slowing the native batched path. | Rejected. |
| 3. Models compiled ahead of time into the slidesim wasm | Generate C for fixed models (SPT) at build time, compile with wasi-sdk, link statically. Only built-in models run. | Fallback for the demo if option 1 slips. |
| 4. An interpreter of the model IR in Rust, or a second integrator in slidesim | Slow, or a second integrator to keep in step. | Rejected. |

### `result/1` layout (decided by the user: b)

| Option | Summary |
|---|---|
| a. Raster on the slide pyramid (master design) | Fields resampled onto the raster of the slide's OME-Zarr pyramid: tile streaming like the slide, overlays align by construction; loses the mesh, and ideal lobules have no slide raster (a synthetic raster would be invented). |
| **b. Mesh-native, plus an optional raster** | The coarse mesh (vertices, triangles, tags) once, nodal fields and per-element cell values per output time as Zarr v3 arrays; the same arrays a live run exposes in memory, so the viewer has one data model for both sources. An optional raster pyramid for slide overlays. |

The user chose b: the live browser source is mesh-native anyway and the ideal lobules have no
slide raster.

## Decided design

### Meshing crate (`mesher`, new)

- Input: a 2D domain as polygons with holes, every boundary chain tagged (wall classes such as
  `portal_wall`, `central_wall`, `vessel_wall`, `section_edge`, `field_cut`), areas tagged, and a
  size field (sizes at tagged walls and in the interior with a gradation, as slideviz's "5:20"
  today: small at walls, large inside).
- Step 1: `spade` constrained Delaunay triangulation of the polygons (boundary edges kept as
  constraints, holes removed), refined just enough for MMG to start from.
- Step 2: the triangulation becomes a DMPlex with the tag labels (edge labels for walls, cell
  labels for areas, the label names and values as the gmsh physical groups of slideviz give them
  today, so the rest of slidesim is unchanged), then `DMAdaptMetric` with Mmg to the metric of the
  size field. Labels survive the adaptation (Mmg's references carry them).
- Output: `Mesh::from_dm` (new) beside `Mesh::load`; the rest of the load sequence (pinched
  vertex split, partitioning, distribution, refinement levels) is unchanged.
- PETSc gets `--download-mmg` (and cmake on the host) in the native builds and, after the spike,
  in the wasm build.
- A small quality report (min angle, minSICN as slideviz prints) per generated mesh.

### Scene sources

- `scene/1` gains a mesh source besides `gmsh`: a geometry description that slidesim meshes at
  load time (format left to the builder, versioned with the contract, a JSON Schema generated as
  for the other contracts):
  - `lobules`: the parameters below;
  - `polygons`: tagged polygons (GeoJSON as slideviz's `mesh/polygons.py` writes) plus the size
    field.
- Maps stay Zarr rasters as today; the lobule source writes (or provides in memory) its own
  `zonation` map. gmsh scenes keep working.

### Ideal lobules (parametric)

- Central veins on a hexagonal lattice: one lobule, or rings of 6, 12, ... lobules around it; an
  optional seeded jitter of the vein positions makes lobules irregular. Lobule borders are the
  Voronoi edges between veins; portal fields sit at the Voronoi vertices where three lobules
  meet; the outer border of the tissue is a `field_cut`.
- Central veins and portal fields are holes (circles of given radius, polygonized at the wall
  mesh size) with walls `central_wall` and `portal_wall`.
- Parameters (with mouse defaults, the controller's call): lobule radius 300 um, rings 0, jitter 0,
  central vein radius 30 um, portal field radius 25 um, mesh size at walls 5 um, interior 20 um,
  seed.
- Zonation `position` (0 periportal, 1 pericentral) from the geometry: the relative distance
  `d_p / (d_p + d_c)` to the nearest portal wall and the nearest central wall (controller's call;
  a Laplace solve between the walls is the alternative if the isolines look wrong at irregular
  lobules). From it, as on slides, come the SPT parameters `position` and, by pattern, the CYP2E1
  `protein_ext`.
- Example studies: SPT on one lobule and on a ring-1 tissue (7 lobules), engine and batched path,
  24 h, constant portal concentration.

### Examples from real slides

- A few fields (the controller proposes three: one control, two dosed, 300 to 500 um, chosen
  around a central vein with its surrounding portal fields) from the slides already used in
  milestones 2 and 3, as slideviz polygons meshed by the Rust mesher, committed to the public
  repository as small example scenes (polygons and maps sized for the repository, a few MB at
  most; decided by the user), with a data license and credit line in the example's README, and
  the same SPT studies as the ideal lobules.
- slideviz: `mesh/polygons.py` output becomes the hand-over; `build_mesh.py` (gmsh) retires once the
  Rust mesher reproduces its meshes (same tags, comparable sizes and quality on the milestone 2 and
  3 fields).

### Speed

Targets (controller's call, a goal reported against, not a merge gate; the baseline re-measured on
a quiet machine first): the 281 mg/kg slide field of milestone 3, 24 h, 8 processes, batched path,
from 573 s to at most 300 s; a single ideal lobule, 24 h, SPT,
batched path, one process: as fast as possible, reported (it sets what the browser can do live).
Candidates, each measured before it is kept:

- Transport linear solves (about 400 of 573 s): the BDF extrapolation as the initial guess, the
  linear tolerance tied to the time integration error instead of a fixed 1e-10, a better coarse
  solve.
- Parity policy (controller's call): the 1e-10 parity across 1, 2 and 3 processes is a property of
  the test settings (fixed transport steps, tight linear tolerance, kept as an option) and stays a
  test in CI. Production settings may use the looser linear tolerance; there runs on different
  process counts agree within the solver tolerances, which the results report.
- Cell work balance (max/mean 1.36 at 24 h): partition weights from the cells' work.
- The exchange (34 s): buffers reused (milestone 3's deferred O1), from a profile.
- Compile the cell model once on rank 0 on a cold cache.

### `result/1`

- Fields are written on the finest mesh level, where they live (the mesh of that level written
  once: vertices, triangles, tags), so the viewer never reproduces DMPlex refinement; cell values
  per coarse element with the partition-independent element id and the map from fine to coarse
  elements (controller's call; a 24 h slide field at 1 refinement is about 170k triangles, small
  enough).
- Zarr v3 store per run, one versioned contract with a Rust definition and a JSON Schema like the
  others. Mesh-native layout (option b) with an optional raster; time is an axis; chunking and compression
  left to the builder within the rule that one output time of one field is readable alone.
- Written by native runs beside the VTU files (VTU stays for ParaView until VTKHDF exists).
- The same arrays are what the browser build exposes in memory.

### Library and browser build

- The coupling driver moves from the binary into a library crate (`Coupled`, `Phases`), as the
  final review of milestone 3 recommended: a `Simulation` built from a scene and a study that
  steps to the next output and exposes fields, cell values and the summary in memory; `slidesim
  run` becomes a thin CLI over it.
- A partition-independent element id (carried through distribution) for joins with the scene and
  comparisons across process counts (milestone 3 final review).
- The wasm build (`wasm32-wasip1`, wasi-sdk, sequential PETSc and libCEED with the spike's
  patches, `/cpu/self` backends) exports a small API to a host page: build a lobule scene from
  parameters, set up a study, step, read fields and cell values. The page side reuses sbmlsim-core's
  JavaScript host for `compiler.wasm` and `model.wasm`. A demo page in the docs runs SPT on a lobule.
- The build scripts of the spike move into `scripts/deps/` (a `build-wasi.sh` beside
  `build-native.sh`), with Mmg added.

## Order of work

1. Spike (throwaway): Mmg through PETSc's `DMAdaptMetric` natively with `--with-cxx=0`, and in
   `wasm32-wasip1`; `spade` to DMPlex to Mmg on one lobule. Decides B against the fallback A.
2. `mesher` crate and `Mesh::from_dm`; scene sources `polygons` and `lobules`; the zonation map.
3. Real-slide examples via polygons; the comparison with slideviz's gmsh meshes.
4. Library crate for the coupling driver, element ids.
5. Speed.
6. `result/1`.
7. Browser build with the lobule demo.

## Verification

- Mesher: tags preserved (every wall edge and area of the input in the mesh), size field honoured
  within a factor, minimum quality bound, deterministic output for a given input and seed.
- Ideal lobules: area and wall lengths against the analytic geometry; zonation 0 on portal walls,
  1 on central walls, monotone along a portal to central line.
- Every milestone 3 verification test still passes on gmsh scenes; the coupled tests also run on a
  generated lobule (conservation, parity, Strang order).
- Speed: before/after per phase, parity and balances unchanged.
- `result/1`: round trip (write, read, compare with the in-memory values), schema valid.
- Browser: on one shared mesh (generated natively, loaded by the wasm build), the wasm run of the
  lobule agrees with the native one-process run (fields within solver tolerances; cell states
  within the batched path's bound), in wasmtime and in a browser (Node and a headless browser in
  CI). Meshes generated in the browser are checked for tags, size and quality and for determinism
  on that target, not for bitwise equality with the native mesh (Mmg runs on each platform's libm;
  controller's call).

## Open questions (for the user)

1. The data license and credit line of the real-slide example scenes (the controller proposes
   CC BY 4.0 with the credit line the user names).

## Left to the builder

The `spade` refinement before Mmg, the metric construction from the size field, the geometry
description format in `scene/1`, the polygonization of circles, the zonation alternative if
needed, chunking and compression of `result/1`, the wasm export mechanism and the JS glue, the
choice among the speed candidates by measurement, and which real-slide fields (within the stated
rule).
