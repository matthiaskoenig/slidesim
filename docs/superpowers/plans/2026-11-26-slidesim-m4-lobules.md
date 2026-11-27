# slidesim M4: meshing in Rust, lobules, speed, `result/1`, the browser build - Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** slidesim meshes its own 2D scenes (tagged polygons and parametric lobules) in Rust,
runs SPT on ideal lobules and on small real-slide fields, runs faster, writes `result/1`, and
runs the same simulation compiled to `wasm32-wasip1` in a browser page.

**Architecture:** a spike first decides Mmg through PETSc's `DMAdaptMetric` (option B) against a
pure-Rust fallback (option A). A new crate `mesher` turns tagged polygons into a `TaggedMesh`
(`spade` constrained Delaunay, then Mmg to a size field) and generates lobule domains with their
zonation; `scene/1` gains the mesh sources `polygons` and `lobules`; slideviz hands over tagged
polygons (`polygons/1`). The coupling driver moves into a library crate `simulation`, with
partition-independent element ids and a canonical finest mesh, which `result/1` (Zarr v3,
mesh-native) and the browser build expose alike. sbmlsim-core gets a batched wasm kernel (model
plus RODAS4 in one module), which the wasm build of slidesim calls once per cell step.

**Tech Stack:** as milestone 3 (Rust 1.100.0, PETSc v3.26.1, libCEED v1.1.2, MPI, cargo-nextest,
zarrs 0.23, sbmlsim-core crates by git `rev`), plus Mmg 5.8.0 (the version the pinned PETSc
downloads, LGPL, with cmake) in the PETSc builds, `spade` 2 and `rstar` 0.12 (MIT/Apache-2.0) in
`mesher`, wasi-sdk 34.0, wasmtime (test host only, outside the workspace), Node 24, Playwright
with Chromium (browser tests), Python with uv/pixi in slideviz.

**Spec:** `docs/superpowers/specs/2026-11-26-slidesim-m4-lobules-design.md` (milestone 4, binding)
and its parent `docs/superpowers/specs/2026-11-24-slidesim-design.md`.

## Global Constraints

- Every constraint of the milestone 1 and 2 plans holds: MIT, pinned PETSc and libCEED,
  `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings` and all tests pass at every
  commit; collective errors (`Comm::max_int`) for anything one process finds alone; `MPI_Abort`
  in the CLI with more than one process.
- No agent attribution anywhere (no `Co-Authored-By`, no "Generated with"); never the em dash.
- No liver-specific code in slidesim or sbmlsim-core: "SPT", "CYP2E1", "necrosis", "portal",
  "central" appear only in test data, studies, pkdb_models and slideviz.
- Units in slidesim: length um, time s, concentration mM. A cell model keeps its own units; the
  study gives the length of its time unit in s (`time_unit_s`, 60 for the SPT model).
- Parity: 1, 2 and 3 processes agree within 1e-10 relative with fixed transport steps (fields,
  cell states, integrals). Cell step control is per cell and never depends on other cells, the
  process count, the libCEED backend or the blocking.
- Generated C is compiled with `-O3 -march=native -ffp-contract=off -fno-fast-math` plus `-fPIC
  -shared`; the compiler is `$CC` or `cc`; the flags, the compiler's `--version` line and the
  source are in the cache key (SHA-256). Default cache directory:
  `$XDG_CACHE_HOME/sbmlsim/c` (or `~/.cache/sbmlsim/c`), overridable by `SBMLSIM_C_CACHE`.
- The generated C uses `<math.h>` functions, not sbmlsim-core's `mathlib`; agreement with the
  WebAssembly model is within integration tolerances, not bitwise.
- Each repository follows its own style and checks (sbmlsim-core: `cargo fmt --check`,
  `cargo clippy --all-targets --all-features -- -D warnings`, `cargo test --all-features`, and
  `node --test js/test`; slideviz: its ruff, pytest and pre-commit configuration).
- sbmlsim-core: slidesim pins sbmlsim-core `develop` 85bfa545db7459ff16cbd9baab293ae26a95c0c0 by
  `rev` until a release contains its PR #119; our own libraries at their latest release
  otherwise. The wasm build takes `compiler.wasm`, `mathlib.wasm` and `js/` of sbmlsim-core at
  the head of the Task 15 branch, pinned as `SBMLSIM_CORE_KERNEL_REV` in `deps/versions.env`,
  until a release contains the kernel.
- A native slidesim run uses no WebAssembly: the crates of the workspace depend neither on
  `sbmlsim-host` nor on wasmtime; only the test host `wasm/host` (its own Cargo workspace,
  excluded like `spike`) does.
- No GPL code in the default builds (gmsh, CGAL), no non-free code (Triangle), no AGPL (TetGen):
  PETSc is never configured with `--download-triangle`, `--download-ctetgen` or similar.
- Tag names are data: `mesh` and `mesher` know no tag name; the lobule source takes its tag
  names and its map id from the scene.
- Generated meshes are deterministic: the same input (and seed) on the same target gives the
  same `TaggedMesh`, bit for bit, on any number of processes (the mesher runs on process 0).
- Machine safety (shared machine, crashed once from overload): one heavy job at a time (a test
  suite, a build, or one `mpiexec` run), never two at once; every command runs with
  `CARGO_BUILD_JOBS=8` and `NEXTEST_TEST_THREADS=8`, dependency builds with `JOBS=8`, and
  `mpiexec` with at most 8 processes. Tests use `. deps/install/native-check/env.sh`, benchmarks
  `native-opt`.
- Repositories and branches:
  - slidesim: `feat/m4-lobules` (exists, holds the design).
  - sbmlsim-core (`~/git/sbmlsim-core`): branch `feat/batched-kernel` from `origin/develop`
    (85bfa545); pushed with a pull request to `develop`.
  - slideviz (`~/git/slideviz`): branch `feat/tagged-polygons` from `main`; pull request to
    `main`.
  - Never merge a pull request, never tag or publish a release: the user does that (release
    notes and version bumps are prepared in the branches).

## Review Focus

1. Polygons whose boundary is not fully tagged, or whose areas overlap or leave a gap (a gap is
   an untagged boundary): an error naming the GeoJSON file and the coordinates, never a mesh
   with an untagged wall (Task 4 `untagged_boundary_segment_is_an_error`,
   `overlapping_areas_are_an_error`).
2. Lobule parameters which make holes overlap or leave no tissue (large radii, large jitter):
   an error naming the parameter at load, or the two holes after generation, before any meshing
   (Task 3 `lobule_parameters_out_of_range`, Task 5 `holes_overlap_is_an_error`).
3. The mesher fails on process 0 in a run on several processes: every process stops with the
   same error, none hangs (Task 6 `mesher_failure_on_process_0_stops_all`).
4. A rerun into an output folder holding the `result/1` of another study, or a run that stops
   between outputs: the result is replaced as a whole, and an unfinished one says so (Task 13
   `earlier_result_is_replaced`, `unfinished_result_is_marked`).
5. A study the browser build cannot run (engine solver, a scene that needs files, a trap or out
   of memory in the module): a JavaScript `Error` carrying slidesim's message, not a hang or a
   dead page (Task 16 `engine_solver_is_refused`, `file_scene_is_refused`; Task 17
   `trap_is_an_error`).

---

### Task 1: Spike (throwaway): Mmg through `DMAdaptMetric`, natively and in wasm32-wasip1

**Files:**
- Create: `spike/mmg/RESULT.md`, `spike/mmg/build.sh` (native and wasi PETSc with Mmg into
  `deps/install/spike-mmg-{native,wasi}`, never into the prefixes the tests use),
  `spike/mmg/lobule/` (standalone Cargo package as `spike/petsc-wasm/diffusion`: raw bindgen
  bindings, `spade`), `spike/mmg/patches/` (if any)

**Interfaces:**
- Produces: the decision GO (option B) or NO-GO (option A) in `RESULT.md`, the configure line
  and patches for Task 2, and the measured quality numbers.

- [ ] **Step 1: Native build:** PETSc v3.26.1 opt with the flags of `build-native.sh` plus
  `--download-mmg=<v5.8.0 tarball>` and `--with-cxx=0`. Expected: `has_external_package("mmg")`.
- [ ] **Step 2: wasm build:** the configure of `spike/petsc-wasm/build-wasi.sh` plus Mmg. PETSc's
  `Mmg.py` requires PT-Scotch, whose package requires MPI (the wasm build has MPIUNI): try PT-Scotch
  on MPIUNI first, else a patch making PT-Scotch optional for Mmg (Mmg builds with
  `-DUSE_SCOTCH=OFF`); record each attempt in `RESULT.md`.
- [ ] **Step 3: Lobule program:** one lobule (radius 300 um, central hole 30 um, six corner
  holes 25 um, circles polygonized at 5 um) as `spade` CDT with the boundary split to the size
  field (5 um at the hole walls, growing linearly from 10 um to 200 um distance up to 20 um),
  then `DMPlexCreateFromCellListPetsc`, one combined face label (value = tag index + 1) and one
  cell label, an isotropic metric `1/h^2` per vertex (`DMPlexMetricCreateIsotropic`),
  `-dm_plex_metric_no_surf` (frozen boundary), `DMAdaptMetric`, labels split back. Prints counts,
  min angle, min SICN (`4 sqrt(3) A / sum l^2`), edge ratios `l / h(midpoint)`, per-tag lengths,
  a hash of the output, the wall time; writes a VTU. Runs natively, in wasmtime and in Node.
- [ ] **Step 4: Go/no-go** (all must hold on both targets for GO):
  - G1: Step 1 builds; G2: Step 2 builds with at most small patches and Step 3 runs in wasmtime
    and Node;
  - G3: every input wall segment's sub-edges carry its tag, per-tag lengths equal the input
    polyline lengths within 1e-12 relative; min angle >= 20 deg; min SICN >= 0.4; every edge
    ratio in [0.4, 2.0];
  - G4: two runs on one target give bitwise identical meshes;
  - G5: native under 2 s, wasm under 10 s.
  One retry with other Mmg parameters (`hgrad`, `hausd`) is allowed for G3. NO-GO on G1 or G2
  means option A on every target (anti-goal: two meshing paths), not B natively and A in wasm.
- [ ] **Step 5: Write `RESULT.md`** (table of criteria, measured values, configure lines, patches,
  sizes of the wasm) and **report the decision to the controller, who confirms it with the user
  before Task 2.** On NO-GO: Task 2 keeps everything but Mmg, Task 4 Step 3 implements option A,
  every other task is unchanged.
- [ ] **Step 6: Commit** `spike(mmg): Mmg through DMAdaptMetric natively and in wasm32-wasip1`

### Task 2: Mmg in the dependency builds, the wasi build script, metric adaptation in `petsc`

**Files:**
- Modify: `deps/versions.env` (`MMG_VERSION=5.8.0`, `MMG_SHA256`), `scripts/deps/fetch.sh`
  (tarball, `check_petsc_package Mmg.py`), `scripts/deps/build-native.sh` (`--download-mmg`,
  `JOBS="${JOBS:-$(nproc)}"`), `.github/workflows/ci.yml` (cache key, `cmake` already installed),
  `README.md`, `docs/development.md`, `docs/licenses.md` (Mmg LGPL-3.0)
- Move: `spike/petsc-wasm/build-wasi.sh` -> `scripts/deps/build-wasi.sh`, `spike/petsc-wasm/patches/`
  and the spike's Mmg patches -> `scripts/deps/patches/`, `spike/petsc-wasm/PATCHES.md` ->
  `scripts/deps/patches/README.md`; the spike's docs point there
- Create: `crates/petsc/src/metric.rs`, `crates/petsc/tests/metric.rs`,
  `scripts/deps/wasi-smoke/smoke.c` (unit square adapted with Mmg, run in wasmtime)

**Interfaces:**
- Produces:
  - `pub struct MetricOptions { pub no_surf: bool, pub gradation: f64, pub hausdorff: f64, pub verbosity: i32 }`
    (defaults `true`, PETSc's 1.3, 0.01, -1).
  - `Dm::isotropic_metric(&self, sizes: &[f64], options: &MetricOptions) -> Result<Vector>`:
    `sizes` per local vertex in the order of the depth-0 stratum, metric `1/h^2`.
  - `Dm::adapt_metric(&self, metric: &Vector, faces: Option<&str>, cells: Option<&str>) -> Result<Dm>`
    (`DMAdaptMetric`; the labels keep their names and values on the new DM).
  - `deps/install/{native-*,wasi}` with Mmg (wasi: per the spike's decision).

- [ ] **Step 1: Write the failing tests** in `crates/petsc/tests/metric.rs`:
  - `mmg_is_available`: `petsc.has_external_package("mmg")` is true.
  - `uniform_metric_on_unit_square`: square of 2 triangles (`plex_from_cells`), size 0.1
    everywhere: cells in [150, 320]; every edge length in [0.04, 0.2].
  - `labels_survive_adaptation`: face label `sides` with values 1 to 4 on the four sides, cell
    label `area` value 7: after adaptation every boundary edge has exactly one value, the edges
    of value k lie on side k, their lengths sum to 1.0 within 1e-12 per side; every cell has
    `area` 7.
- [ ] **Step 2: Run** `cargo nextest run -p petsc --test metric`. Expected: FAIL (no Mmg, no API).
- [ ] **Step 3: Implement** the build changes, rebuild `native-check` then `native-opt` (one at a
  time), `build-wasi.sh` with the spike's route, and `metric.rs`.
- [ ] **Step 4: Run** `cargo nextest run -p petsc`, then the wasi smoke in wasmtime. Expected:
  PASS; smoke prints the adapted cell count.
- [ ] **Step 5: Commit** `build: Mmg in the PETSc builds, build-wasi.sh from the spike, metric adaptation in petsc`

### Task 3: `scene/1` mesh sources and the `polygons/1` contract

**Files:**
- Modify: `crates/contracts/src/scene.rs`, `crates/contracts/src/lib.rs`,
  `crates/contracts/src/study.rs` (`check_maps` uses `Scene::map_ids`),
  `crates/contracts/examples/write-schemas.rs`, `crates/mesh/src/lib.rs` (match on the source),
  `schemas/scene-1.schema.json`, `docs/contracts.md`
- Create: `crates/contracts/src/polygons.rs`, `schemas/polygons-1.schema.json`,
  `testdata/contracts/scene-lobules.json`, `scene-polygons.json`, `scene-lobules-bad-*.json`,
  `polygons-minimal.geojson`, `polygons-bad-*.geojson`
- Test: `crates/contracts/tests/contracts.rs`

**Interfaces:**
- Produces (the version stays `scene/1`: the change is additive, every existing scene loads):
  - `Scene::mesh: MeshSource`, `#[serde(tag = "format")] pub enum MeshSource { Gmsh { path: PathBuf }, Polygons { path: PathBuf, size: SizeField }, Lobules(LobuleSpec) }`
    (`"gmsh"`, `"polygons"`, `"lobules"`); paths resolved as today. `MeshRef` and `MeshFormat`
    are removed.
  - `pub struct SizeField { pub min_um: f64, pub max_um: f64, pub dist_min_um: f64, pub dist_max_um: f64, pub walls: Vec<String> }`:
    size `min_um` up to distance `dist_min_um` from the nearest edge of a tag in `walls`, linear
    to `max_um` at `dist_max_um`, `max_um` beyond (slideviz's Threshold field); no walls:
    `max_um` everywhere.
  - `pub struct LobuleSpec { lobule_radius_um: f64 = 300, rings: u32 = 0, jitter: f64 = 0, seed: u64 = 0, centre: HoleSpec, corner: HoleSpec, outer_tag: String, area_tag: String, size: LobuleSize, zonation: ZonationSpec }`,
    `HoleSpec { radius_um: f64, tag: String }` (centre default 30, corner default 25),
    `LobuleSize { wall_um: f64 = 5, interior_um: f64 = 20 }`,
    `ZonationSpec { map: String, um_per_px: f64 = 2 }`. `lobule_radius_um` is the circumradius of
    the hexagonal lobule; `jitter` the largest displacement of a vein as a fraction of it.
    Tag names and the map id have no default.
  - `LobuleSpec::size_field(&self) -> SizeField`: min `wall_um`, max `interior_um`, `dist_min_um
    = 2 wall_um`, `dist_max_um = 10 interior_um`, walls the centre and corner tags.
  - `Scene::map_ids(&self) -> Vec<&str>`: the ids of `maps` plus the zonation map of a lobule
    source.
  - `polygons/1`: a GeoJSON FeatureCollection with the foreign member `"version": "polygons/1"`
    and free `"meta"`; features `{"properties": {"kind": "area" | "edge", "tag": <name>, "value": <int > 0, default 1>}}`,
    areas `Polygon`/`MultiPolygon`, edges `LineString`/`MultiLineString`, coordinates in the
    scene's um. `Polygons::load(path) -> Result<Polygons, ContractError>` with
    `Polygons { areas: Vec<AreaPolygon { tag, value, rings: Vec<Vec<[f64; 2]>> }>, edges: Vec<EdgeChain { tag, value, points: Vec<[f64; 2]> }> }`
    (one `AreaPolygon` per polygon, outer ring first). Geometry checks belong to `mesher`.
  - `schema_polygons() -> serde_json::Value`.

- [ ] **Step 1: Write the failing tests:** `lobules_scene_loads_with_defaults` (only tags and map
  id given: radius 300.0, rings 0, jitter 0.0, centre 30.0, corner 25.0, wall 5.0, interior 20.0,
  um_per_px 2.0; `map_ids()` is `["zonation"]`); `polygons_scene_resolves_its_path`;
  `gmsh_scenes_still_load` (every existing `testdata` scene); `lobule_parameters_out_of_range`:
  each of radius 0, jitter 0.26, centre radius + corner radius >= (1 - 2 jitter) radius / 2,
  2 corner radius >= (1 - 2 jitter) radius / 2, wall_um > interior_um, a zonation map id equal
  to an id of `maps`, two equal tag names: `ContractError::Invalid` naming the key;
  `size_field_rules` (min > max, dist_min >= dist_max, negative: error naming `size.<key>`);
  `polygons_load` (minimal file: 2 areas, 3 edge chains, values default 1);
  `polygons_refuse_bad_kinds` (unknown kind, a LineString area, a tag used as area and as edge,
  value 0); `schemas_are_current` and `schemas_match_load` extended to `polygons-1`.
- [ ] **Step 2: Run** `cargo nextest run -p contracts`. Expected: FAIL.
- [ ] **Step 3: Implement**, regenerate the schemas (`cargo run -p contracts --example write-schemas`),
  `Mesh::load` returns `MeshError::Source { format }` for the generated formats until Task 6.
- [ ] **Step 4: Run** `cargo nextest run -p contracts -p mesh`. Expected: PASS.
- [ ] **Step 5: Commit** `feat(contracts): polygons and lobules mesh sources in scene/1, polygons/1`

### Task 4: Crate `mesher`: tagged polygons to an adapted `TaggedMesh`

**Files:**
- Create: `crates/mesher/Cargo.toml` (`contracts`, `petsc`, `spade`, `rstar`, `serde`,
  `thiserror`), `crates/mesher/src/{lib.rs, domain.rs, size.rs, triangulate.rs, adapt.rs, tagged.rs, quality.rs}`,
  `crates/mesher/tests/mesher.rs`
- Modify: `docs/licenses.md` (spade, rstar)

**Interfaces:**
- Consumes: `contracts::{Polygons, SizeField}`, Task 2's `Dm::isotropic_metric`, `Dm::adapt_metric`,
  `Dm::plex_from_cells`, `Dm::set_label_value`.
- Produces:
  - `pub struct MeshTag { pub name: String, pub value: i32, pub dim: u8 }`.
  - `pub struct Domain { pub tags: Vec<MeshTag>, pub areas: Vec<(usize, Vec<Vec<[f64; 2]>>)>, pub edges: Vec<(usize, Vec<[f64; 2]>)> }`
    (tag index, rings / polyline); `Domain::from_polygons(p: &Polygons) -> Domain`.
  - `pub struct Size`; `Size::new(spec: &SizeField, domain: &Domain) -> Result<Size, MesherError>`
    (unknown wall tag is an error), `Size::at(&self, x: [f64; 2]) -> f64` (distance by an R-tree
    of the wall segments).
  - `#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)] pub struct TaggedMesh { pub tags: Vec<MeshTag>, pub vertices: Vec<[f64; 2]>, pub triangles: Vec<[u32; 3]>, pub triangle_tags: Vec<u16>, pub edges: Vec<[u32; 2]>, pub edge_tags: Vec<u16> }`:
    triangles counterclockwise, one area tag each; edges `[a, b]` with `a < b`, one entry per
    tagged edge and tag.
  - `TaggedMesh::to_dm(&self, comm: &Comm) -> petsc::Result<Dm>` (process 0's mesh, the input of
    others ignored; one label per tag, named after it, with its value; cells and vertices in the
    order of the arrays); `TaggedMesh::from_dm(dm: &Dm) -> Result<TaggedMesh, MesherError>` (a
    serial DM; labels but PETSc's own); `TaggedMesh::quality(&self, size: Option<&Size>) -> Quality`.
  - `#[derive(Serialize)] pub struct Quality { pub triangles: usize, pub vertices: usize, pub min_angle_deg: f64, pub min_sicn: f64, pub mean_sicn: f64, pub edge_ratio: Option<[f64; 2]> }`;
    SICN of a straight triangle `4 sqrt(3) A / (l1^2 + l2^2 + l3^2)` (gmsh's for linear
    triangles, Task 8 checks it against a slideviz record); `edge_ratio` the min and max of
    `l / Size::at(midpoint)`.
  - `triangulate(domain: &Domain, size: &Size) -> Result<TaggedMesh, MesherError>`; `adapt(petsc: &Petsc, mesh: &TaggedMesh, size: &Size) -> Result<TaggedMesh, MesherError>`
    (on `PETSC_COMM_SELF`); `generate(petsc: &Petsc, domain: &Domain, size: &Size) -> Result<(TaggedMesh, Quality), MesherError>`.
  - `MesherError { UntaggedBoundary { at: [f64; 2] }, Overlap { at: [f64; 2] }, EdgeOffBoundary { tag: String, at: [f64; 2] }, HolesOverlap { a: String, b: String }, AreaTags { triangle: usize, count: usize }, Triangulation(String), Petsc(petsc::Error) }`
    (`AreaTags`: `from_dm` of a triangle without exactly one area tag),
    each naming coordinates in um.

- [ ] **Step 1: Write the failing tests** in `crates/mesher/tests/mesher.rs` (fixture built in
  code: a 100 x 100 um square split at x = 50 into areas `left` and `right`, the border tagged
  `border`, a hole of radius 20 at (25, 50) polygonized at 2 um tagged `wall`, the outer sides
  `cut`; size min 2, max 10, dist 4 to 40, walls `["wall"]`):
  - `tags_are_preserved`: every boundary edge carries exactly one tag; per tag the edge lengths
    sum to the input polyline length within 1e-12 relative (`border` 100.0); the areas of the
    triangles of `left` and `right` equal the polygon areas within 1e-12 relative.
  - `size_field_is_honoured`: `edge_ratio` within [0.4, 2.0].
  - `quality_bound`: `min_angle_deg >= 20.0`, `min_sicn >= 0.4`.
  - `deterministic`: `generate` twice gives equal `TaggedMesh` (`==`, bitwise).
  - `untagged_boundary_segment_is_an_error`: one segment of `cut` dropped:
    `UntaggedBoundary { at }` with `at` its midpoint exactly.
  - `overlapping_areas_are_an_error`: `right` shifted 1 um left: `Overlap`.
  - `edge_not_on_a_boundary_is_an_error`: an edge chain inside `left`: `EdgeOffBoundary`.
  - `dm_round_trip`: `TaggedMesh::from_dm(&mesh.to_dm(&comm_self)?)? == mesh`.
  - `quality_of_known_triangles`: equilateral: SICN 1.0 within 1e-15, min angle 60.0 within
    1e-12; right isosceles: SICN 0.8660254037844386 within 1e-15, min angle 45.0 within 1e-12.
- [ ] **Step 2: Run** `cargo nextest run -p mesher`. Expected: FAIL (crate missing).
- [ ] **Step 3: Implement.** `triangulate`: CDT of all ring and chain segments (shared borders
  inserted once, vertices matched by exact coordinates), boundary segments split to `Size::at`
  so the boundary is final, interior refined just enough for Mmg to start; triangles classified
  by the area polygon containing their centroid, the others removed; every sub-edge carries the
  tag of its input segment. `adapt` (option B): combined face and cell labels as in the spike,
  `no_surf`, labels split back. On a NO-GO of Task 1, `adapt` (option A) refines the CDT with
  `spade`'s angle limit (20 deg) and splits triangles whose longest edge exceeds `Size::at` at
  their centroid until none does; the tests are the same.
- [ ] **Step 4: Run** `cargo nextest run -p mesher`. Expected: PASS.
- [ ] **Step 5: Commit** `feat(mesher): tagged polygons to a graded mesh (spade, Mmg through PETSc)`

### Task 5: Ideal lobules and their zonation

**Files:**
- Create: `crates/mesher/src/lobules.rs`, `crates/mesher/tests/lobules.rs`

**Interfaces:**
- Consumes: `contracts::LobuleSpec`, Task 4's `Domain`, `Size`, `generate`.
- Produces:
  - `pub struct Lobules`; `Lobules::new(spec: &LobuleSpec) -> Result<Lobules, MesherError>`
    (geometry only, cheap, the same on every process); `veins(&self) -> &[[f64; 2]]`,
    `corners(&self) -> &[[f64; 2]]`, `domain(&self) -> &Domain`, `size(&self) -> Size`,
    `bounds(&self) -> [[f64; 2]; 2]`.
  - `Lobules::position(&self, x: [f64; 2]) -> f64`: `d_p / (d_p + d_c)` with `d_p` the distance
    to the nearest corner wall and `d_c` to the nearest centre wall (both clamped at 0 inside a
    hole): 0 on corner walls, 1 on centre walls.
  - Geometry: veins on a hexagonal lattice of spacing `sqrt(3) R` within ring `rings + 1`, each
    displaced by a seeded uniform point of the disc of radius `jitter R` (SplitMix64 of `seed`,
    as sbmlsim-c's tests have it); the tissue the union of the Voronoi cells of the veins within
    ring `rings` (the outer ring only bounds them); corners the vertices of these cells; holes
    circles polygonized with `max(12, ceil(2 pi r / wall_um))` vertices on the circle; the outer
    border `outer_tag`, hole walls `centre.tag` and `corner.tag`, the tissue `area_tag`.

- [ ] **Step 1: Write the failing tests** (`R` 300, centre 30, corner 25, wall 5, interior 20):
  - `single_lobule_area`: rings 0: tissue area 227072.43481658038 um^2 within 1e-3 relative.
  - `single_lobule_wall_lengths`: centre 188.49555921538757, corner 314.1592653589793 (six
    thirds), outer 1500.0, each within 5e-3 relative.
  - `ring_1_geometry`: 7 veins, 24 corners; area 1589507.0437160626 within 1e-3; corner walls
    2199.114857512855 (14 full circles), centre walls 1319.4689145077132, outer 4500.0, each
    within 5e-3 relative.
  - `jitter_is_seeded`: jitter 0.1: the same seed gives equal domains, seeds 1 and 2 differ;
    jitter 0: domains equal for any seed.
  - `holes_overlap_is_an_error`: a `LobuleSpec` built directly (not through the contract's
    checks), rings 0, corner radius 160 (adjacent corners are 300 um apart):
    `HolesOverlap` naming two corner holes, no domain.
  - `position_on_walls`: 0.0 at every corner wall vertex and 1.0 at every centre wall vertex,
    within 1e-12.
  - `position_is_monotone`: rings 0, on the line from a corner to the vein, 100 samples between
    the walls: strictly increasing.
  - `lobule_meshes`: `generate` of rings 0: tags `{area, centre, corner, outer}` present,
    `min_angle_deg >= 20.0`, `edge_ratio` within [0.4, 2.0], twice equal.
- [ ] **Step 2: Run** `cargo nextest run -p mesher --test lobules`. Expected: FAIL.
- [ ] **Step 3: Implement** `lobules.rs` (Voronoi from `spade`'s Delaunay triangulation of the
  veins or directly from the lattice; the builder's choice).
- [ ] **Step 4: Run** `cargo nextest run -p mesher`. Expected: PASS.
- [ ] **Step 5: Commit** `feat(mesher): parametric lobules and their zonation`

### Task 6: Generated meshes in runs: `Mesh::from_dm`, `slidesim mesh`, the lobule studies

**Files:**
- Modify: `crates/mesh/src/lib.rs`, `crates/mesh/Cargo.toml` (`mesher`), `crates/maps/src/lib.rs`,
  `crates/slidesim/src/{main.rs, run.rs}`, `docs/{examples.md, usage.md, contracts.md}`
- Create: `crates/slidesim/src/mesh_command.rs`, `docs/methods/meshing.md`,
  `testdata/studies/lobules/{scene.json, ring1.scene.json, spt.study.json, spt_batched.study.json, ring1_spt.study.json, ring1_spt_batched.study.json, linear.study.json}`,
  `testdata/contracts/polygons-self-intersecting.geojson` with its scene
- Test: `crates/mesh/tests/mesh.rs`, `crates/maps/tests/maps.rs`, `crates/slidesim/tests/{cli.rs, coupled.rs}`

**Interfaces:**
- Consumes: Tasks 3 to 5.
- Produces:
  - `Mesh::from_dm(petsc: &Petsc, coarse: Dm, refinements: u32) -> Result<Mesh, MeshError>`:
    the load sequence after reading (pinched split, partition, distribution, refinement).
  - `Mesh::from_tagged(petsc: &Petsc, mesh: &TaggedMesh, refinements: u32) -> Result<Mesh, MeshError>`.
  - `Mesh::load` meshes `polygons` and `lobules` sources on process 0 (`mesher::generate` on
    `PETSC_COMM_SELF`), broadcasts success or the error text (`Comm::broadcast_bytes`), then
    `from_tagged`; `MeshError::Mesher { message: String }` on every process.
  - `Mesh::generated(&self) -> Option<&(TaggedMesh, Quality)>` (process 0 only).
  - `Map::from_fn(id: &str, um_per_px: f64, origin_um: [f64; 2], shape: [usize; 2], bounds: [[f64; 2]; 2], f: impl Fn([f64; 2]) -> f64) -> Map`
    (values at the pixel centres of the window `Map::read` would read); `Maps::insert(&mut self, map: Map)`.
  - The run builds the zonation map of a lobule source with `Lobules::position` over the local
    bounds (grid: `um_per_px`, origin and shape covering `Lobules::bounds`).
  - `slidesim mesh <scene.json> --out <dir>` (one process; more is a usage error): writes
    `<dir>/mesh.json` (`TaggedMesh`, serde JSON, any source; a gmsh file read serially and
    converted with `TaggedMesh::from_dm`), `<dir>/mesh.vtu` (coarse level, one cell array per area tag) and
    `<dir>/quality.json` (`Quality`); prints the quality in one line.
  - `testdata/studies/lobules`: `scene.json` (rings 0) and `ring1.scene.json` (rings 1) with tags
    `tissue`, `central_wall`, `portal_wall`, `field_cut` and map `zonation`; the SPT studies on
    each scene as `cells_toy/spt.study.json` (engine and batched, constant 5 mM at
    `portal_wall`, 24 h, hourly outputs); `linear.study.json` the linear uptake model for 1 h.

- [ ] **Step 1: Write the failing tests:**
  - `mesh.rs`: `from_tagged_matches_load_of_the_same_mesh` (the `TaggedMesh` of `lobule_toy.msh`
    via `from_dm`: cell count, tags and area equal `Mesh::load`); `lobule_scene_loads` (rings 0,
    refinements 1: tags as above, area 227072.43481658038 within 1e-3 relative, 4 times the
    coarse cells).
  - `maps.rs`: `from_fn_samples_exactly_at_pixel_centres` (f = x + 2y: `sample` equals f within
    1e-12 inside the window, `gradient` [1, 2] within 1e-12).
  - `cli.rs`: `mesh_command_writes_mesh_and_quality` (lobule scene: three files, `min_angle_deg
    >= 20`, `mesh.json` loads into a `TaggedMesh` with the four tags); `lobule_spt_runs` (the
    batched study shortened to 30 min with `variant`: exit 0, every balance closes as
    `assert_closes`); `cells_parallel_matches_one_on_generated_lobule`
    (`cells_parallel_matches_one_on` with the lobule study, 2 and 3 processes);
    `mesher_failure_on_process_0_stops_all` (the self-intersecting polygons scene on 2
    processes: both exit 1 within the timeout, stderr of each names the GeoJSON file and the
    crossing point).
  - `coupled.rs`: `coupled_conservation_on_generated_lobule` (as `coupled_conservation`);
    `strang_order_two_on_generated_lobule` (linear study, split steps 240, 120, 60 s: the ratio
    of the successive differences of the cell amounts at 1 h >= 3.5).
- [ ] **Step 2: Run** `cargo nextest run -p mesh -p maps` and `cargo nextest run -p slidesim`
  (one after the other). Expected: FAIL.
- [ ] **Step 3: Implement**; document the sources, `slidesim mesh` and the method (CDT, metric,
  Mmg, frozen boundary, zonation formula) in the docs.
- [ ] **Step 4: Run** the same commands. Expected: PASS, every milestone 3 test included.
- [ ] **Step 5: Commit** `feat: meshes generated from polygons and lobules in runs, slidesim mesh`

### Task 7: slideviz: tagged polygons as the hand-over

**Files (in `~/git/slideviz`, branch `feat/tagged-polygons` from `main`):**
- Create: `mesh/edges.py` (boundary chains and their classes, without gmsh), `tests/test_edges.py`
- Modify: `mesh/polygons.py` (writes `polygons/1`: area features with `kind`, `tag`, `value` of
  `AREA_TAGS`, edge features of `EDGE_TAGS`; option `--variant plain|follow` as `build_mesh.py`
  has it; reads the lumen layers of the inputs), `mesh/build_mesh.py` (imports the moved
  classification), `scripts/write_scene.py` (`--polygons <geojson> --size-um SMALL:LARGE` as an
  alternative to `--mesh`), `schemas/scene-1.schema.json` and `schemas/polygons-1.schema.json`
  (copied from slidesim), `tests/test_write_scene.py`, `scripts/README.md`

**Interfaces:**
- Consumes: slidesim's `polygons/1` and `scene/1` schemas (Task 3).
- Produces: `<animal>_polygons_<tol>um.geojson` in `polygons/1`; a scene with
  `"mesh": {"format": "polygons", "path": ..., "size": {"min_um": SMALL, "max_um": LARGE, "dist_min_um": 2 SMALL, "dist_max_um": 10 LARGE, "walls": ["vessel_wall", "portal_wall", "central_wall", "section_edge", "necrosis_border"]}}`
  (`REFINED`, slideviz's sizes); every boundary segment of the domain in exactly one edge
  feature: `necrosis_border` between two areas, `field_cut` on the field square, a wall
  (`portal_wall`, `central_wall` by the lumen class vote of `classify_walls`, else
  `vessel_wall`) beside a lumen or around a hole, else `section_edge`.

- [ ] **Step 1: Write the failing tests:** `test_edges.py::test_edges_cover_the_boundary`
  (synthetic field: square with a portal lumen hole and a necrotic patch: each boundary segment
  once; hole ring `portal_wall`, square sides `field_cut`, patch border `necrosis_border`);
  `test_edges.py::test_wall_classes_match_build_mesh` (the fixture of `test_wall_tags.py`: the
  same class per wall chain as `build_mesh.classify_walls`);
  `test_write_scene.py::test_polygons_scene_validates` (schema-valid, the size as above for
  `--size-um 5:20`).
- [ ] **Step 2: Run** `uv run pytest tests/test_edges.py tests/test_write_scene.py`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the repository's checks (pytest, ruff, pre-commit). Expected: PASS.
- [ ] **Step 5: Commit, push, open a pull request** to `main`:
  `Tagged polygons (polygons/1) for slidesim's mesher, polygons scenes`.

### Task 8: Real-slide example fields and the comparison with gmsh

**Files:**
- Create: `testdata/studies/slide_fields/<field>/{scene.json, polygons.geojson, zonation.zarr, cyp2e1.zarr, necrosis.zarr, spt.study.json, spt_batched.study.json}`
  for three fields, `testdata/studies/slide_fields/README.md` (license, credit line, provenance:
  slide, centre, size, slideviz commit, commands), `docs/results/2026-11-m4-meshing.md`
- Modify: `docs/examples.md`, `crates/slidesim/tests/cli.rs`
- slideviz (branch of Task 7): delete `mesh/build_mesh.py` and its gmsh/DOLFINx-only parts of
  `mesh/pixi.toml` if Step 4 passes

**Interfaces:**
- Consumes: Tasks 6 and 7.

- [ ] **Step 1: Confirm the data license and credit line with the user (the controller asks;
  CC BY 4.0 proposed).** Nothing under `testdata/studies/slide_fields` is committed before the
  answer.
- [ ] **Step 2: Generate** with the commands of `docs/results/2026-11-m3-wall-classes.md`
  ("Reproduce"), `polygons.py` of Task 7 and `write_scene.py --polygons`: three fields of 300 to
  500 um around a central vein with its portal fields (one control `000mg_*`, two dosed), maps
  cropped to the field; the folder at most 5 MB (`du -sh`).
- [ ] **Step 3: Write the failing tests** in `cli.rs`: `slide_field_examples_mesh` (each scene:
  `slidesim mesh` exits 0, tags include `portal_wall` and `central_wall`, area equals the polygon
  area within 1e-9 relative, `min_sicn >= 0.2`); `slide_field_spt_runs` (one field, batched,
  shortened to 10 min: balances close).
- [ ] **Step 4: Compare** on the milestone 2 field (`000mg_m1`) and the milestone 3 field
  (`281mg_m1`), 1500 um, sizes 5:20: `slidesim mesh` on the gmsh scene and on the polygons scene.
  Pass when: the same tag names; tissue area within 0.5 %; per tag edge length within 1 %;
  triangles within 25 %; `min_sicn` of the Rust mesh >= gmsh's minus 0.1; and `min_sicn` of the
  gmsh mesh equals the value of its slideviz record within 1e-6 (pins the SICN definition).
  Table and verdict in `docs/results/2026-11-m4-meshing.md`. If it passes, retire
  `build_mesh.py` in slideviz (commit `Retire build_mesh.py: slidesim meshes the polygons`,
  push); if not, keep it and record why.
- [ ] **Step 5: Run** `cargo nextest run -p slidesim --test cli`. Expected: PASS.
- [ ] **Step 6: Commit** `feat(examples): real-slide fields meshed from slideviz polygons, comparison with gmsh`

### Task 9: Partition-independent element ids and the canonical finest mesh

**Files:**
- Modify: `crates/petsc/src/{lib.rs, dm.rs}`, `crates/mesh/src/lib.rs`, `crates/cells/src/{lib.rs, parameters.rs}`,
  `crates/pde/src/problem.rs`
- Create: `crates/mesh/src/canonical.rs`
- Test: `crates/petsc/tests/basics.rs`, `crates/mesh/tests/mesh.rs`, `crates/cells/tests/batched.rs`

**Interfaces:**
- Produces:
  - `Comm::gather_bytes(&self, root: usize, data: &[u8]) -> Result<Option<Vec<Vec<u8>>>>` (Some
    on `root`, per process); `Dm::vertex_numbering(&self) -> Result<Vec<PetscInt>>`
    (`DMPlexGetVertexNumbering`, as `cell_numbering`).
  - `Mesh::element_ids(&self) -> &[u64]`: per local coarse cell its number in the undistributed
    coarse mesh (after the pinched split), carried through `DMPlexDistribute` by the migration SF.
  - `Mesh::fine_element_id(&self, level: usize, cell: usize) -> u64`:
    `(element_ids[cell >> (dim level)] << (dim level)) | (cell & ((1 << (dim level)) - 1))`.
  - `Mesh::canonical(&self) -> Result<Option<CanonicalMesh>>` (collective, Some on process 0):
    `CanonicalMesh { vertices: Vec<[f64; 2]>, triangles: Vec<[u32; 3]>, element: Vec<u64>, tags: Vec<CanonicalTag> }`,
    the finest level's triangles sorted by fine element id, vertices numbered by first
    appearance, `element` the coarse element id of each triangle,
    `CanonicalTag { name, dim, triangles: Vec<u32>, edges: Vec<[u32; 2]> }`.
  - `Mesh::gather_vertex_values(&self, values: &[f64]) -> Result<Option<Vec<f64>>>` (local
    vertices of the finest level, depth-0 order, to canonical order on process 0) and
    `Mesh::gather_element_values(&self, ids: &[u64], values: &[f64], width: usize) -> Result<Option<(Vec<u64>, Vec<f64>)>>`
    (rows sorted by element id).
  - `Problem::vertex_values(&self) -> Result<Vec<f64>>` (per local finest vertex, depth-0 order,
    Dirichlet values included).
  - `CellsError::Element` names the element by its element id (milestone 3 final review).

- [ ] **Step 1: Write the failing tests:** `basics.rs::gather_bytes_collects_in_rank_order` (2
  and 3 processes); `mesh.rs::element_ids_are_partition_independent` (on `lobule_toy.msh` and the
  lobule scene, 1, 2, 3 processes: each id 0..n-1 exactly once over the processes, the centroid
  of each id equal to 1 process's within 1e-12); `fine_element_ids_are_partition_independent`
  (levels 1 and 2, the same); `canonical_mesh_is_partition_independent` (all arrays of
  `canonical()` equal across 1, 2, 3 processes, bitwise); `gathered_vertex_values_follow_canonical_order`
  (values `x + 2y` at the local vertices: gathered equals `x + 2y` of the canonical vertices
  exactly); `batched.rs::failure_on_one_process_stops_all` asserts the element id in the message.
- [ ] **Step 2: Run** `cargo nextest run -p petsc -p mesh`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `cargo nextest run -p petsc -p mesh -p cells -p pde`. Expected: PASS.
- [ ] **Step 5: Commit** `feat(mesh): partition-independent element ids and the canonical finest mesh`

### Task 10: Library crate `simulation`; `slidesim run` a thin CLI

**Files:**
- Create: `crates/simulation/Cargo.toml`, `crates/simulation/src/{lib.rs, inputs.rs, phases.rs, summary.rs, vtu.rs}`,
  `crates/simulation/tests/simulation.rs`
- Move: `crates/slidesim/src/coupled.rs` -> `crates/simulation/src/coupled.rs`; `Phases` and the
  summary types of `run.rs` -> the library
- Modify: `crates/slidesim/src/{main.rs, run.rs}`, `crates/slidesim/Cargo.toml`

**Interfaces:**
- Consumes: Task 9's canonical gathers; Task 6's `Mesh::from_tagged`.
- Produces:
  - `pub struct Inputs { pub study: Study, pub scene: Scene, pub model_sbml: Option<String>, pub mesh: Option<TaggedMesh> }`;
    `Inputs::load(study_path: &Path) -> Result<Inputs>` (the SBML relative to the study);
    `mesh: Some` replaces the scene's meshing (a generated scene keeps its zonation map).
  - `pub struct Simulation<'p>`; `Simulation::new(petsc: &'p Petsc, inputs: Inputs, started: Instant) -> Result<Simulation<'p>>`
    (all checks, setup and the flow, as `run` does today);
    `next_output(&mut self) -> Result<Option<usize>>` (steps to the next output time, records
    its summary values, `Some(index)`; after the last one steps to `time.end` and returns `None`);
    `time(&self) -> f64`, `output_times(&self) -> &[f64]`, `mesh(&self) -> &Mesh`,
    `field_ids(&self) -> Vec<&str>`, `field_values(&self, field: usize) -> Result<Vec<f64>>`
    (`Problem::vertex_values`), `cell_value_names(&self) -> Vec<String>`,
    `cell_values(&self) -> Option<(&[u64], Vec<f64>)>` (element ids of the local cells, one row
    per cell), `summary(&self) -> &Summary` (`Serialize`, the `summary.json` of today),
    `write_vtu(&self, out: &Path, index: usize) -> Result<()>`, `write_flow(&self, out: &Path) -> Result<()>`.
  - `slidesim run` = `Inputs::load`, `Simulation::new`, the loop over `next_output` writing the
    VTU files, `summary.json`. Every output and message of milestone 3 unchanged.

- [ ] **Step 1: Write the failing tests** in `simulation.rs`: `simulation_matches_cli`
  (`cells_toy/spt.study.json`: `summary()` serialized equals the CLI's `summary.json` but for
  `phases`); `next_output_visits_every_output` (indices 0..n in order, `time()` equals each
  output time, then `None` with `time()` equal to `time.end`); `mesh_override_matches_generated`
  (lobule linear study: with `mesh: Some(mesh.json of slidesim mesh)` the summary equals the run
  without, bitwise, 1 process).
- [ ] **Step 2: Run** `cargo nextest run -p simulation`. Expected: FAIL.
- [ ] **Step 3: Implement** (move, no behaviour change).
- [ ] **Step 4: Run** `cargo nextest run -p simulation`, then `cargo nextest run -p slidesim`.
  Expected: PASS, the CLI tests unchanged.
- [ ] **Step 5: Commit** `refactor: the coupling driver as library crate simulation, slidesim run over it`

### Task 11: Speed: transport linear solves and the linear tolerance

**Files:**
- Modify: `crates/contracts/src/study.rs` (`solver.linear_rtol`), `schemas/study-1.schema.json`,
  `crates/pde/src/{problem.rs, multigrid.rs}`, `crates/simulation/src/summary.rs`,
  `docs/{contracts.md, usage.md}`
- Create: `scripts/bench/coupled.sh` (the 281 mg/kg 24 h batched study on 8 processes; the lobule
  24 h batched study on 1 process; prints the phases), `docs/results/2026-11-m4-performance.md`
- Test: `crates/contracts/tests/contracts.rs`, `crates/pde/tests/verification.rs`,
  `crates/slidesim/tests/cli.rs`

**Interfaces:**
- Produces: `SolverOptions::linear_rtol: f64` (default 1e-10, in (0, 1)): the KSP relative
  tolerance of the transport (the parity setting is the default; `petsc_options` still override);
  `summary.json` gains `"solver": {"linear_rtol": ...}`. The time integration has no error
  estimate with fixed steps, so the tolerance is a number the study sets, not derived.

- [ ] **Step 1: Re-measure the baseline** on a quiet machine (load average below 1, recorded with
  `uptime` before and after): `scripts/bench/coupled.sh` both studies; phases into the results
  doc.
- [ ] **Step 2: Write the failing tests:** `linear_rtol_default_and_bounds` (default 1e-10; 0 and
  1 refused naming `solver.linear_rtol`); `linear_rtol_sets_ksp_rtol` (1e-6 read back from the
  KSP; `-ksp_rtol 1e-8` in `petsc_options` wins); `summary_reports_linear_rtol`.
- [ ] **Step 3: Run** `cargo nextest run -p contracts -p pde`. Expected: FAIL.
- [ ] **Step 4: Implement** the option; then each candidate, measured with `coupled.sh` and kept
  only if the transport phase drops by at least 3 % with every test passing: the BDF
  extrapolation as the initial guess, the coarse solve (direct or redundant LU on the coarse
  level, against today's), and for production `linear_rtol` 1e-6 and 1e-8 (the largest relative
  field difference and the largest cell state difference at 24 h against 1e-10, reported). The
  example studies take the chosen production value; the parity tests pin the tight settings
  (linear rtol 1e-12, cells 1e-12/1e-16). `lobule_spt_runs` runs the example study itself: its
  `variant` pins `linear_rtol`, or its balance check scales with the storage magnitude, once the
  example studies get a production `linear_rtol`.
- [ ] **Step 5: Run** `cargo nextest run --workspace`. Expected: PASS (parity within 1e-10, the
  balances and the Strang order tests unchanged).
- [ ] **Step 6: Commit** `perf(pde): faster transport solves, solver.linear_rtol`

### Task 12: Speed: cell work balance, the exchange, one compile on a cold cache

**Files:**
- Modify: `crates/petsc/src/lib.rs`, `crates/cells/src/{lib.rs, batched.rs, engine.rs, model.rs}`,
  `crates/contracts/src/study.rs` (`cells.balance`), `schemas/study-1.schema.json`,
  `crates/pde/src/exchange.rs`, `crates/simulation/src/coupled.rs`, `docs/results/2026-11-m4-performance.md`
- Create: `crates/cells/src/balance.rs`
- Test: `crates/cells/tests/{batched.rs, engine.rs}`, `crates/slidesim/tests/cli.rs`

**Interfaces:**
- Produces:
  - `Comm::exchange_bytes(&self, sends: &[Vec<u8>]) -> Result<Vec<Vec<u8>>>` (all-to-all, one
    buffer per process).
  - `CellsSpec::balance: bool` (default `true`): before each cell step the cells are spread over
    the processes by their step attempts of the step before (their own work, known exactly; the
    mesh partition stays), states, parameters, blood means and step control sent with them and
    the results sent back. A cell's steps do not change, so results are bitwise those without.
  - `CellModel::new` compiles on process 0 first, then the others load from the cache.

- [ ] **Step 1: Write the failing tests:** `batched.rs::balanced_cells_match_unbalanced` and
  `engine.rs::balanced_cells_match_unbalanced` (2 and 3 processes: states and counts bitwise equal
  with `balance` true and false); `batched.rs::balance_evens_the_work` (cells of process 0 given
  10 times the attempts by their parameters: after one step the largest attempts per process
  over the mean <= 1.1); `cli.rs::compile_once_on_cold_cache` (`CC` a wrapper script appending
  to a log, empty cache, 3 processes: one compiler call).
- [ ] **Step 2: Run** `cargo nextest run -p cells`. Expected: FAIL.
- [ ] **Step 3: Implement**; then reuse the exchange buffers (milestone 3's O1) from a profile;
  measure each change with `coupled.sh`.
- [ ] **Step 4: Run** `cargo nextest run --workspace`, then `scripts/bench/coupled.sh`. Expected:
  PASS; the 24 h 281 mg/kg run against the target 300 s and the lobule run reported in the
  results doc (before/after per phase, parity and balances unchanged).
- [ ] **Step 5: Commit** `perf(cells): cell work balance, reused exchange buffers, one compile on a cold cache`

### Task 13: `result/1`

**Files:**
- Create: `crates/contracts/src/result.rs`, `schemas/result-1.schema.json`,
  `crates/simulation/src/result.rs`, `crates/simulation/tests/result.rs`
- Modify: `crates/contracts/src/lib.rs`, `crates/contracts/examples/write-schemas.rs`,
  `crates/simulation/Cargo.toml` (`zarrs`), `crates/slidesim/src/run.rs`, `docs/{usage.md, contracts.md}`
- Test: `crates/slidesim/tests/cli.rs`

**Interfaces:**
- Consumes: Task 9's `canonical`, `gather_vertex_values`, `gather_element_values`; Task 10's
  `Simulation`.
- Produces:
  - Store `<out>/result.zarr` (Zarr v3, written by process 0), root attributes `ResultMeta`:
    `{ version: "result/1", study: String, scene: String, refinements: u32, times: Vec<f64>, written: usize, complete: bool, fields: Vec<{id, units: "mM"}>, cell_values: Vec<String>, tags: Vec<{name, dim}>, raster: Option<RasterMeta> }`.
  - Arrays: `time` f64 [T]; `mesh/vertices` f64 [V, 2]; `mesh/triangles` u32 [N, 3];
    `mesh/element` u64 [N]; `mesh/tags/<name>` u32 [k] (area: triangles) or [k, 2] (edge: vertex
    pairs); `fields/<id>` f64 [T, V], chunks [1, V]; `cells/element` u64 [C]; `cells/<value>` f64
    [T, C], chunks [1, C]. zstd. These names are also the names of the browser build's arrays.
  - `ResultWriter::create(out: &Path, sim: &Simulation) -> Result<ResultWriter>` (removes an
    earlier `result.zarr` whole), `ResultWriter::write(&mut self, sim: &Simulation, index: usize) -> Result<()>`
    (updates `written`, sets `complete` with the last output);
    `read_result(path: &Path) -> Result<ResultData>` (all arrays, for tests and tools).
  - `slidesim run` writes it beside the VTU files.

- [ ] **Step 1: Write the failing tests** in `result.rs`: `result_round_trip` (`cells_toy/spt`:
  the mesh equals `canonical()`, every field and cell value at every time equals the gathered
  in-memory values, bitwise); `result_metadata_is_schema_valid`; `one_output_time_reads_alone`
  (chunk shapes [1, V] and [1, C]); `result_is_partition_independent` (1, 2, 3 processes: mesh
  arrays bitwise equal, fields within 1e-10 relative); `unfinished_result_is_marked` (writer
  dropped after 2 of 4 outputs: `written` 2, `complete` false); in `cli.rs`
  `earlier_result_is_replaced` (`lobule_toy` then `cells_toy` into one `--out`: no
  `fields/sucrose`, `fields` exactly `S`, `P`); `schemas_are_current` extended.
- [ ] **Step 2: Run** `cargo nextest run -p simulation --test result`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `cargo nextest run -p contracts -p simulation`, then `-p slidesim`.
  Expected: PASS.
- [ ] **Step 5: Commit** `feat: result/1, the mesh-native result store`

### Task 14: Optional raster pyramid in `result/1`

**Files:**
- Modify: `crates/contracts/src/{study.rs, result.rs}` (`result.raster`), `schemas/*`,
  `crates/simulation/src/result.rs`, `docs/{contracts.md, usage.md}`
- Create: `crates/simulation/src/raster.rs`
- Test: `crates/simulation/tests/result.rs`

**Interfaces:**
- Produces: `study/1` optional `"result": {"raster": {"um_per_px": f64 > 0, "origin_um": [x, y] (default: the mesh's lower corner), "levels": u32 >= 1}}`
  (a slide's frame for overlays); arrays `raster/<field>/<level>` f32 [T, rows, cols], chunks
  [1, 256, 256]; level 0 the P1 value at each pixel centre (NaN outside the mesh), level k+1 the
  mean of the valid pixels of 2 x 2 of level k; `RasterMeta { um_per_px, origin_um, levels, shapes }`.

- [ ] **Step 1: Write the failing tests:** `raster_of_a_linear_field_is_exact` (synthetic
  `CanonicalMesh` and values `x + 2y`: every pixel inside equals `x + 2y` at its centre within
  1e-6 (f32), NaN outside); `pyramid_levels_are_means` (level 1 pixel = mean of its valid level 0
  pixels within 1e-6); `no_raster_without_the_option`.
- [ ] **Step 2: Run** `cargo nextest run -p simulation --test result`. Expected: FAIL.
- [ ] **Step 3: Implement** (point location on process 0 with a uniform grid of triangle boxes).
- [ ] **Step 4: Run** `cargo nextest run -p contracts -p simulation`. Expected: PASS.
- [ ] **Step 5: Commit** `feat: optional raster pyramid in result/1`

### Task 15: sbmlsim-core: the batched wasm kernel

**Files (in `~/git/sbmlsim-core`, branch `feat/batched-kernel` from `origin/develop`):**
- Create: `crates/codegen/src/kernel.rs` (and submodules for the integrator), `crates/codegen/tests/kernel.rs`,
  `js/kernel.mjs`, `js/test/kernel.test.mjs`
- Modify: `crates/abi/src/lib.rs` (`KernelDescription`, `KERNEL_CONTRACT = "kernel/1"`),
  `crates/compiler/src/lib.rs`, `crates/wasm/src/lib.rs` (export `compile_kernel`), `crates/cli`
  (`compile --kernel`), `js/compile.mjs`, `docs/javascript.md`, `docs/compiling.md`, the release
  notes of the next version (prepared, not released)

**Interfaces:**
- Consumes: `sbmlsim_c::{generate_qfunction, run_points, RosenbrockContext, CONTROL, COUNTS}` for
  the comparison.
- Produces:
  - `sbmlsim_codegen::emit_kernel(model: &Model) -> Result<(Vec<u8>, KernelDescription), CompileError>`;
    `sbmlsim_compiler::compile_kernel(sbml: &str) -> Result<CompiledKernel { wasm, description }, CompileError>`;
    `compiler.wasm` export `compile_kernel`; JS `compiler.compileKernel(sbml) -> { wasm, description }`;
    CLI `sbmlsim-core compile <sbml> --kernel -o <dir>` writing `kernel.wasm` and `kernel.json`.
  - `kernel/1` module: imports `env.memory` (the caller's) and `math.<f>` of mathlib, as
    `model.wasm`; no data segments, no mutable globals, no memory growth; exports
    `sbmlsim_kernel_init(n: i32, parameters: i32, state: i32) -> i32` (initial values and
    initial assignments from the parameters) and
    `sbmlsim_kernel_advance(n: i32, state: i32, parameters: i32, control: i32, counts: i32, context: i32, scratch: i32) -> i32`
    (pointers into the memory, every array point by point as `Points`; `context` the five f64 of
    `RosenbrockContext`; `scratch` `scratch_per_point` f64 per point; per point RODAS4 step
    attempts as the QFunction's, up to `attempts`, until done or failed; returns the failed
    points, -1 for invalid arguments). Scalar per point (no SIMD128 in this milestone).
  - `KernelDescription { version, state_ids, parameter_ids, control: 4, counts: 2, scratch_per_point, model: ModelDescription }`.
  - `js/kernel.mjs`: `instantiateKernel({ wasm, mathlib, memory })` for hosts and tests.

- [ ] **Step 1: Write the failing tests:** `kernel.rs::kernel_matches_qfunction` (SPT cell model
  and the ODE cases of the QFunction tests, 200 SplitMix64 points, rtol 1e-6, atol 1e-9: states
  within `4e-6 |y| + 4e-9` of `run_points`); `kernel_init_matches_engine` (initial states equal
  the engine's within 1e-14 relative); `kernel_independent_of_block` (one call of all points
  against one call per point: states and counts bitwise equal); `kernel_refuses_events` (the
  `Unsupported` message family of the QFunction); `kernel_module_imports_only_memory_and_math`;
  `js/test/kernel.test.mjs`: `compileKernel` bytes equal the CLI's, and the decay model advanced
  to t = 1 in Node equals `exp(-k)` within 1e-6.
- [ ] **Step 2: Run** `cargo test -p sbmlsim-codegen --test kernel`. Expected: FAIL.
- [ ] **Step 3: Implement** (the integrator emitted in wasm with the model's sizes known, dense
  LU as `sbmlsim_rosenbrock.h`).
- [ ] **Step 4: Run** the repository's checks (fmt, clippy all features, tests all features,
  `node --test js/test`). Expected: PASS.
- [ ] **Step 5: Commit, push, open a pull request** to `develop`:
  `Batched wasm kernel: a model with its RODAS4 integrator in one module (kernel/1)`.

### Task 16: slidesim in `wasm32-wasip1`: kernel cells, exports, the wasmtime host, agreement

**Files:**
- Modify: `crates/petsc-sys/build.rs`, `crates/ceed-sys/build.rs` (wasm: static archives of
  `deps/install/wasi`, the emulation libraries, the sysroot, `-fvisibility=default`),
  `crates/petsc/src/lib.rs` (`Comm` on MPIUNI: one process), `crates/pde/build.rs` (QFunctions
  with wasi-sdk's clang), `crates/cells/{Cargo.toml, src/lib.rs}` (feature `native`, default:
  engine and batched paths, `sbmlsim-c`, `sbmlsim-engine`), `crates/simulation/{Cargo.toml, src/inputs.rs}`
  (feature `native`; on wasi `Inputs::kernel: KernelLayout` in place of `model_sbml`), `deps/versions.env`
  (`SBMLSIM_CORE_KERNEL_REV`), `.github/workflows/ci.yml` (job `wasm`: cached wasi deps, build,
  host tests), `Cargo.toml` (`exclude = ["spike", "wasm"]`)
- Create: `crates/cells/src/kernel.rs` (`KernelCells`), `crates/slidesim-wasm/{Cargo.toml, src/lib.rs}`
  (cdylib, reactor), `scripts/wasm/build.sh` (`slidesim.wasm`; sbmlsim-core artifacts at the
  pinned rev into `deps/install/sbmlsim-js/`), `wasm/host/{Cargo.toml, src/lib.rs, tests/agreement.rs}`
  (wasmtime, wasmtime-wasi), `testdata/studies/lobules/spt_1h.study.json` (batched, refinements
  0, linear_rtol 1e-10)

**Interfaces:**
- Consumes: Task 10's `Simulation`, Task 13's array names, Task 15's `kernel/1`.
- Produces:
  - `cells::KernelLayout { state_ids: Vec<String>, parameter_ids: Vec<String>, scratch_per_point: usize, model: sbmlsim_abi::ModelDescription }`,
    deserialized from `kernel.json` (other keys ignored), so the Cargo pin of sbmlsim-core stays;
    the checks of `CellModel` against the study run on its `model`.
  - `KernelCells` (`CellSolver`, wasi only): parameters from `CellSetup`, states from
    `sbmlsim_kernel_init`, one `sbmlsim_kernel_advance` per cell step for all cells (attempts
    `MAX_ATTEMPTS`), scratch owned by slidesim. In the browser build `"solver": "batched"` runs
    on the kernel; `"engine"` is refused with
    `cells.solver "engine" needs native code; the browser build runs "batched"`.
  - `slidesim.wasm` imports `slidesim_kernel.init` and `slidesim_kernel.advance` (the kernel's
    signatures; hosts bind them late to a kernel instantiated with slidesim's exported memory)
    and exports, over its memory: `slidesim_alloc(len) -> ptr`, `slidesim_free(ptr, len)`,
    `slidesim_create(ptr, len) -> i32` (JSON `{"scene", "study", "kernel", "mesh"?}`, `kernel`
    the `kernel.json`;
    handle or -1), `slidesim_next_output(h) -> i32` (index, -1 done, -2 error),
    `slidesim_time(h) -> f64`, `slidesim_array(h, name_ptr, name_len) -> ptr` with
    `slidesim_array_len() -> u32` (bytes; the names of `result/1`, plus `summary`, `meta`
    (`ResultMeta`) and `quality` (of a generated mesh) as JSON), `slidesim_error_ptr() -> ptr`, `slidesim_error_len() -> u32`,
    `slidesim_destroy(h)`.
  - `wasm/host`: `run(slidesim: &Path, kernel_dir: &Path, mathlib: &Path, inputs: &str) -> Result<Outputs>`
    (all arrays per output).

- [ ] **Step 1: Write the failing tests** in `wasm/host/tests/agreement.rs` (artifact paths from
  `scripts/wasm/build.sh`'s env file):
  - `lobule_matches_native`: native `slidesim mesh` of `lobules/scene.json` and native `slidesim
    run` of `spt_1h.study.json` (1 process) against the wasm run on that `mesh.json`: mesh arrays
    bitwise equal; fields at every output within 1e-5 of the field's maximum; cell states within
    `1e-4 |y| + 1e-8` (the batched bound after one split step is checked in Task 15; larger
    differences are investigated before any bound moves).
  - `wasm_generated_lobule_is_valid`: the wasm build meshes the lobule itself: tags present,
    `min_angle_deg >= 20`, `min_sicn >= 0.4`, `edge_ratio` within [0.4, 2.0]; two runs bitwise
    equal (not compared with the native mesh: Mmg runs on each platform's libm).
  - `engine_solver_is_refused`: the message above; `file_scene_is_refused`: a gmsh scene: an error
    naming its path.
- [ ] **Step 2: Run** `scripts/wasm/build.sh` then `cargo test --manifest-path wasm/host/Cargo.toml`.
  Expected: FAIL (no wasm target support).
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the same, then `cargo nextest run --workspace` (native unchanged). Expected:
  PASS; the CI job `wasm` passes locally in its Docker image.
- [ ] **Step 5: Commit** `feat(wasm): slidesim in wasm32-wasip1 with the batched kernel, wasmtime host`

### Task 17: The browser: JavaScript glue, Node and headless browser tests, the lobule demo

**Files:**
- Create: `wasm/js/{package.json, slidesim.mjs}` (Playwright as dev dependency),
  `wasm/js/test/{node.test.mjs, browser.test.mjs}`, `docs/demo/lobule/{index.html, demo.mjs}`,
  `docs/demo.md`
- Modify: `zensical.toml` (nav), `.github/workflows/docs.yml` (a job builds the bundle with
  cached wasi deps; the site gets it under `demo/lobule/`; nothing binary committed),
  `.github/workflows/ci.yml` (Node and browser tests in job `wasm`), `docs/usage.md`

**Interfaces:**
- Consumes: Task 16's exports, sbmlsim-core's `js/compile.mjs` and `js/kernel.mjs`.
- Produces:
  - `lobuleScene(params, tags) -> object` (a `scene/1` with a lobule source, the defaults of
    Task 3; the tag and map names are supplied by the caller, the demo passes `tissue`,
    `central_wall`, `portal_wall`, `field_cut` and `zonation`; `wasm/js/slidesim.d.ts` is the
    fixed declaration, Task 3b).
  - `createSimulation({ slidesim, compiler, mathlib, sbml, scene, study, mesh?, memory? }) -> Promise<Simulation>`
    (`memory: { maximumPages }` caps the wasm memory in 64 KiB pages, exceeding it is an `Error`)
    (modules or bytes; compiles the kernel with `compileKernel`); `sim.mesh` `{ vertices: Float64Array, triangles: Uint32Array, element: BigUint64Array, tags }`,
    `sim.times`, `sim.time`, `sim.step() -> number | null`, `sim.field(id) -> Float64Array`,
    `sim.cellIds -> BigUint64Array`, `sim.cellValues(name) -> Float64Array`, `sim.summary() -> object`,
    `sim.meshQuality() -> object | null`, `sim.free()`; slidesim's errors rejected or thrown as `Error` with its message.
  - The demo page: lobule parameters, start/stop, field and cell value select, the field drawn
    on a canvas, time and the summary's totals.

- [ ] **Step 1: Write the failing tests:** `node.test.mjs`: `lobule_runs_in_node` (lobule SPT
  study, 30 min: `step()` visits each output, fields finite, every balance closes within 1e-8 of
  its largest term); `node_matches_wasmtime` (on the shared `mesh.json`: arrays bitwise equal to
  the wasmtime run of Task 16); `engine_solver_is_refused_in_the_browser`; `trap_is_an_error`
  (`rings` 30 with a memory limit of 64 MiB: rejects with an `Error` naming memory, the next
  `createSimulation` works). `browser.test.mjs` (Playwright Chromium, headless, the demo served
  statically): run 10 min, canvas pixels not uniform, status `t = 600 s`, no console errors, the
  page's field L2 norm equals Node's bitwise.
- [ ] **Step 2: Run** `npm test --prefix wasm/js`. Expected: FAIL.
- [ ] **Step 3: Implement**; check the page with `chrome-devtools-axi` screenshots at desktop and
  phone width, light and dark, and fix what looks off.
- [ ] **Step 4: Run** `npm test --prefix wasm/js` and the docs build
  (`uvx --from zensical==0.1.3 zensical build --clean --strict`). Expected: PASS.
- [ ] **Step 5: Commit** `feat(wasm): browser glue, Node and browser tests, the lobule demo page`
