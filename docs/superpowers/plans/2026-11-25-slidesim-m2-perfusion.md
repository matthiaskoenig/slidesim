# slidesim M2: maps, Darcy perfusion and advection - Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring position fields ("maps") from slideviz into slidesim as coefficients of the PDEs,
solve steady anisotropic Darcy perfusion from portal to central walls, carry the species by that
flow with SUPG-stabilized advection-diffusion-reaction, export real scenes from slideviz, and run
the literature parameter set on a real slide.

**Architecture:** `contracts` gains `maps` in `scene/1` and coefficient tables, porosity,
advection and `perfusion` in `study/1`. A new crate `maps` reads Zarr v3 rasters per rank and
samples them (bilinear, gradients). In `pde` one generalized libCEED operator, `Transport`
(mass, reaction, tensor diffusion, advection, SUPG, all as values at quadrature points) replaces
`DiffusionReaction`; `Flow` solves Darcy with it and provides velocities at the quadrature points
of every level; `Problem` takes spatial coefficients and an optional flow. slideviz persists the
zonation position and lumen classes, splits the vessel wall tag, and writes `scene.json` with
Zarr maps.

**Tech Stack:** as milestone 1 (Rust 1.100.0, PETSc v3.26.1, libCEED v1.1.2, MPI, cargo-nextest),
plus the `zarrs` crate (latest release) in slidesim; in slideviz Python with uv (`zarr>=3.3`,
`jsonschema`, `pytest` as dev dependency) and the pixi mesh environment (gmsh, DOLFINx 0.11).

**Spec:** `docs/superpowers/specs/2026-11-25-slidesim-m2-perfusion-design.md` (milestone 2, the
more specific document) and its parent `docs/superpowers/specs/2026-11-24-slidesim-design.md`.
Parameters: `docs/parameters/m2-perfusion-mouse.md`.

## Global Constraints

- Every constraint of the milestone 1 plan holds: MIT, pinned PETSc and libCEED, C flags
  `-O3 -DNDEBUG -ffp-contract=off -fno-fast-math`, `cargo fmt --check`,
  `cargo clippy --all-targets -- -D warnings` and all tests pass at every commit.
- No agent attribution anywhere (no `Co-Authored-By`, no "Generated with"); never the em dash.
- No liver-specific code in slidesim: "zonation", "portal", "central" appear only in test data,
  studies and slideviz. The core knows maps, tables, tags and coefficients.
- Units: length um, time s, concentration mM, pressure Pa, permeability um^2, viscosity Pa*s,
  velocity um/s, porosity dimensionless, `g0` 1/um.
- A map error (point outside the raster, no-value or NaN pixel) is an error naming the map, the
  point and the area tag; never a fallback value.
- Every error which one process can find and others cannot is made collective (reduced with
  `Comm::max_int` before returning), so no process waits in a collective call; the CLI keeps the
  milestone 1 rule (`MPI_Abort` after printing the error when more than one process runs).
- Results on 1, 2 and 3 processes agree within 1e-10 relative (L2 norms and integrals).
- Elements stay P1 on triangles.
- slideviz work happens in `~/git/slideviz` on a branch `feat/scene-export` from `dev`, with its
  own pull request; it follows that repository's style (module docstrings, argparse scripts,
  `np.savez_compressed` with a JSON `meta` for intermediate files).

## Review Focus

1. Maps and mesh in different frames (origin or axis swapped): every quadrature point still finds
   data, and the run gives wrong physics silently. Pinned by the slideviz writer test which samples
   the zonation map at the midpoints of `portal_wall` and `central_wall` edges and requires
   p < 0.3 and p > 0.7 respectively (Task 8), and by the `maps` test reading a Python-written
   array whose values differ in x and y (Task 2).
2. A map that covers only part of the mesh, so that the bad points lie on one process: all
   processes return the error, none hangs (Task 3, test on 2 processes).
3. A study whose tables name a map the scene lacks, or whose perfusion names a tag the mesh lacks:
   an error before any solve listing what exists (Task 1, Task 4).
4. A field with advection and no `dirichlet` condition on a tag with net inflow: an error naming
   the tag and the field before the first step (Task 5).
5. Map values beyond the ends of a table: the end values hold (Task 3); a table which yields a
   non-positive diffusion, porosity or permeability is rejected by the contract (Task 1).

---

### Task 1: Contracts: maps, coefficients, porosity, advection, perfusion

**Files:**
- Modify: `crates/contracts/src/scene.rs`, `crates/contracts/src/study.rs`,
  `crates/contracts/src/lib.rs`, `crates/contracts/src/error.rs`
- Modify: `schemas/scene-1.schema.json`, `schemas/study-1.schema.json` (regenerated)
- Create: `testdata/contracts/scene-maps.json`, `study-perfusion.json`,
  `study-table-not-increasing.json`, `study-table-negative-diffusion.json`,
  `study-perfusion-no-boundary.json`, `study-perfusion-duplicate-tag.json`
- Modify: `testdata/studies/tissue_oxygen/study.json` only if its shape must change (numbers stay
  valid as constants)
- Test: `crates/contracts/tests/contracts.rs`

**Interfaces:**
- Produces:
  - `Scene.maps: Vec<MapSpec>` (default empty);
    `MapSpec { id: String, path: PathBuf, units: String, um_per_px: f64, origin_um: [f64; 2], no_value: Option<f64> }`.
    `path` resolved against the scene file's directory on load, as `mesh.path` is.
  - `enum Coefficient { Constant(f64), Table(Table) }`, serde untagged (a JSON number or an
    object); `Table { map: String, points: Vec<[f64; 2]> }`.
  - `impl Coefficient { pub fn maps(&self) -> Option<&str>; pub fn eval(&self, value: f64) -> f64 }`:
    `eval` interpolates linearly between points and holds the end values beyond them; for
    `Constant(c)` it returns `c`.
  - `FieldSpec`: `diffusion: Coefficient`, `decay: Coefficient` (default `Constant(0.0)`),
    `source: Coefficient` (default `Constant(0.0)`), `porosity: Coefficient` (default
    `Constant(1.0)`), `advection: bool` (default `false`); `id`, `initial`, `boundaries` unchanged.
  - `Study.perfusion: Option<Perfusion>`;
    `Perfusion { viscosity: f64, k_along: Coefficient, k_across: Coefficient, direction: Direction, boundaries: Vec<PressureBoundary> }`;
    `Direction { map: String, g0: f64 }`; `PressureBoundary { tag: String, pressure: f64 }`.
  - `pub fn check_maps(study: &Study, scene: &Scene) -> Result<(), ContractError>`: every map a
    table or `direction` names is in `scene.maps`; else `ContractError::UnknownMap { map, available: Vec<String> }`.
- Validation rules (in schema and in `from_value`):
  - table: at least 2 points, values `v` strictly increasing, all numbers finite;
  - `diffusion`, `porosity`, `k_along`, `k_across`: constant > 0, table outputs all > 0;
    `decay`: >= 0; `source`: any finite;
  - `viscosity` > 0, `g0` > 0, `um_per_px` > 0, at least one pressure boundary, pressure tags
    unique, map ids unique and identifiers (`^[A-Za-z][A-Za-z0-9_]*$`).

- [ ] **Step 1: Write the failing tests** in `crates/contracts/tests/contracts.rs`:
  - `scene_maps_load_and_resolve`: `scene-maps.json` has one map `zonation`, its `path` is
    absolute after load and ends with the relative path given.
  - `study_perfusion_round_trip`: `study-perfusion.json` (diffusion a table over `zonation`
    `[[0, 500], [1, 900]]`, porosity 0.15, advection true, perfusion with the baseline of
    `docs/parameters/m2-perfusion-mouse.md`) loads, serializes and loads back equal.
  - `table_eval`: `points [[0,1],[1,3]]`: `eval(-1) == 1`, `eval(0.5) == 2`, `eval(2) == 3`.
  - `table_not_increasing`, `table_negative_diffusion`, `perfusion_without_boundary`,
    `perfusion_duplicate_tag`: each fixture fails to load, the message names the offending key.
  - `unknown_map_lists_available`: `check_maps` on a study naming `zonation2` against
    `scene-maps.json` errs; the message contains `zonation2` and `zonation`.
  - `milestone_1_studies_still_load`: every existing `testdata/contracts/study-minimal.json` and
    `testdata/studies/*/study.json` loads with constants.
  - The existing `schemas_are_current` and `schemas_match_load` cover the new shapes.
- [ ] **Step 2: Run** `cargo nextest run -p contracts`. Expected: the new tests fail.
- [ ] **Step 3: Implement** the types and rules above; regenerate the schemas the way
  `schemas_are_current` says.
- [ ] **Step 4: Run** `cargo nextest run -p contracts`. Expected: all pass. Run the workspace
  build: `pde` and `slidesim` stop compiling where they read `field.diffusion` as `f64`; make them
  compile by taking `Coefficient::Constant` and returning `PdeError::Unsupported("coefficient
  tables")` otherwise (removed in Task 3).
- [ ] **Step 5: Commit** `feat(contracts): maps in scene/1; coefficient tables, porosity, advection and perfusion in study/1`

### Task 2: Crate `maps`: Zarr rasters, bilinear samples and gradients

**Files:**
- Create: `crates/maps/{Cargo.toml,src/lib.rs}`
- Create: `testdata/maps/make_fixtures.py` (run with `uvx --with zarr>=3.3 --with numpy python`),
  `testdata/maps/linear.zarr`, `testdata/maps/no_value.zarr` (committed, small)
- Test: `crates/maps/tests/maps.rs`

**Interfaces:**
- Consumes: `contracts::MapSpec`.
- Produces:
  - `pub struct Map` (one raster window in memory) with
    `pub fn read(spec: &MapSpec, bounds: [[f64; 2]; 2]) -> Result<Map, MapError>`: reads the
    pixels covering `bounds` (`[[xmin, ymin], [xmax, ymax]]` in um) plus a margin of 2 pixels,
    clipped to the array; only the chunks needed.
  - `Map::sample(&self, x: [f64; 2]) -> Result<f64, MapError>`: bilinear between pixel centres
    (pixel (row r, col c) centred at `origin_um + [c, r] * um_per_px`).
  - `Map::gradient(&self, x: [f64; 2]) -> Result<[f64; 2], MapError>`: central differences at
    pixel centres (one-sided at the array border), then bilinear; per um.
  - `Map::id(&self) -> &str`.
  - `pub struct Maps` (`BTreeMap<String, Map>`): `Maps::read(specs: &[MapSpec], ids: &[&str], bounds) -> Result<Maps, MapError>`
    reads only the maps named; `Maps::get(&self, id) -> Result<&Map, MapError>`.
  - `pub enum MapError { Outside { map, x: [f64; 2] }, NoValue { map, x: [f64; 2] }, Unknown { map, available: Vec<String> }, Format { map, message }, Zarr { map, message } }`
    (`thiserror`), messages naming the map and the point in um.
- Also produces, for the tests of other crates: `#[doc(hidden)] pub fn write_test_map(path: &Path,
  values: &[f64], rows: usize, cols: usize, um_per_px: f64, origin_um: [f64; 2]) -> MapSpec`
  (float64, zstd), so tests build analytic maps over their meshes.
- `zarrs` with its filesystem store and the `zstd` codec; arrays 2D float32 or float64, else
  `Format`.

- [ ] **Step 1: Fixtures.** `make_fixtures.py` writes, with zstd compression and chunks of 16x16:
  `linear.zarr`, 40 x 60 float32 (rows x cols), value `2 + 0.5 x - 0.25 y` at the pixel centres,
  `um_per_px` 2.0, origin `[10, 20]` (stored in the fixture's own JSON beside it, `linear.json`,
  as the `MapSpec` fields); `no_value.zarr`, the same with a 3 x 3 block of NaN at rows 10-12,
  cols 30-32 and the value -1 at row 0 col 0 (`no_value` -1).
- [ ] **Step 2: Write the failing tests:**
  - `bilinear_is_exact_for_linear` at 50 random points inside: `|sample - f(x)| < 1e-5`.
  - `gradient_is_exact_for_linear`: `[0.5, -0.25]` within 1e-5 at interior points and at points
    in the border pixel row.
  - `outside_is_error`: a point 1 um left of the first pixel centre: `MapError::Outside`, message
    contains `linear` and the coordinates.
  - `nan_and_no_value_are_errors`: points whose 4 neighbours include a NaN or the -1 pixel.
  - `window_reads_only_needed_chunks`: `read` with a box of 5 x 5 um returns a window of at most
    (5/2 + 1 + 2 * 2)^2 pixels and samples equal those of a window over the whole array.
  - `unknown_map_lists_available`.
  - `test_map_round_trip`: `write_test_map` then `read` gives back the values.
- [ ] **Step 3: Implement** `crates/maps`; add it to the workspace.
- [ ] **Step 4: Run** `cargo nextest run -p maps`. Expected: all pass.
- [ ] **Step 5: Commit** `feat(maps): Zarr v3 rasters read per window, bilinear samples and gradients`

### Task 3: Coefficients at quadrature points, and the `Transport` operator without advection

**Files:**
- Modify: `crates/mesh/src/lib.rs` (local bounds)
- Create: `crates/pde/src/coefficients.rs`
- Create: `crates/pde/qfunctions/transport.h` (replaces `diffusion.h`); modify
  `crates/pde/qfunctions/qfunctions.c`, `crates/pde/src/lib.rs` (QFunction list)
- Modify: `crates/pde/src/operators.rs` (`DiffusionReaction` becomes `Transport`),
  `crates/pde/src/multigrid.rs`, `crates/pde/src/problem.rs`, `crates/pde/src/discretization.rs`
  (cell area tags)
- Test: `crates/pde/tests/coefficients.rs`, `crates/pde/tests/operators.rs`,
  `crates/pde/tests/verification.rs`, `crates/pde/tests/multigrid.rs`

**Interfaces:**
- Consumes: `contracts::{Coefficient, FieldSpec}`, `maps::{Maps, MapError}`.
- Produces:
  - `Mesh::local_bounds(&self) -> Option<[[f64; 2]; 2]>`: bounding box of the coordinates of
    this process's cells on all levels (`None` without local cells).
  - `Discretization::cell_area_tag(&self, cell: usize) -> Option<String>`: the name of the area
    tag (a tag of dimension 2) of local cell `cell`, if any.
  - `pub fn coefficient_values(disc: &Discretization, c: &Coefficient, maps: &Maps) -> Result<Vec<f64>>`:
    one value per quadrature point (layout of `Discretization::quadrature_points`). Collective:
    a `MapError` on any process becomes `PdeError::Map { message }` on all, the message from the
    lowest failing rank (map, point, area tag) and the count of failing points.
  - `pub struct PointData`: per quadrature point `mass` (the factor of `u_t`, e.g. porosity),
    `reaction` (e.g. `phi k`), `diffusion: [f64; 3]` (symmetric tensor `D00, D01, D11`),
    `velocity: [f64; 2]`, `tau: f64` (SUPG; zero here). Constructors:
    `PointData::isotropic(disc, mass, reaction, diffusion: &[f64]) -> PointData` and
    `PointData::with_velocity(self, velocity: Vec<[f64; 2]>) -> PointData` (used in Task 5).
  - `pub enum Terms { Steady, Mass, Jacobian }`: which part `Transport` applies: `Steady` is
    `R + K_D (+ A + SUPG)`, `Mass` is `M_m` (with its SUPG part in Task 5), `Jacobian` is
    `shift * Mass + Steady`.
  - `pub struct Transport` with `Transport::new(disc: &Discretization, data: Rc<PointData>, terms: Terms) -> Result<Transport>`,
    `pub shift: f64`, and the methods `DiffusionReaction` has now (`apply_local`, `assemble`,
    `reassemble`, `discretization`, `ShellOperator` with diagonal). `SharedOperator` wraps
    `Rc<RefCell<Transport>>`.
  - `Multigrid::set_shift(&mut self, shift: f64) -> Result<()>` replaces `set_coefficients`.
  - `Problem::new(mesh, field, time, ceed, petsc_options, inputs: &Inputs) -> Result<Problem>`
    with `pub struct Inputs<'a> { pub maps: &'a Maps }` (Task 5 adds `pub flow: Option<&'a Flow>`).
  - `PdeError::Map { message: String }`.
- QFunction `Transport2D` (in `transport.h`): inputs `u`, `du`, `q_data` (geometry), `p_data`
  (8 values per point: `mass, reaction, D00, D01, D11, u0, u1, tau`); context
  `{ shift, terms }`; outputs `v`, `dv`. Equation: `shift m u + r u - div(D grad u)` in Task 3.
- The equation of a field: `phi u_t - div(phi D grad u) + phi k u = phi f` with `phi` the
  porosity: `mass = phi`, `reaction = phi k`, `diffusion = phi d I`, source integrated as `phi f`.

- [ ] **Step 1: Write the failing tests:**
  - `coefficients.rs::table_over_map_at_points`: a map `x + 2 y` written with
    `maps::write_test_map` over `unit_square.msh` (pixel 0.01 um, covering [-0.05, 1.05]^2), table
    `[[0, 1], [3, 4]]`: values equal `Coefficient::eval(map.sample(x_q))` at every point.
  - `coefficients.rs::values_beyond_table_hold_end_values`.
  - `coefficients.rs::map_error_names_map_point_and_tag`: a mesh partly outside the raster; the
    error contains the map id, a point and the area tag.
  - `coefficients.rs::map_error_on_one_process_is_error_on_all`, run by
    `parallel_on_2_processes` (the map covers only the left half of the mesh).
  - `operators.rs::transport_matches_diffusion_reaction`: with constant data, `Transport` applies
    the same as the milestone 1 operator did (keep the old expected values of the existing tests,
    now through `PointData::isotropic`).
  - `operators.rs::anisotropic_tensor_integrates_exactly`: for `u = x`, `D = [[2, 1], [1, 3]]`,
    `v = y`: `int D grad u . grad v` over the unit square equals 1 within 1e-12.
  - `verification.rs::spatial_order_variable_coefficients`: manufactured solution
    `u = sin(pi x) sin(pi y) e^-t` with `d(x) = 0.01 (1 + x)` and `phi(x) = 0.5 + 0.25 y`
    (coefficients from analytic maps written as test Zarr arrays), L2 error order >= 1.9 over
    3 refinements.
  - All milestone 1 tests in `pde` pass unchanged in their assertions.
- [ ] **Step 2: Run** `cargo nextest run -p pde -p mesh`. Expected: the new tests fail.
- [ ] **Step 3: Implement** `Mesh::local_bounds`, `cell_area_tag`, `coefficient_values`,
  `PointData`, `Transport2D` and `Transport`; adapt `Multigrid` (coefficients fixed per level,
  only the shift changes) and `Problem` (coefficients from the field through `coefficient_values`
  on every level; the source vector scaled by porosity). Remove the `Unsupported("coefficient
  tables")` of Task 1.
- [ ] **Step 4: Run** `cargo nextest run --workspace`. Expected: all pass, including the parallel
  tests.
- [ ] **Step 5: Commit** `feat(pde): coefficients from maps at quadrature points, Transport operator with tensor diffusion and porosity`

### Task 4: Darcy perfusion (`Flow`)

**Files:**
- Create: `crates/pde/src/flow.rs`, `crates/pde/qfunctions/velocity.h`
- Modify: `crates/pde/src/lib.rs`, `crates/pde/qfunctions/qfunctions.c`
- Test: `crates/pde/tests/flow.rs`

**Interfaces:**
- Consumes: `Transport`, `PointData`, `coefficient_values`, `configure_multigrid`,
  `contracts::Perfusion`, `maps::Maps`.
- Produces:
  - `pub struct Flow` with
    `Flow::solve(mesh: &Mesh, perfusion: &Perfusion, maps: &Maps, ceed: &Ceed, petsc_options: &str) -> Result<Flow>`:
    the steady pressure on the finest level (and what the levels need, see below). Options prefix
    `perfusion_`; CG with the geometric multigrid; the defaults of `Problem` (errors if not
    converged, `-ksp_rtol 1e-10`).
  - `Flow::velocity(&self, level: usize) -> &[[f64; 2]]`: `u = -K grad p / mu` at the quadrature
    points of level `level` (layout of `Discretization::quadrature_points` of any discretization
    of that level). How coarse levels get their velocity is the implementer's choice (spec: left
    to the builder); recommended: the pressure solved on every level, which is cheap and keeps
    each level's velocity consistent with its own mesh.
  - `Flow::tag_flows(&self) -> &[(String, f64)]`: net volume flow into the domain through each
    pressure boundary tag (um^2/s in 2D, positive = inflow), from the residual of the full
    operator at the constrained dofs (consistent fluxes; spec: verification item 4).
  - `Flow::pressure(&self) -> &Vector`, `Flow::local_pressure(&self) -> Result<Vector>` (for
    output, field name `pressure`, unit `Pa`), `Flow::nodal_velocity(&self) -> Result<Vector>`
    (lumped L2 projection to the nodes, 2 components, unit `um/s`; output only).
  - `#[doc(hidden)] Flow::solve_manufactured(mesh, perfusion, maps, ceed, petsc_options, exact: &dyn Fn([f64; 2]) -> f64, source: &dyn Fn([f64; 2]) -> f64) -> Result<Flow>`:
    for the convergence tests only: Dirichlet values from `exact` at the nodes of the pressure
    tags, and a volume source.
- Permeability tensor at a point with direction map gradient `g`, `t = g/|g|`,
  `w = |g|^2/(|g|^2 + g0^2)`: `K = k_across I + w (k_along - k_across) t t^T`; the operator's
  `diffusion` is `K / mu`, `mass` and `reaction` zero.
- `PdeError::Mesh` (existing) for an unknown pressure tag, listing the tags.
- QFunction `Velocity2D` (in `velocity.h`): inputs `dp` (Grad), `q_data`, `p_data`; output the
  velocity per point (`EvalMode::None`).

- [ ] **Step 1: Write the failing tests** in `crates/pde/tests/flow.rs` (unit square, pressure 1 on
  `left`, 0 on `right`, no flow on `top`, `bottom`, maps as test Zarr arrays):
  - `uniform_flow`: isotropic `k = 2`, `mu = 1`: velocity `[2, 0]` at every point within 1e-10;
    `tag_flows` `left` +2, `right` -2 within 1e-10.
  - `pressure_order_isotropic`: `solve_manufactured` with `p = sin(pi x/2) cosh(pi y/2)` (harmonic,
    source zero), pressure tags all four sides: L2 order of pressure >= 1.9, of velocity >= 0.9,
    over 4 levels.
  - `pressure_order_anisotropic`: direction map `x + 0.5 y` (constant gradient), `k_along = 3`,
    `k_across = 1`, `g0 = 1e-6`, `p = sin(pi x) sin(pi y) + x` with the matching source
    `-div(K grad p)/mu`: the same orders.
  - `isotropic_where_gradient_vanishes`: direction map constant: `K = k_across I` at every point.
  - `unknown_pressure_tag_lists_tags`.
  - `parallel_on_2_processes`, `parallel_on_3_processes`: `uniform_flow` and the anisotropic order
    test agree with one process within 1e-10 relative (use `parallel_with_env` like
    `two_processes_match_one` in `multigrid.rs`).
- [ ] **Step 2: Run** `cargo nextest run -p pde --test flow`. Expected: fail.
- [ ] **Step 3: Implement** `flow.rs` and `Velocity2D`.
- [ ] **Step 4: Run** `cargo nextest run -p pde`. Expected: all pass.
- [ ] **Step 5: Commit** `feat(pde): steady Darcy perfusion with anisotropic permeability along a map gradient`

### Task 5: Advection with SUPG in `Problem`

**Files:**
- Modify: `crates/pde/qfunctions/transport.h`, `crates/pde/src/operators.rs`,
  `crates/pde/src/coefficients.rs`, `crates/pde/src/problem.rs`, `crates/pde/src/multigrid.rs`
- Test: `crates/pde/tests/advection.rs`

**Interfaces:**
- Consumes: `Flow::velocity`, `Flow::tag_flows`, `PointData::with_velocity`, `Terms`.
- Produces:
  - `Transport2D` adds `u . grad u` tested with `v`, and the SUPG term
    `tau (vel . grad v) (shift m u + r u + vel . grad u)` (in `Jacobian`; `Mass` gets
    `tau (vel . grad v) m u`, `Steady` the rest). `tau` per point is computed in Rust when the data
    is built (formula and element length: implementer's choice, documented in the code; spec
    leaves it to the builder); zero where the velocity is zero.
  - `Inputs` gains `pub flow: Option<&'a Flow>`. `Problem` with `field.advection == true` requires
    `inputs.flow` (else `PdeError::Unsupported`); uses GMRES instead of CG,
    does not mark the shell symmetric, and `configure_multigrid(.., symmetric: bool)` estimates
    the Chebyshev interval with GMRES instead of CG when `symmetric` is false.
  - Before the first step: every tag with `Flow::tag_flows` > 0 (inflow) needs a `dirichlet`
    condition for each field with advection, else
    `PdeError::Inflow { tag: String, field: String }` ("tag 'X' has inflow; field 'Y' with
    advection needs a dirichlet condition there").
  - `Problem::mass_balance(&self) -> Result<MassBalance>` with
    `MassBalance { storage_rate, source, boundary: Vec<(String, f64)>, advection_defect }`
    (mM*um^2/s), each from the discrete equations: boundary terms at Dirichlet tags from the
    residual at their constrained dofs, at other boundary tags `-int (u . n) c` (outflow), and
    `advection_defect = int (u . grad c) dx - int_boundary (u . n) c ds`, the non-conservation of
    the primal velocity (jumps of `u . n` between elements), reported, not gated (spec:
    verification item 4).
- Free outflow: no boundary term for advection (the non-conservative form needs none); the
  diffusive flux is zero where no condition is given.

- [ ] **Step 1: Write the failing tests** in `crates/pde/tests/advection.rs` (unit square, flow
  from `uniform_flow` scaled to the cell Peclet number wanted):
  - `orders_low_peclet`: manufactured `c = sin(pi x) sin(pi y) e^-t` with source, velocity
    `[1, 0.5] * a` for cell Peclet 0.5 on the finest level: L2 order >= 1.9.
  - `orders_high_peclet`: cell Peclet about 100: L2 order >= 0.9 (SUPG's rate for this regime
    in the L2 norm is between 1 and 2; the test fixes the lower bound).
  - `front_without_oscillations`: inflow `dirichlet` 1 on `left`, initial 0, cell Peclet 100:
    at t when the front is at x = 0.5, min >= -0.05 and max <= 1.05.
  - `inflow_without_dirichlet_is_error`: message names `left` and the field.
  - `mass_balance_closes`: `storage_rate = source + sum(boundary) - advection_defect` relative to
    the largest term within 1e-8 (with `-ksp_rtol 1e-12`); the exact split of the boundary terms
    follows from testing the discrete equations with `v = 1` (Dirichlet rows give the consistent
    fluxes), which the implementer derives and documents on `MassBalance`. The test prints
    `advection_defect`.
  - `parallel_on_2_processes`, `parallel_on_3_processes`: `front_without_oscillations` L2 norm and
    total mass agree with one process within 1e-10 relative.
  - The milestone 1 tests still pass (advection false: CG, symmetric).
- [ ] **Step 2: Run** `cargo nextest run -p pde --test advection`. Expected: fail.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `cargo nextest run --workspace`. Expected: all pass.
- [ ] **Step 5: Commit** `feat(pde): advection with SUPG stabilization carried by the Darcy velocity`

### Task 6: `slidesim run` with maps and perfusion

**Files:**
- Modify: `crates/slidesim/src/run.rs`, `crates/slidesim/Cargo.toml`
- Modify: `testdata/meshes/make_fixtures.py`; create `testdata/meshes/lobule_toy.msh`: a 400 x 400
  um square with a hole tagged `portal_wall` near one corner and a hole tagged `central_wall` near
  the opposite one, outer edges `section_edge`, area `tissue`
- Create: `testdata/studies/lobule_toy/{scene.json,study.json,make_maps.py,zonation.zarr}`:
  zonation p linear from the portal to the central hole; study with the baseline parameters of
  `docs/parameters/m2-perfusion-mouse.md` and a field `sucrose` with advection, `dirichlet` 1 on
  `portal_wall`
- Test: `crates/slidesim/tests/cli.rs`

**Interfaces:**
- Consumes: `contracts::check_maps`, `maps::Maps::read`, `Mesh::local_bounds`, `Flow::solve`,
  `Problem::new(.., &Inputs)`.
- Produces:
  - `run` order: load study and scene, `check_maps`, load mesh, read the maps named by the study
    for `local_bounds` (all ranks), solve `Flow` if `study.perfusion` is present, build the
    problems, step and write.
  - Outputs: `pressure.vtu` and `velocity.vtu` once (from `Flow`), the fields as before;
    `summary.json` gains `"perfusion": {"tag_flows": {"<tag>": <um^2/s>}}` and, per field with
    advection, `"advection_defect"` at each output.
  - Earlier outputs: `remove_earlier_outputs` also removes `pressure.vtu` and `velocity.vtu`.

- [ ] **Step 1: Write the failing tests** in `cli.rs`:
  - `lobule_toy_flows_from_portal_to_central`: `summary.json` has `portal_wall` flow > 0 and
    `central_wall` flow < 0, their sum within 1e-8 relative of zero; `pressure.vtu` exists.
  - `lobule_toy_tracer_reaches_central`: the mean of `sucrose` at the last output over the nodes
    within 10 um of the central hole is > 0.1.
  - `unknown_map_fails_before_solving`: a study naming `zonation2`: exit code nonzero, stderr
    lists `zonation`, no `.vtu` written.
  - `two_ranks_match_one` extended to `lobule_toy`.
- [ ] **Step 2: Run** `cargo nextest run -p slidesim`. Expected: fail.
- [ ] **Step 3: Implement**; generate the fixture with `make_fixtures.py` and `make_maps.py`.
- [ ] **Step 4: Run** `cargo nextest run --workspace` and `cargo test --doc --workspace`.
  Expected: all pass. Render `pressure.vtu`, `velocity.vtu` and the last `sucrose` output to PNG
  (pvpython if present, else meshio + matplotlib via `uvx`) into the scratch directory and inspect
  them: pressure falls from portal to central, velocity points from portal to central, tracer
  front between the walls.
- [ ] **Step 5: Commit** `feat: slidesim run with maps and Darcy perfusion`

### Task 7: slideviz: zonation position, lumen classes, wall tags

Repository `~/git/slideviz`, branch `feat/scene-export` from `dev`.

**Files:**
- Modify: `pyproject.toml` (dev dependency group: `pytest`, `jsonschema`)
- Modify: `src/slideviz/analysis/zonation.py` (lumen classes out of `portality`)
- Modify: `scripts/zonation_layers.py` (save `position` and `lumen_class` on the working grid)
- Modify: `scripts/mesh_inputs.py` (layer `lumen_class` cropped like `lumen`)
- Modify: `mesh/build_mesh.py` (tags `portal_wall` 15, `central_wall` 16; `vessel_wall` 11 for
  unclassified)
- Create: `tests/test_zonation_export.py`, `tests/test_wall_tags.py`

**Interfaces:**
- Produces:
  - `Portality.lumen_class: np.ndarray` (uint8 on the working grid: 0 no lumen, 1 portal,
    2 central, 3 unclassified), from the `holes`, `portal` and `central` sets `portality` already
    computes.
  - `<stem>_lobules.npz` gains `position` (float32, `Portality.points`, NaN outside `inside`),
    `lumen_class` (uint8), `working_um_per_px` (float) and `working_origin_um` ([row, col] of
    pixel (0, 0)'s centre in H&E level 0 um, the convention of `mesh_inputs.frame()`).
  - `mesh_inputs.npz` gains the layer `lumen_class` (nearest-neighbour onto its grid if the grids
    differ).
  - `build_mesh.edge_groups`: an edge that is a vessel wall gets the class by majority vote of
    the `lumen_class` pixels (classes 1 and 2) within 10 um of the midpoints of the edges of its
    connected vessel-wall chain; ties or none: `vessel_wall`. `EDGE_TAGS` gains
    `portal_wall: 15, central_wall: 16`; `REFINED` includes both.

- [ ] **Step 1: Write the failing tests** (pytest; synthetic arrays, no slides):
  - `test_lumen_classes`: a synthetic landscape with one peak and one valley and two holes, one
    beside each: `lumen_class` is 2 in the hole by the peak, 1 by the valley, 0 elsewhere; a hole
    beside both is 3.
  - `test_position_saved`: `zonation_layers.measure`'s save step (factored into a function
    `save_layers(path, land, port, inside, um, offset)`) writes `position` and `lumen_class` with
    the frame keys.
  - `test_wall_tags` (pixi env, gmsh): a square with two holes and a `lumen_class` raster marking
    one portal, one central: the `.msh` physical groups contain `portal_wall` and `central_wall`
    with the edges of the matching hole; a hole without class stays `vessel_wall`.
- [ ] **Step 2: Run** `uv run pytest tests/test_zonation_export.py` and
  `pixi run --manifest-path mesh/pixi.toml python -m pytest tests/test_wall_tags.py` (add
  `pytest` to `mesh/pixi.toml`). Expected: fail.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** both test commands. Expected: pass. Run `zonation_layers.py` and
  `mesh_inputs.py` and `build_mesh.py` on one real CYP2E1/H&E pair from `SLIDEVIZ_DATA` and
  inspect the `_view.npz` plot of edge tags: portal and central walls where the zonation map has
  its valleys and peaks.
- [ ] **Step 5: Commit** `Persist the zonation position and lumen classes; tag portal and central walls in meshes`

### Task 8: slideviz: scene writer

**Files:**
- Create: `scripts/write_scene.py`, `schemas/scene-1.schema.json` (copy of slidesim's, with a
  header line in `schemas/README.md` naming the source commit)
- Test: `tests/test_write_scene.py`

**Interfaces:**
- Consumes: the mesh `.json` record and `.msh` of `build_mesh.py`, `<stem>_lobules.npz` of Task 7.
- Produces: `write_scene.py --mesh <record.json> --lobules <stem>_lobules.npz --out <dir>` writes
  `<dir>/scene.json` (`scene/1`, `mesh.path` relative), `<dir>/maps/zonation.zarr` (float32, zstd,
  chunks 256 x 256): `position` cropped to the mesh bounding box plus 2 pixels, NaN inside the
  window filled by nearest neighbour (`scipy.ndimage.distance_transform_edt` indices), with
  `um_per_px` and `origin_um` converted to the scene's `[x, y]` order.
- Validates the written `scene.json` against `schemas/scene-1.schema.json` before exit.

- [ ] **Step 1: Write the failing tests:**
  - `test_scene_valid`: on the synthetic mesh of `test_wall_tags`, the written scene validates
    against the schema and the Zarr array opens with `zarr.open_array` with the expected shape.
  - `test_frame_matches_walls`: sampling the written map bilinearly (a NumPy reimplementation of
    `Map::sample` in the test) at the midpoints of `portal_wall` edges gives p < 0.3 and at
    `central_wall` edges p > 0.7.
  - `test_no_nan_in_window`.
- [ ] **Step 2: Run** `uv run pytest tests/test_write_scene.py`. Expected: fail.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `uv run pytest`. Expected: all pass.
- [ ] **Step 5: Commit** `Write slidesim scenes: scene.json with the zonation map as Zarr`, then
  push the branch and open the slideviz pull request (via `gh-axi`).

### Task 9: End to end on a real slide

**Files:**
- Create: `docs/results/2026-11-m2-real-slide.md` (in slidesim; filled with the run's numbers and
  images)
- Create: `scripts/e2e/m2_real_slide.sh` (runs the slideviz steps of Tasks 7-8 for one
  configured pair from `SLIDEVIZ_DATA`, then `slidesim run` for the baseline and the three
  sensitivity studies)
- Create: `testdata/studies/m2_real_slide/{baseline,calibrated,whole_blood,radial10}.study.json`
  (scene path given by the script; parameter values exactly those of
  `docs/parameters/m2-perfusion-mouse.md`)

**Interfaces:**
- Consumes: everything above. Not part of CI (the slides are not in the repository).

- [ ] **Step 1: Run** `scripts/e2e/m2_real_slide.sh` on one mouse H&E/CYP2E1 pair (a control
  animal, not necrotic) with 2 refinements on 4 processes.
- [ ] **Step 2: Check** in each run's `summary.json`: every `portal_wall` flow > 0, every
  `central_wall` flow < 0, total within 1e-8 relative of zero.
- [ ] **Step 3: Check** the mean interstitial velocity `|u|/phi` over the tissue: baseline about
  73 um/s (50-100), calibrated about 180 um/s.
- [ ] **Step 4: Render and inspect** pressure, velocity arrows over the H&E thumbnail, and the
  sucrose front at 1, 3 and 10 s; record the images and numbers in
  `docs/results/2026-11-m2-real-slide.md`, with the deviations from the expectations of the
  parameter document.
- [ ] **Step 5: Commit** `docs: milestone 2 end-to-end run on a real slide`

---

## Self-review notes

- Spec coverage: maps (T1, T2, T3, T8), tables (T1, T3), anisotropic Darcy (T4), advection with
  SUPG, porosity, inflow rule, free outflow (T3, T5), wall split and lumen classes (T7), scene
  writer with schema validation and frame conversion (T8), verification items 1-5 (T2-T6), item 6
  (T9), pressure solve as its own step (T4 `Flow::solve`), output of pressure and velocity (T6).
- Left to the implementer, as the spec says: `tau` and the element length, outflow terms within
  the stated behaviour, coarse-level velocity, Zarr chunking (fixed in T8 for the writer only),
  VTU layout, slideviz writer structure.
