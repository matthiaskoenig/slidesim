# slidesim M3: cell models coupled to perfusion - Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run one SBML cell model (the SPT liver cell model with CYP2E1 from the slide) in every
coarse element of a real slide, coupled to the transport of the blood species by Strang
splitting, on two cell solvers (sbmlsim-core's native engine, and a batched libCEED QFunction
with a Rosenbrock integrator), fast, and compare the simulated necrosis with the measured one.

**Architecture:** slidesim first gets a conservative Darcy velocity and a faster transport.
sbmlsim-core gets a C target (point functions of a model, compiled at study start into a shared
library implementing `ModelFns`), a `Stepper` which advances one engine instance from a given
state, and a QFunction emitter with a Rosenbrock integrator. pkdb_models gets the
`spt_liver_cell` variant, slideviz the `cyp2e1` and `necrosis` maps and the portal/central check.
slidesim gets `cells` in `study/1`, a new crate `cells` with the trait `CellSolver` and its two
paths, the exchange between blood fields and cells, and the coupled run.

**Tech Stack:** as milestone 2 (Rust 1.100.0, PETSc v3.26.1, libCEED v1.1.2, MPI, cargo-nextest,
zarrs), plus sbmlsim-core crates (`sbmlsim-ir`, `sbmlsim-abi`, `sbmlsim-engine` with its
`csolvers`: SUNDIALS 7.9.0 CVODE/IDA, KLU; the new `sbmlsim-c`) by git dependency, `libloading`
and `sha2` in sbmlsim-c; the system C compiler at study start; Python with uv in pkdb_models and
slideviz.

**Spec:** `docs/superpowers/specs/2026-11-25-slidesim-m3-cells-design.md` (milestone 3, the more
specific document) and its parent `docs/superpowers/specs/2026-11-24-slidesim-design.md`.

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
- No WebAssembly in a slidesim run. slidesim does not depend on `sbmlsim-host` or wasmtime.
- Generated C is compiled with `-O3 -march=native -ffp-contract=off -fno-fast-math` plus `-fPIC
  -shared`; the compiler is `$CC` or `cc`; the flags, the compiler's `--version` line and the
  source are in the cache key (SHA-256). Default cache directory:
  `$XDG_CACHE_HOME/sbmlsim/c` (or `~/.cache/sbmlsim/c`), overridable by `SBMLSIM_C_CACHE`.
- The generated C uses `<math.h>` functions, not sbmlsim-core's `mathlib`; agreement with the
  WebAssembly model is within integration tolerances, not bitwise.
- Repositories and branches:
  - slidesim: `feat/m3-cells` (exists, holds the design).
  - sbmlsim-core (`~/git/sbmlsim-core`): branch `feat/c-target` from `main`; pushed with a pull
    request; slidesim depends on it by `git` + `rev` until the release tag exists.
  - pkdb_models (`~/git/pkdb_models`): branch `feat/spt-liver-cell` from `develop`, pull request
    to `develop` (its rules in its `CLAUDE.md`).
  - slideviz (`~/git/slideviz`): branch `feat/cell-maps` from `feat/scene-export`, pull request
    with base `feat/scene-export`.
  - Never merge a pull request, never tag or publish a release: the user does that (release
    notes and version bumps are prepared in the branches).
- Each repository follows its own style and checks (sbmlsim-core: `cargo fmt --check`,
  `cargo clippy --all-targets --all-features -- -D warnings`, `cargo test --all-features`;
  pkdb_models and slideviz: their ruff, pytest and pre-commit configuration).

## Review Focus

1. A coarse element whose fine children lie partly outside the area tags where the model runs,
   or a scene whose maps have no-value pixels inside the tissue: parameters are means over the
   element's tissue only, a no-value pixel is an error naming the map and the element (Task 10,
   `parameters_are_area_means`, `no_value_in_tissue_is_an_error`).
2. A cell failure (step size underflow, non-finite state) on one process only: every process
   stops with an error naming the element, none hangs (Task 10 engine path and Task 12 batched
   path, `failure_on_one_process_stops_all` on 2 processes).
3. No C compiler, or generated code the compiler rejects: an error naming the command and its
   stderr before any solve (Task 6 `missing_compiler_is_an_error`, Task 11
   `run_without_compiler_fails_before_solving`).
4. A model with events or algebraic rules run with `"solver": "batched"`: an error naming the
   feature at study start; the same model with `"solver": "engine"` runs (Task 12
   `batched_refuses_events`, using an SBML Test Suite case with an event).
5. Steep uptake which would drive nodal blood concentrations below zero after an exchange: no
   negative concentration, total amount still conserved (Task 11
   `steep_uptake_stays_nonnegative`).

---

### Task 1: Conservative Darcy velocity

**Files:**
- Create: `crates/pde/src/fluxes.rs` (equilibrated edge fluxes and the Raviart-Thomas velocity)
- Modify: `crates/pde/src/flow.rs`, `crates/pde/src/lib.rs`, `crates/pde/src/balance.rs`
- Test: `crates/pde/tests/flow.rs`, `crates/pde/tests/advection.rs`,
  `crates/slidesim/tests/cli.rs`

**Interfaces:**
- Consumes: `Flow` (milestone 2), its residual-based `tag_flows`.
- Produces:
  - `Flow::edge_fluxes(&self, level: usize) -> &[f64]`: per local edge of level `level`, the
    volume flux (um^2/s in 2D) in the direction of the edge's reference normal; per element the
    signed sum of its three edge fluxes is zero; on edges of walls without a pressure condition
    exactly zero.
  - `Flow::velocity(level)` now returns the lowest-order Raviart-Thomas velocity of these fluxes
    at the quadrature points (same layout as before); `Flow::primal_velocity(level)` keeps the
    milestone 2 velocity (`-K grad p / mu`) for output and comparison.
  - `Flow::edge_velocity(level)` (used by `MassBalance` for the backflow term) follows the new
    velocity.
  - `velocity.vtu` gains the array `primal_velocity`.

- [ ] **Step 1: Write the failing tests**:
  - `crates/pde/tests/flow.rs`:
    - `element_fluxes_sum_to_zero`: on `lobule_toy.msh` (refinement 1) and the unit square:
      for every local element `|sum of signed edge fluxes| <= 1e-12 * max |flux|`.
    - `no_flux_through_walls_without_pressure`: every edge of a tag without a pressure condition
      has flux exactly 0.0.
    - `tag_flows_equal_edge_flux_sums`: per pressure tag, the sum of its edge fluxes equals
      `tag_flows` within 1e-10 relative.
    - `conservative_velocity_order`: the manufactured cases of milestone 2 (`pressure_order_*`,
      including `pressure_order_on_tagged_mesh`): L2 order of the new velocity >= 0.9 over 4
      levels.
    - `parallel_on_2_processes`, `parallel_on_3_processes` extended: edge flux sums and the new
      velocity agree with 1 process within 1e-10 relative.
  - `crates/pde/tests/advection.rs`: `no_advection_defect_with_conservative_velocity`: the
    `lobule_toy` transport of milestone 2: `advection_defect` of every output's `MassBalance`
    within 1e-10 of the portal inflow, and the advective outflow through untagged walls and
    `section_edge` zero to 1e-12 relative.
  - `crates/slidesim/tests/cli.rs`: `lobule_toy_flows_from_portal_to_central` tightened: the
    `advection_defect` in `summary.json` is below 1e-10 of the portal flow.
- [ ] **Step 2: Run** `cargo nextest run -p pde --test flow --test advection`. Expected: fail.
- [ ] **Step 3: Implement** in `fluxes.rs`: equilibrated fluxes from the residual of the discrete
  Darcy equations (patch-wise as Larson and Niklasson 2004, or a global solve on the edge graph;
  the builder's choice, spec: left to the builder), on every level from that level's own
  pressure. The quadrature-point velocity is the RT0 field `u = sum_e F_e phi_e` of the element.
- [ ] **Step 4: Run** `cargo nextest run -p pde` and `cargo nextest run -p slidesim`. Expected:
  all pass (update assertions of milestone 2 tests which bounded the old defect, and say so in the
  commit message).
- [ ] **Step 5: Commit** `feat(pde): conservative Darcy velocity from equilibrated edge fluxes`

### Task 2: Optimized test build of the native dependencies

**Files:**
- Modify: `scripts/deps/build-native.sh` (a third mode `check`), `.github/workflows/ci.yml`,
  `README.md`, `.config/nextest.toml` (only if timeouts can be tightened)

**Interfaces:**
- Produces: `deps/install/native-check` with `env.sh`; CI and the README use it for the tests.

- [ ] **Step 1: Add** mode `check` to `build-native.sh`: PETSc `--with-debugging=1` (argument and
  memory checks kept), `FLAGS="-O2 -g $FP"`, no `-march=native` (CI cache portability); libCEED
  with the same flags.
- [ ] **Step 2: Build** `scripts/deps/build-native.sh check` and run the whole suite with
  `. deps/install/native-check/env.sh && cargo nextest run --workspace && cargo test --doc --workspace`.
  Expected: all pass. Record the wall time of the suite and of `temporal_order_bdf2` before (debug)
  and after (check) in the commit message.
- [ ] **Step 3: Switch** CI (`ci.yml`, cache key includes the mode) and the README's test
  instructions to `native-check`; keep `native-debug` documented for gdb sessions.
- [ ] **Step 4: Run** both CI jobs locally in their Docker images as in milestone 1 (Open MPI on
  Ubuntu 24.04, MPICH on 22.04). Expected: pass.
- [ ] **Step 5: Commit** `build: optimized check build of PETSc and libCEED for the tests`

### Task 3: Transport performance and phase timers

**Files:**
- Modify: `crates/pde/src/problem.rs`, `crates/pde/src/multigrid.rs`,
  `crates/pde/src/operators.rs`, `crates/slidesim/src/run.rs`
- Create: `scripts/bench/transport.sh` (runs the milestone 2 baseline study on the real field at
  1 refinement on 1, 4 and 8 processes with `-log_view`, prints seconds per step)
- Create: `docs/results/2026-11-m3-performance.md` (started here, completed in Task 13)
- Test: `crates/pde/tests/problem.rs` (or the existing file of `Problem` tests),
  `crates/slidesim/tests/cli.rs`

**Interfaces:**
- Consumes: `Problem`, `Multigrid`, `Transport` (milestone 2).
- Produces:
  - `summary.json` gains `"phases": {"<name>": seconds}` with at least `setup`, `pressure`,
    `transport`, `output` (Task 11 adds `exchange`, `cells`), measured on rank 0 after a barrier
    per phase boundary (call: rank 0's wall time).
  - `Problem` keeps its operators, assembled coarse matrices and multigrid between steps and
    between `step_to` calls; it reassembles only when the TS shift changes (adaptive steps).

- [ ] **Step 1: Profile** the milestone 2 baseline study on the real field (`scripts/e2e/m2_real_slide.sh`
  outputs, 1 refinement) with `native-opt`, 4 processes, `-log_view`, fixed and adaptive steps;
  write the top five events by time into `docs/results/2026-11-m3-performance.md`.
- [ ] **Step 2: Write the failing tests:**
  - `operators_are_built_once`: a `Problem` stepped 20 fixed steps in two `step_to` calls builds
    its `Transport` operators and multigrid once (a test-only counter,
    `#[doc(hidden)] Problem::operator_builds(&self) -> usize`, equals the number of levels).
  - `same_result_with_reuse`: the L2 norm after the 20 steps equals that of a fresh `Problem`
    per call within 1e-12 relative.
  - `cli.rs` `summary_has_phases`: the four phase names present, each >= 0, their sum within 10 %
    of the run's wall time.
- [ ] **Step 3: Implement** the reuse, then whatever else the profile of Step 1 shows at the top
  (each change with its before/after seconds per step in the results doc).
- [ ] **Step 4: Run** `cargo nextest run --workspace`, then `scripts/bench/transport.sh`.
  Expected: tests pass; the seconds per step on 4 processes recorded before and after.
- [ ] **Step 5: Commit** `perf(pde): reuse operators and multigrid across steps, phase timers in summary.json`

### Task 4: pkdb_models: the `spt_liver_cell` variant

**Files (in `~/git/pkdb_models`, branch `feat/spt-liver-cell` from `develop`):**
- Modify: `pkdb_models/tpm/spt/models/zonation.py` (pattern 7), `pkdb_models/tpm/spt/models/spt_liver.py`
  (a builder option for the cell variant), `pkdb_models/tpm/spt/models/factory.py`,
  `pkdb_models/tpm/spt/README.md`
- Create: `pkdb_models/tpm/spt/models/output/models/spt_liver_cell.xml` and `.md` (generated)
- Create: `release-notes/<next version>.md` entry (or extend the open one), following the format
  of `release-notes/0.9.0.md`
- Test: `tests/models/test_tpm_spt.py`

**Interfaces:**
- Produces: `spt_liver_cell.xml`: model id `spt_liver_cell`, SBML L3V2 core only (no `comp`, no
  `distrib` namespace or `required` attribute), every element of `spt_liver` 0.9.0 except:
  - no ports;
  - `protein_random` a constant parameter 0.5 without an initial assignment;
  - a new constant parameter `protein_ext` (dimensionless, value 0.5, "protein set from outside,
    in [0, 1]");
  - pattern 7 "external" in `PATTERNS`: `protein = protein_ext`;
  - model version `0.10.0` with a changelog entry naming pattern 7 and the cell variant
    (`spt_liver` itself also gains pattern 7 and the version, so both stay one model).

- [ ] **Step 1: Write the failing tests** in `tests/models/test_tpm_spt.py`:
  - `test_cell_variant_has_no_packages`: the XML has no `comp` or `distrib` namespace and no
    `required` package attribute; libsbml reads it without errors.
  - `test_cell_variant_matches_liver_model`: roadrunner simulations of `spt_liver` and
    `spt_liver_cell` with `pattern` 6, `position` 0.3 and 0.8, `S_ext` initial 2 mM, 1440 min:
    every state equal within 1e-8 relative.
  - `test_pattern_7_is_protein_ext`: `pattern` 7, `protein_ext` 0.37: `protein` is 0.37.
- [ ] **Step 2: Run** `uv run pytest tests/models/test_tpm_spt.py`. Expected: fail.
- [ ] **Step 3: Implement**, regenerate with `python pkdb_models/tpm/spt/models/factory.py`.
- [ ] **Step 4: Run** the SPT tests and the repository's checks (pre-commit, ruff). Expected: pass.
  Compile the variant with sbmlsim-core: `~/git/sbmlsim-core/target/release/sbmlsim-core compile spt_liver_cell.xml -o /tmp/...`
  (scratchpad). Expected: compiles (no package error).
- [ ] **Step 5: Commit, push, open a pull request** to `develop`:
  `SPT liver cell variant for slidesim: pattern 7, no comp or distrib`.

### Task 5: slideviz: `cyp2e1` and `necrosis` maps, portal/central check

**Files (in `~/git/slideviz`, branch `feat/cell-maps` from `feat/scene-export`):**
- Modify: `scripts/write_scene.py` (options `--cyp2e1 <readout.npz>` and `--necrosis <calls.npz>`)
- Create: `scripts/wall_class_check.py` (the check on dosed slides)
- Test: `tests/test_write_scene.py`, `tests/test_wall_class_check.py`
- Create (in slidesim): `docs/results/2026-11-m3-wall-classes.md`

**Interfaces:**
- Consumes: the scene writer of milestone 2; `<stem>_cyp2e1_readout.npz` (`position` uint8 0-250,
  255 outside the tissue, `um_per_px`, `reference` the H&E stem; on the H&E working grid);
  `<animal>_necrosis.npz` (`call_E` uint8: 1 necrotic, 0 not, 255 not measured; `meta` JSON with
  `um_per_px`, `level`, `origin_level_rc`); the `_lobules.npz` of milestone 2.
- Produces:
  - scene maps `cyp2e1` (float32, readout / 250, units `1`) and `necrosis` (float32, 0 or 1,
    units `1`), on the grid and crop of the `zonation` map; pixels outside the tissue or not
    measured filled by nearest neighbour as `zonation` is (`fill_nearest`), the filled share
    stated in the map's provenance in `scene.json`.
  - `wall_class_check.py <animal> ...`: per slide the counts of lumina by class (portal 1,
    central 2, other 3), the portal/central ratio, and among the lumina inside or within 50 um of
    `call_E` necrosis the counts by class; JSON to stdout and a table.

- [ ] **Step 1: Write the failing tests:**
  - `test_write_scene.py::test_cell_maps_share_the_zonation_grid`: synthetic readout and calls:
    `cyp2e1` and `necrosis` maps have the shape, `um_per_px` and `origin_um` of `zonation`;
    values: readout 125 gives 0.5, call 1 gives 1.0; a 255 pixel takes its nearest neighbour.
  - `test_write_scene.py::test_necrosis_resampled_from_level_grid`: a calls array at 0.25 times
    the working pixel size with a necrotic block: the map is 1 inside the block's footprint and
    0 outside (block edges excepted).
  - `test_wall_class_check.py::test_counts_by_class_near_necrosis`: synthetic lumen classes and a
    necrotic disc: the counts within 50 um are exact.
- [ ] **Step 2: Run** `uv run pytest tests/test_write_scene.py tests/test_wall_class_check.py`.
  Expected: fail.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the slideviz tests (uv; the pixi mesh test) and the check on
  `mouse_apap_281mg_m1`, `mouse_apap_375mg_m1`, `mouse_apap_500mg_m1` and the three control
  mice (`SLIDEVIZ_DATA=/data/koeniglab/image.analysis/APAP/APAP_zarr`, read only). Write the
  results into slidesim's `docs/results/2026-11-m3-wall-classes.md`.
- [ ] **Step 5: Gate** (spec: portal and central check): the classification holds when at most 1
  in 10 lumina inside or within 50 um of the necrosis is portal and the portal/central ratio of
  each dosed slide lies within the range of the controls. If it does not hold on a slide, stop
  and report to the controller (the user decides); otherwise continue.
- [ ] **Step 6: Commit, push, open a pull request** in slideviz (base `feat/scene-export`):
  `CYP2E1 and necrosis maps in the scene, wall class check on dosed slides`; commit the results
  doc in slidesim: `docs: portal and central classification on dosed slides`.

### Task 6: sbmlsim-core: the C target and native model code

**Files (in `~/git/sbmlsim-core`, branch `feat/c-target` from `main`):**
- Create: crate `crates/c` (package `sbmlsim-c`): `src/lib.rs`, `src/emit.rs` (expressions and
  point functions), `src/build.rs` (compile, cache, load), `src/model.rs` (`CModel: ModelFns`)
- Modify: root `Cargo.toml` (member), `crates/cli/src/testsuite.rs` (option
  `--model-code <wasm|c>` running the native engine with `CModel` for `c`), `.github/workflows/ci.yml`
  (a job running the test suite with `--model-code c`), `docs/rust.md` (a section on the C target)
- Test: `crates/c/tests/c_target.rs`

**Interfaces:**
- Consumes: `sbmlsim_sbml::read`, `sbmlsim_ir::build`, `sbmlsim_ir::derive::jacobian`,
  `sbmlsim_ir::optimize::plan` (as `sbmlsim-codegen` uses them), `sbmlsim_engine::ModelFns`,
  the `ModelDescription` the codegen builds today (reuse its builder, do not duplicate it).
- Produces:
  - `pub fn generate(sbml: &str) -> Result<CSource, CError>`; `pub struct CSource { pub code:
    String, pub description: ModelDescription }`. The C defines, with a model-specific prefix,
    one function per non-default `ModelFns` method (`defaults`, `init`, `rhs`, `jac`, `roots`,
    `trigger`, `priority`, `delay`, `event_values`, `event_apply`, `selections`, `assign`),
    each taking `t`, pointers to `y`, `p` and outputs, and no globals.
  - `pub struct Library`; `Library::build(source: &str) -> Result<Arc<Library>, CError>` (cache as
    in the Global Constraints); `Library::symbol` for the loaded functions.
  - `pub struct CModel`; `CModel::new(lib: Arc<Library>, description: &ModelDescription) -> Result<CModel, CError>`;
    `impl ModelFns for CModel`; `CModel` is cheap to clone (one per engine instance).
  - `pub enum CError { Unsupported { feature: String }, NoCompiler { command: String },
    Compiler { command: String, stderr: String }, Load(String), Model(String) }` with messages.
  - Features the C target refuses in this task (an `Unsupported` error naming them): delays
    (`Op::Delay`, `Op::HistoryQuery`) unless implemented; models with `JacobianIr::ColoredDifference`.
- Expression semantics: as `crates/codegen/src/wasm.rs` `f64`/`bool`/`apply`/`piecewise`
  (piecewise without `otherwise` gives NaN; booleans as in `bool`); math from `<math.h>`.

- [ ] **Step 1: Write the failing tests** in `crates/c/tests/c_target.rs`:
  - `spt_matches_wasm_model`: `spt_liver_cell.xml` (copied into `testdata/models/` with its
    provenance line), 50 parameter sets (seeded: `pattern` 7, `protein_ext` in [0, 1],
    `position` in [0, 1], `S_ext` initial in [0, 5] mM), simulated 1440 min by the native engine
    with `CModel` and by the wasmtime host: every state at 100 output points within
    `1e-6 * |y| + 1e-9`.
  - `rhs_and_jacobian_match_wasm`: at 100 random states per model, `rhs` and `jac` of `CModel`
    equal those of the wasm model within 1e-13 relative (same expressions, different math library).
  - `event_model_matches_wasm`: an SBML Test Suite case with events and one with algebraic
    rules (names chosen from `testdata/cases`), native engine with `CModel` against the
    baseline with the suite's tolerances.
  - `cache_reuses_library`: `Library::build` twice with the same source compiles once (a second
    call takes under 50 ms and the cache file's mtime does not change).
  - `missing_compiler_is_an_error`: `CC=/nonexistent/cc` gives `CError::NoCompiler` naming it.
- [ ] **Step 2: Run** `cargo test -p sbmlsim-c`. Expected: fail.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `cargo test --all-features`, `target/release/sbmlsim-core testsuite
  testdata/cases --model-code c`. Expected: every case of the baseline passes except those the C
  target refuses, which are listed with their feature in the commit message and in
  `docs/rust.md`.
- [ ] **Step 5: Commit** `The C target: native model code for the engine`; push `feat/c-target`
  (no pull request yet).

### Task 7: sbmlsim-core: `Stepper`, an engine instance advanced from a given state

**Files (sbmlsim-core, `feat/c-target`):**
- Create: `crates/engine/src/stepper.rs`
- Modify: `crates/engine/src/lib.rs` (export), `docs/rust.md`
- Test: `crates/engine/tests/stepper.rs` (with `CModel` from Task 6 as a dev-dependency, and a
  wasm model through the host for the comparison)

**Interfaces:**
- Consumes: `ModelFns`, `Integrator` (`start`, `integrate`), `segments::integrate_events`,
  `IntegratorSettings`, `LinearSolver`.
- Produces:
  - `pub struct Stepper`;
    `Stepper::new(model: Box<dyn ModelFns>, desc: ModelDescription, settings: IntegratorSettings, changes: &[(String, f64)]) -> Result<Stepper, EngineError>`:
    defaults, the `changes` (parameter ids, fixed), `init`; solver memory created once.
  - `Stepper::state(&self) -> &[f64]` (initial state after `new`, then the last state).
  - `Stepper::advance(&mut self, y: &[f64], t0: f64, t1: f64) -> Result<StepStats, EngineError>`:
    integrates from `y` at `t0` to `t1` with the events and discontinuities of the interval as
    `simulate` handles them; the solver is reinitialized from `y` and starts with the last step
    size; the result is `state()`.
  - `pub struct StepStats { pub steps: u64, pub rejected: u64 }`.
  - Memory per instance is reported by `Stepper::memory_bytes(&self) -> usize` (estimate) for the
    performance report.

- [ ] **Step 1: Write the failing tests:**
  - `advance_matches_simulate`: the 50 SPT parameter sets of Task 6, advanced 0 to 1440 min in
    steps of 1, 10 and 60 min with nothing changed between calls: every state at every step end
    within `1e-6 * |y| + 1e-9` of one `simulate` with outputs at those times.
  - `advance_from_changed_state`: after 60 min, `S_ext` set to 0, then advanced 60 min: equals a
    `simulate` from the changed state (the request's initial values) within the same tolerance.
  - `events_inside_interval`: the event case of Task 6 advanced in 7 uneven steps equals its
    baseline.
  - `dae_model_advances`: the algebraic-rule case of Task 6, same check (IDA).
- [ ] **Step 2: Run** `cargo test -p sbmlsim-engine --test stepper`. Expected: fail.
- [ ] **Step 3: Implement.** Linear solver: `LinearSolver::Auto` (dense under `KLU_THRESHOLD`).
- [ ] **Step 4: Run** `cargo test --all-features`. Expected: pass. Record `memory_bytes` of an SPT
  stepper in the commit message.
- [ ] **Step 5: Commit** `Stepper: an engine instance advanced from a given state`; push.

### Task 8: sbmlsim-core: QFunction emitter with a Rosenbrock integrator

**Files (sbmlsim-core, `feat/c-target`):**
- Create: `crates/c/src/qfunction.rs`, `crates/c/include/sbmlsim_rosenbrock.h` (the integrator,
  written once, plain C99, no allocation, no host calls), `crates/c/tests/qfunction.rs`,
  `crates/c/benches/` or `bench/` entry (the repository's benchmark style)
- Modify: `crates/c/src/lib.rs`, `docs/rust.md`, `release-notes/0.27.0.md` (prepared, covering
  Tasks 6-8), the version locations of a release (as commit 273973f "Release 0.26.0" touched),
  only in a final commit of this task

**Interfaces:**
- Consumes: Task 6's emitter (rates, Jacobian dense in the QFunction, initial evaluations).
- Produces:
  - `pub fn generate_qfunction(sbml: &str) -> Result<QSource, CError>`;
    `pub struct QSource { pub code: String, pub function: String, pub state_ids: Vec<String>,
    pub parameter_ids: Vec<String>, pub description: ModelDescription }`. Refuses models with
    events, delays or algebraic rules (`CError::Unsupported` naming the feature).
  - The QFunction (libCEED interior signature `int f(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out)`,
    `CEED_QFUNCTION` macro, includes resolvable by libCEED's JIT):
    - inputs: `state` (n_states per point), `parameters` (n_parameters per point), `control`
      (4 per point: `t`, `h`, `done`, `failed`);
    - outputs: `state_out`, `control_out`, `counts` (2 per point: accepted, rejected steps,
      cumulative);
    - context `struct SbmlsimRosenbrockContext { double t1; double rtol; double atol; double h_min; }`
      in model time units;
    - one application = one step attempt per point which is not done (spec: the sequence of
      steps of a point does not depend on other points); a done or failed point is copied.
  - Rosenbrock method: order 3 or 4, L-stable, embedded error estimate (builder's choice, e.g.
    ROS3P or RODAS4), dense LU with partial pivoting of the `n x n` matrix per point.
  - A harness `pub fn run_points(lib: &Library, q: &QSource, ...)` for tests and benchmarks that
    calls the compiled QFunction on arrays exactly as libCEED would (layout `[size][Q]`), until
    every point is done.

- [ ] **Step 1: Write the failing tests** in `crates/c/tests/qfunction.rs`:
  - `rosenbrock_matches_stepper`: the 50 SPT parameter sets, 0 to 1440 min in 60 min intervals,
    `rtol 1e-6`, `atol 1e-9`: every state at every interval end within `1e-5 * |y| + 1e-8` of the
    `Stepper` (tighter if the method allows; the test states the bound).
  - `order_of_the_method`: a linear test ODE with a known solution, fixed steps: the observed
    order equals the method's order within 0.2.
  - `independent_of_blocking`: the same points applied with Q = 1, 7 and 64 per call and in a
    shuffled order: bitwise equal states.
  - `refuses_events`: the event case of Task 6: `CError::Unsupported` naming events.
  - `step_size_underflow_flags_failed`: a point whose `h` falls below `h_min` is `failed` and
    keeps its last state; other points finish.
- [ ] **Step 2: Run** `cargo test -p sbmlsim-c --test qfunction`. Expected: fail.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `cargo test --all-features`; the benchmark: SPT point-steps per second per
  core for the QFunction harness and for `Stepper` with `CModel`, on the machine of record;
  numbers in the commit message.
- [ ] **Step 5: Commit** `QFunction target: the model with a Rosenbrock integrator for libCEED`,
  then the release preparation commit `Prepare 0.27.0: C target, Stepper, QFunction target`;
  push, open the pull request to `main` (not merged, no tag).

### Task 9: `cells` in `study/1`

**Files:**
- Modify: `crates/contracts/src/study.rs`, `crates/contracts/schemas/study-1.schema.json`
  (generated as before), `crates/contracts/src/lib.rs`
- Test: `crates/contracts/tests/contracts.rs`

**Interfaces:**
- Produces:
  - `Study.cells: Option<CellsSpec>`;
  - `pub struct CellsSpec { pub model: PathBuf, pub solver: CellSolverKind, pub time_unit_s: f64,
    pub areas: Option<Vec<String>>, pub species: Vec<SpeciesLink>, pub volumes: VolumeLink,
    pub parameters: Vec<CellParameter>, pub split_step: f64, pub rtol: f64, pub atol: f64,
    pub outputs: Vec<String> }`
    (`model` relative to the study file; `solver` default `batched`; `areas` default all area
    tags; `rtol` default 1e-6, `atol` default 1e-9; `split_step` in s, > 0; `outputs`: extra
    assignment rule targets written per cell, default none);
  - `pub enum CellSolverKind { Batched, Engine }` (JSON `"batched"`, `"engine"`);
  - `pub struct SpeciesLink { pub species: String, pub field: String }`;
  - `pub struct VolumeLink { pub blood: String, pub cells: String }` (compartment ids of the
    model; set per element to porosity and 1 - porosity);
  - `pub struct CellParameter { pub id: String, pub value: Coefficient, pub replace: Option<Replace> }`;
  - `pub struct Replace { pub map: String, pub above: f64, pub value: Coefficient }`: where the
    element mean of `map` is above `above`, the parameter takes `value` instead;
  - `check_cells(study, scene)`: every linked field exists in `fields`, every map named in
    `parameters` exists in the scene, `split_step > 0` and `time.dt <= split_step`, every
    `outputs` id is a state or an assignment rule target (checked at study start, Task 10);
    errors list what exists.
- The SPT study (Task 13) writes `pattern` 7 with `replace {map: "necrosis", above: 0.5, value: 6}`,
  `protein_ext` as a table over `cyp2e1` `[[0, 0], [0.7405, 1]]` (held at the ends, so capped at
  1; spec: CYP2E1 replacement rule, scale), `position` as a table `[[0, 0], [1, 1]]` over
  `zonation`.

- [ ] **Step 1: Write the failing tests:** `cells_round_trip` (all fields), `cells_defaults`,
  `cells_unknown_field_lists_fields`, `cells_unknown_map_lists_maps`,
  `cells_split_step_positive`, `example_studies_are_valid` covering a new
  `testdata/studies/cells_toy/` study (written in Task 11, a placeholder file now).
- [ ] **Step 2: Run** `cargo nextest run -p contracts`. Expected: fail.
- [ ] **Step 3: Implement** and regenerate the schema.
- [ ] **Step 4: Run** `cargo nextest run -p contracts`. Expected: pass.
- [ ] **Step 5: Commit** `feat(contracts): cell models in study/1`

### Task 10: Crate `cells`: the engine path, parameters per element, the exchange

**Files:**
- Create: `crates/cells/` (`Cargo.toml`, `src/lib.rs`, `src/engine.rs`, `src/parameters.rs`)
- Create: `crates/pde/src/exchange.rs`
- Modify: `Cargo.toml` (workspace member; `sbmlsim-c`, `sbmlsim-engine`, `sbmlsim-abi` by `git` +
  `rev` of `feat/c-target`), `crates/mesh/src/lib.rs`, `crates/pde/src/problem.rs`,
  `crates/pde/src/lib.rs`
- Create: `testdata/models/spt_liver_cell.xml` (from Task 4, with a provenance comment: pkdb_models
  commit), `testdata/models/linear_uptake.xml` (one species `A_ext` in `Vext`, one `A` in `Vli`,
  uptake `k (A_ext - A)` and decay `d A`, plain SBML)
- Test: `crates/cells/tests/engine.rs`, `crates/pde/tests/exchange.rs`, `crates/mesh/tests/...`

**Interfaces:**
- Consumes: `CellsSpec` (Task 9), `sbmlsim_c::{generate, Library, CModel}`,
  `sbmlsim_engine::{Stepper, IntegratorSettings}`, `coefficient_values` and `Maps` (milestone 2).
- Produces:
  - `Mesh::coarse_cell(&self, level: usize, cell: usize) -> usize`: the local coarse cell (level
    0) containing local cell `cell` of `level` (from the regular refinement's numbering, checked by
    a test that every fine centroid lies in its coarse cell).
  - `pub trait CellSolver { fn n_cells(&self) -> usize; fn species(&self) -> &[String];
    fn state_ids(&self) -> &[String]; fn advance(&mut self, t0: f64, t1: f64, blood: &mut [f64]) -> Result<CellStats>;
    fn states(&self) -> &[f64]; }`: `blood` holds per cell the concentrations of the linked
    species (`n_cells * n_species`, cell-major) and is replaced by their values after the step;
    times in s (the solver converts with `time_unit_s`); `states` is `n_cells * n_states`.
  - `pub struct CellStats { pub steps: u64, pub rejected: u64, pub seconds: f64 }`.
  - `pub struct CellSetup { pub cells: Vec<usize> (local coarse cells with a model), pub parameters: Vec<Vec<(String, f64)>> }`;
    `CellSetup::new(mesh: &Mesh, spec: &CellsSpec, maps: &Maps, porosity: &Coefficient, ceed: &Ceed) -> Result<CellSetup>`:
    per coarse cell of the `areas`, every `CellParameter` as the mean of its coefficient over the
    quadrature points of the cell's finest children (spec: mean over the element's tissue), the
    `replace` rule applied, `Vext = mean porosity`, `Vli = 1 - mean porosity`.
  - `pub struct EngineCells: CellSolver`; `EngineCells::new(setup: &CellSetup, spec: &CellsSpec, model_sbml: &str) -> Result<EngineCells>`
    (one `Stepper` per cell, one `Library` per process).
  - `pub struct Exchange`; `Exchange::new(problem_disc: &Discretization, mesh: &Mesh, cells: &[usize], porosity: &Coefficient, maps: &Maps) -> Result<Exchange>`;
    `Exchange::means(&self, problem: &Problem, out: &mut [f64]) -> Result<()>` (per cell the
    porosity-weighted mean of the field over the cell); `Exchange::give_back(&self, problem: &mut Problem, delta: &[f64]) -> Result<()>`
    (per cell the change of concentration times the cell's blood volume added to the field so that
    `int phi u` changes by exactly the sum; built from libCEED operators; positivity safeguard).
  - `Problem::modify(&mut self, f: impl FnOnce(&mut Vector) -> Result<()>) -> Result<()>`: changes
    the solution between steps and restarts the time integrator's history (BDF2 restarts).
  - `cells` errors are `CellsError` variants naming the element (global coarse index and centroid)
    and are collective.

- [ ] **Step 1: Write the failing tests:**
  - `crates/mesh`: `coarse_cell_contains_fine_centroids` (lobule_toy, 2 refinements, 1 and 3
    processes).
  - `crates/pde/tests/exchange.rs`:
    - `means_of_constant_field_are_the_constant`;
    - `give_back_conserves_total` (random deltas: `int phi u` changes by `sum delta_i V_i` within
      1e-12 relative);
    - `steep_uptake_stays_nonnegative` (deltas removing 99.9 % of each cell's amount: no nodal
      value below zero, total still exact);
    - `parallel_on_2_processes`.
  - `crates/cells/tests/engine.rs`:
    - `parameters_are_area_means` (a map `x` over a two-area mesh, model only in one area: each
      cell's parameter equals the mean of `x` over its children within 1e-12);
    - `replace_rule_switches_value` (a `necrosis`-like 0/1 map: cells above 0.5 get the replacement);
    - `no_value_in_tissue_is_an_error`;
    - `engine_cells_match_stepper` (SPT, 10 cells, blood held fixed: each cell equals a standalone
      `Stepper` within 1e-12, since it is the same code);
    - `failure_on_one_process_stops_all` (a parameter making one cell on rank 1 fail, 2
      processes: both ranks return the error naming the cell).
- [ ] **Step 2: Run** `cargo nextest run -p mesh -p pde -p cells`. Expected: fail.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `cargo nextest run --workspace`. Expected: pass.
- [ ] **Step 5: Commit** `feat(cells): cell models per element on sbmlsim-core's engine, exchange with the blood fields`

### Task 11: Coupled run: Strang splitting, mass balance, output

**Files:**
- Create: `crates/slidesim/src/coupled.rs`
- Modify: `crates/slidesim/src/run.rs`, `crates/slidesim/src/main.rs` (help text),
  `crates/pde/src/balance.rs` (exchange term), `README.md`
- Create: `testdata/studies/cells_toy/{scene.json,cells.study.json}` (lobule_toy with
  `linear_uptake.xml`, engine path; and an SPT variant `spt.study.json`)
- Create: `testdata/studies/cells_manufactured/` (unit square, closed walls, `linear_uptake.xml`)
- Test: `crates/slidesim/tests/cli.rs`, `crates/slidesim/tests/coupled.rs`

**Interfaces:**
- Consumes: Tasks 3, 9, 10.
- Produces:
  - `slidesim run` with `cells`: per split step `dt`: cells `dt/2`, transport of every linked
    field `dt` (each its `Problem::step_to`), cells `dt/2`; the exchange before and after every
    cell step.
  - `MassBalance` gains `exchange` (amount per s given by the cells to the field, per output
    time); `summary.json` per field and in total.
  - Outputs: `cells_<index>.vtu` with cell data per coarse element: every model state, `protein`
    and the assignment rule targets of `CellsSpec::outputs` (`protein` in the SPT studies), and
    the element's mean parameters; fields as before.
  - `summary.json` `phases` gains `exchange`, `cells`; `cells` gains `steps`, `rejected`.

- [ ] **Step 1: Write the failing tests:**
  - `coupled.rs::strang_order_two`: `cells_manufactured` (well mixed: blood `u` and cell `a`
    with `u' = -k (u - a) V_c/V_b`, `a' = k (u - a) - d a`, closed walls, uniform initial values,
    so the analytic solution is the linear ODE's): split steps 8, 4, 2, 1 s with transport
    steps 1/16 of the split step: the error at 32 s falls at order >= 1.9.
  - `coupled.rs::coupled_conservation`: `cells_toy` with the linear model and decay 0: the
    amount in blood and cells changes by the boundary flows only, within 1e-8 relative, at every
    output.
  - `cli.rs::cells_parallel_matches_one` (2 and 3 processes, fixed transport steps, engine path):
    fields, cell states and `summary.json` integrals within 1e-10 relative.
  - `cli.rs::run_without_compiler_fails_before_solving` (`CC=/nonexistent/cc`): exit code 1, the
    message names the command, no output files beyond the log.
  - `cli.rs::spt_toy_runs` (SPT on `lobule_toy`, 60 min, portal `S` 5 mM): `necrosis` in
    `cells_*.vtu` rises in cells with high `protein` first.
- [ ] **Step 2: Run** `cargo nextest run -p slidesim`. Expected: fail.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `cargo nextest run --workspace`, then a first real-slide baseline: the
  281 mg/kg slide scene of Task 5, SPT engine path, 24 h, split step 60 s, 1 refinement, 8
  processes, `native-opt`; record the phases in `docs/results/2026-11-m3-performance.md`.
  Expected: tests pass; the run finishes (whatever its time).
- [ ] **Step 5: Commit** `feat: coupled runs of cell models and blood fields by Strang splitting`

### Task 12: The batched path in slidesim

**Files:**
- Create: `crates/cells/src/batched.rs`
- Modify: `crates/cells/src/lib.rs`, `crates/cells/Cargo.toml` (`ceed`, `petsc`),
  `crates/slidesim/src/coupled.rs` (path choice)
- Test: `crates/cells/tests/batched.rs`, `crates/slidesim/tests/cli.rs`

**Interfaces:**
- Consumes: `sbmlsim_c::{generate_qfunction, QSource, Library}`, the QFunction layout of Task 8,
  `ceed::{QFunction::interior, Operator, ElemRestriction::strided}`, `CellSetup`.
- Produces:
  - `pub struct BatchedCells: CellSolver`;
    `BatchedCells::new(setup: &CellSetup, spec: &CellsSpec, model_sbml: &str, ceed: &Ceed) -> Result<BatchedCells>`:
    one libCEED operator over the process's cells, one point per cell (strided restrictions,
    no basis), the generated QFunction from the compiled `Library` (function pointer) with its
    source path for JIT backends; `advance` applies the operator until every point is done
    (swapping or copying `state`/`state_out`, `control`/`control_out`), then reduces `failed`
    collectively.
  - `Cells::new(...) -> Result<Box<dyn CellSolver>>` choosing by `CellsSpec::solver`; `batched`
    on a model with events, delays or algebraic rules is an error naming the feature and
    suggesting `"engine"`.

- [ ] **Step 1: Write the failing tests:**
  - `batched_matches_engine`: SPT, the 50 parameter sets as cells on `lobule_toy`, blood held
    fixed, 1440 min in 60 min steps: within the bound of Task 8's `rosenbrock_matches_stepper`.
  - `same_on_backends`: `/cpu/self/ref/serial` and `/cpu/self/opt/blocked`: states within 1e-12
    relative.
  - `parallel_on_2_processes`, `parallel_on_3_processes`: the cell states equal 1 process within
    1e-12 relative.
  - `failure_on_one_process_stops_all` (as Task 10, batched).
  - `batched_refuses_events`.
  - `cli.rs::cells_parallel_matches_one` extended to the batched path.
- [ ] **Step 2: Run** `cargo nextest run -p cells`. Expected: fail.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `cargo nextest run --workspace`; then the Task 11 baseline run on the batched
  path; record the phases beside the engine path's.
- [ ] **Step 5: Commit** `feat(cells): batched cell models as a libCEED QFunction with a Rosenbrock integrator`

### Task 13: End to end on dosed slides, performance and scaling

**Files:**
- Create: `scripts/e2e/m3_dosed_slides.sh` (slideviz chain of milestone 2 plus the cell maps for
  one animal, then the slidesim runs), `scripts/e2e/m3_dosed_slides/` (analysis, calibration and
  figure scripts, run with `uv run --project $SLIDEVIZ_REPO`, as in milestone 2)
- Create: `testdata/studies/m3_dosed_slides/{measured,pattern6}.study.json`
- Create: `docs/results/2026-11-m3-dosed-slides.md` and images under
  `docs/results/2026-11-m3-dosed-slides/`; complete `docs/results/2026-11-m3-performance.md`
- Modify: `Cargo.toml` (sbmlsim-core dependencies on the release tag, once the user has released
  0.27.0; until then the `rev`)

**Interfaces:**
- Consumes: everything above; slides `mouse_apap_281mg_m1`, `mouse_apap_375mg_m1`,
  `mouse_apap_500mg_m1` (read only); the M2 perfusion baseline parameters; `necrosis` map.

- [ ] **Step 1: Studies:** perfusion as the milestone 2 baseline; fields `S` and `P` (diffusion
  880 um^2/s for S, 700 for P, porosity 0.15, advection, `S` dirichlet at `portal_wall` the
  calibrated value, `P` 0 there; the diffusion of P is a call, stated in the results); cells as in Task 9's interface note; `time_unit_s` 60;
  `split_step` 60 s; 24 h; outputs every hour; 1 refinement.
- [ ] **Step 2: Calibrate** per slide the portal `S` by bisection over runs (batched path) so that
  the simulated necrotic share (cells with `necrosis > 0.5`, area weighted) equals the measured
  share of the field (`necrosis` map mean) within 0.01; record the runs.
- [ ] **Step 3: Compare** per slide, for `measured` and `pattern6`: Dice coefficient of
  simulated (`necrosis > 0.5`) and measured (`necrosis` map > 0.5) per coarse element; the
  necrotic fraction in 10 position bins of `zonation`, simulated against measured; the necrotic
  share per lobule (lobules from the `_lobules.npz`), correlation. Check the split step by
  halving it once on one slide (Dice and shares change by less than 0.01).
- [ ] **Step 4: Paths and performance:** one slide with both paths: the same cells necrotic
  (differences listed) and states within the bound of Task 12; phases per path; a strong scaling
  series 1, 2, 4, 8, 12 processes for both paths; memory per process; against the target
  (24 h at 1 refinement in at most 10 minutes on 8 processes, batched path).
- [ ] **Step 5: Render and inspect** per slide: measured and simulated necrosis side by side over
  the H&E thumbnail, `protein` map, `S` and `T` at 1, 6 and 24 h; pixel perfection as in
  milestone 2; write `docs/results/2026-11-m3-dosed-slides.md` with every number, the wall class
  check of Task 5, the CYP2E1 scale check (readout against pattern 6 by position on
  non-necrotic tissue), and the deviations.
- [ ] **Step 6: Commit** `docs: milestone 3 end-to-end runs on dosed slides, performance and scaling`

---

## Self-review notes

- Spec coverage: conservative velocity (T1); optimized test build, operator reuse, adaptive
  transport, phase timers (T2, T3); SPT variant (T4); cyp2e1/necrosis maps and the wall class
  check with its stop (T5); C target and native engine path (T6, T10); advance call (T7); QFunction
  and Rosenbrock (T8, T12); study syntax (T9); coupling, exchange, conservation, Strang order,
  parity, output (T10, T11); verification items 1-5 (T1, T6-T12), item 6 with calibration,
  comparisons, both CYP2E1 inputs, both paths, scaling (T13); releases prepared, not made (T4, T8).
- Order follows the spec's order of work, with the contracts (T9) placed before the coupling.
- Left to the implementer, as the spec says: the flux algorithm, the Rosenbrock method and its
  stepping scheme within the stated rules, the advance call's internals, the cache location
  (default given), the split and transport steps of the end-to-end study (checked by halving),
  the metrics' thresholds beyond those stated.
