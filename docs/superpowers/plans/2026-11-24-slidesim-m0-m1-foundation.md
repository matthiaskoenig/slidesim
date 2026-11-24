# slidesim M0-M1: wasm spike and 2D diffusion-reaction foundation - Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decide by a spike whether PETSc + libCEED run in `wasm32-wasip1`, and build the native
CPU foundation of slidesim: a Rust core which solves 2D diffusion-reaction on slideviz gmsh meshes
with libCEED matrix-free operators, PETSc time stepping and geometric multigrid, verified by
manufactured solutions.

**Architecture:** A Cargo workspace. `petsc-sys` and `ceed-sys` are `bindgen` bindings against
PETSc and libCEED built by our own scripts into `deps/install/<arch>`. Thin safe crates (`petsc`,
`ceed`) wrap only the calls used. `contracts` defines `scene/1` and `study/1`. `mesh` loads gmsh
into DMPlex with named tags and a uniform refinement hierarchy. `pde` builds libCEED operators on
it (QFunctions in C headers, so the same source later JIT-compiles on GPU), wraps them as PETSc
shell matrices and integrates with `TS`. The `slidesim` crate is the CLI.

**Tech Stack:** Rust 1.100.0 (edition 2024), PETSc and libCEED (latest releases, pinned), MPI
(system `mpicc`), bindgen, cc, serde, schemars, cargo-nextest, wasi-sdk (`~/opt/wasi-sdk`),
wasmtime and Node for the spike, gmsh (Python, via `uvx`) for test fixtures.

**Spec:** `docs/superpowers/specs/2026-11-24-slidesim-design.md`

This plan covers milestones 0 and 1 of the spec. Milestones 2-8 get their own plans; milestone 5
(browser build of the real crates) is planned only after the spike result of Task 3.

## Global Constraints

- Rust toolchain `1.100.0` with targets `wasm32-wasip1`, `wasm32-unknown-unknown`, as in sbmlsim-core.
- slidesim is MIT. p4est is not linked in this plan.
- PETSc, libCEED pinned to one version each in `deps/versions.env`, built the same way on all targets.
- No Python, or any interpreter, in the hot loop.
- No model-specific or tissue-specific code in the core: tags, fields and ids are data from the contracts.
- Units in the contracts: length um, time s, concentration mM; diffusion um^2/s, decay 1/s, source mM/s, flux mM*um/s.
- C code compiled with `-O3 -DNDEBUG -ffp-contract=off -fno-fast-math`, as in sbmlsim-core.
- `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings` and all tests pass at every commit.
- No agent attribution anywhere: no `Co-Authored-By`, no "Generated with" lines (user's CLAUDE.md).
- Never use the em dash character in docs, comments or messages.

## Review Focus

1. A study names a boundary tag that the mesh does not have: the run fails before solving with an error naming the tag and listing the available tags (Task 7).
2. A PETSc or libCEED call fails: the run returns a Rust error carrying the library's message; the process never aborts or prints only a code (Task 4, Task 5).
3. A linear or nonlinear solve does not converge: the run fails with the solver's reason instead of writing a wrong result (Task 9).
4. The same study on 1 and 2 MPI ranks: results agree within 1e-10 relative in the L2 norm (Task 11).
5. A slideviz mesh with sliver triangles or a scene in a length unit other than `um`: a clear error for a non-positive element Jacobian, and a refusal of the unit (Task 6, Task 8).

---

## Part A - Spike (milestone 0)

### Task 1: Repository bootstrap and native dependency build

**Files:**
- Create: `Cargo.toml` (workspace, `resolver = "3"`, edition 2024, `license = "MIT"`)
- Create: `rust-toolchain.toml` (copy of sbmlsim-core's)
- Create: `LICENSE` (MIT), `README.md` (one paragraph from the spec's Intent, build steps)
- Create: `.gitignore` (`target/`, `deps/build/`, `deps/install/`, `deps/src/`)
- Create: `deps/versions.env`
- Create: `scripts/deps/fetch.sh`, `scripts/deps/build-native.sh`
- Create: `.github/workflows/ci.yml` (fmt, clippy, nextest on Ubuntu with the native deps cached by the hash of `deps/versions.env` and the scripts)

- [ ] **Step 1: `git init`, default branch `main`**

- [ ] **Step 2: Pin versions.** Find the latest release tags:
  `git ls-remote --tags https://gitlab.com/petsc/petsc.git 'v3.*'` and
  `git ls-remote --tags https://github.com/CEED/libCEED.git 'v0.*'`.
  Write `PETSC_VERSION=<tag>` and `LIBCEED_VERSION=<tag>` into `deps/versions.env`.

- [ ] **Step 3: `scripts/deps/fetch.sh`** clones both tags shallowly into `deps/src/petsc` and `deps/src/libceed`; idempotent.

- [ ] **Step 4: `scripts/deps/build-native.sh <debug|opt>`** builds into `deps/install/native-<mode>`:
  - libCEED: `make install prefix=... OPT='-O3 ...'` (debug: `-O0 -g`), CPU backends only.
  - PETSc: `./configure --prefix=... --with-cc=mpicc --with-cxx=0 --with-fc=0 --download-f2cblaslapack --with-shared-libraries=1 --with-debugging=<1|0> COPTFLAGS=<flags>`, then `make all install`.
  - Writes `deps/install/native-<mode>/env.sh` exporting `PETSC_DIR`, `CEED_DIR`, `PKG_CONFIG_PATH`, `LD_LIBRARY_PATH`.
  - Do not use the system PETSc 3.19.6.

- [ ] **Step 5: Verify.** Run `scripts/deps/build-native.sh debug && . deps/install/native-debug/env.sh && pkg-config --modversion PETSc ceed`.
  Expected: the two pinned versions.

- [ ] **Step 6: Commit** `chore: bootstrap workspace and native PETSc/libCEED build`

### Task 2 (spike, throwaway): PETSc + libCEED for `wasm32-wasip1`

**Files:**
- Create: `spike/petsc-wasm/build-wasi.sh`, `spike/petsc-wasm/PATCHES.md`

- [ ] **Step 1: Build libCEED** with `$WASI_SDK_PATH/bin/clang --sysroot=$WASI_SDK_PATH/share/wasi-sysroot -msimd128` as a static library, `/cpu/self` backends only, into `deps/install/wasi`.

- [ ] **Step 2: Build PETSc** for wasi with `--with-batch --known-*` cross-configure: `--with-mpi=0 --with-fc=0 --with-cxx=0 --download-f2cblaslapack --with-shared-libraries=0 --with-x=0 --with-ssl=0 --with-pthread=0 --with-debugging=0`, the same compile flags. Record every configure workaround and source patch (missing POSIX calls such as signals, `getpwuid`, `gethostname`, `popen`) in `PATCHES.md`, with the reason for each.

- [ ] **Step 3: If wasip1 is blocked** after patches of reasonable size: repeat with emscripten (`wasm32-unknown-emscripten`), the route FEM on Browser uses, and record the result in `PATCHES.md`.

- [ ] **Step 4: Verify.** Run `ls -l deps/install/wasi/lib/{libpetsc.a,libceed.a}`. Expected: both exist.

- [ ] **Step 5: Commit** `spike: build PETSc and libCEED for wasm32-wasip1`

### Task 3 (spike, throwaway): 2D diffusion in wasm, native versus wasm, decision

**Files:**
- Create: `spike/petsc-wasm/diffusion/` (a standalone Cargo package, not in the workspace), `spike/petsc-wasm/run-node.mjs`, `spike/petsc-wasm/RESULT.md`

- [ ] **Step 1: Program.** A Rust binary with raw `bindgen` bindings:
  - `DMPlexCreateBoxMesh`, 2D, 64x64 simplices, P1;
  - a libCEED diffusion operator applied through a `MATSHELL`;
  - CG with Jacobi on `-Δu = 2π² sin(πx) sin(πy)`, zero Dirichlet;
  - prints the L2 error and the solution's 2-norm with 17 digits.

- [ ] **Step 2: Run** natively, under `wasmtime`, and in Node through `node:wasi` (`run-node.mjs`).

- [ ] **Step 3: `RESULT.md`** records:
  - go or no-go;
  - the target used (wasip1 or emscripten);
  - the `.wasm` size (raw and gzip);
  - run times on the three hosts;
  - the relative difference of the 2-norms native versus wasm (go requires below 1e-12);
  - the list of patches.

  No-go means the spec's fallback (backend trait with a pure-Rust solver) is planned for milestone 5.

- [ ] **Step 4: Commit** `spike: 2D diffusion in wasm, result`

---

## Part B - Foundation (milestone 1)

### Task 4: `petsc-sys` and `petsc`

**Files:**
- Create: `crates/petsc-sys/{Cargo.toml,build.rs,wrapper.h,src/lib.rs}`
- Create: `crates/petsc/{Cargo.toml,src/lib.rs,src/error.rs,src/vector.rs,src/matrix.rs,src/dm.rs,src/ksp.rs,src/ts.rs}`
- Test: `crates/petsc/tests/basics.rs`

**Interfaces:**
- Produces:
  - `petsc::Petsc::init(args: &[&str]) -> Result<Petsc>`. Sets `PetscPushErrorHandler(PetscReturnErrorHandler)`; `Drop` calls `PetscFinalize`.
  - `Petsc::comm(&self) -> Comm`.
  - `Comm::rank(&self) -> usize`, `Comm::size(&self) -> usize`.
  - `petsc::Error { code: i32, message: String }`, `petsc::Result<T>`; macro `chk!(call)` builds `Error` with `PetscErrorMessage` plus the traceback text.
  - `Vector`: `create_seq(p: &Petsc, n: usize)`, `create_global(dm: &Dm)`, `create_local(dm: &Dm)`, `norm2()`, `dot(&Vector)`, `axpy(f64, &Vector)`, `set(f64)`, `with_array(|&[f64]|)`, `with_array_mut(|&mut [f64]|)`.
  - `Matrix::shell(comm, local_rows, global_rows, ctx: Box<dyn ShellOperator>)`, with `trait ShellOperator { fn mult(&self, x: &Vector, y: &mut Vector) -> Result<()>; fn diagonal(&self, d: &mut Vector) -> Result<()>; }`. Plus `Matrix::aij_coo(comm, rows, cols: &[i64]...)` with `set_values_coo(&[f64])`.
  - `Dm` (owned DMPlex handle, `Clone` via reference count), `Ksp`, `Ts`: thin owners with `set_from_options()` and the setters used by Tasks 8-10.
- bindgen: `wrapper.h` includes `petsc.h`; link via `pkg-config PETSc`.

- [ ] **Step 1: Failing tests**

```rust
#[test] fn init_and_finalize() { let p = Petsc::init(&[]).unwrap(); assert_eq!(p.comm().size(), 1); }
#[test] fn error_carries_message() {
    let p = Petsc::init(&[]).unwrap();
    let e = Vector::create_seq(&p, 3).unwrap().axpy(1.0, &Vector::create_seq(&p, 4).unwrap()).unwrap_err();
    assert!(e.message.contains("Nonconforming object sizes"), "{e}");
}
#[test] fn shell_matrix_applies() { /* 3x3 shell doubling x; y = A x; assert y == 2 x */ }
```

- [ ] **Step 2: Run** `cargo nextest run -p petsc`. Expected: FAIL (crate empty).

- [ ] **Step 3: Implement** the bindings and wrappers above. Tests run one process per test (nextest), so `Petsc::init` runs once per process.

- [ ] **Step 4: Run** `cargo nextest run -p petsc`. Expected: PASS.

- [ ] **Step 5: Commit** `feat(petsc): bindings and safe wrappers`

### Task 5: `ceed-sys` and `ceed`

**Files:**
- Create: `crates/ceed-sys/{Cargo.toml,build.rs,wrapper.h,src/lib.rs}`
- Create: `crates/ceed/{Cargo.toml,src/lib.rs}`
- Test: `crates/ceed/tests/basics.rs`

**Interfaces:**
- Produces:
  - `ceed::Ceed::new(resource: &str) -> Result<Ceed>` (default `"/cpu/self/opt"`).
  - `ceed::Error { message: String }`; libCEED's error handler set to `CeedErrorStore`, the message read with `CeedGetErrorMessage`.
  - `CeedVector` wrapper `CVector` with `from_slice`, `with_array`.
  - `ElemRestriction::new(ceed, num_elem, elem_size, num_comp, comp_stride, l_size, offsets: &[i32])`.
  - `Basis::h1_lagrange_simplex(ceed, dim, num_comp, degree, q_order)`.
  - `QFunction::interior(ceed, vlength, f: CeedQFunctionUser, source: &str)` with `add_input`/`add_output`, and `set_context(&[f64])`.
  - `Operator` with `set_field`, `apply(&CVector, &mut CVector)`, `assemble_diagonal(&mut CVector)`, `assemble_coo() -> (Vec<i64>, Vec<i64>, CVector)`.
- We use our own bindings rather than the `libceed` crate, so that C QFunctions with source paths (needed for GPU JIT later) and the wasi build go through one controlled path.

- [ ] **Step 1: Failing tests:** `mass_operator_integrates_one` asserts that the 1D P1 mass operator applied to ones, summed, equals the interval length 1.0 (1e-14). `error_carries_message` asserts that an invalid resource `"/nonexistent"` gives an error whose message contains `"/nonexistent"`.
- [ ] **Step 2: Run** `cargo nextest run -p ceed`. Expected: FAIL.
- [ ] **Step 3: Implement** bindings and wrappers. Link via `pkg-config ceed`.
- [ ] **Step 4: Run** `cargo nextest run -p ceed`. Expected: PASS.
- [ ] **Step 5: Commit** `feat(ceed): bindings and safe wrappers`

### Task 6: `contracts`: `scene/1` and `study/1`

**Files:**
- Create: `crates/contracts/{Cargo.toml,src/lib.rs,src/scene.rs,src/study.rs}`
- Create: `schemas/scene-1.schema.json`, `schemas/study-1.schema.json` (generated, checked by a test)
- Test: `crates/contracts/tests/contracts.rs`, fixtures `testdata/contracts/*.json`

**Interfaces:**
- Produces:
  - `Scene { version: String, dimension: u8, units: Units { length: LengthUnit }, mesh: MeshRef { format: MeshFormat::Gmsh, path: PathBuf } }`.
  - `Study { version, scene: PathBuf, fields: Vec<FieldSpec>, time: TimeSpec, mesh: MeshOptions { refinements: u32 }, solver: SolverOptions { petsc_options: String } }`.
  - `FieldSpec { id: String, diffusion: f64, decay: f64, source: f64, initial: f64, boundaries: Vec<BoundarySpec> }`.
  - `BoundarySpec { tag: String, condition: Condition }`, where `enum Condition { Dirichlet { value: f64 }, Flux { value: f64 } }`, serialized as `{"tag": "...", "dirichlet": 0.1}` or `{"tag": "...", "flux": 1.0}`. Tags not listed are zero flux.
  - `TimeSpec { end: f64, dt: f64, adaptive: bool, outputs: Vec<f64> }`.
  - `Scene::load(path) -> Result<Scene, ContractError>`, `Study::load(path) -> Result<Study, ContractError>`. Relative paths resolve against the file's directory.
  - `fn schema_scene() -> serde_json::Value`, `fn schema_study() -> serde_json::Value` (schemars).
- Errors (exact text):
  - `unsupported contract version 'study/2', expected 'study/1'`;
  - `unsupported length unit 'mm', expected 'um'`;
  - `field 'oxygen': diffusion must be > 0, got -1`;
  - `time.outputs must be sorted and within [0, end]`.

- [ ] **Step 1: Failing tests:**
  - round trip of `testdata/contracts/study-minimal.json`;
  - each of the four error texts above, from its own fixture;
  - `schemas_are_current`: the generated schema equals the checked-in file. On mismatch the error message tells the reader to run `cargo run -p contracts --example write-schemas`.
- [ ] **Step 2: Run** `cargo nextest run -p contracts`. Expected: FAIL.
- [ ] **Step 3: Implement** with serde (`deny_unknown_fields`) plus validation in `load`, and the example `write-schemas`.
- [ ] **Step 4: Run** `cargo nextest run -p contracts`. Expected: PASS.
- [ ] **Step 5: Commit** `feat(contracts): scene/1 and study/1`

### Task 7: `mesh`: gmsh into DMPlex with tags and a refinement hierarchy

**Files:**
- Create: `crates/mesh/{Cargo.toml,src/lib.rs}`
- Create: `testdata/meshes/make_fixtures.py` (gmsh API, run with `uvx --from gmsh python`), producing:
  - `unit_square.msh`: 1 x 1 um, edges `left`, `right`, `top`, `bottom`;
  - `tissue_small.msh`: a 200 x 200 um square with two circular holes, physical names as slideviz writes them: area `tissue` (3), edges `vessel_wall` (11), `section_edge` (12).
- Test: `crates/mesh/tests/mesh.rs`

**Interfaces:**
- Consumes: `petsc::{Petsc, Dm}`, `contracts::Scene`.
- Produces:
  - `Mesh::load(petsc: &Petsc, scene: &Scene, refinements: u32) -> Result<Mesh, MeshError>`. Reads with `-dm_plex_gmsh_use_regions` so labels carry the physical names, distributes, then refines uniformly `refinements` times.
  - `Mesh::levels(&self) -> &[Dm]` (coarse first; the finest is the PDE level).
  - `Mesh::finest(&self) -> &Dm`.
  - `Mesh::tag(&self, name: &str) -> Result<Tag, MeshError>`, `Tag { name: String, dim: u8 }`.
  - `MeshError::UnknownTag { name: String, available: Vec<String> }`, displayed as `mesh has no tag 'portal_wall'; available: section_edge, tissue, vessel_wall`.

- [ ] **Step 1: Failing tests:**
  - `unit_square_has_edges`: the four tags exist.
  - `refinement_quadruples_cells`: global cell count at level 1 is 4x level 0.
  - `unknown_tag_lists_available`: exact message above for `tissue_small.msh`.
  - `tissue_area`: integral of 1 over the finest level equals 40000 - 2·π·r² within the polygonal approximation (relative 1e-2).
- [ ] **Step 2: Run** `cargo nextest run -p mesh`. Expected: FAIL.
- [ ] **Step 3: Implement.** Generate the fixtures once and commit them; the generator stays for reproducibility.
- [ ] **Step 4: Run** `cargo nextest run -p mesh`. Expected: PASS.
- [ ] **Step 5: Commit** `feat(mesh): gmsh to DMPlex with tags and refinement`

### Task 8: `pde` discretization and operators

**Files:**
- Create: `crates/pde/{Cargo.toml,build.rs,src/lib.rs,src/discretization.rs,src/operators.rs}`
- Create: `crates/pde/qfunctions/{geometry.h,mass.h,diffusion.h}` (`CEED_QFUNCTION` style; `build.rs` compiles them with `cc` and exports their absolute paths for later GPU JIT)
- Test: `crates/pde/tests/operators.rs`

**Interfaces:**
- Consumes: `mesh::Mesh`, `petsc::{Dm, Vector, Matrix, ShellOperator}`, `ceed::*`.
- Produces:
  - `Discretization::new(mesh: &Mesh, level: usize, ceed: &Ceed, degree: u32) -> Result<Discretization>`. Builds a P`degree` `PetscFE` on the level's DM (section, boundary closure, interpolation) and libCEED restrictions from `DMPlexGetLocalOffsets`, as Ratel and HONEE do. Quadrature geometry (`w·det J`, `J^{-1}`) is stored per point. A non-positive `det J` gives `PdeError::InvertedElement { cell: i64, det: f64 }`.
  - `Discretization::quadrature_points(&self) -> &[[f64; 2]]` (local).
  - `Discretization::num_quadrature_points(&self) -> usize`.
  - `struct DiffusionReaction { d: f64, k: f64, shift: f64 }` implementing `ShellOperator`: `y = shift·M x + d·K x + k·M x`, with essential boundary rows as identity. Diagonal from `CeedOperatorLinearAssembleDiagonal`; `assemble(&self) -> Matrix` via COO for coarse levels.
  - `SourceField`: a `CVector` of quadrature-point values (mM/s), applied as `∫ f v`.

- [ ] **Step 1: Failing tests:**
  - `mass_integrates_area`: `1ᵀ M 1` equals 1.0 on `unit_square` (1e-12).
  - `diffusion_kills_constants`: `K 1 = 0` (1e-12).
  - `diagonal_matches_assembled`: shell diagonal equals the diagonal of the COO-assembled matrix (1e-13).
  - `inverted_element_is_error`: a fixture with one clockwise triangle gives `InvertedElement`.
- [ ] **Step 2: Run** `cargo nextest run -p pde --test operators`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the same. Expected: PASS.
- [ ] **Step 5: Commit** `feat(pde): libCEED diffusion-reaction operators on DMPlex`

### Task 9: `pde` time integration and manufactured-solution verification

**Files:**
- Create: `crates/pde/src/problem.rs`
- Test: `crates/pde/tests/verification.rs`

**Interfaces:**
- Consumes: Task 8, `contracts::{FieldSpec, TimeSpec}`.
- Produces:
  - `Problem::new(mesh: &Mesh, field: &FieldSpec, time: &TimeSpec, ceed: &Ceed, petsc_options: &str) -> Result<Problem>`. Resolves boundary tags through `Mesh::tag`, so unknown tags fail here.
  - `Problem::source_mut(&mut self) -> &mut [f64]`: quadrature-point source, initialized from `field.source`. Later milestones write ODE sources here.
  - `Problem::step_to(&mut self, t: f64) -> Result<()>`.
  - `Problem::time(&self) -> f64`, `Problem::solution(&self) -> &Vector`.
  - `Problem::l2_error(&self, exact: impl Fn([f64; 2], f64) -> f64) -> Result<f64>` (quadrature on the finest level).
  - `Problem::total_mass(&self) -> Result<f64>`.
- `TS`: `TSBDF` order 2. With `adaptive = false`, fixed `dt`; with `adaptive = true`, `TSAdapt` basic.
  - Implicit `IFunction` and `IJacobian` (shift times M plus K plus k·M) as the shell operator.
  - `-ksp_error_if_not_converged`, `-snes_error_if_not_converged`, `-ts_error_if_step_fails` are always set, before `petsc_options`.
  - On divergence, `PdeError::Diverged { reason: String, t: f64 }`.

- [ ] **Step 1: Failing tests** (unit square, `d = 0.01`, `k = 0.5`, zero Dirichlet on all four edges):

```rust
// spatial: steady source f = (2π²d + k) φ, φ = sin(πx) sin(πy); integrate to t = 200, dt = 1
// on refinements 2, 3, 4; observed order log2(e_l / e_{l+1}) in [1.9, 2.1]
#[test] fn spatial_order_p1() { ... }
// temporal: initial φ, no source, refinement 3, t_end = 1, dt = 0.1, 0.05, 0.025;
// error against a reference run with dt = 0.003125 on the same mesh (removes the spatial error);
// observed order in [1.8, 2.2]
#[test] fn temporal_order_bdf2() { ... }
// zero-flux everywhere, k = 0, no source, initial bump: |mass(t=10) - mass(0)| / mass(0) < 1e-10
#[test] fn mass_conserved_with_zero_flux() { ... }
// d = 0.01 with dt = 1e9 and -ksp_max_it 1: run fails with PdeError::Diverged
#[test] fn divergence_is_error() { ... }
```

- [ ] **Step 2: Run** `cargo nextest run -p pde --test verification`. Expected: FAIL.
- [ ] **Step 3: Implement.** The spatial test writes the manufactured source into `source_mut` from `quadrature_points`.
- [ ] **Step 4: Run** the same. Expected: PASS.
- [ ] **Step 5: Commit** `feat(pde): BDF2 time integration verified by manufactured solutions`

### Task 10: Geometric multigrid preconditioner

**Files:**
- Create: `crates/pde/src/multigrid.rs`
- Modify: `crates/pde/src/problem.rs` (default preconditioner)
- Test: `crates/pde/tests/multigrid.rs`

**Interfaces:**
- Consumes: `Mesh::levels`, Task 8 operators per level.
- Produces:
  - `fn configure_multigrid(ksp: &Ksp, mesh: &Mesh, ops: &[DiffusionReaction]) -> Result<()>`.
  - `PCMG` over all levels, interpolation from `DMCreateInterpolation` between consecutive `PetscFE` levels.
  - Smoother: Chebyshev with Jacobi (shell diagonal), 2 sweeps.
  - Coarse level: assembled COO matrix with `PCGAMG`.
  - Default for `Problem`; overridable through `petsc_options`.

- [ ] **Step 1: Failing test:** `iterations_bounded`. The steady problem of `spatial_order_p1` is solved on `tissue_small` at refinements 1 to 4, with CG plus multigrid to `rtol = 1e-10`. Assert: iterations at refinement 4 are at most 20, and at most 1.5x the iterations at refinement 1.
- [ ] **Step 2: Run** `cargo nextest run -p pde --test multigrid`. Expected: FAIL (Jacobi-only iterations grow).
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `cargo nextest run -p pde`. Expected: PASS, the Task 9 tests included.
- [ ] **Step 5: Commit** `feat(pde): geometric multigrid on the refinement hierarchy`

### Task 11: `slidesim run`: end to end on a slideviz-like mesh, parallel parity

**Files:**
- Create: `crates/slidesim/{Cargo.toml,src/main.rs,src/run.rs}`
- Create: `testdata/studies/tissue_oxygen/{scene.json,study.json}` (`tissue_small.msh`, field `oxygen`: d = 2000 um²/s, k = 0.1 1/s, Dirichlet 0.1 mM on `vessel_wall`, zero flux elsewhere, end 60 s, outputs 0, 10, 30, 60)
- Test: `crates/slidesim/tests/cli.rs`

**Interfaces:**
- Consumes: all earlier tasks.
- Produces: the CLI `slidesim run <study.json> --out <dir> [-- <petsc options>]`.
  - Writes `<dir>/<field>_<index>.vtu` per output time (PETSc VTK viewer), and `<dir>/summary.json`: `{"fields": {"oxygen": {"times": [...], "l2_norm": [...], "total_mass": [...]}}}`.
  - Exit code 0 on success. Otherwise 1, with the error on stderr.
- Note: `.vtu` is the interim output for ParaView. `result/1` (Zarr v3) comes in milestone 4.

- [ ] **Step 1: Failing tests:**
  - `runs_tissue_oxygen`: exit 0; four `.vtu` files; `total_mass` increasing towards steady state; the final value differs from the value at 30 s by less than 1 %.
  - `unknown_tag_fails_cleanly`: the study with tag `portal_wall` exits 1, and stderr contains `mesh has no tag 'portal_wall'; available:`.
  - `two_ranks_match_one`: spawns `mpiexec -n 2 slidesim run ...`; `l2_norm` and `total_mass` agree with the serial run within 1e-10 relative.
- [ ] **Step 2: Run** `cargo nextest run -p slidesim`. Expected: FAIL.
- [ ] **Step 3: Implement.** Uses `Petsc::init` with the passthrough options; one `Problem` per field.
- [ ] **Step 4: Run** `cargo nextest run --workspace && cargo clippy --all-targets -- -D warnings && cargo fmt --check`. Expected: all pass.
- [ ] **Step 5: E2E check by eye.** Open `out/oxygen_3.vtu` in ParaView. The field is highest at the vessel walls and decays into the tissue, with no artifacts at the refinement seams. Fix anything that looks off.
- [ ] **Step 6: Commit** `feat: slidesim run CLI with VTU output and parallel parity`
