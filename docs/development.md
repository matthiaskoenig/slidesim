# Development

How slidesim is tested, measured and changed. The toolchain and the build of the native
dependencies are in [Installation](installation.md).

## Repository

| Path | Content |
| --- | --- |
| `crates/petsc-sys`, `crates/ceed-sys` | `bindgen` bindings of PETSc and libCEED |
| `crates/petsc`, `crates/ceed` | thin safe wrappers of the calls slidesim uses |
| `crates/contracts` | `scene/1` and `study/1` (serde), and their JSON Schemas in `schemas/` |
| `crates/mesh` | gmsh files and generated meshes to DMPlex with tags, refinement, partitioning, partition-independent element ids |
| `crates/mesher` | tagged polygons and ideal lobules to a graded, adapted triangle mesh (spade, Mmg through PETSc) |
| `crates/maps` | Zarr v3 maps: windowed reads, bilinear samples and gradients |
| `crates/pde` | P1 discretization, libCEED operators (QFunctions in `crates/pde/qfunctions/`), multigrid, transport, Darcy flow, mass balance, exchange with cells |
| `crates/cells` | cell models per element: the engine path and the batched path, natively (feature `native`); the batched kernel of sbmlsim-core in the browser build |
| `crates/simulation` | a study as a simulation in memory: setup, the coupled runs of fields and cells, stepped from output time to output time, the summary and the VTU outputs |
| `crates/slidesim` | the command line: `slidesim run` over `crates/simulation`, `slidesim mesh` |
| `crates/slidesim-wasm` | the browser build: slidesim in `wasm32-wasip1`, a reactor with a C ABI over its memory (not a member of the native workspace) |
| `wasm/host` | a wasmtime host of the browser build and its agreement tests with the native build (a workspace of its own) |
| `wasm/js` | the live API of the browser build: the JavaScript glue `slidesim.mjs`, its TypeScript declarations and its Node and browser tests |
| `docs/demo/lobule` | the page of the [lobule demo](demo.md) |
| `crates/test-support` | helpers of the tests, among them runs under `mpiexec` |
| `testdata/` | meshes, maps, models, contracts and example studies of the tests |
| `scripts/deps/` | the builds of the dependencies, native (`build-native.sh`) and wasm32-wasip1 (`build-wasi.sh`, with its smoke test `wasi-smoke/`), and their patches (`patches/`, each explained in its `README.md`) |
| `scripts/wasm/` | the build of the browser build and of what its tests need (`build.sh`), and the bundle of the lobule demo (`bundle.sh`) |
| `scripts/e2e/`, `scripts/bench/` | the end-to-end runs on real slides, and the benchmarks |
| `spike/petsc-wasm/`, `spike/mmg/` | the spikes of milestone 0 (PETSc and libCEED from Rust in `wasm32-wasip1`) and milestone 4 (Mmg through `DMAdaptMetric`, natively and in `wasm32-wasip1`) |
| `docs/` | these pages; `docs/superpowers/` holds the design documents and plans of the milestones, which are not part of the site |

## Tests

```bash
. deps/install/native-check/env.sh
cargo nextest run --workspace   # one process per test; some start mpiexec
cargo test --doc --workspace    # doctests, among them compile_fail tests
```

`.config/nextest.toml` runs the tests which start `mpiexec` alone, because MPI processes stall
when other tests take their CPUs, and ends a test which hangs after ten minutes. The tests run
against the `native-check` build (`-O2` with PETSc's debugging checks).

The tests are the acceptance of each milestone: manufactured solutions with their convergence
rates, conservation of the mass balances, agreement across 1, 2 and 3 processes within 1e-10
at tight test settings (fixed transport steps; on the generated lobule also the linear solves and
the cells to 1e-12, see [Running a study](usage.md)),
the cell paths against each other and against sbmlsim-core, and end-to-end runs of the CLI on the
[example studies](examples.md). The compiled cell models of the tests go to a cache directory of
the test binary in the temporary directory, not to the cache of the user.

The JSON Schemas in `schemas/` are generated from the contracts; a test fails when they are
stale:

```bash
cargo run -p contracts --example write-schemas
```

## Lint

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
```

## WebAssembly dependencies

The browser build links PETSc (with Mmg) and libCEED compiled for `wasm32-wasip1` with wasi-sdk,
as static libraries in `deps/install/wasi`:

```bash
scripts/deps/build-wasi.sh       # into deps/install/wasi
scripts/deps/wasi-smoke/run.sh   # in wasmtime: Mmg adapts a square, and prints nothing itself
```

Both builds, native and wasm, copy the sources and apply the patches of `scripts/deps/patches`;
its `README.md` gives the reason and the upstream status of each patch, and the configure lines
of the wasm build.

## The browser build

`crates/slidesim-wasm` is slidesim for `wasm32-wasip1` (its crate documentation lists the
exports, [Contracts](contracts.md#live-api) the arrays). The cells run on the batched kernel of
sbmlsim-core (`kernel/1`, see [Cell models](methods/cells.md#kernel-path)), which the host
compiles from the SBML of the study and links to slidesim; the engine path needs native code and
is refused. The build reads no files: the inputs are JSON, a scene whose mesh or maps are files
needs its mesh in the inputs. The crates choose by the feature `native` of `cells` and
`simulation` (on by default; off in the browser build) and by `target_os = "wasi"` for Zarr's
files (`maps`, `result.zarr`). MPI is PETSc's MPIUNI there, one process: every MPI call of the
Rust code goes through a function of `crates/petsc-sys/shim.c`, since MPIUNI defines them as
macros.

```bash
scripts/wasm/build.sh                     # target/wasm/slidesim.wasm and what the tests need
CARGO_TARGET_DIR=target/wasm/host cargo test --manifest-path wasm/host/Cargo.toml
```

`build.sh` builds sbmlsim-core's `compiler.wasm` and `mathlib.wasm` at
`SBMLSIM_CORE_KERNEL_REV` of `deps/versions.env` from a clean checkout of that revision
(`deps/src/sbmlsim-core`) into `deps/install/sbmlsim-js`, compiles the kernel of the SPT model
with that compiler in Node (`scripts/wasm/compile_kernel.mjs`, as a page compiles it), builds the
native `slidesim` (`deps/install/native-check`, or `NATIVE_DEPS`) and `slidesim.wasm`, linked by
wasi-sdk's `wasm-ld` against its wasi-libc, and writes their paths to `target/wasm/artifacts.env`.

The tests of `wasm/host` run `slidesim.wasm` in wasmtime as a page runs it: the kernel
instantiated with slidesim's memory and mathlib, slidesim's imports `slidesim_kernel.init`,
`slidesim_kernel.advance` and `slidesim_kernel.assign` bound to it. `lobule_matches_native` runs the SPT study of an hour on
the lobule (`testdata/studies/lobules/spt_1h.study.json`) natively and in wasm on the native
mesh: the mesh arrays agree bitwise, the fields within 1e-5 of their maximum and the cell values
within `1e-4 |y| + 1e-8` (measured: about 1e-14 relative; the math functions of mathlib and of the
C library differ in the last bits). `rule_outputs_match_native` does the same for the targets of
assignment rules which read a state (`necrosis_trigger` of the SPT model, its threshold lowered so
that it moves within the hour), evaluated by the kernel's `sbmlsim_kernel_assign`. The others
check the lobule meshed in wasm against the
mesher's bounds and twice the same, the refusals of the engine path and of a file scene, and a
cap of the memory (an error naming it, the host lives on). `mesher_wall_times` (ignored) prints
the time of the mesher in wasm.

The JavaScript glue `wasm/js/slidesim.mjs` implements the live API
([Contracts](contracts.md#live-api)) over the exports of `slidesim.wasm`: WASI preview 1 over
nothing (no files, stdout and stderr kept for the messages of a trap), the kernel compiled from
the SBML with sbmlsim-core's `compileKernel` and bound late to slidesim's imports, one instance
per simulation. Its tests:

```bash
scripts/wasm/build.sh                     # also wasm/js/sbmlsim-core/ and the host's dump
npm ci --prefix wasm/js
npx --prefix wasm/js playwright install chromium
npm test --prefix wasm/js
npx --yes -p typescript@5.6.3 tsc --noEmit -p wasm/js/tsconfig.json
```

`test/node.test.mjs` runs the SPT study of 30 min on the generated lobule in Node (every mass
balance closes within 1e-8 of its largest term), the same study on the native mesher's `mesh.json`
against the wasmtime host (`target/wasm/host/debug/dump`, `wasm/host/src/bin/dump.rs`): every
array of every output time bitwise equal; the refusals of the engine path and of a file scene,
a memory cap (an `Error` naming it, then the next simulation runs), a trap (the simulation
unusable after it, another untouched) and an error which is not a trap. `test/browser.test.mjs`
serves the demo's bundle (`scripts/wasm/bundle.sh`) and runs 10 min in headless Chromium
(Playwright): the drawing is not uniform, no console errors, and the page's L2 norm of S equals
that of the same run in Node bitwise.

Lint of the browser build's paths (the crates without `native`, for the wasm target):

```bash
. deps/install/wasi/env.sh
CARGO_TARGET_DIR=target/wasm cargo clippy --target wasm32-wasip1 -p simulation -p cells -p maps \
  --no-default-features --lib -- -D warnings
CARGO_TARGET_DIR=target/wasm cargo clippy --target wasm32-wasip1 \
  --manifest-path crates/slidesim-wasm/Cargo.toml --release -- -D warnings
```

## CI

`.github/workflows/ci.yml` runs on every push to `main` and every pull request: the format, Clippy,
the tests and the doctests, on Ubuntu 24.04 with Open MPI and on Ubuntu 22.04 with MPICH, against
the `native-check` dependencies, which it caches (the key covers `deps/versions.env` but for the
revision of sbmlsim-core, the build scripts and the patches). The job `wasm` then runs the
browser build on Ubuntu 24.04: the wasi dependencies and sbmlsim-core's artifacts cached,
`scripts/wasm/build.sh`, the format and Clippy of the browser build's crates, the tests of
`wasm/host`, the type check of `slidesim.d.ts` and the Node and browser tests of `wasm/js`.
`.github/workflows/docs.yml` builds these pages (below), and in a job of its own the browser build
(`build.sh --browser`, the wasi dependencies from the cache of the job `wasm`), whose bundle goes
to `demo/lobule/` of the site.

## Benchmarks and end-to-end runs

These need the slides, which are not in the repository, and run on the optimized build
(`native-opt`, `cargo build --release`).

- `scripts/e2e/m2_real_slide.sh`: the chain from a slide pair through slideviz to the four
  perfusion studies of milestone 2 and their analysis and figures
  ([results](results/2026-11-m2-real-slide.md#reproduce)).
- `scripts/bench/transport.sh`: the transport of the milestone 2 baseline on 1, 4 and 8
  processes with `-log_view`, seconds per step and the phases of `summary.json`
  ([performance](results/2026-11-m3-performance.md)).
- `scripts/bench/coupled.sh`: the batched SPT studies of cells and fields, shortened, with
  `-log_view`, the Krylov iterations per solve, the phases and the load of the machine; its
  default case, one coarse lobule for 1 h, runs in seconds and needs no slide
  ([performance](results/2026-11-m4-performance.md)). It starts every simulation through
  `run_sim` of `scripts/lib/proc.sh`: a wall-clock limit, CPUs 0-9 at nice 10, and no process
  left behind when the script ends, however it ends.
- `scripts/bench/split_step.sh`: the batched SPT study of one lobule at several split steps of
  the Strang splitting against a reference, on the lobule's mesh (1 h), a coarser mesh (4 h,
  with necrosis) and the 7 lobules (15 min), compared by `scripts/bench/split_step.py` (uptake,
  cell states, necrotic share and Dice, observed orders, cost); about 12 min, no slide needed
  ([split step](results/2026-11-m4-split-step.md)).

All of these scripts source `scripts/lib/proc.sh` and start every `slidesim` and `mpiexec` run
through its `run_sim <limit> <cmd...>`: a wall-clock limit (`LIMIT`), the CPUs `PROC_CPUS` (default
`0-9`, so at most 10 CPUs) at nice 10, and at most 4 MPI processes by default. The script runs in
a session of its own and takes down every process it started when it ends, however it ends: normally,
by an error, by TERM, INT or HUP, or by KILL of the script or of the shell which started it.
`scripts/tests/proc_kill.sh` checks this with a dummy script (no slidesim needed, a few seconds):
it stops the script in each of these ways, checks that no process of it remains within 15 s, and
checks the exit status of a normal run, a failing command and a time-limit hit. CI runs it.

## Contributing

Work happens on branches and reaches `main` through pull requests, which CI must pass. The commit
messages follow the conventional style of the history (`feat(pde): ...`, `fix(mesh): ...`,
`perf: ...`, `docs: ...`, `test: ...`). Every commit keeps `cargo fmt --check`, Clippy and the
tests clean.

- No model-specific or tissue-specific code in the crates: tags, maps, fields and models are data
  of the contracts. Liver names belong in the scenes, studies, test data and these pages.
- C code (the QFunctions, the dependencies, the generated cell models) is compiled with
  `-ffp-contract=off -fno-fast-math`, so that the numbers are the same on every target.
- A milestone starts with a design in `docs/superpowers/specs/` and a plan in
  `docs/superpowers/plans/`, and ends with its results in `docs/results/`.

## Documentation

These pages are built with [Zensical](https://zensical.org/) from `docs/` and `zensical.toml`
into the gitignored `site/`:

```bash
uvx --from zensical==0.1.3 zensical build --clean --strict
uvx --from zensical==0.1.3 zensical serve      # a preview on http://localhost:8000
```

The workflow `docs` builds them on every pull request and every push to `main`, and publishes
them from `main` to [GitHub Pages](https://matthiaskoenig.github.io/slidesim/).

A new results document goes into `docs/results/` with its figures in a folder of the same name,
gets an entry in the "Results" section of the `nav` in `zensical.toml` and a row in
`docs/results/index.md`.
