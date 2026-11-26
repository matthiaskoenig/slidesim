# Development

How slidesim is tested, measured and changed. The toolchain and the build of the native
dependencies are in [Installation](installation.md).

## Repository

| Path | Content |
| --- | --- |
| `crates/petsc-sys`, `crates/ceed-sys` | `bindgen` bindings of PETSc and libCEED |
| `crates/petsc`, `crates/ceed` | thin safe wrappers of the calls slidesim uses |
| `crates/contracts` | `scene/1` and `study/1` (serde), and their JSON Schemas in `schemas/` |
| `crates/mesh` | gmsh to DMPlex with tags, refinement, partitioning |
| `crates/maps` | Zarr v3 maps: windowed reads, bilinear samples and gradients |
| `crates/pde` | P1 discretization, libCEED operators (QFunctions in `crates/pde/qfunctions/`), multigrid, transport, Darcy flow, mass balance, exchange with cells |
| `crates/cells` | cell models per element: the engine path and the batched path |
| `crates/slidesim` | the command line, `slidesim run`, and the coupled runs |
| `crates/test-support` | helpers of the tests, among them runs under `mpiexec` |
| `testdata/` | meshes, maps, models, contracts and example studies of the tests |
| `scripts/deps/` | the build of the native dependencies |
| `scripts/e2e/`, `scripts/bench/` | the end-to-end runs on real slides, and the benchmarks |
| `spike/petsc-wasm/` | the milestone 0 spike: PETSc and libCEED from Rust in `wasm32-wasip1` |
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
rates, conservation of the mass balances, agreement across 1, 2 and 3 processes within 1e-10,
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

## CI

`.github/workflows/ci.yml` runs on every push to `main` and every pull request: the format, Clippy,
the tests and the doctests, on Ubuntu 24.04 with Open MPI and on Ubuntu 22.04 with MPICH, against
the `native-check` dependencies, which it caches. `.github/workflows/docs.yml` builds these pages
(below).

## Benchmarks and end-to-end runs

These need the slides, which are not in the repository, and run on the optimized build
(`native-opt`, `cargo build --release`).

- `scripts/e2e/m2_real_slide.sh`: the chain from a slide pair through slideviz to the four
  perfusion studies of milestone 2 and their analysis and figures
  ([results](results/2026-11-m2-real-slide.md#reproduce)).
- `scripts/bench/transport.sh`: the transport of the milestone 2 baseline on 1, 4 and 8
  processes with `-log_view`, seconds per step and the phases of `summary.json`
  ([performance](results/2026-11-m3-performance.md)).

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
