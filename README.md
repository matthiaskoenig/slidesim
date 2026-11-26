# slidesim

slidesim simulates spatial PDE-ODE systems on real tissue geometry. The geometry and the local
parameters come from whole-slide scans and spatial transcriptomics, preprocessed by slideviz.
On this geometry, PDEs describe transport (diffusion, perfusion along the sinusoids, advection),
and every mesh element carries an SBML ODE network which supplies local sources and sinks and
which changes coefficients of the PDEs. The results are explored in a web viewer.

## Requirements

- Rust: the toolchain of `rust-toolchain.toml` (rustup installs it on the first `cargo` call in
  the repository), and [cargo-nextest](https://nexte.st) for the tests
  (`cargo install cargo-nextest --locked`).
- An MPI with `mpicc` and `mpiexec`, and a C compiler behind `mpicc`: Open MPI
  (`apt install openmpi-bin libopenmpi-dev`) or MPICH (`apt install mpich libmpich-dev`); CI tests
  both. Not the MPICH package of Ubuntu 24.04, whose `mpiexec` runs every process alone.
- For the native dependencies: git, curl, make, cmake, m4, python3 and pkg-config. bison and
  flex are built from pinned tarballs.
- [wasi-sdk](https://github.com/WebAssembly/wasi-sdk), the version `WASI_SDK_VERSION` of
  `deps/versions.env`: bindgen uses its clang and libclang for the bindings of every target,
  native ones included. It is looked for in `~/opt/wasi-sdk`; set `WASI_SDK_PATH` to its
  directory if it is elsewhere. For example (the sha256 `WASI_SDK_SHA256` of
  `deps/versions.env`):

  ```bash
  . deps/versions.env
  curl -fsSLO "https://github.com/WebAssembly/wasi-sdk/releases/download/wasi-sdk-${WASI_SDK_VERSION%%.*}/wasi-sdk-${WASI_SDK_VERSION}-x86_64-linux.tar.gz"
  echo "$WASI_SDK_SHA256  wasi-sdk-${WASI_SDK_VERSION}-x86_64-linux.tar.gz" | sha256sum --check
  mkdir -p ~/opt/wasi-sdk
  tar xzf "wasi-sdk-${WASI_SDK_VERSION}-x86_64-linux.tar.gz" --strip-components=1 -C ~/opt/wasi-sdk
  ```

## Build

Build the pinned native dependencies (PETSc with PT-Scotch and libCEED, versions and sha256 in
`deps/versions.env`) into `deps/install/native-check` (`-O2` with PETSc's debugging checks, for the tests and CI),
`native-debug` (`-O0`, for gdb sessions) or `native-opt` (`-O3 -march=native`, without PETSc's
debugging), then slidesim against them:

```bash
scripts/deps/build-native.sh check   # or: debug, opt
. deps/install/native-check/env.sh   # PETSC_DIR, CEED_DIR, PKG_CONFIG_PATH, LD_LIBRARY_PATH
cargo build --workspace              # with native-opt: cargo build --release
```

The `slidesim` binary has an absolute RUNPATH into the installation it was built against
(`deps/install/native-<mode>/lib`, for PETSc, PT-Scotch and libCEED): it runs without `env.sh`,
but not after `deps/install` is moved or removed.

## Test

```bash
. deps/install/native-check/env.sh
cargo nextest run --workspace   # one process per test; some start mpiexec
cargo test --doc --workspace    # doctests, among them compile_fail tests of the bindings
```

`.config/nextest.toml` runs the tests which start `mpiexec` alone, because MPI processes stall
when other tests take their CPUs, and ends a test which hangs after ten minutes.

## Run

```bash
target/debug/slidesim run testdata/studies/tissue_oxygen/study.json --out out
mpiexec -n 2 target/debug/slidesim run testdata/studies/tissue_oxygen/study.json --out out
```

`out` gets `<field id>_<index>.vtu` per output time (for ParaView) and `summary.json`; outputs of
an earlier run there are removed first. `slidesim run --help` describes them.

`summary.json` has, per field and output time, the mass balance over the interval since the
output time before (`mass_balance`, each term its mean rate there: the rate of change of the
amount, sources, advective and diffusive inflow per boundary tag, the `advection_defect` of the
velocity, and with cells their `exchange` and the `resupply` of the Dirichlet tags); the terms
add up to the change of the amount exactly, up to the residuals of the linear solves.

A study with a `perfusion` section solves the Darcy flow first (maps named by the study are read
from the scene) and also writes `pressure.vtu` and `velocity.vtu`; `summary.json` then has the flow
through each pressure tag (`perfusion.tag_flows`). A part of the mesh which no pressure tag reaches
(an island bounded by walls alone) has pressure 0 and no flow; `perfusion.pressure_islands` counts
these parts and gives their area. `testdata/studies/lobule_toy` is an example:

```bash
target/debug/slidesim run testdata/studies/lobule_toy/study.json --out out
```

A study with a `cells` section runs an SBML cell model on every coarse element of the mesh (the
mesh before its refinements), coupled to the fields by Strang splitting: per split step
`cells.split_step`, the cells over half of it, the fields over all of it, the cells over the
other half; the linked species of the model exchange with the means of their fields over the
element. The model is translated to C and compiled with `$CC` (or `cc`) at the start of the run,
cached in `$SBMLSIM_C_CACHE` (default `$XDG_CACHE_HOME/sbmlsim/c` or `~/.cache/sbmlsim/c`). The
run also writes `cells_<index>.vtu` (the states, the outputs and the parameters of every element
as cell data), and `summary.json` gets the steps of the cells and the amounts they exchanged and
hold (`cells`). `testdata/studies/cells_toy` has examples, a linear model (`cells.study.json`)
and a larger one with two linked fields (`spt.study.json`):

```bash
target/debug/slidesim run testdata/studies/cells_toy/spt.study.json --out out
```

PETSc options follow `--`. The solvers of a field take the prefix `<field id>_`, for example
`-oxygen_ts_monitor` (the time steps of the field `oxygen`), `-oxygen_ksp_monitor` (the
residuals of its linear solves) or `-oxygen_ksp_rtol 1e-8`:

```bash
target/debug/slidesim run testdata/studies/tissue_oxygen/study.json --out out -- -oxygen_ts_monitor
```

Precedence of the options, lowest first: slidesim's defaults (`-ksp_rtol 1e-10`, and solvers
which do not converge are errors), `solver.petsc_options` of the study, the options after `--`.

## License

slidesim is MIT licensed (`LICENSE`).

### Third-party licenses

The native dependencies are linked into slidesim (`deps/versions.env` pins their versions):

- PETSc: BSD-2-Clause.
- libCEED: BSD-2-Clause.
- PT-Scotch (Scotch): CeCILL-C.
- f2cblaslapack (PETSc's C translation of the reference BLAS and LAPACK): LAPACK's modified BSD
  license.

bison and flex are build tools of PT-Scotch only and are not linked. MPI, and the compression
libraries Scotch uses, come from the system.
