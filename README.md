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
- An MPI with `mpicc` and `mpiexec`, for example MPICH (`apt install mpich libmpich-dev`, as CI
  uses) or Open MPI, and a C compiler behind `mpicc`.
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
`deps/versions.env`) into `deps/install/native-debug` (or `native-opt`, with `-O3 -march=native`
and without PETSc's debugging), then slidesim against them:

```bash
scripts/deps/build-native.sh debug   # or: opt
. deps/install/native-debug/env.sh   # PETSC_DIR, CEED_DIR, PKG_CONFIG_PATH, LD_LIBRARY_PATH
cargo build --workspace              # with native-opt: cargo build --release
```

The `slidesim` binary has an absolute RUNPATH into the installation it was built against
(`deps/install/native-<mode>/lib`, for PETSc, PT-Scotch and libCEED): it runs without `env.sh`,
but not after `deps/install` is moved or removed.

## Test

```bash
. deps/install/native-debug/env.sh
cargo nextest run --workspace   # one process per test; some start mpiexec
cargo test --doc --workspace    # doctests, among them compile_fail tests of the bindings
```

`.config/nextest.toml` gives the tests which start `mpiexec` a test thread per process, and ends
a test which hangs after ten minutes.

## Run

```bash
target/debug/slidesim run testdata/studies/tissue_oxygen/study.json --out out
mpiexec -n 2 target/debug/slidesim run testdata/studies/tissue_oxygen/study.json --out out
```

`out` gets `<field id>_<index>.vtu` per output time (for ParaView) and `summary.json`; outputs of
an earlier run there are removed first. `slidesim run --help` describes them.

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
