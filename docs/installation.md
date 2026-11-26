# Installation

slidesim is built from source: a Rust workspace against PETSc, PT-Scotch and libCEED, which
`scripts/deps/build-native.sh` builds at the versions pinned in `deps/versions.env`.

## Requirements

- **Rust**: the toolchain of `rust-toolchain.toml` (1.100.0; rustup installs it on the first
  `cargo` call in the repository), and [cargo-nextest](https://nexte.st) for the tests
  (`cargo install cargo-nextest --locked`).
- **MPI** with `mpicc` and `mpiexec`, and a C compiler behind `mpicc`: Open MPI
  (`apt install openmpi-bin libopenmpi-dev`) or MPICH (`apt install mpich libmpich-dev`); CI tests
  both. Not the MPICH package of Ubuntu 24.04, whose `mpiexec` runs every process alone.
- **For the native dependencies**: git, curl, make, cmake, m4, python3 and pkg-config. bison and
  flex are built from pinned tarballs.
- **[wasi-sdk](https://github.com/WebAssembly/wasi-sdk)**, the version `WASI_SDK_VERSION` of
  `deps/versions.env`: bindgen uses its clang and libclang for the bindings of every target,
  native ones included. It is looked for in `~/opt/wasi-sdk`; set `WASI_SDK_PATH` to its
  directory if it is elsewhere. For example (the sha256 is `WASI_SDK_SHA256` of
  `deps/versions.env`):

    ```bash
    . deps/versions.env
    curl -fsSLO "https://github.com/WebAssembly/wasi-sdk/releases/download/wasi-sdk-${WASI_SDK_VERSION%%.*}/wasi-sdk-${WASI_SDK_VERSION}-x86_64-linux.tar.gz"
    echo "$WASI_SDK_SHA256  wasi-sdk-${WASI_SDK_VERSION}-x86_64-linux.tar.gz" \
      | sha256sum --check
    mkdir -p ~/opt/wasi-sdk
    tar xzf "wasi-sdk-${WASI_SDK_VERSION}-x86_64-linux.tar.gz" \
      --strip-components=1 -C ~/opt/wasi-sdk
    ```

- **A C compiler at run time** for studies with cell models: the model is translated to C and
  compiled with `$CC` (or `cc`) at the start of the run (see [Running a study](usage.md#cell-models)).

## Native dependencies

| Dependency | Version | Role |
| --- | --- | --- |
| PETSc | 3.26.1 | meshes (DMPlex), time stepping (`TS`), Krylov solvers and multigrid (`KSP`, `PC`) |
| libCEED | 1.1.2 | matrix-free finite element operators and the batched cell QFunctions, CPU backends |
| PT-Scotch | 7.0.15 | partitioning of the mesh over the processes |
| f2cblaslapack | 3.8.0.q2 | BLAS and LAPACK for PETSc, in C |

`scripts/deps/build-native.sh <mode>` builds them into `deps/install/native-<mode>`, in one of
three modes:

| Mode | Flags | For |
| --- | --- | --- |
| `check` | `-O2 -g` with PETSc's debugging checks, no `-march=native` | the tests and CI (portable, so CI can cache it) |
| `debug` | `-O0 -g` with PETSc's debugging | gdb sessions |
| `opt` | `-O3 -march=native` without PETSc's debugging | production runs and benchmarks |

Every mode compiles with `-ffp-contract=off -fno-fast-math`: no FMA contraction, no fast math,
so that the numbers are the same on every target.

## Build

Build the dependencies, then slidesim against them:

```bash
scripts/deps/build-native.sh check   # or: debug, opt
. deps/install/native-check/env.sh   # PETSC_DIR, CEED_DIR, PKG_CONFIG_PATH, ...
cargo build --workspace              # with native-opt: cargo build --release
```

The `slidesim` binary has an absolute RUNPATH into the installation it was built against
(`deps/install/native-<mode>/lib`, for PETSc, PT-Scotch and libCEED): it runs without `env.sh`,
but not after `deps/install` is moved or removed.

For long runs and timings, build the optimized dependencies and a release binary:

```bash
scripts/deps/build-native.sh opt
. deps/install/native-opt/env.sh
cargo build --release
```

The [performance results](results/2026-11-m3-performance.md) are measured this way.

Next: [run a study](usage.md).
