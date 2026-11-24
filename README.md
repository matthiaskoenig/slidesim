# slidesim

slidesim simulates spatial PDE-ODE systems on real tissue geometry. The geometry and the local
parameters come from whole-slide scans and spatial transcriptomics, preprocessed by slideviz.
On this geometry, PDEs describe transport (diffusion, perfusion along the sinusoids, advection),
and every mesh element carries an SBML ODE network which supplies local sources and sinks and
which changes coefficients of the PDEs. The results are explored in a web viewer.

## Build

Requirements: a Rust toolchain (see `rust-toolchain.toml`), an MPI C compiler (`mpicc`), git,
make, python3 and pkg-config.

Build the pinned native dependencies (PETSc and libCEED, versions in `deps/versions.env`):

```bash
scripts/deps/build-native.sh debug   # or: opt
. deps/install/native-debug/env.sh
pkg-config --modversion PETSc ceed
cargo build --workspace
```

## License

MIT
