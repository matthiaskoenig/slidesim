# Spike result: PETSc + libCEED from Rust, native and wasm

**Decision: GO.** slidesim keeps one stack (PETSc + libCEED, driven from Rust) on every target.
The fallback of the spec (backend trait with a pure-Rust solver) is not planned for milestone 5.

| Criterion | Required | Measured |
| --- | --- | --- |
| Relative difference of the solution 2-norms, native vs wasm | < 1e-12 | 3.3e-16 (platform `sin()`), 0 with one `sin()` everywhere |
| L2 error consistent with P1 at h = 1/64 | O(h^2) | 3.3799233483411807e-4, ratio 4.00 per halving of h |
| Runs in wasmtime and in Node (`node:wasi`) | yes | yes, bitwise identical to each other |

Target: **wasm32-wasip1** (Rust 1.100.0, wasi-sdk 34.0). Emscripten was not needed.

## What runs

`diffusion/` is a standalone Cargo package (own `[workspace]`, not a member of the root
workspace) with raw `bindgen` bindings of `petscdmplex.h`, `petscdmplextransform.h`,
`petscfe.h`, `petscksp.h` and `ceed.h`. It solves -Δu = 2π² sin(πx) sin(πy) on the unit square,
u = 0 on the boundary, exact solution sin(πx) sin(πy):

- mesh: `DMPlexCreateBoxMesh` 64 x 64, two triangles per square (8192 triangles, h = 1/64);
- P1 `PetscFE` (quadrature degree 3, 12 points), zero Dirichlet via `DMAddBoundary` (constrained
  dofs of the section; 3969 unknowns, local vector 4225);
- libCEED on `/cpu/self/opt/blocked` (requested explicitly, resolved resource printed and
  checked): element restrictions from `DMPlexGetLocalOffsets` for the solution and the
  coordinate DM, H1 bases from the `PetscFE` tabulation (as Ratel and HONEE do), gallery
  `Poisson2DBuild` / `Poisson2DApply`, and the right-hand side as a libCEED QFunction written in
  Rust (`extern "C"` function pointer, also in wasm);
- a `MATSHELL` whose `MatMult` is `DMGlobalToLocal` (boundary entries zero), the libCEED operator
  on the local vectors (`CEED_USE_POINTER` on the PETSc arrays), `DMLocalToGlobal` with
  `ADD_VALUES`; `MatGetDiagonal` from `CeedOperatorLinearAssembleDiagonal`;
- CG + Jacobi, rtol 1e-12; L2 error with `DMComputeL2Diff`; the solution's 2-norm and an
  FNV-1a hash over the bits of all solution entries (bitwise comparison).

Deviation from the plan: `DMPlexCreateBoxMesh` with `simplex = true` calls a mesh generator
(Triangle) in 2D, which neither the native nor the wasm build has. The program creates the box
of quadrilaterals and splits each square along one diagonal with the `refine_tosimplex`
`DMPlexTransform`, the same structured triangulation without a generator.

Reproduce: `spike/petsc-wasm/diffusion/run.sh [-n 64] [-rust_libm]` builds against
`deps/install/{native-opt,native-debug,wasi}` and runs native-opt, native-debug, wasmtime and
Node (`spike/petsc-wasm/run-node.mjs`).

## Results (n = 64)

| Host | iterations | relative residual | L2 error | solution 2-norm | hash |
| --- | --- | --- | --- | --- | --- |
| native-opt | 114 | 8.6310400610120091e-13 | 3.3799233483411807e-4 | 3.1993575272559472e1 | 269f0a5eb146767d |
| native-debug | 114 | 8.6310400610120091e-13 | 3.3799233483411807e-4 | 3.1993575272559472e1 | 269f0a5eb146767d |
| wasmtime | 114 | 8.5774848795535524e-13 | 3.3799233483425772e-4 | 3.1993575272559461e1 | 52967720494f804b |
| Node | 114 | 8.5774848795535524e-13 | 3.3799233483425772e-4 | 3.1993575272559461e1 | 52967720494f804b |

Relative difference of the 2-norms native vs wasm: **3.331e-16** (1.5 ulp). native-opt and
native-debug are bitwise identical, as are wasmtime and Node.

The only source of the native/wasm difference is `sin()`: glibc natively, wasi-libc (musl) in
wasm. With `-rust_libm` the program evaluates sin() with the `libm` crate (a port of musl's) on
every target, and then all four runs are **bitwise identical** (hash `52967720494f804b`, 2-norm
3.1993575272559461e1, L2 error 3.3799233483425772e-4 everywhere). PETSc, libCEED,
f2cblaslapack and the Rust code otherwise round identically on x86-64 (AVX2, `-march=native`)
and in wasm (SIMD128): the `-ffp-contract=off -fno-fast-math` flags of both builds and the
explicit libCEED backend do their job. Consequence for slidesim: transcendental functions in
user code (QFunctions, initial and boundary values) should come from one Rust implementation
(`libm`) when bitwise reproducibility across targets is wanted.

### L2 error (P1 convergence)

| n | h | L2 error | ratio |
| --- | --- | --- | --- |
| 16 | 1/16 | 5.3774349987809949e-3 | |
| 32 | 1/32 | 1.3504362483758936e-3 | 3.982 |
| 64 | 1/64 | 3.3799233483411807e-4 | 3.995 |
| 128 | 1/128 | 8.4522098073640953e-5 | 3.999 |

Second order as expected for P1 in L2; at h = 1/64 the error is 3.38e-4 = 1.38 h^2 (native-opt;
wasm differs in the 13th digit).

## Size of the `.wasm`

| Module | raw | gzip -9 |
| --- | --- | --- |
| `diffusion.wasm` (release) | 16,290,070 B | 4,131,349 B |
| after `llvm-strip` (wasi-sdk) | 15,922,046 B | 4,023,074 B |

`wasm-strip` (wabt) and `wasm-opt` (binaryen) are not installed; `llvm-strip` of wasi-sdk does
what `wasm-strip` does (drops the custom name sections). The code section is 13.9 MB, data
2.0 MB. The C smoke test of Task 2 is the same size (16.6 MB), so the size is PETSc's: its
`*RegisterAll` functions reference every implementation of every class (KSP, PC, Mat, DM, TS,
SNES, Tao types), which keeps them alive through `--gc-sections`. Size work for later (not a
blocker): `wasm-opt -Oz`, a PETSc build without the unused packages, Brotli on the wire.

## Run times

i7-12700, 1 process. Setup (mesh, FE, libCEED objects, qdata, right-hand side, diagonal) and
solve (`KSPSolve`) are the best of three runs (native-debug: one run); totals are from one run.

| Host | setup | solve | in-program total | process wall |
| --- | --- | --- | --- | --- |
| native-opt | 29 ms | 82 ms | 450 ms | 461 ms |
| native-debug | 274 ms | 538 ms | 1313 ms | 1317 ms |
| wasmtime 50 (precompiled `.cwasm`) | 36 ms | 128 ms | 228 ms | 236 ms |
| Node 24 | 69 ms | 93 ms | 229 ms | 286 ms |

- The native total is dominated by `MPI_Init` of OpenMPI (about 300 ms); the wasm build uses
  MPIUNI.
- wasm solve: 1.56x native in wasmtime, 1.13x in Node (V8's optimizing tier wins here).
- wasmtime compile of the module: 0.97 s wall (15 s CPU, parallel Cranelift); `wasmtime run`
  without precompilation takes 1.41 s in total. Node's `WebAssembly.compile` takes 17 to 22 ms
  (lazy Liftoff, then tier-up), which is what a browser does too.

## Link setup per target

- native: `pkg-config --libs PETSc ceed` of `deps/install/native-{opt,debug}`, only `-L`/`-l`
  used (ceed.pc's `-fopenmp-simd` is a compile flag), shared `libpetsc.so`/`libceed.so` with an
  rpath. bindgen gets the MPI include paths from `mpicc -show`.
- wasm32-wasip1: `pkg-config --static --libs PETSc ceed` of `deps/install/wasi` (static
  archives, f2cblaslapack, `-ldl`, `-lwasi-emulated-{signal,process-clocks,getpid}`, `-lm`).
  Linker `$WASI_SDK_PATH/bin/clang` with `-C link-self-contained=no --target=wasm32-wasip1
  --sysroot=$WASI_SDK_PATH/share/wasi-sysroot`, so Rust's std links against the same wasi-libc
  (and emulation libraries) the C libraries were compiled against, not Rust's bundled one.
- bindgen (0.73) on both targets uses wasi-sdk's libclang 23 with its resource directory (the
  system libclang 18 has no builtin headers). Bindings are generated per target (pointer and
  `long` size differ).

## Pitfalls found (relevant for the real `petsc-sys` / `ceed-sys` crates)

- clang defaults to hidden visibility for wasm, and bindgen skips hidden declarations; PETSc's
  wasm headers carry no visibility attribute (libCEED's do), so without `-fvisibility=default`
  bindgen silently emits almost no PETSc functions for wasm.
- wasi-sdk's libclang defaults to target `wasm32-wasip1` and the wasi sysroot; bindgen passes
  `--target` only when cross compiling, so the native bindings need explicit
  `--target=<host> --sysroot=/`.
- `PetscBool` is C `bool` in PETSc 3.26 (bindgen: Rust `bool`); `PetscErrorCode` is `int`.
- PETSc callbacks are type-erased (`PetscVoidFn`, `PetscErrorCodeFn`); PETSc casts them back to
  the real signature before calling, so wasm's typed `call_indirect` is satisfied.
- OpenMPI's hwloc probes X displays at `MPI_Init` and prints "Authorization required" eight
  times on a desktop session without X authorization; `HWLOC_COMPONENTS=-gl` silences it
  (`run.sh` sets it). Host environment, not a PETSc issue.

## Patches

No new patches for this task. The wasm libraries are the ones of Task 2 (`PATCHES.md`):

- `libceed-0001-no-wordexp-on-wasi.patch`
- `libceed-0002-optional-cpu-gen-backend.patch`
- `petsc-0001-no-mkdtemp-on-wasi.patch`
- `petsc-0002-wasm-memory-usage.patch`
- `petsc-0003-dmnetwork-view-no-tmpnam-system-on-wasi.patch`
- `petsc-0004-segv-checkpointer-without-setjmp.patch`
- `petsc-0005-no-pic-flag-for-static-libraries.patch`

plus the wasi-libc emulation libraries `wasi-emulated-signal`, `wasi-emulated-process-clocks`
and `wasi-emulated-getpid`.
