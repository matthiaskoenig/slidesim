# PETSc and libCEED for wasm32-wasip1: patches and workarounds

Result: **wasip1 works.** PETSc v3.26.1 (sequential, MPIUNI, f2cblaslapack) and libCEED v1.1.2
(`/cpu/self` ref, blocked and opt backends) build as static libraries with wasi-sdk 34.0 and run
under wasmtime 50 and Node 24 (`smoke/run.sh`). The emscripten fallback was not needed.

`build-wasi.sh` copies `deps/src/{libceed,petsc}` into `deps/build/{libceed,petsc}-wasi` and applies
`patches/<name>-*.patch` there with `git apply`; `deps/src` is never modified. Five small source
patches and one configure patch were needed; everything else is covered by wasi-libc emulation
libraries or configure options.

## Compiler and flags

- `$WASI_SDK_PATH/bin/clang --target=wasm32-wasip1 --sysroot=$WASI_SDK_PATH/share/wasi-sysroot`
- `-O3 -DNDEBUG -ffp-contract=off -fno-fast-math -msimd128` for all C (libCEED, PETSc, f2cblaslapack).
- Archiver `llvm-ar`, `llvm-ranlib` from wasi-sdk.

## wasi-libc emulation libraries (instead of patches)

PETSc is compiled with `-D_WASI_EMULATED_SIGNAL -D_WASI_EMULATED_PROCESS_CLOCKS
-D_WASI_EMULATED_GETPID` and linked with the matching `-lwasi-emulated-*` (each one verified
necessary: without it a `--whole-archive` link of `libpetsc.a` has undefined symbols). libCEED
needs none.

| Library | Needed for |
| --- | --- |
| `wasi-emulated-signal` | `signal()`, `raise()` and `SIG*` in PETSc's error and signal handlers (`src/sys/error/signal.c`, `fp.c`). No signal is ever delivered in wasm; a trap stays a trap. |
| `wasi-emulated-process-clocks` | `times()`, `clock()`, `getrusage()` (`PetscGetCPUTime`, memory usage). |
| `wasi-emulated-getpid` | `getpid()` (PETSc error messages, `/proc/<pid>` paths that are never opened). |

`wasi-emulated-mman` is not needed: `<sys/mman.h>` is only included by PETSc's shared-memory
code, which configure leaves out.

`-ldl` (in PETSc's link line) resolves to wasi-libc's `libdl.a`, weak `dlopen`/`dlsym`/`dlclose`/
`dlerror` stubs that fail at run time; PETSc only calls them for dynamic loading, which is off for
static libraries.

## Configure (PETSc)

The final command is in `build-wasi.sh`:

```
./configure PETSC_ARCH=arch-wasi --prefix=deps/install/wasi \
  --with-cc=$WASI_SDK_PATH/bin/clang --with-cxx=0 --with-fc=0 --with-mpi=0 \
  --with-ar=$WASI_SDK_PATH/bin/llvm-ar --with-ranlib=$WASI_SDK_PATH/bin/llvm-ranlib \
  CFLAGS+="--target=wasm32-wasip1 --sysroot=$WASI_SDK_PATH/share/wasi-sysroot" \
  CPPFLAGS="-D_WASI_EMULATED_SIGNAL -D_WASI_EMULATED_PROCESS_CLOCKS -D_WASI_EMULATED_GETPID" \
  COPTFLAGS="-O3 -DNDEBUG -ffp-contract=off -fno-fast-math -msimd128" \
  LIBS="-lwasi-emulated-signal -lwasi-emulated-process-clocks -lwasi-emulated-getpid" \
  --download-f2cblaslapack=deps/src/f2cblaslapack-3.8.0.q2.tar.gz \
  --with-shared-libraries=0 --with-x=0 --with-ssl=0 --with-pthread=0 \
  --with-debugging=0 --with-avx512-kernels=0 --with-batch \
  --known-sdot-returns-double=0 --known-snrm2-returns-double=0
```

- `--with-batch`: the conftest executables are wasm modules which configure cannot run. PETSc
  3.26 determines `sizeof` (long 4, pointer 4, size_t 4, ...) by compile-time assertions, so no
  `--known-sizeof-*` exists or is needed. In batch mode it only cannot probe the return type of
  the single precision BLAS `sdot`/`snrm2`; `--known-sdot-returns-double=0
  --known-snrm2-returns-double=0` are the values the native configure determines for
  f2cblaslapack (no `PETSC_BLASLAPACK_*_RETURNS_DOUBLE` in its `petscconf.h`).
- `CFLAGS+=` appends target and sysroot to PETSc's default flags (`-Wall ... -fstack-protector
  -fvisibility=hidden`), the same defaults the native build gets.
- `--with-avx512-kernels=0`: the default is on and was set from the build host's CPU; the
  kernels are guarded by `__AVX512F__` and never compiled for wasm, so this only keeps
  `petscconf.h` honest.
- `--with-cc` takes the bare compiler; target and sysroot go in `CFLAGS` so configure does not
  see a compiler with arguments.
- f2cblaslapack: `build-wasi.sh` fetches the tarball the pinned PETSc downloads
  (`f2cblaslapack-3.8.0.q2.tar.gz`, URL as in PETSc's `config/BuildSystem/config/packages/f2cblaslapack.py`)
  once into `deps/src`, checks its sha256 (pinned in the script) and gives configure that file.
  The script fails if PETSc's package file no longer names that URL (a PETSc bump then has to
  update URL and hash). The build depends on nothing outside `deps/src`.

## Patches

### libCEED

| Patch | Reason |
| --- | --- |
| `libceed-0001-no-wordexp-on-wasi.patch` | `interface/ceed.c` includes `<wordexp.h>` and calls `wordexp()` to shell-expand the JiT cache directory (`CeedGetCacheDir`). wasi-libc has no `wordexp`. Under `__wasi__` the base directory is used verbatim. The cache directory is only used by the JiT backends, which are not built for wasm. |
| `libceed-0002-optional-cpu-gen-backend.patch` | The `/cpu/self/gen/*` backends are always built: they are C++ and compile generated code at run time with a C++ compiler and `dlopen()` it, neither of which exists in wasip1. The patch adds a Makefile switch `CPU_GEN=0` which leaves them out and adds a weak registration stub `backends/weak/ceed-cpu-gen-weak.c` (same pattern as the AVX/CUDA/... stubs), so `/cpu/self/gen` reports "Backend not currently compiled" and `/cpu/self` resolves to `/cpu/self/opt/blocked`. |

libCEED is then built with `make install STATIC=1 CPU_GEN=0 MARCHFLAG= OPT="<flags>"`
(`MARCHFLAG=` drops the host `-march=native` probe, `OPT` replaces the default
`-ffp-contract=fast`).

### libCEED backend: always the full resource string

The `/cpu/self` prefix resolves to the registered backend of highest priority, which differs per
build (checked with `CeedInit("/cpu/self")` + `CeedGetResource`):

| Build | `/cpu/self` resolves to |
| --- | --- |
| `deps/install/native-debug` (`-O0`, no AVX) | `/cpu/self/opt/blocked` |
| `deps/install/native-opt` (`-march=native`, AVX) | `/cpu/self/avx/blocked` |
| `deps/install/wasi` | `/cpu/self/opt/blocked` |

The JiT backends `/cpu/self/gen/*` are chosen by neither native build, so dropping them in wasm
(libceed-0002) changes nothing; the difference is AVX, which only the native opt build has.
Different backends may round differently, so **slidesim always passes the full resource string
`/cpu/self/opt/blocked` to `CeedInit` on CPU and in wasm**, never the `/cpu/self` prefix. That
backend exists in all three builds (`CeedInit("/cpu/self/opt/blocked")` returns exactly that
resource in native-debug, native-opt and wasi), so every target runs the same backend. The smoke
test requests it and fails if the resource returned differs.

### PETSc

| Patch | Reason |
| --- | --- |
| `petsc-0001-no-mkdtemp-on-wasi.patch` | `PetscMkdtemp()` (`src/sys/fileio/fdir.c`) calls `mkdtemp()`, which wasi-libc does not declare (compile error). Under `__wasi__` it returns `PETSC_ERR_SUP`. |
| `petsc-0002-wasm-memory-usage.patch` | `PetscMemoryGetCurrentUsage()` (`src/sys/memory/mem.c`) reads `ru_maxrss` of `getrusage()`; the emulated `struct rusage` of wasi-libc has no such member (compile error). Under `__wasm__` it reports the size of the linear memory (`__builtin_wasm_memory_size(0) * 64 KiB`), the true resident size of a wasm instance. `-memory_view` works. |
| `petsc-0003-dmnetwork-view-no-tmpnam-system-on-wasi.patch` | The matplotlib viewer of DMNetwork (`src/dm/impls/network/networkview.c`) falls back to `tmpnam()` and `system()`; wasi-libc declares both as unavailable (deprecated/undefined), which leaves undefined symbols in `libpetsc.a`. Under `__wasi__` that viewer returns `PETSC_ERR_SUP_SYS`. |
| `petsc-0004-segv-checkpointer-without-setjmp.patch` | Upstream bug exposed by wasip1: without `<setjmp.h>` (wasi-sdk only provides it with the wasm exception-handling `-mllvm -wasm-enable-sjlj` mode, which we do not enable), `checkptr.c` does not define `PetscSignalSegvCheckPointerOrMpi()`, which `signal.c` still calls (undefined symbol at link). The patch adds an empty definition for that case; `PetscCheckPointer` already falls back to a NULL check via a macro in `petscimpl.h`. |
| `petsc-0005-no-pic-flag-for-static-libraries.patch` | Configure (`config/BuildSystem/config/setCompilers.py`, `checkSharedLinker`) adds `-fPIC` to `CFLAGS` even with `--with-shared-libraries=0` (also with `--with-pic=0`), because it tries the PIC guesses with its archiver fallback, which accepts anything. In wasm, `-fPIC` routes every global access through a GOT import (`global.get g@GOT` instead of an `i32.const` address). The patch skips the PIC guesses for the static archiver. Without it, appending `-fno-pic` to `COPTFLAGS` covers PETSc but not the `-O0` `CNOOPT` files of f2cblaslapack (`*lamch.c`, `*laruv.c`). |

Not patched, because unreachable or harmless in wasip1:

- `setjmp`/`longjmp`: not available (see 0004); PETSc only uses them for pointer probing in
  debug mode and the X11 viewer. Nothing needs `-mllvm -wasm-enable-sjlj`.
- `gethostname`, `getdomainname`, `uname`, `getcwd`, `realpath`, `access`, `sleep`: present in
  wasi-libc. The host name is reported as `(none)`, the user as `Unknown` (no `getpwuid`/`pwd.h`).
- `popen`, `fork`, `getpwuid`, threads: not detected by configure, so PETSc's own fallbacks are used.
- Compiler warnings left as they are: unused variables in code paths for `popen`/`setjmp`
  (`filev.c`, `err.c`, `binv.c`, `smatlab.c`, `fretrieve.c`, `pythonsys.c`, `networkview.c`,
  `checkptr.c`; upstream, on any platform without those functions) and `-Wshorten-64-to-32` in
  `sorti.c`, `kdtree.c`, `mpiaij.c`, `mpiuni/mpi.c` (64 bit to `long`, which is 32 bit in wasm32;
  upstream on every 32 bit target, irrelevant below 2^31 entries). libCEED warns "loop not
  vectorized" for the `CEED_PRAGMA_SIMD` loops (also natively, since `OPT` drops `-fopenmp-simd`).

## Verification

```
WASI_SDK_PATH=~/opt/wasi-sdk spike/petsc-wasm/build-wasi.sh   # 1 to 2 min on 16 cores
ls -l deps/install/wasi/lib/{libpetsc.a,libceed.a}
spike/petsc-wasm/smoke/run.sh                                 # wasmtime and Node
```

The smoke test checks `VecNorm`, a 1D Laplacian solve with CG + Jacobi (50 iterations, LAPACK
singular value estimate, `-memory_view`), `CeedInit("/cpu/self/opt/blocked")`, `CeedVectorNorm` and the gallery
mass operator (`Mass1DBuild` + `MassApply`) on [0, 1]. Both runtimes print identical numbers.
Additionally, linking a trivial `main` with `--whole-archive` of `libpetsc.a` and of `libceed.a`
(and `--no-gc-sections`) leaves no undefined symbol, so every object of both archives links in
wasip1.

## Emscripten

Not attempted: wasip1 was not blocked.
