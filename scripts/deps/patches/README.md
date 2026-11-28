# Patches of the dependencies

`scripts/deps/build-native.sh` and `scripts/deps/build-wasi.sh` copy each library from
`deps/src` (PETSc and libCEED clones, the Mmg tarball) into `deps/build/<name>-<build>` and apply
the patches of this directory there with `git apply` (`apply_patches` of `scripts/deps/lib.sh`);
`deps/src` is never modified. The name of a patch says where it applies:

- `<name>-<NNNN>-*.patch`: every build of `<name>` (native and wasm32-wasip1);
- `<name>-wasi-<NNNN>-*.patch`: the wasm32-wasip1 build only, after the former.

None of the patches has been submitted upstream yet; the table of each library says which are
meant to be.

| Patch | Builds | Upstream |
| --- | --- | --- |
| [`petsc-0001-mmg-ptscotch-optional.patch`](#petsc-every-build) | all | to submit (PETSc) |
| [`mmg-0001-memory-size-on-wasm.patch`](#mmg) | all (changes wasm32 only) | to submit (Mmg) |
| [`mmg-0002-no-common-symbols.patch`](#mmg) | all | to submit (Mmg) |
| [`petsc-wasi-0001` to `0005`](#petsc-wasm32-wasip1) | wasm32-wasip1 | 0004 to submit (PETSc bug); the others are wasip1 specific, to submit as a set with a wasip1 port |
| [`libceed-wasi-0001`, `0002`](#libceed-wasm32-wasip1) | wasm32-wasip1 | to submit with a wasip1 port |

The wasm32-wasip1 build: PETSc v3.26.1 (sequential, MPIUNI, f2cblaslapack, Mmg) and libCEED
v1.1.2 (`/cpu/self` ref, blocked and opt backends) build as static libraries with wasi-sdk 34.0
and run under wasmtime 50 and Node 24 (the milestone 0 spike, `spike/petsc-wasm/RESULT.md`; Mmg:
the milestone 4 spike, `spike/mmg/RESULT.md`). The patches are small source patches and two
configure patches; everything else is covered by wasi-libc emulation libraries or configure
options.

## Compiler and flags

- `$WASI_SDK_PATH/bin/clang --target=wasm32-wasip1 --sysroot=$WASI_SDK_PATH/share/wasi-sysroot`
- `-O3 -DNDEBUG -ffp-contract=off -fno-fast-math -msimd128` for all C (libCEED, Mmg, PETSc,
  f2cblaslapack), and `-ffile-prefix-map=<repository>/deps/build/=
  -ffile-prefix-map=$WASI_SDK_PATH/=wasi-sdk/`, so that `__FILE__` (PETSc's tracebacks, which the
  browser build shows on a page) holds no path of the machine.
- Archiver `llvm-ar`, `llvm-ranlib` from wasi-sdk.

## wasi-libc emulation libraries (instead of patches)

PETSc and Mmg are compiled with `-D_WASI_EMULATED_SIGNAL -D_WASI_EMULATED_PROCESS_CLOCKS
-D_WASI_EMULATED_GETPID` and linked with the matching `-lwasi-emulated-*` (each one verified
necessary: without it a `--whole-archive` link of `libpetsc.a` has undefined symbols). libCEED
needs none; `libmmg.a` needs `signal()` and `getrusage()`, covered by the same libraries.

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
  COPTFLAGS="-O3 -DNDEBUG -ffp-contract=off -fno-fast-math -msimd128 -ffile-prefix-map=..." \
  LIBS="-lwasi-emulated-signal -lwasi-emulated-process-clocks -lwasi-emulated-getpid" \
  --download-f2cblaslapack=deps/src/f2cblaslapack-3.8.0.q2.tar.gz \
  --with-mmg-dir=deps/install/wasi \
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
- f2cblaslapack: the tarball the pinned PETSc downloads, fetched into `deps/src` and checked
  against its sha256 by `scripts/deps/fetch.sh` (pinned in `deps/versions.env`, as for the
  native build). The build depends on nothing outside `deps/src`.

## Mmg

Mmg 5.8.0 (the version of PETSc's `Mmg.py`, LGPL-3.0) is the remesher of `DMAdaptMetric`. Both
builds compile it themselves (`build_mmg` of `scripts/deps/lib.sh`) and give it to PETSc's
configure with `--with-mmg-dir`, instead of PETSc's `--download-mmg`: PETSc applies no patches
to the packages it downloads, and Mmg needs two (below). Both builds configure it the same:
static libraries, `-DUSE_SCOTCH=OFF` (2D Mmg never uses Scotch; PT-Scotch is not available in
wasm, and slidesim keeps PT-Scotch as its mesh partitioner only), no Elas, no VTK, the private
headers PETSc's adaptor includes, no Fortran headers (`CMAKE_DISABLE_FIND_PACKAGE_Perl`: their
generator is a program of the target, so wasm). Natively the library is position independent and
linked into `libpetsc.so` (Mmg's CMake means to make its static libraries position independent,
but compares the strings `ON` as numbers); in wasm it is a plain static archive.

The wasm build uses wasi-sdk's CMake toolchain (`share/cmake/wasi-sdk-p1.cmake`: compilers,
target, the sysroot for `find_library`) plus `CMAKE_LIBRARY_PATH=/lib/wasm32-wasip1` (relative to
the sysroot, where wasi-libc's `libm.a` lies). Mmg's command line programs are always built and
cannot be linked in wasip1 (`system()` does not exist): they are linked with
`-Wl,--allow-undefined` and removed after the install, never run. The spike built Mmg through
`--download-mmg` instead, with a compile rule override for `-fno-common` and `M_LIB` set by hand;
the separate build replaces both.

| Patch | Reason |
| --- | --- |
| `mmg-0001-memory-size-on-wasm.patch` | `MMG5_memSize` (`src/common/tools.c`) asks the system for its physical memory to set Mmg's memory limit (half of it, `MMG5_MEMPERCENT`). It knows macOS, Unix and Windows, not WASI: in wasm it warned "unknown system" on stderr, and Mmg then printed "Maximum memory set to default value: 800 MB." on **stdout**, twice per adaptation and regardless of the verbosity, and capped itself at a silent 800 MB. Under `__wasm32__` the patch returns the 32 bit address space (`SIZE_MAX`, 65536 pages of 64 KiB), the most a wasm32 instance can ever have, so the limit is half of it, 2 GiB, by Mmg's own rule; nothing is printed. A caller can still set a limit with `MMG2D_IPARAM_mem` (PETSc's adaptor does not). Natively nothing changes. Chosen over a PETSc patch of `mmgadapt.c` setting `MMG2D_IPARAM_mem`: that could not silence the first of the two prints, which comes from `MMG2D_Init_mesh` before any parameter can be set, and would leave the stderr warning. |
| `mmg-0002-no-common-symbols.patch` | Mmg adds `-fcommon` to its libraries for clang newer than 10 (`cmake/modules/macros.cmake`). wasm has no common symbols (`int8_t ddb;` in `mmg3d1_delone.c` is a tentative definition: "common symbols are not yet implemented for Wasm"). Mmg 5.8.0 needs no common symbols: gcc 10 and newer compile it with `-fno-common` by default, natively and in the spike's wasm build. The patch drops the flag. |

## PETSc (every build)

| Patch | Reason |
| --- | --- |
| `petsc-0001-mmg-ptscotch-optional.patch` | `Mmg.py` requires PT-Scotch, whose package requires an MPI that Scotch's CMake can use, not MPIUNI (the Mmg spike, attempt 1). Mmg uses Scotch only to renumber in mmgs and mmg3d (`USE_SCOTCH`), never in mmg2d. The patch makes PT-Scotch an optional dependency of Mmg and passes `-DUSE_SCOTCH=OFF` to a downloaded Mmg without it; with PT-Scotch present (native) nothing changes. Applied to every build, so that both configure PETSc's Mmg package alike. |

## Patches of the wasm32-wasip1 build

### libCEED (wasm32-wasip1)

| Patch | Reason |
| --- | --- |
| `libceed-wasi-0001-no-wordexp-on-wasi.patch` | `interface/ceed.c` includes `<wordexp.h>` and calls `wordexp()` to shell-expand the JiT cache directory (`CeedGetCacheDir`). wasi-libc has no `wordexp`. Under `__wasi__` the base directory is used verbatim. The cache directory is only used by the JiT backends, which are not built for wasm. |
| `libceed-wasi-0002-optional-cpu-gen-backend.patch` | The `/cpu/self/gen/*` backends are always built: they are C++ and compile generated code at run time with a C++ compiler and `dlopen()` it, neither of which exists in wasip1. The patch adds a Makefile switch `CPU_GEN=0` which leaves them out and adds a weak registration stub `backends/weak/ceed-cpu-gen-weak.c` (same pattern as the AVX/CUDA/... stubs), so `/cpu/self/gen` reports "Backend not currently compiled" and `/cpu/self` resolves to `/cpu/self/opt/blocked`. |

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
(libceed-wasi-0002) changes nothing; the difference is AVX, which only the native opt build has.
Different backends may round differently, so **slidesim always passes the full resource string
`/cpu/self/opt/blocked` to `CeedInit` on CPU and in wasm**, never the `/cpu/self` prefix. That
backend exists in all three builds (`CeedInit("/cpu/self/opt/blocked")` returns exactly that
resource in native-debug, native-opt and wasi), so every target runs the same backend. The smoke
test requests it and fails if the resource returned differs.

### PETSc (wasm32-wasip1)

| Patch | Reason |
| --- | --- |
| `petsc-wasi-0001-no-mkdtemp-on-wasi.patch` | `PetscMkdtemp()` (`src/sys/fileio/fdir.c`) calls `mkdtemp()`, which wasi-libc does not declare (compile error). Under `__wasi__` it returns `PETSC_ERR_SUP`. |
| `petsc-wasi-0002-wasm-memory-usage.patch` | `PetscMemoryGetCurrentUsage()` (`src/sys/memory/mem.c`) reads `ru_maxrss` of `getrusage()`; the emulated `struct rusage` of wasi-libc has no such member (compile error). Under `__wasm__` it reports the size of the linear memory (`__builtin_wasm_memory_size(0) * 64 KiB`), the true resident size of a wasm instance. `-memory_view` works. |
| `petsc-wasi-0003-dmnetwork-view-no-tmpnam-system-on-wasi.patch` | The matplotlib viewer of DMNetwork (`src/dm/impls/network/networkview.c`) falls back to `tmpnam()` and `system()`; wasi-libc declares both as unavailable (deprecated/undefined), which leaves undefined symbols in `libpetsc.a`. Under `__wasi__` that viewer returns `PETSC_ERR_SUP_SYS`. |
| `petsc-wasi-0004-segv-checkpointer-without-setjmp.patch` | Upstream bug exposed by wasip1: without `<setjmp.h>` (wasi-sdk only provides it with the wasm exception-handling `-mllvm -wasm-enable-sjlj` mode, which we do not enable), `checkptr.c` does not define `PetscSignalSegvCheckPointerOrMpi()`, which `signal.c` still calls (undefined symbol at link). The patch adds an empty definition for that case; `PetscCheckPointer` already falls back to a NULL check via a macro in `petscimpl.h`. |
| `petsc-wasi-0005-no-pic-flag-for-static-libraries.patch` | Configure (`config/BuildSystem/config/setCompilers.py`, `checkSharedLinker`) adds `-fPIC` to `CFLAGS` even with `--with-shared-libraries=0` (also with `--with-pic=0`), because it tries the PIC guesses with its archiver fallback, which accepts anything. In wasm, `-fPIC` routes every global access through a GOT import (`global.get g@GOT` instead of an `i32.const` address). The patch skips the PIC guesses for the static archiver. Without it, appending `-fno-pic` to `COPTFLAGS` covers PETSc but not the `-O0` `CNOOPT` files of f2cblaslapack (`*lamch.c`, `*laruv.c`). |

Not patched, because unreachable or harmless in wasip1:

- `setjmp`/`longjmp`: not available (see petsc-wasi-0004); PETSc only uses them for pointer probing in
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
WASI_SDK_PATH=~/opt/wasi-sdk scripts/deps/build-wasi.sh
ls -l deps/install/wasi/lib/{libpetsc.a,libmmg.a,libceed.a}
scripts/deps/wasi-smoke/run.sh      # Mmg through DMAdaptMetric in wasmtime, silent
spike/petsc-wasm/smoke/run.sh       # PETSc and libCEED in wasmtime and Node
```

`scripts/deps/wasi-smoke` adapts the unit square of two triangles to the size 0.1 with Mmg
(`DMAdaptMetric`, the calls of `crates/petsc/src/metric.rs`), prints the number of cells and fails
unless it lies in [150, 320] and that line is all the output, on stdout and stderr: Mmg prints
nothing (mmg-0001). The milestone 0 smoke test checks `VecNorm`, a 1D Laplacian solve with CG +
Jacobi (50 iterations, LAPACK singular value estimate, `-memory_view`),
`CeedInit("/cpu/self/opt/blocked")`, `CeedVectorNorm` and the gallery mass operator
(`Mass1DBuild` + `MassApply`) on [0, 1]. Both runtimes print identical numbers. Additionally,
linking a trivial `main` with `--whole-archive` of `libpetsc.a` and of `libceed.a` (and
`--no-gc-sections`) leaves no undefined symbol, so every object of both archives links in wasip1.

## Emscripten

Not attempted: wasip1 was not blocked.
