#!/usr/bin/env bash
# Build libCEED (/cpu/self backends), Mmg and PETSc (sequential, f2cblaslapack, Mmg) as static
# libraries for wasm32-wasip1 with wasi-sdk, into deps/install/wasi.
#
# Usage: WASI_SDK_PATH=~/opt/wasi-sdk scripts/deps/build-wasi.sh
# JOBS (default: the number of CPUs) bounds the parallel builds.
#
# The sources in deps/src are never modified: each library is copied into deps/build/*-wasi and
# the patches of scripts/deps/patches are applied there (see its README.md).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=SCRIPTDIR/../../deps/versions.env
. "$ROOT/deps/versions.env"
SRC="$ROOT/deps/src"
PREFIX="$ROOT/deps/install/wasi"
JOBS="${JOBS:-$(nproc)}"
DEPS_SRC="$SRC"
DEPS_PATCHES="$ROOT/scripts/deps/patches"
# shellcheck source=SCRIPTDIR/lib.sh
. "$ROOT/scripts/deps/lib.sh"

WASI_SDK_PATH="${WASI_SDK_PATH:-$HOME/opt/wasi-sdk}"
[[ -x "$WASI_SDK_PATH/bin/clang" ]] || { echo "WASI_SDK_PATH ($WASI_SDK_PATH) has no bin/clang" >&2; exit 2; }
CC="$WASI_SDK_PATH/bin/clang"
AR="$WASI_SDK_PATH/bin/llvm-ar"
RANLIB="$WASI_SDK_PATH/bin/llvm-ranlib"
SYSROOT="$WASI_SDK_PATH/share/wasi-sysroot"
TARGET="--target=wasm32-wasip1 --sysroot=$SYSROOT"

# Reproducible floating point (same as the native build) plus wasm SIMD. The paths of the build
# relative in the libraries (`__FILE__` in PETSc's tracebacks and libCEED's errors, which a page
# of the browser build shows): deps/build/petsc-wasi/src/... as petsc-wasi/src/..., wasi-sdk's
# headers as wasi-sdk/...; without -g there is no other path to map.
FLAGS="-O3 -DNDEBUG -ffp-contract=off -fno-fast-math -msimd128"
FLAGS+=" -ffile-prefix-map=$ROOT/deps/build/= -ffile-prefix-map=$WASI_SDK_PATH/=wasi-sdk/"
# wasi-libc emulations used instead of source patches (signal(), times()/clock(), getpid()).
EMU_DEFS="-D_WASI_EMULATED_SIGNAL -D_WASI_EMULATED_PROCESS_CLOCKS -D_WASI_EMULATED_GETPID"
EMU_LIBS="-lwasi-emulated-signal -lwasi-emulated-process-clocks -lwasi-emulated-getpid"

# The flags of the environment must not leak into the build (e.g. a stray -ffast-math).
unset CFLAGS CXXFLAGS CPPFLAGS LDFLAGS LIBS PETSC_DIR PETSC_ARCH

"$ROOT/scripts/deps/fetch.sh"
rm -rf "$PREFIX"
mkdir -p "$PREFIX" "$ROOT/deps/build"

# Fresh copy of deps/src/<name> into deps/build/<name>-wasi with its patches applied.
prepare() {
  local name="$1" dir="$ROOT/deps/build/$1-wasi"
  rm -rf "$dir"
  cp -a "$SRC/$name" "$dir"
  apply_patches "$name" "$dir" wasi
}

# libCEED: static, without the /cpu/self/gen JiT backends (they compile C++ at run time and
# dlopen() the result, neither exists in wasip1; CPU_GEN=0 comes from libceed-wasi-0002).
prepare libceed
make -C "$ROOT/deps/build/libceed-wasi" -j"$JOBS" install prefix="$PREFIX" \
  STATIC=1 CPU_GEN=0 CC="$CC $TARGET" AR="$AR" MARCHFLAG= OPT="$FLAGS"

# Mmg, configured as in build-native.sh, with wasi-sdk's CMake toolchain (compilers, target,
# sysroot). libm lies in the sysroot's lib/wasm32-wasip1, which find_library does not search by
# itself (the path is taken relative to the sysroot). No position independent code (as for PETSc,
# petsc-wasi-0005). Mmg's command line programs are always built, and cannot be linked in
# wasip1 (system() does not exist): linked with undefined symbols and removed after the install,
# never run.
build_mmg wasi "$ROOT/deps/build/mmg-wasi" "$PREFIX" Release "$FLAGS $EMU_DEFS" \
  -DCMAKE_TOOLCHAIN_FILE="$WASI_SDK_PATH/share/cmake/wasi-sdk-p1.cmake" \
  -DWASI_SDK_PREFIX="$WASI_SDK_PATH" -DCMAKE_SYSROOT="$SYSROOT" \
  -DCMAKE_LIBRARY_PATH=/lib/wasm32-wasip1 -DBUILD_SHARED_LIBS=OFF \
  -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-undefined
rm -f "$PREFIX"/bin/mmg2d_O3 "$PREFIX"/bin/mmgs_O3 "$PREFIX"/bin/mmg3d_O3
rmdir "$PREFIX/bin" 2>/dev/null || true

# PETSc: cross configure (--with-batch: the conftest executables are wasm and cannot be run;
# the --known-* values are the ones the native configure determines for f2cblaslapack).
# CFLAGS+= keeps PETSc's default warning and visibility flags, as in the native build.
# No -fPIC: in wasm it turns every global access into a GOT indirection (petsc-wasi-0005).
# f2cblaslapack is the tarball the pinned PETSc downloads, fetched and checked by fetch.sh.
prepare petsc
(
  cd "$ROOT/deps/build/petsc-wasi"
  ./configure PETSC_ARCH=arch-wasi --prefix="$PREFIX" \
    --with-cc="$CC" --with-cxx=0 --with-fc=0 --with-mpi=0 \
    --with-ar="$AR" --with-ranlib="$RANLIB" \
    CFLAGS+="$TARGET" CPPFLAGS="$EMU_DEFS" COPTFLAGS="$FLAGS" LIBS="$EMU_LIBS" \
    --download-f2cblaslapack="$SRC/f2cblaslapack-$F2CBLASLAPACK_VERSION.tar.gz" \
    --with-mmg-dir="$PREFIX" \
    --with-shared-libraries=0 --with-x=0 --with-ssl=0 --with-pthread=0 \
    --with-debugging=0 --with-avx512-kernels=0 --with-batch \
    --known-sdot-returns-double=0 --known-snrm2-returns-double=0
  make PETSC_DIR="$PWD" PETSC_ARCH=arch-wasi -j"$JOBS" all
  make PETSC_DIR="$PWD" PETSC_ARCH=arch-wasi install
)
grep -q '^#define PETSC_HAVE_MMG 1' "$PREFIX/include/petscconf.h" ||
  { echo "error: PETSc in $PREFIX has no Mmg" >&2; exit 1; }

cat > "$PREFIX/env.sh" <<ENV
export WASI_SDK_PATH="$WASI_SDK_PATH"
export PETSC_DIR="$PREFIX"
export CEED_DIR="$PREFIX"
export PKG_CONFIG_PATH="$PREFIX/lib/pkgconfig"
# Link line for an executable using both libraries (order matters for static archives).
export WASI_PETSC_CEED_LIBS="-L$PREFIX/lib -lpetsc -lmmg -lceed -lf2clapack -lf2cblas $EMU_LIBS -ldl -lm"
ENV
ls -l "$PREFIX/lib/libpetsc.a" "$PREFIX/lib/libmmg.a" "$PREFIX/lib/libceed.a"
echo "built wasm32-wasip1 into $PREFIX"
