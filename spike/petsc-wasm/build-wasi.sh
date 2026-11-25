#!/usr/bin/env bash
# Spike: build libCEED (/cpu/self backends) and PETSc (sequential, f2cblaslapack) as static
# libraries for wasm32-wasip1 with wasi-sdk, into deps/install/wasi.
#
# Usage: WASI_SDK_PATH=~/opt/wasi-sdk spike/petsc-wasm/build-wasi.sh
#
# The sources in deps/src are never modified: each library is copied into deps/build/*-wasi
# and the patches of spike/petsc-wasm/patches are applied there (see PATCHES.md).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SPIKE="$ROOT/spike/petsc-wasm"
PREFIX="$ROOT/deps/install/wasi"
JOBS="$(nproc)"

WASI_SDK_PATH="${WASI_SDK_PATH:-$HOME/opt/wasi-sdk}"
[[ -x "$WASI_SDK_PATH/bin/clang" ]] || { echo "WASI_SDK_PATH ($WASI_SDK_PATH) has no bin/clang" >&2; exit 2; }
CC="$WASI_SDK_PATH/bin/clang"
AR="$WASI_SDK_PATH/bin/llvm-ar"
RANLIB="$WASI_SDK_PATH/bin/llvm-ranlib"
TARGET="--target=wasm32-wasip1 --sysroot=$WASI_SDK_PATH/share/wasi-sysroot"

# Reproducible floating point (same as the native build) plus wasm SIMD.
FLAGS="-O3 -DNDEBUG -ffp-contract=off -fno-fast-math -msimd128"
# wasi-libc emulations used instead of source patches (signal(), times()/clock(), getpid()).
EMU_DEFS="-D_WASI_EMULATED_SIGNAL -D_WASI_EMULATED_PROCESS_CLOCKS -D_WASI_EMULATED_GETPID"
EMU_LIBS="-lwasi-emulated-signal -lwasi-emulated-process-clocks -lwasi-emulated-getpid"

# The flags of the environment must not leak into the build (e.g. a stray -ffast-math).
unset CFLAGS CPPFLAGS LDFLAGS LIBS PETSC_DIR PETSC_ARCH

"$ROOT/scripts/deps/fetch.sh"
rm -rf "$PREFIX"
mkdir -p "$PREFIX" "$ROOT/deps/build"

# Fresh copy of deps/src/<name> into deps/build/<name>-wasi with the patches <name>-*.patch applied.
prepare() {
  local name="$1" dir="$ROOT/deps/build/$1-wasi"
  rm -rf "$dir"
  cp -a "$ROOT/deps/src/$name" "$dir"
  for p in "$SPIKE"/patches/"$name"-*.patch; do
    git -C "$dir" apply --whitespace=nowarn "$p"
  done
}

# libCEED: static, without the /cpu/self/gen JiT backends (they compile C++ at run time and
# dlopen() the result, neither exists in wasip1; CPU_GEN=0 comes from libceed-0002).
prepare libceed
make -C "$ROOT/deps/build/libceed-wasi" -j"$JOBS" install prefix="$PREFIX" \
  STATIC=1 CPU_GEN=0 CC="$CC $TARGET" AR="$AR" MARCHFLAG= OPT="$FLAGS"

# PETSc: cross configure (--with-batch: the conftest executables are wasm and cannot be run;
# the --known-* values are the ones the native configure determines for f2cblaslapack).
# CFLAGS+= keeps PETSc's default warning and visibility flags, as in the native build.
# No -fPIC: in wasm it turns every global access into a GOT indirection (petsc-0005).
prepare petsc

# f2cblaslapack: the tarball the pinned PETSc downloads, fetched once into deps/src and checked
# against its sha256, so the build depends on nothing but deps/src.
F2C_URL="https://web.cels.anl.gov/projects/petsc/download/externalpackages/f2cblaslapack-3.8.0.q2.tar.gz"
F2C_SHA256="12fa8d5001e313aeeea22eb2054883f59e3e3c656f11129f31abb02ddbf8ad60"
F2C="$ROOT/deps/src/$(basename "$F2C_URL")"
grep -qF "'$F2C_URL'" "$ROOT/deps/build/petsc-wasi/config/BuildSystem/config/packages/f2cblaslapack.py" ||
  { echo "PETSc no longer downloads $F2C_URL: update F2C_URL and F2C_SHA256" >&2; exit 1; }
if [[ ! -f "$F2C" ]]; then
  curl -fsSL -o "$F2C.part" "$F2C_URL"
  mv "$F2C.part" "$F2C"
fi
echo "$F2C_SHA256  $F2C" | sha256sum --check --quiet
(
  cd "$ROOT/deps/build/petsc-wasi"
  ./configure PETSC_ARCH=arch-wasi --prefix="$PREFIX" \
    --with-cc="$CC" --with-cxx=0 --with-fc=0 --with-mpi=0 \
    --with-ar="$AR" --with-ranlib="$RANLIB" \
    CFLAGS+="$TARGET" CPPFLAGS="$EMU_DEFS" COPTFLAGS="$FLAGS" LIBS="$EMU_LIBS" \
    --download-f2cblaslapack="$F2C" \
    --with-shared-libraries=0 --with-x=0 --with-ssl=0 --with-pthread=0 \
    --with-debugging=0 --with-avx512-kernels=0 --with-batch \
    --known-sdot-returns-double=0 --known-snrm2-returns-double=0
  make PETSC_DIR="$PWD" PETSC_ARCH=arch-wasi -j"$JOBS" all
  make PETSC_DIR="$PWD" PETSC_ARCH=arch-wasi install
)

cat > "$PREFIX/env.sh" <<ENV
export WASI_SDK_PATH="$WASI_SDK_PATH"
export PETSC_DIR="$PREFIX"
export CEED_DIR="$PREFIX"
export PKG_CONFIG_PATH="$PREFIX/lib/pkgconfig"
# Link line for an executable using both libraries (order matters for static archives).
export WASI_PETSC_CEED_LIBS="-L$PREFIX/lib -lpetsc -lceed -lf2clapack -lf2cblas $EMU_LIBS -ldl -lm"
ENV
ls -l "$PREFIX/lib/libpetsc.a" "$PREFIX/lib/libceed.a"
echo "built wasm32-wasip1 into $PREFIX"
