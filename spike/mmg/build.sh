#!/usr/bin/env bash
# Spike (throwaway): PETSc v3.26.1 with Mmg 5.8.0 for DMAdaptMetric, natively and for
# wasm32-wasip1, into deps/install/spike-mmg-{native,wasi} (never the prefixes the tests use).
#
# Usage: spike/mmg/build.sh <native|wasi>
#   native: opt build of scripts/deps/build-native.sh (mpicc, f2cblaslapack, PT-Scotch, bison)
#           plus Mmg; no libCEED (the lobule program does not need it).
#   MMG_CMAKE_ARGS: extra CMake arguments of Mmg (native default: none, Mmg finds PT-Scotch).
#   wasi:   the configure of the milestone 0 build-wasi.sh (MPIUNI, static) plus Mmg, with the
#           patches petsc-wasi-* and petsc-0001-mmg-ptscotch-optional of scripts/deps/patches
#           (the spike's patch, moved there).
# JOBS (default 8) bounds the parallel make.
set -euo pipefail

MODE="${1:-}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SPIKE="$ROOT/spike/mmg"
# shellcheck source=SCRIPTDIR/../../deps/versions.env
. "$ROOT/deps/versions.env"
SRC="$ROOT/deps/src"
JOBS="${JOBS:-8}"
PREFIX="$ROOT/deps/install/spike-mmg-$MODE"
BUILD="$ROOT/deps/build/spike-mmg-$MODE"

# The Mmg tarball PETSc v3.26.1 downloads (Mmg.py: version 5.8.0, github archive v5.8.0).
MMG_VERSION=5.8.0
MMG_SHA256=686eaab84de79c072f3aedf26cd11ced44c84b435d51ce34e016ad203172922f
MMG="$SRC/mmg-v$MMG_VERSION.tar.gz"

# Reproducible floating point: no FMA contraction, no fast-math (same numbers on every target).
FP="-ffp-contract=off -fno-fast-math"

unset CFLAGS CPPFLAGS LDFLAGS LIBS PETSC_DIR PETSC_ARCH

"$ROOT/scripts/deps/fetch.sh"
grep -qE "self\.version *= *'${MMG_VERSION//./\\.}'" "$SRC/petsc/config/BuildSystem/config/packages/Mmg.py" ||
  { echo "PETSc no longer downloads Mmg $MMG_VERSION" >&2; exit 1; }
if [[ ! -f "$MMG" ]]; then
  curl -fsSL -o "$MMG.part" "https://github.com/MmgTools/mmg/archive/v$MMG_VERSION.tar.gz"
  mv "$MMG.part" "$MMG"
fi
echo "$MMG_SHA256  $MMG" | sha256sum --check --quiet

rm -rf "$PREFIX" "$BUILD"
mkdir -p "$PREFIX" "$BUILD"

# flex, a build tool of PT-Scotch (its strategy parser), as in build-native.sh.
FLEX="$BUILD/flex"
mkdir -p "$FLEX/src"
tar xzf "$SRC/flex-$FLEX_VERSION.tar.gz" -C "$FLEX/src" --strip-components=1
(
  cd "$FLEX/src"
  ./configure --quiet --prefix="$FLEX/install" --disable-nls --disable-shared \
    CFLAGS="-O2 -D_GNU_SOURCE"
  make -j"$JOBS" >/dev/null
  make install >/dev/null
)
export PATH="$FLEX/install/bin:$PATH"

cp -a "$SRC/petsc" "$BUILD/petsc"
apply_patches() { # <glob>
  local p
  for p in $1; do
    [[ -e "$p" ]] || continue
    echo "applying $(basename "$p")"
    git -C "$BUILD/petsc" apply --whitespace=nowarn "$p"
  done
}

case "$MODE" in
native)
  (
    cd "$BUILD/petsc"
    ./configure --prefix="$PREFIX" --with-cc=mpicc --with-cxx=0 --with-fc=0 \
      --download-f2cblaslapack="$SRC/f2cblaslapack-$F2CBLASLAPACK_VERSION.tar.gz" \
      --download-ptscotch="$SRC/scotch-v$PTSCOTCH_VERSION.tar.gz" \
      --download-ptscotch-cmake-arguments="-DTHREADS=OFF -DSCOTCH_DETERMINISTIC=FULL" \
      --download-bison="$SRC/bison-$BISON_VERSION.tar.gz" \
      --download-mmg="$MMG" \
      ${MMG_CMAKE_ARGS:+--download-mmg-cmake-arguments="$MMG_CMAKE_ARGS"} \
      --with-shared-libraries=1 \
      --with-debugging=0 COPTFLAGS="-O3 -DNDEBUG -march=native $FP"
    make -j"$JOBS" all
    make install
  )
  cat >"$PREFIX/env.sh" <<ENV
export PETSC_DIR="$PREFIX"
export PKG_CONFIG_PATH="$PREFIX/lib/pkgconfig"
export WASI_SDK_PATH="\${WASI_SDK_PATH:-\$HOME/opt/wasi-sdk}"
ENV
  ;;
wasi)
  WASI_SDK_PATH="${WASI_SDK_PATH:-$HOME/opt/wasi-sdk}"
  [[ -x "$WASI_SDK_PATH/bin/clang" ]] || { echo "WASI_SDK_PATH ($WASI_SDK_PATH) has no bin/clang" >&2; exit 2; }
  CC="$WASI_SDK_PATH/bin/clang"
  TARGET="--target=wasm32-wasip1 --sysroot=$WASI_SDK_PATH/share/wasi-sysroot"
  FLAGS="-O3 -DNDEBUG $FP -msimd128"
  EMU_DEFS="-D_WASI_EMULATED_SIGNAL -D_WASI_EMULATED_PROCESS_CLOCKS -D_WASI_EMULATED_GETPID"
  EMU_LIBS="-lwasi-emulated-signal -lwasi-emulated-process-clocks -lwasi-emulated-getpid"
  apply_patches "$ROOT/scripts/deps/patches/petsc-wasi-*.patch"
  apply_patches "$ROOT/scripts/deps/patches/petsc-0001-mmg-ptscotch-optional.patch"
  # Mmg's CMake (the arguments go into a shell command line, hence the inner quotes):
  # - no Perl: it would run the wasm genheader to write the Fortran header;
  # - libm of the wasi sysroot (find_library looks on the host);
  # - the C++ compiler of wasi-sdk with the target flags: inoutcpp_*.cpp (VTK I/O stubs without
  #   VTK) is always part of the libraries; without this CMake takes the host c++;
  # - -fno-common after Mmg's own -fcommon (added for clang > 10): wasm has no common symbols
  #   (`int8_t ddb;` in mmg3d1_delone.c); gcc builds Mmg with -fno-common natively anyway;
  # - --allow-undefined for Mmg's executables (mmg2d_O3, mmgs_O3, mmg3d_O3, genheader), built
  #   but never run: system() does not exist in wasi-libc, signal() needs the emulation library.
  MMG_WASI_CMAKE="-DCMAKE_DISABLE_FIND_PACKAGE_Perl=ON \
-DM_LIB=$WASI_SDK_PATH/share/wasi-sysroot/lib/wasm32-wasip1/libm.a \
-DCMAKE_CXX_COMPILER=$WASI_SDK_PATH/bin/clang++ -DCMAKE_CXX_FLAGS=\"$TARGET $FLAGS $EMU_DEFS\" \
-DCMAKE_C_COMPILE_OBJECT='<CMAKE_C_COMPILER> <DEFINES> <INCLUDES> <FLAGS> -fno-common -o <OBJECT> -c <SOURCE>' \
-DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-undefined"
  # WASI_PTSCOTCH=1: attempt 1, PT-Scotch on MPIUNI (as natively; bison built for the host).
  extra=()
  if [[ "${WASI_PTSCOTCH:-0}" == 1 ]]; then
    extra=(--download-ptscotch="$SRC/scotch-v$PTSCOTCH_VERSION.tar.gz"
      --download-ptscotch-cmake-arguments="-DTHREADS=OFF -DSCOTCH_DETERMINISTIC=FULL"
      --download-bison="$SRC/bison-$BISON_VERSION.tar.gz" --download-bison-cc=gcc)
  fi
  (
    cd "$BUILD/petsc"
    ./configure PETSC_ARCH=arch-wasi --prefix="$PREFIX" \
      --with-cc="$CC" --with-cxx=0 --with-fc=0 --with-mpi=0 \
      --with-ar="$WASI_SDK_PATH/bin/llvm-ar" --with-ranlib="$WASI_SDK_PATH/bin/llvm-ranlib" \
      CFLAGS+="$TARGET" CPPFLAGS="$EMU_DEFS" COPTFLAGS="$FLAGS" LIBS="$EMU_LIBS" \
      --download-f2cblaslapack="$SRC/f2cblaslapack-$F2CBLASLAPACK_VERSION.tar.gz" \
      --download-mmg="$MMG" \
      --download-mmg-cmake-arguments="${MMG_CMAKE_ARGS:-$MMG_WASI_CMAKE}" \
      "${extra[@]}" \
      --with-shared-libraries=0 --with-x=0 --with-ssl=0 --with-pthread=0 \
      --with-debugging=0 --with-avx512-kernels=0 --with-batch \
      --known-sdot-returns-double=0 --known-snrm2-returns-double=0
    make PETSC_DIR="$PWD" PETSC_ARCH=arch-wasi -j"$JOBS" all
    make PETSC_DIR="$PWD" PETSC_ARCH=arch-wasi install
  )
  cat >"$PREFIX/env.sh" <<ENV
export WASI_SDK_PATH="$WASI_SDK_PATH"
export PETSC_DIR="$PREFIX"
export PKG_CONFIG_PATH="$PREFIX/lib/pkgconfig"
ENV
  ;;
*)
  echo "usage: $0 <native|wasi>" >&2
  exit 2
  ;;
esac

grep -q '^#define PETSC_HAVE_MMG 1' "$PREFIX/include/petscconf.h" ||
  { echo "error: PETSc in $PREFIX has no Mmg" >&2; exit 1; }
echo "built PETSc with Mmg ($MODE) into $PREFIX"
