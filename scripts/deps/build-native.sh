#!/usr/bin/env bash
# Build PETSc and libCEED (CPU backends) into deps/install/native-<mode>.
# Usage: build-native.sh <debug|opt>
set -euo pipefail

# Reproducible floating point: no FMA contraction, no fast-math (same numbers on every target).
FP="-ffp-contract=off -fno-fast-math"
MODE="${1:-}"
case "$MODE" in
  debug) DEBUGGING=1; FLAGS="-O0 -g $FP" ;;
  opt) DEBUGGING=0; FLAGS="-O3 -DNDEBUG -march=native $FP" ;;
  *) echo "usage: $0 <debug|opt>" >&2; exit 2 ;;
esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PREFIX="$ROOT/deps/install/native-$MODE"
JOBS="$(nproc)"

"$ROOT/scripts/deps/fetch.sh"
mkdir -p "$PREFIX" "$ROOT/deps/build"

# libCEED first (no dependency on PETSc); CPU backends only.
rm -rf "$ROOT/deps/build/libceed-$MODE"
cp -a "$ROOT/deps/src/libceed" "$ROOT/deps/build/libceed-$MODE"
(
  cd "$ROOT/deps/build/libceed-$MODE"
  make -j"$JOBS" install prefix="$PREFIX" OPT="$FLAGS" CC=mpicc
)

# PETSc with its own BLAS/LAPACK, no C++ or Fortran.
rm -rf "$ROOT/deps/build/petsc-$MODE"
cp -a "$ROOT/deps/src/petsc" "$ROOT/deps/build/petsc-$MODE"
(
  cd "$ROOT/deps/build/petsc-$MODE"
  unset PETSC_DIR PETSC_ARCH
  ./configure --prefix="$PREFIX" --with-cc=mpicc --with-cxx=0 --with-fc=0 \
    --download-f2cblaslapack --with-shared-libraries=1 \
    --with-debugging="$DEBUGGING" COPTFLAGS="$FLAGS"
  make -j"$JOBS" all
  make install
)

cat > "$PREFIX/env.sh" <<ENV
export PETSC_DIR="$PREFIX"
export CEED_DIR="$PREFIX"
export PKG_CONFIG_PATH="$PREFIX/lib/pkgconfig\${PKG_CONFIG_PATH:+:\$PKG_CONFIG_PATH}"
export LD_LIBRARY_PATH="$PREFIX/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
ENV
echo "built $MODE into $PREFIX"
