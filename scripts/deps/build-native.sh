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
# shellcheck source=SCRIPTDIR/../../deps/versions.env
. "$ROOT/deps/versions.env"
SRC="$ROOT/deps/src"
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

# flex, a build tool of PT-Scotch (its strategy parser), which PETSc does not download. Not
# linked into anything. -D_GNU_SOURCE declares reallocarray for flex 2.6.4 on glibc >= 2.26.
FLEX="$ROOT/deps/build/flex-$MODE"
rm -rf "$FLEX"
mkdir -p "$FLEX/src"
tar xzf "$SRC/flex-$FLEX_VERSION.tar.gz" -C "$FLEX/src" --strip-components=1
(
  cd "$FLEX/src"
  ./configure --quiet --prefix="$FLEX/install" --disable-nls --disable-shared \
    CFLAGS="-O2 -D_GNU_SOURCE"
  make -j"$JOBS"
  make install
)

# PETSc with its own BLAS/LAPACK and the PT-Scotch graph partitioner (which needs bison and flex
# to build), no C++ or Fortran. Every package comes from a pinned tarball in deps/src (fetch.sh).
# PT-Scotch is compiled with PETSc's C flags, so with COPTFLAGS. Its CMake build would add threads
# (THREADS=ON by default, despite PETSc's -USCOTCH_PTHREAD), whose results may depend on the
# scheduling: THREADS=OFF, and SCOTCH_DETERMINISTIC=FULL for its deterministic algorithms with a
# fixed random seed. Then a partition depends on the graph and the number of processes only.
rm -rf "$ROOT/deps/build/petsc-$MODE"
cp -a "$ROOT/deps/src/petsc" "$ROOT/deps/build/petsc-$MODE"
(
  cd "$ROOT/deps/build/petsc-$MODE"
  unset PETSC_DIR PETSC_ARCH
  export PATH="$FLEX/install/bin:$PATH"
  ./configure --prefix="$PREFIX" --with-cc=mpicc --with-cxx=0 --with-fc=0 \
    --download-f2cblaslapack="$SRC/f2cblaslapack-$F2CBLASLAPACK_VERSION.tar.gz" \
    --download-ptscotch="$SRC/scotch-v$PTSCOTCH_VERSION.tar.gz" \
    --download-ptscotch-cmake-arguments="-DTHREADS=OFF -DSCOTCH_DETERMINISTIC=FULL" \
    --download-bison="$SRC/bison-$BISON_VERSION.tar.gz" \
    --with-shared-libraries=1 \
    --with-debugging="$DEBUGGING" COPTFLAGS="$FLAGS"
  make -j"$JOBS" all
  make install
)

# PT-Scotch built without threads: none of its libraries imports pthread_create.
for lib in "$PREFIX"/lib/lib*scotch*.so; do
  symbols="$(nm -D --undefined-only "$lib")"
  if grep -qw pthread_create <<<"$symbols"; then
    echo "error: $lib imports pthread_create: PT-Scotch was built with threads" >&2
    exit 1
  fi
done

cat > "$PREFIX/env.sh" <<ENV
export PETSC_DIR="$PREFIX"
export CEED_DIR="$PREFIX"
export PKG_CONFIG_PATH="$PREFIX/lib/pkgconfig\${PKG_CONFIG_PATH:+:\$PKG_CONFIG_PATH}"
export LD_LIBRARY_PATH="$PREFIX/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
ENV
echo "built $MODE into $PREFIX"
