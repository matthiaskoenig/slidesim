#!/usr/bin/env bash
# Build PETSc (with PT-Scotch and Mmg) and libCEED (CPU backends) into deps/install/native-<mode>.
# Usage: build-native.sh <debug|check|opt>
# debug: -O0 -g with PETSc's debugging (gdb sessions); check: -O2 -g with PETSc's debugging checks, no
# -march=native (portable, so CI can cache it): the tests; opt: -O3 -march=native without debugging.
# JOBS (default: the number of CPUs) bounds the parallel builds.
set -euo pipefail

# Reproducible floating point: no FMA contraction, no fast-math (same numbers on every target).
FP="-ffp-contract=off -fno-fast-math"
MODE="${1:-}"
case "$MODE" in
  debug) DEBUGGING=1; BUILD_TYPE=Debug; FLAGS="-O0 -g $FP" ;;
  check) DEBUGGING=1; BUILD_TYPE=Debug; FLAGS="-O2 -g $FP" ;;
  opt) DEBUGGING=0; BUILD_TYPE=Release; FLAGS="-O3 -DNDEBUG -march=native $FP" ;;
  *) echo "usage: $0 <debug|check|opt>" >&2; exit 2 ;;
esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# `__FILE__` relative (PETSc's tracebacks name deps/build/petsc-<mode>/src/... as
# petsc-<mode>/src/...), as in the wasm build; the debug information keeps the absolute paths,
# which a debugger needs.
FLAGS+=" -fmacro-prefix-map=$ROOT/deps/build/="
# shellcheck source=SCRIPTDIR/../../deps/versions.env
. "$ROOT/deps/versions.env"
SRC="$ROOT/deps/src"
PREFIX="$ROOT/deps/install/native-$MODE"
JOBS="${JOBS:-$(nproc)}"
DEPS_SRC="$SRC"
DEPS_PATCHES="$ROOT/scripts/deps/patches"
# shellcheck source=SCRIPTDIR/lib.sh
. "$ROOT/scripts/deps/lib.sh"

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

# Mmg, the remesher of DMAdaptMetric, as a static library which libpetsc.so takes in (so
# position independent: Mmg's CMake means to make its static libraries so, but its test compares
# strings as numbers): built here rather than by PETSc's --download-mmg for its patches
# (scripts/deps/patches/README.md), configured as in build-wasi.sh. Compiled with the C flags of
# PETSc, by its compiler.
build_mmg native "$ROOT/deps/build/mmg-$MODE" "$PREFIX" "$BUILD_TYPE" "$FLAGS" \
  -DCMAKE_C_COMPILER=mpicc -DBUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON

# PETSc with its own BLAS/LAPACK, the PT-Scotch graph partitioner (which needs bison and flex
# to build) and Mmg, no C++ or Fortran. Every package comes from a pinned tarball in deps/src
# (fetch.sh), and the patches of every build are applied (scripts/deps/patches).
# PT-Scotch is compiled with PETSc's C flags, so with COPTFLAGS. Its CMake build would add threads
# (THREADS=ON by default, despite PETSc's -USCOTCH_PTHREAD), whose results may depend on the
# scheduling: THREADS=OFF, and SCOTCH_DETERMINISTIC=FULL for its deterministic algorithms with a
# fixed random seed. Then a partition depends on the graph and the number of processes only.
rm -rf "$ROOT/deps/build/petsc-$MODE"
cp -a "$ROOT/deps/src/petsc" "$ROOT/deps/build/petsc-$MODE"
apply_patches petsc "$ROOT/deps/build/petsc-$MODE" native
(
  cd "$ROOT/deps/build/petsc-$MODE"
  unset PETSC_DIR PETSC_ARCH
  export PATH="$FLEX/install/bin:$PATH"
  ./configure --prefix="$PREFIX" --with-cc=mpicc --with-cxx=0 --with-fc=0 \
    --download-f2cblaslapack="$SRC/f2cblaslapack-$F2CBLASLAPACK_VERSION.tar.gz" \
    --download-ptscotch="$SRC/scotch-v$PTSCOTCH_VERSION.tar.gz" \
    --download-ptscotch-cmake-arguments="-DTHREADS=OFF -DSCOTCH_DETERMINISTIC=FULL" \
    --download-bison="$SRC/bison-$BISON_VERSION.tar.gz" \
    --with-mmg-dir="$PREFIX" \
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

grep -q '^#define PETSC_HAVE_MMG 1' "$PREFIX/include/petscconf.h" ||
  { echo "error: PETSc in $PREFIX has no Mmg" >&2; exit 1; }

cat > "$PREFIX/env.sh" <<ENV
export PETSC_DIR="$PREFIX"
export CEED_DIR="$PREFIX"
export PKG_CONFIG_PATH="$PREFIX/lib/pkgconfig\${PKG_CONFIG_PATH:+:\$PKG_CONFIG_PATH}"
export LD_LIBRARY_PATH="$PREFIX/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
ENV
echo "built $MODE into $PREFIX"
