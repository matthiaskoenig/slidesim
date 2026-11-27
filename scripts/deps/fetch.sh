#!/usr/bin/env bash
# Fetch the pinned sources into deps/src (idempotent): shallow clones of the PETSc and libCEED
# tags, and the tarballs of the dependency builds, each checked against its sha256.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=SCRIPTDIR/../../deps/versions.env
. "$ROOT/deps/versions.env"
SRC="$ROOT/deps/src"

fetch() {
  local url="$1" tag="$2" dir="$3"
  if [ -d "$dir/.git" ] && [ "$(git -C "$dir" describe --tags --exact-match 2>/dev/null || true)" = "$tag" ]; then
    echo "$dir already at $tag"
    return
  fi
  rm -rf "$dir"
  git clone --quiet --depth 1 --branch "$tag" "$url" "$dir"
}

# The tarball at `url` as deps/src/<file> (default: the file name of the URL), unless it is
# there; fails (and removes it) unless its sha256 is `sha256`.
fetch_tarball() {
  local url="$1" sha256="$2" file
  file="$SRC/${3:-$(basename "$url")}"
  if [ ! -f "$file" ]; then
    curl -fsSL -o "$file.part" "$url"
    mv "$file.part" "$file"
  fi
  if ! echo "$sha256  $file" | sha256sum --check --quiet; then
    rm -f "$file"
    echo "$file does not have the sha256 $sha256 pinned in deps/versions.env" >&2
    exit 1
  fi
}

# Fails unless the PETSc package file `package` (in config/BuildSystem/config/packages) matches
# the extended regular expression `pattern`: the pinned tarball is the one PETSc downloads.
check_petsc_package() {
  local package="$1" pattern="$2"
  grep -qE "$pattern" "$SRC/petsc/config/BuildSystem/config/packages/$package" || {
    echo "PETSc $PETSC_VERSION ($package) does not download the version pinned in deps/versions.env" >&2
    exit 1
  }
}

mkdir -p "$SRC"
fetch https://gitlab.com/petsc/petsc.git "$PETSC_VERSION" "$SRC/petsc"
fetch https://github.com/CEED/libCEED.git "$LIBCEED_VERSION" "$SRC/libceed"

check_petsc_package f2cblaslapack.py "f2cblaslapack-${F2CBLASLAPACK_VERSION//./\\.}\.tar\.gz"
check_petsc_package PTSCOTCH.py "self\.version *= *'${PTSCOTCH_VERSION//./\\.}'"
check_petsc_package Bison.py "self\.version *= *'${BISON_VERSION//./\\.}'"
check_petsc_package Mmg.py "self\.version *= *'${MMG_VERSION//./\\.}'"
fetch_tarball "https://web.cels.anl.gov/projects/petsc/download/externalpackages/f2cblaslapack-$F2CBLASLAPACK_VERSION.tar.gz" "$F2CBLASLAPACK_SHA256"
fetch_tarball "https://web.cels.anl.gov/projects/petsc/download/externalpackages/scotch-v$PTSCOTCH_VERSION.tar.gz" "$PTSCOTCH_SHA256"
fetch_tarball "https://ftp.gnu.org/gnu/bison/bison-$BISON_VERSION.tar.gz" "$BISON_SHA256"
fetch_tarball "https://github.com/MmgTools/mmg/archive/v$MMG_VERSION.tar.gz" "$MMG_SHA256" "mmg-v$MMG_VERSION.tar.gz"
fetch_tarball "https://github.com/westes/flex/releases/download/v$FLEX_VERSION/flex-$FLEX_VERSION.tar.gz" "$FLEX_SHA256"
