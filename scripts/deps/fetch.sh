#!/usr/bin/env bash
# Shallow-clone the pinned PETSc and libCEED tags into deps/src (idempotent).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=../../deps/versions.env
. "$ROOT/deps/versions.env"

fetch() {
  local url="$1" tag="$2" dir="$3"
  if [ -d "$dir/.git" ] && [ "$(git -C "$dir" describe --tags --exact-match 2>/dev/null || true)" = "$tag" ]; then
    echo "$dir already at $tag"
    return
  fi
  rm -rf "$dir"
  git clone --quiet --depth 1 --branch "$tag" "$url" "$dir"
}

mkdir -p "$ROOT/deps/src"
fetch https://gitlab.com/petsc/petsc.git "$PETSC_VERSION" "$ROOT/deps/src/petsc"
fetch https://github.com/CEED/libCEED.git "$LIBCEED_VERSION" "$ROOT/deps/src/libceed"
