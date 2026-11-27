# Functions shared by build-native.sh and build-wasi.sh (sourced, not run).
# shellcheck shell=bash

# Applies the patches of scripts/deps/patches for the library `name` to its copy in `dir`, in
# the order of their numbers: `<name>-<NNNN>-*.patch` in every build, then, for the target
# `wasi`, `<name>-wasi-<NNNN>-*.patch` (scripts/deps/patches/README.md lists them).
apply_patches() { # <name> <dir> <native|wasi>
  local name="$1" dir="$2" target="$3" patches p
  patches=("$DEPS_PATCHES/$name"-[0-9]*.patch)
  if [[ "$target" == wasi ]]; then
    patches+=("$DEPS_PATCHES/$name"-wasi-[0-9]*.patch)
  fi
  for p in "${patches[@]}"; do
    [[ -e "$p" ]] || continue
    echo "applying $(basename "$p")"
    # The ceiling keeps git from taking slidesim's repository for that of a copy without its own
    # (Mmg from its tarball): the paths of the patch are relative to `dir`.
    GIT_CEILING_DIRECTORIES="$(dirname "$dir")" git -C "$dir" apply --whitespace=nowarn "$p"
  done
}

# Builds Mmg (deps/src/mmg-v$MMG_VERSION.tar.gz, with its patches for `target`) in `build` and
# installs it into `prefix`, with the C flags `flags` (CMake build type `type`) and the extra
# CMake arguments that follow. Configured the same on every target: without Scotch (2D Mmg never
# uses it, and PT-Scotch is not available in wasm), Elas and VTK, with the private headers PETSc's
# Mmg adaptor includes, without the Fortran headers (whose generator would run a wasm program).
build_mmg() { # <native|wasi> <build> <prefix> <Debug|Release> <flags> [cmake args...]
  local target="$1" build="$2" prefix="$3" type="$4" flags="$5"
  shift 5
  rm -rf "$build"
  mkdir -p "$build/src"
  tar xzf "$DEPS_SRC/mmg-v$MMG_VERSION.tar.gz" -C "$build/src" --strip-components=1
  apply_patches mmg "$build/src" "$target"
  # Mmg records the commit of the git repository around its sources: none (not slidesim's).
  GIT_CEILING_DIRECTORIES="$build" cmake -S "$build/src" -B "$build/build" \
    -DCMAKE_INSTALL_PREFIX="$prefix" -DCMAKE_INSTALL_LIBDIR=lib \
    -DCMAKE_BUILD_TYPE="$type" \
    -DCMAKE_C_FLAGS_"${type^^}"="$flags" -DCMAKE_CXX_FLAGS_"${type^^}"="$flags" \
    -DUSE_SCOTCH=OFF -DUSE_ELAS=OFF -DUSE_VTK=OFF -DMMG_INSTALL_PRIVATE_HEADERS=ON \
    -DCMAKE_DISABLE_FIND_PACKAGE_Perl=ON \
    "$@"
  GIT_CEILING_DIRECTORIES="$build" cmake --build "$build/build" -j"$JOBS"
  cmake --install "$build/build" >/dev/null
}
