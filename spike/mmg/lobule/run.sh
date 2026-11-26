#!/usr/bin/env bash
# Build the lobule spike natively (deps/install/spike-mmg-native) and for wasm32-wasip1
# (deps/install/spike-mmg-wasi), run both fixtures twice natively, in wasmtime and in Node,
# compare the hashes and write the VTUs into target/runs.
#
# Usage: spike/mmg/lobule/run.sh [native|wasi|all] [program options, e.g. -dm_plex_metric_gradation_factor 1.2]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
export WASI_SDK_PATH="${WASI_SDK_PATH:-$HOME/opt/wasi-sdk}"
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-8}"
# OpenMPI's hwloc probes X displays (GL plugin); the spike needs no topology from it.
export HWLOC_COMPONENTS=-gl
WHICH="${1:-all}"
shift || true
cd "$HERE"

build_native() {
  (
    # shellcheck source=/dev/null
    source "$ROOT/deps/install/spike-mmg-native/env.sh"
    CARGO_TARGET_DIR="$HERE/target/native" cargo build --quiet --release
  )
}

build_wasm() {
  (
    # shellcheck source=/dev/null
    source "$ROOT/deps/install/spike-mmg-wasi/env.sh"
    # Link with wasi-sdk's clang and its wasi-libc (the one PETSc was compiled against, with
    # the emulation libraries) instead of Rust's bundled wasi-libc.
    export CARGO_TARGET_WASM32_WASIP1_LINKER="$WASI_SDK_PATH/bin/clang"
    export CARGO_TARGET_WASM32_WASIP1_RUSTFLAGS="-C link-self-contained=no \
      -C link-arg=--target=wasm32-wasip1 -C link-arg=--sysroot=$WASI_SDK_PATH/share/wasi-sysroot"
    CARGO_TARGET_DIR="$HERE/target/wasi" cargo build --quiet --release --target wasm32-wasip1
  )
}

OUT="$HERE/target/runs"
mkdir -p "$OUT"
run() { # <name> <command...>
  local name="$1"
  shift
  echo "== $name"
  "$@" >"$OUT/$name.txt"
  cat "$OUT/$name.txt"
}

hosts=()
if [[ "$WHICH" == native || "$WHICH" == all ]]; then
  build_native
  for fixture in lobule split; do
    for i in 1 2; do
      run "native-$fixture-$i" "$HERE/target/native/release/lobule" -fixture "$fixture" \
        -vtu "$OUT/native-$fixture.vtu" "$@"
    done
  done
  hosts+=(native)
fi
if [[ "$WHICH" == wasi || "$WHICH" == all ]]; then
  build_wasm
  WASM="$HERE/target/wasi/wasm32-wasip1/release/lobule.wasm"
  wasmtime compile -o "$OUT/lobule.cwasm" "$WASM"
  for fixture in lobule split; do
    for i in 1 2; do
      run "wasmtime-$fixture-$i" wasmtime run --dir "$OUT::/out" --allow-precompiled \
        "$OUT/lobule.cwasm" -fixture "$fixture" -vtu "/out/wasmtime-$fixture.vtu" "$@"
      run "node-$fixture-$i" node --no-warnings "$HERE/../run-node.mjs" "$OUT" "$WASM" \
        -fixture "$fixture" -vtu "/out/node-$fixture.vtu" "$@"
    done
  done
  hosts+=(wasmtime node)
  echo "== wasm size"
  STRIPPED="$OUT/lobule.stripped.wasm"
  "$WASI_SDK_PATH/bin/llvm-strip" -o "$STRIPPED" "$WASM"
  for f in "$WASM" "$STRIPPED"; do
    printf "%-24s raw %9d  gzip -9 %9d\n" "$(basename "$f")" "$(stat -c %s "$f")" "$(gzip -9c "$f" | wc -c)"
  done
fi

echo "== hashes (G4: run 1 == run 2 on each host)"
for fixture in lobule split; do
  for host in "${hosts[@]}"; do
    h1="$(awk '/^hash/ { print $2 }' "$OUT/$host-$fixture-1.txt")"
    h2="$(awk '/^hash/ { print $2 }' "$OUT/$host-$fixture-2.txt")"
    printf "%-8s %-9s %s %s %s\n" "$fixture" "$host" "$h1" "$h2" "$([[ "$h1" == "$h2" ]] && echo same || echo DIFFERENT)"
  done
done
