#!/usr/bin/env bash
# Build the diffusion spike natively (deps/install/native-opt and native-debug) and for
# wasm32-wasip1 (deps/install/wasi), run it natively, in wasmtime and in Node, and compare.
#
# Usage: spike/petsc-wasm/diffusion/run.sh [program options, e.g. -n 32 -rust_libm]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
export WASI_SDK_PATH="${WASI_SDK_PATH:-$HOME/opt/wasi-sdk}"
# OpenMPI's hwloc probes X displays (GL plugin); without an X authorization that prints
# "Authorization required" eight times. The spike needs no topology from it.
export HWLOC_COMPONENTS=-gl
cd "$HERE"

build_native() { # <config>
  (
    # shellcheck source=/dev/null
    source "$ROOT/deps/install/$1/env.sh"
    CARGO_TARGET_DIR="$HERE/target/$1" cargo build --quiet --release
  )
}

build_wasm() {
  (
    # shellcheck source=/dev/null
    source "$ROOT/deps/install/wasi/env.sh"
    # Link with wasi-sdk's clang and its wasi-libc (the one PETSc and libCEED were compiled
    # against, with the emulation libraries) instead of Rust's bundled wasi-libc.
    export CARGO_TARGET_WASM32_WASIP1_LINKER="$WASI_SDK_PATH/bin/clang"
    export CARGO_TARGET_WASM32_WASIP1_RUSTFLAGS="-C link-self-contained=no \
      -C link-arg=--target=wasm32-wasip1 -C link-arg=--sysroot=$WASI_SDK_PATH/share/wasi-sysroot"
    CARGO_TARGET_DIR="$HERE/target/wasi" cargo build --quiet --release --target wasm32-wasip1
  )
}

build_native native-opt
build_native native-debug
build_wasm
WASM="$HERE/target/wasi/wasm32-wasip1/release/diffusion.wasm"

OUT="$HERE/target/runs"
mkdir -p "$OUT"
run() { # <name> <command...>
  local name="$1"
  shift
  echo "== $name"
  local t0 t1
  t0=$(date +%s%N)
  "$@" > "$OUT/$name.txt"
  t1=$(date +%s%N)
  cat "$OUT/$name.txt"
  echo "wall (process)       $(((t1 - t0) / 1000000)) ms"
}

run native-opt "$HERE/target/native-opt/release/diffusion" "$@"
run native-debug "$HERE/target/native-debug/release/diffusion" "$@"
wasmtime compile -o "$OUT/diffusion.cwasm" "$WASM"
run wasmtime wasmtime run --allow-precompiled "$OUT/diffusion.cwasm" "$@"
run node node --no-warnings "$HERE/../run-node.mjs" "$WASM" "$@"

echo "== comparison (relative difference of the solution 2-norm to native-opt, go: < 1e-12)"
norm() { awk '/solution 2-norm/ { print $3 }' "$OUT/$1.txt"; }
for host in native-debug wasmtime node; do
  awk -v a="$(norm native-opt)" -v b="$(norm "$host")" -v h="$host" \
    'BEGIN { d = (a - b) / a; if (d < 0) d = -d; printf "%-20s %.3e\n", h, d; exit !(d < 1e-12) }'
done

echo "== wasm size"
STRIPPED="$OUT/diffusion.stripped.wasm"
"$WASI_SDK_PATH/bin/llvm-strip" -o "$STRIPPED" "$WASM"
for f in "$WASM" "$STRIPPED"; do
  printf "%-28s raw %9d  gzip -9 %9d\n" "$(basename "$f")" "$(stat -c %s "$f")" "$(gzip -9c "$f" | wc -c)"
done
