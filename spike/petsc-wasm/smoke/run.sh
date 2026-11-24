#!/usr/bin/env bash
# Compile smoke.c against the static libraries of deps/install/wasi and run it in wasmtime and Node.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
# shellcheck source=/dev/null
source "$ROOT/deps/install/wasi/env.sh"
OUT="$ROOT/deps/build/smoke-wasi"
mkdir -p "$OUT"

"$WASI_SDK_PATH/bin/clang" --target=wasm32-wasip1 --sysroot="$WASI_SDK_PATH/share/wasi-sysroot" \
  -O3 -DNDEBUG -ffp-contract=off -fno-fast-math -msimd128 \
  -D_WASI_EMULATED_SIGNAL -D_WASI_EMULATED_PROCESS_CLOCKS -D_WASI_EMULATED_GETPID \
  -I"$PETSC_DIR/include" "$(dirname "${BASH_SOURCE[0]}")/smoke.c" $WASI_PETSC_CEED_LIBS -o "$OUT/smoke.wasm"
ls -l "$OUT/smoke.wasm"

# CG + Jacobi (50 iterations), LAPACK singular value estimate, wasm memory usage (petsc-0002).
OPTS="-ksp_type cg -pc_type jacobi -ksp_converged_reason -ksp_view_singularvalues -memory_view"

echo "== wasmtime"
wasmtime run "$OUT/smoke.wasm" $OPTS
echo "== node"
node --no-warnings "$(dirname "${BASH_SOURCE[0]}")/../run-node.mjs" "$OUT/smoke.wasm" $OPTS
