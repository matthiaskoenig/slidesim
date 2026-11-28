#!/usr/bin/env bash
# Compile smoke.c against deps/install/wasi (scripts/deps/build-wasi.sh) and run it in wasmtime.
# Fails unless it succeeds with its one line on stdout and nothing on stderr: Mmg prints nothing
# (scripts/deps/patches/mmg-0001-memory-size-on-wasm.patch).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
# shellcheck source=/dev/null
. "$ROOT/deps/install/wasi/env.sh"
OUT="$ROOT/deps/build/wasi-smoke"
mkdir -p "$OUT"

# shellcheck disable=SC2086 # WASI_PETSC_CEED_LIBS is a list of flags
"$WASI_SDK_PATH/bin/clang" --target=wasm32-wasip1 --sysroot="$WASI_SDK_PATH/share/wasi-sysroot" \
  -O3 -DNDEBUG -ffp-contract=off -fno-fast-math -msimd128 \
  -D_WASI_EMULATED_SIGNAL -D_WASI_EMULATED_PROCESS_CLOCKS -D_WASI_EMULATED_GETPID \
  -I"$PETSC_DIR/include" "$HERE/smoke.c" $WASI_PETSC_CEED_LIBS -o "$OUT/smoke.wasm"

status=0
wasmtime run "$OUT/smoke.wasm" >"$OUT/stdout" 2>"$OUT/stderr" || status=$?
cat "$OUT/stdout"
cat "$OUT/stderr" >&2
if [[ $status -ne 0 ]]; then
  echo "error: the smoke test failed (exit $status)" >&2
  exit 1
fi
if [[ -s "$OUT/stderr" || "$(wc -l <"$OUT/stdout")" -ne 1 ]] ||
  ! grep -qE '^adapted cells [0-9]+$' "$OUT/stdout"; then
  echo "error: output other than the line of the smoke test (Mmg prints)" >&2
  exit 1
fi
echo "ok"
