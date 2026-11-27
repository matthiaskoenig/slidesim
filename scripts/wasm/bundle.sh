#!/usr/bin/env bash
# Assembles the lobule demo with the browser build into a folder a static server serves:
#
# - the page of docs/demo/lobule (index.html, demo.mjs, worker.mjs);
# - the JavaScript glue wasm/js/slidesim.mjs with sbmlsim-core's compile.mjs and kernel.mjs in
#   sbmlsim-core/;
# - slidesim.wasm, sbmlsim-core's compiler.wasm and mathlib.wasm;
# - the SBML of the SPT cell model (testdata/models/spt_liver_cell.xml).
#
# Usage: scripts/wasm/bundle.sh <out dir>
# Needs: scripts/wasm/build.sh (or build.sh --browser) first. The docs workflow bundles into
# site/demo/lobule after the site is built; locally a folder under target/ is ignored by git.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${1:?usage: scripts/wasm/bundle.sh <out dir>}"
JS="$ROOT/deps/install/sbmlsim-js"
files=(
  "$ROOT/target/wasm/slidesim.wasm"
  "$JS/compiler.wasm"
  "$JS/mathlib.wasm"
  "$ROOT/wasm/js/sbmlsim-core/compile.mjs"
  "$ROOT/wasm/js/sbmlsim-core/kernel.mjs"
)
for file in "${files[@]}"; do
  [[ -f "$file" ]] || { echo "no $file: run scripts/wasm/build.sh" >&2; exit 2; }
done

mkdir -p "$OUT/sbmlsim-core"
cp "$ROOT"/docs/demo/lobule/{index.html,demo.mjs,worker.mjs} "$OUT/"
cp "$ROOT/wasm/js/slidesim.mjs" "$OUT/"
cp "$ROOT/wasm/js/sbmlsim-core/compile.mjs" "$ROOT/wasm/js/sbmlsim-core/kernel.mjs" "$OUT/sbmlsim-core/"
cp "$ROOT/target/wasm/slidesim.wasm" "$JS/compiler.wasm" "$JS/mathlib.wasm" "$OUT/"
cp "$ROOT/testdata/models/spt_liver_cell.xml" "$OUT/"
echo "the lobule demo in $OUT ($(du -sh "$OUT" | cut -f1))"
