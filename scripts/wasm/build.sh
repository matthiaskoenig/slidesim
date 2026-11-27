#!/usr/bin/env bash
# Builds the browser build of slidesim and what its tests need:
#
# - target/wasm/slidesim.wasm: slidesim for wasm32-wasip1, a reactor (crates/slidesim-wasm),
#   linked against the wasi builds of PETSc, libCEED and Mmg in deps/install/wasi
#   (scripts/deps/build-wasi.sh) with wasi-sdk's wasm-ld and its wasi-libc;
# - deps/install/sbmlsim-js/: compiler.wasm and mathlib.wasm of sbmlsim-core at
#   SBMLSIM_CORE_KERNEL_REV (deps/versions.env), built from a clean checkout of that revision in
#   deps/src/sbmlsim-core, with its js/compile.mjs and js/kernel.mjs, and in
#   kernels/<model>/ the batched kernel (kernel.wasm, kernel.json; kernel/1) of each cell model
#   of KERNEL_MODELS, compiled with that compiler.wasm in Node as a page compiles it;
# - the native slidesim (target/release/slidesim, with deps/install/$NATIVE_DEPS, default
#   native-check), which the agreement tests of wasm/host run beside the wasm build;
# - target/wasm/artifacts.env: the paths of all of these, which wasm/host's tests read.
#
# Usage: scripts/wasm/build.sh
# Needs: wasi-sdk (WASI_SDK_PATH, default ~/opt/wasi-sdk), Node (the kernels), git and network
# for the first checkout of sbmlsim-core. The Cargo target directories lie under target/.
# CARGO_BUILD_JOBS bounds the parallel jobs of cargo.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=SCRIPTDIR/../../deps/versions.env
. "$ROOT/deps/versions.env"
NATIVE_DEPS="${NATIVE_DEPS:-native-check}"
KERNEL_MODELS=("$ROOT/testdata/models/spt_liver_cell.xml")
export WASI_SDK_PATH="${WASI_SDK_PATH:-$HOME/opt/wasi-sdk}"
[[ -x "$WASI_SDK_PATH/bin/clang" ]] || { echo "WASI_SDK_PATH ($WASI_SDK_PATH) has no bin/clang" >&2; exit 2; }
[[ -f "$ROOT/deps/install/wasi/env.sh" ]] || { echo "no deps/install/wasi: run scripts/deps/build-wasi.sh" >&2; exit 2; }

CORE="$ROOT/deps/src/sbmlsim-core"
JS="$ROOT/deps/install/sbmlsim-js"
OUT="$ROOT/target/wasm"
mkdir -p "$ROOT/deps/src" "$OUT"

# compiler.wasm and mathlib.wasm as sbmlsim-core's crates/host/build.rs builds them, from a
# clean checkout of the pinned revision (a shallow fetch of exactly that commit), unless
# deps/install/sbmlsim-js holds those of the revision.
if [[ "$(cat "$JS/REV" 2>/dev/null || true)" != "$SBMLSIM_CORE_KERNEL_REV" ]]; then
  if [[ "$(git -C "$CORE" rev-parse HEAD 2>/dev/null || true)" != "$SBMLSIM_CORE_KERNEL_REV" ]]; then
    rm -rf "$CORE"
    git init --quiet "$CORE"
    git -C "$CORE" fetch --quiet --depth 1 "$SBMLSIM_CORE_REPO" "$SBMLSIM_CORE_KERNEL_REV"
    git -C "$CORE" checkout --quiet --detach FETCH_HEAD
  fi
  rm -rf "$JS"
  (
    cd "$CORE"
    export CARGO_TARGET_DIR="$ROOT/target/sbmlsim-core"
    cargo build --quiet --locked -p sbmlsim-mathlib --lib --target wasm32-unknown-unknown --release
    cargo build --quiet --locked -p sbmlsim-wasm --lib --target wasm32-unknown-unknown --profile wasm
  )
  mkdir -p "$JS"
  cp "$ROOT/target/sbmlsim-core/wasm32-unknown-unknown/release/sbmlsim_mathlib.wasm" "$JS/mathlib.wasm"
  cp "$ROOT/target/sbmlsim-core/wasm32-unknown-unknown/wasm/sbmlsim_wasm.wasm" "$JS/compiler.wasm"
  cp "$CORE/js/compile.mjs" "$CORE/js/kernel.mjs" "$JS/"
  echo "$SBMLSIM_CORE_KERNEL_REV" > "$JS/REV"
fi

# The kernels of the cell models.
for model in "${KERNEL_MODELS[@]}"; do
  name="$(basename "$model" .xml)"
  node "$ROOT/scripts/wasm/compile_kernel.mjs" "$JS/compile.mjs" "$JS/compiler.wasm" "$model" \
    "$JS/kernels/$name"
done

# The native slidesim.
(
  # shellcheck source=/dev/null
  . "$ROOT/deps/install/$NATIVE_DEPS/env.sh"
  cd "$ROOT"
  cargo build --quiet --release -p slidesim
)

# slidesim.wasm: a reactor, linked by wasi-sdk's wasm-ld against its wasi-libc (the one PETSc,
# libCEED and Mmg were compiled against, with the emulation libraries), not against Rust's
# bundled one. wasm-ld itself, not clang: for a cdylib rustc gives clang `-shared`, which makes a
# module of position independent code. So the link line is the one clang gives a reactor
# (`-mexec-model=reactor`): wasi-libc's crt1-reactor.o, whose `_initialize` (exported) runs the
# constructors, libc and the builtins of compiler-rt.
(
  # shellcheck source=/dev/null
  . "$ROOT/deps/install/wasi/env.sh"
  sysroot="$WASI_SDK_PATH/share/wasi-sysroot"
  builtins="$("$WASI_SDK_PATH/bin/clang" -print-resource-dir)/lib/wasm32-unknown-wasip1"
  export CARGO_TARGET_WASM32_WASIP1_LINKER="$WASI_SDK_PATH/bin/wasm-ld"
  export CARGO_TARGET_WASM32_WASIP1_RUSTFLAGS="-C linker-flavor=wasm-ld -C link-self-contained=no \
    -C link-arg=-L$sysroot/lib/wasm32-wasip1 -C link-arg=$sysroot/lib/wasm32-wasip1/crt1-reactor.o \
    -C link-arg=--export=_initialize -C link-arg=-lc -C link-arg=$builtins/libclang_rt.builtins.a"
  export CARGO_TARGET_DIR="$OUT"
  cargo build --quiet --locked --release --target wasm32-wasip1 \
    --manifest-path "$ROOT/crates/slidesim-wasm/Cargo.toml"
)
cp "$OUT/wasm32-wasip1/release/slidesim_wasm.wasm" "$OUT/slidesim.wasm"

cat > "$OUT/artifacts.env" <<EOF
# Written by scripts/wasm/build.sh: the artifacts the tests of wasm/host use.
SLIDESIM_WASM=$OUT/slidesim.wasm
SLIDESIM_NATIVE=$ROOT/target/release/slidesim
SBMLSIM_MATHLIB=$JS/mathlib.wasm
SBMLSIM_KERNELS=$JS/kernels
EOF
echo "slidesim.wasm: $(stat -c %s "$OUT/slidesim.wasm") bytes; artifacts in $OUT/artifacts.env"
