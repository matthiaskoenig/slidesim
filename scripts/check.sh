#!/usr/bin/env bash
# The checks before a push: what CI runs (.github/workflows/ci.yml) and what it leaves to a
# local run, against the native-check dependencies.
#
# 1. the format, the process hygiene of the scripts (scripts/tests/proc_kill.sh) and Clippy;
# 2. the full test suite (the default profile of .config/nextest.toml; CI runs the profile ci,
#    the fast tests) and the doctests;
# 3. the heavy quality sweep of the mesher (quality_sweep_heavy: rings 2 to 4, about 30 meshes,
#    ignored in the debug test run) in a release build;
# 4. ShellCheck and Ruff on the scripts, when uvx is there;
# 5. the browser build when its prerequisites are there (deps/install/wasi and wasi-sdk, see
#    docs/development.md): scripts/wasm/build.sh, the format and Clippy of its crates, the tests
#    of wasm/host and, when the packages of wasm/js are installed (npm ci --prefix wasm/js), its
#    type check and its Node and browser tests.
#
# Usage: scripts/check.sh [--ci]
#   --ci: only what the job check of CI runs (1 and the profile ci with the doctests).
# Every step runs through `run_sim` of scripts/lib/proc.sh: on the CPUs PROC_CPUS (default 0-9)
# at nice 10, with a wall-clock limit, and no process left behind when the script ends. The tests
# run on 4 threads (TEST_THREADS); the tests which start mpiexec run alone, on at most 3
# processes.
# shellcheck source=scripts/lib/proc.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib/proc.sh"
set -euo pipefail

CI_ONLY=0
case "${1:-}" in
  "") ;;
  --ci) CI_ONLY=1 ;;
  *) echo "usage: scripts/check.sh [--ci]" >&2; exit 2 ;;
esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
TEST_THREADS="${TEST_THREADS:-4}"
LIMIT="${LIMIT:-60m}"
STARTED=$SECONDS

# step <name> <cmd...>: the command through run_sim, with its name and wall time.
step() {
  local name=$1 start=$SECONDS
  shift
  echo "==> $name"
  run_sim "$LIMIT" "$@"
  echo "    $name: $((SECONDS - start)) s"
}

[[ -f deps/install/native-check/env.sh ]] ||
  { echo "no deps/install/native-check: run scripts/deps/build-native.sh check" >&2; exit 2; }
# shellcheck source=/dev/null
. deps/install/native-check/env.sh

step format cargo fmt --all --check
step "process hygiene" bash scripts/tests/proc_kill.sh
step clippy cargo clippy --workspace --all-targets -- -D warnings
if [[ "$CI_ONLY" == 1 ]]; then
  step "tests (profile ci)" cargo nextest run --workspace --profile ci \
    --test-threads "$TEST_THREADS"
else
  step "tests (full suite)" cargo nextest run --workspace --test-threads "$TEST_THREADS"
fi
step doctests cargo test --doc --workspace

if [[ "$CI_ONLY" == 0 ]]; then
  step "heavy mesher sweep" cargo test --release --locked -p mesher --test lobules -- \
    --ignored --exact quality_sweep_heavy

  if command -v uvx >/dev/null; then
    mapfile -t shell_scripts < <(git ls-files '*.sh')
    mapfile -t python_scripts < <(git ls-files '*.py')
    step shellcheck uvx --from shellcheck-py shellcheck -x "${shell_scripts[@]}"
    step ruff uvx ruff check "${python_scripts[@]}"
  else
    echo "==> shellcheck and ruff skipped: no uvx"
  fi

  WASI_SDK_PATH="${WASI_SDK_PATH:-$HOME/opt/wasi-sdk}"
  if [[ -f deps/install/wasi/env.sh && -x "$WASI_SDK_PATH/bin/clang" ]]; then
    export WASI_SDK_PATH
    step "browser build" bash scripts/wasm/build.sh
    step "format of the browser build" cargo fmt --manifest-path crates/slidesim-wasm/Cargo.toml \
      --check
    step "format of the host" cargo fmt --manifest-path wasm/host/Cargo.toml --check
    (
      # shellcheck source=/dev/null
      . deps/install/wasi/env.sh
      step "clippy of the browser crates" env CARGO_TARGET_DIR=target/wasm cargo clippy --locked \
        --target wasm32-wasip1 -p simulation -p cells -p maps --no-default-features --lib -- \
        -D warnings
      step "clippy of slidesim-wasm" env CARGO_TARGET_DIR=target/wasm cargo clippy --locked \
        --target wasm32-wasip1 --manifest-path crates/slidesim-wasm/Cargo.toml --release -- \
        -D warnings
    )
    step "clippy of the host" env CARGO_TARGET_DIR=target/wasm/host cargo clippy \
      --manifest-path wasm/host/Cargo.toml --all-targets --locked -- -D warnings
    step "host tests" env CARGO_TARGET_DIR=target/wasm/host cargo test \
      --manifest-path wasm/host/Cargo.toml --locked
    if [[ -d wasm/js/node_modules ]]; then
      step "type check of slidesim.d.ts" npm run typecheck --prefix wasm/js
      step "Node and browser tests" npm test --prefix wasm/js
    else
      echo "==> tests of wasm/js skipped: no wasm/js/node_modules (npm ci --prefix wasm/js)"
    fi
  else
    echo "==> browser build skipped: no deps/install/wasi or no wasi-sdk at $WASI_SDK_PATH"
  fi
fi

echo "all checks passed in $((SECONDS - STARTED)) s"
