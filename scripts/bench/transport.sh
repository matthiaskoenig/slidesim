#!/usr/bin/env bash
# Transport benchmark: the baseline study of milestone 2 on the real field
# (testdata/studies/m2_real_slide/baseline.study.json) at REFINEMENTS refinements, on each process
# count of NPS, with PETSc's -log_view. Prints per run the steps (TSStep events of -log_view), the
# transport phase of summary.json, its seconds per step, and the other phases.
#
# Input: the scene of the end to end run of milestone 2 on the real slide, E2E/scene/scene.json.
# Without it, make it first (slides and slideviz required, see the script):
#   SLIDEVIZ_DATA=<slides> STEPS=scene scripts/e2e/m2_real_slide.sh
#
# Settings (environment variables, defaults in brackets):
#   E2E            output folder of scripts/e2e/m2_real_slide.sh [target/e2e/m2_real_slide]
#   OUT            output folder of the runs [E2E/bench/transport]
#   NPS            process counts [1 4 8]
#   REFINEMENTS    mesh refinements [1]
#   ADAPTIVE       1 for adaptive steps (time.adaptive), 0 for the fixed steps of the study [0]
#   MPIEXEC_FLAGS  extra mpiexec flags, e.g. --oversubscribe []
#   PETSC_ENV      env.sh of the PETSc/libCEED install [deps/install/native-opt/env.sh]
#   SLIDESIM       the slidesim binary, e.g. of an earlier commit [target/release/slidesim, built
#                  by the script]
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
E2E=$(realpath -m "${E2E:-$ROOT/target/e2e/m2_real_slide}")
OUT=$(realpath -m "${OUT:-$E2E/bench/transport}")
NPS=${NPS:-1 4 8}
REFINEMENTS=${REFINEMENTS:-1}
ADAPTIVE=${ADAPTIVE:-0}
MPIEXEC_FLAGS=${MPIEXEC_FLAGS:-}
PETSC_ENV=${PETSC_ENV:-$ROOT/deps/install/native-opt/env.sh}
SLIDESIM=${SLIDESIM:-}

scene=$E2E/scene/scene.json
if [[ ! -f $scene ]]; then
    echo "no scene $scene: make it with" >&2
    echo "  SLIDEVIZ_DATA=<slides> STEPS=scene $ROOT/scripts/e2e/m2_real_slide.sh" >&2
    exit 1
fi
# shellcheck source=/dev/null
source "$PETSC_ENV"
if [[ -z $SLIDESIM ]]; then
    (cd "$ROOT" && cargo build --quiet --release -p slidesim)
    SLIDESIM=$ROOT/target/release/slidesim
fi

mkdir -p "$OUT"
adaptive=$([[ $ADAPTIVE == 1 ]] && echo true || echo false)
study=$OUT/baseline_r${REFINEMENTS}_$([[ $ADAPTIVE == 1 ]] && echo adaptive || echo fixed).study.json
jq --arg scene "$scene" --argjson level "$REFINEMENTS" --argjson adaptive "$adaptive" \
    '.scene = $scene | .mesh.refinements = $level | .time.adaptive = $adaptive' \
    "$ROOT/testdata/studies/m2_real_slide/baseline.study.json" >"$study"

printf '%-9s %6s %12s %12s %9s %9s %9s\n' processes steps transport_s s_per_step setup_s \
    pressure_s output_s
for np in $NPS; do
    dir=$OUT/$(basename "$study" .study.json)_np$np
    mkdir -p "$dir"
    # shellcheck disable=SC2086 # MPIEXEC_FLAGS is a list of flags
    mpiexec -n "$np" $MPIEXEC_FLAGS "$SLIDESIM" run "$study" --out "$dir" \
        -- -log_view ":$dir/log_view.txt" >"$dir/run.log" 2>&1 ||
        { tail -20 "$dir/run.log" >&2; exit 1; }
    steps=$(awk '$1 == "TSStep" { n += $2 } END { print n + 0 }' "$dir/log_view.txt")
    jq -r --arg np "$np" --argjson steps "$steps" '.phases as $p
        | [$np, $steps, $p.transport, $p.transport / $steps, $p.setup, $p.pressure, $p.output]
        | "\(.[0]) \(.[1]) \(.[2]) \(.[3]) \(.[4]) \(.[5]) \(.[6])"' "$dir/summary.json" |
        awk '{ printf "%-9s %6d %12.2f %12.4f %9.2f %9.2f %9.2f\n", $1, $2, $3, $4, $5, $6, $7 }'
done
echo "logs: $OUT"
