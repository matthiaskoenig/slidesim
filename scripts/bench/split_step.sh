#!/usr/bin/env bash
# Split-step convergence of the coupled SPT runs: the 1-lobule SPT study on the batched path at
# several split steps between cells and transport, each run against a reference at a smaller
# split step and transport step, and a short confirmation on the 7 lobules. The results and the
# recommendation are in docs/results/2026-11-m4-split-step.md.
#
# The studies (one process each, the transport step min(DT_MAX, split step)):
# - lobule: testdata/studies/lobules/spt_batched.study.json as it is (its mesh, its refinements),
#   shortened to END, at the split steps SPLITS and the reference REF_SPLIT with the transport
#   step REF_DT;
# - coarse: the same study on a coarser mesh of the lobule (the scene's wall and interior sizes
#   COARSE_SIZE, COARSE_REFINEMENTS refinements), so that its cells are fewer and the run long
#   enough for the necrosis, END_COARSE; the same split steps and reference;
# - lobules7: the 7 lobules of testdata/studies/lobules/ring1_spt_batched.study.json, shortened
#   to END7, at the split steps SPLITS7, compared with the last of them.
#
# Every run goes through `run_sim` of scripts/lib/proc.sh: confined to the CPUs PROC_CPUS (0-9)
# at nice 10, with a wall-clock limit, one run at a time, and taken down with the script however
# it ends.
#
# Prints per run the phases of summary.json (s), the steps of the cells, the wall time and the
# load average before and after (uptime), then per study the comparison of
# scripts/bench/split_step.py: the uptake (the amount `exchanged` per field), the cell states at
# the output times, the necrotic share and the Dice coefficient of the necrosis pattern against
# the reference, the observed orders and the cost. The run folders keep summary.json, the
# cells_<index>.vtu, run.log and wall.txt (the wall time and the loads).
#
# Settings (environment variables, defaults in brackets):
#   STUDIES             the studies to run: lobule, coarse, lobules7 [lobule coarse lobules7]
#   SPLITS              split steps of lobule and coarse (s) [120 60 30 15 10]
#   REF_SPLIT           split step of their reference (s) [5]
#   REF_DT              transport step of their reference (s) [2]
#   SPLITS7             split steps of lobules7 (s), the last one its reference [120 60 10]
#   DT_MAX              largest transport step (s): time.dt = min(DT_MAX, split step) [10]
#   END, EVERY          simulated time and output interval of lobule (s) [3600, 1800]
#   END_COARSE, EVERY_COARSE  the same of coarse (s) [14400, 3600]
#   END7, EVERY7        the same of lobules7 (s) [900, 300]
#   COARSE_SIZE         wall and interior size of the coarse mesh (um) [15 60]
#   COARSE_REFINEMENTS  refinements of the coarse mesh [0]
#   LIMIT               wall-clock limit of each run, a `timeout` duration [150s]
#   OUT                 output folder of the runs [target/bench/split_step]
#   REF_RUN             0: no reference runs of lobule and coarse, those in OUT are used [1]
#   COMPARE             1: only the comparison of the runs in OUT, no runs [0]
#   PROC_CPUS           the CPUs of the runs and the build, for taskset [0-9]
#   PETSC_ENV           env.sh of the PETSc/libCEED install [deps/install/native-opt/env.sh]
#   SLIDESIM            the slidesim binary [target/release/slidesim, built by the script]
# shellcheck source=scripts/lib/proc.sh
source "$(dirname "${BASH_SOURCE[0]}")/../lib/proc.sh"
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
STUDIES=${STUDIES:-lobule coarse lobules7}
SPLITS=${SPLITS:-120 60 30 15 10}
REF_SPLIT=${REF_SPLIT:-5}
REF_DT=${REF_DT:-2}
SPLITS7=${SPLITS7:-120 60 10}
DT_MAX=${DT_MAX:-10}
END=${END:-3600}
EVERY=${EVERY:-1800}
END_COARSE=${END_COARSE:-14400}
EVERY_COARSE=${EVERY_COARSE:-3600}
END7=${END7:-900}
EVERY7=${EVERY7:-300}
COARSE_SIZE=${COARSE_SIZE:-15 60}
COARSE_REFINEMENTS=${COARSE_REFINEMENTS:-0}
LIMIT=${LIMIT:-150s}
OUT=$(realpath -m "${OUT:-$ROOT/target/bench/split_step}")
REF_RUN=${REF_RUN:-1}
COMPARE=${COMPARE:-0}
PETSC_ENV=${PETSC_ENV:-$ROOT/deps/install/native-opt/env.sh}
SLIDESIM=${SLIDESIM:-}

lobules=$ROOT/testdata/studies/lobules
model=$ROOT/testdata/models/spt_liver_cell.xml
mkdir -p "$OUT"
read -r wall_um interior_um <<<"$COARSE_SIZE"
jq --argjson wall "$wall_um" --argjson interior "$interior_um" \
    '.mesh.size.wall_um = $wall | .mesh.size.interior_um = $interior' \
    "$lobules/scene.json" >"$OUT/coarse.scene.json"

# setting <study>: the study file, scene, simulated time, output interval and refinements of a
# study (refinements "keep": those of the study file).
setting() {
    case $1 in
    lobule) echo "$lobules/spt_batched.study.json $lobules/scene.json $END $EVERY keep" ;;
    coarse)
        echo "$lobules/spt_batched.study.json $OUT/coarse.scene.json $END_COARSE $EVERY_COARSE" \
            "$COARSE_REFINEMENTS"
        ;;
    lobules7) echo "$lobules/ring1_spt_batched.study.json $lobules/ring1.scene.json $END7 $EVERY7 keep" ;;
    *)
        echo "unknown study $1 (lobule, coarse, lobules7)" >&2
        exit 1
        ;;
    esac
}

# run <study> <name> <split> <dt>: one run into OUT/<name>, a line of the table.
run() {
    local source scene end every level dir=$OUT/$2 path=$OUT/$2.study.json
    read -r source scene end every level < <(setting "$1")
    jq --arg scene "$scene" --arg model "$model" --argjson split "$3" --argjson dt "$4" \
        --argjson end "$end" --argjson every "$every" --arg level "$level" \
        '.scene = $scene | .cells.model = $model | .cells.split_step = $split | .time.dt = $dt
         | .time.end = $end | .time.outputs = [range(0; $end + 1; $every)]
         | if (.time.outputs | last) != $end then .time.outputs += [$end] else . end
         | if $level == "keep" then . else .mesh.refinements = ($level | tonumber) end' \
        "$source" >"$path"
    rm -rf "$dir"
    mkdir -p "$dir"
    local load0 load1 start wall
    load0=$(uptime | sed 's/.*load average: //; s/,.*//')
    start=$(date +%s.%N)
    run_sim "$LIMIT" mpiexec -n 1 --bind-to none "$SLIDESIM" run "$path" --out "$dir" \
        >"$dir/run.log" 2>&1 ||
        { echo "run $2 failed (status $?, 124: over $LIMIT)" >&2; tail -20 "$dir/run.log" >&2; exit 1; }
    wall=$(echo "$(date +%s.%N) - $start" | bc)
    load1=$(uptime | sed 's/.*load average: //; s/,.*//')
    echo "$wall $load0 $load1" >"$dir/wall.txt"
    jq -r '[.phases.setup, .phases.pressure, .phases.transport, .phases.exchange, .phases.cells,
            .phases.output, .cells.count, .cells.steps] | map(tostring) | join(" ")' \
        "$dir/summary.json" |
        awk -v run="$2" -v sp="$3" -v dt="$4" -v wall="$wall" -v l0="$load0" -v l1="$load1" '{
            printf "%-18s %5s %3s %6.1f %8.1f %9.1f %8.1f %7.1f %6.1f %6d %11d %7.1f %6s %6s\n",
                run, sp, dt, $1, $2, $3, $4, $5, $6, $7, $8, wall, l0, l1 }'
}

# runs <study>: the run folders of a study, the reference first.
runs() {
    local split
    case $1 in
    lobules7)
        echo -n "$OUT/lobules7_s${SPLITS7##* }"
        for split in $SPLITS7; do echo -n " $OUT/lobules7_s$split"; done
        ;;
    *)
        echo -n "$OUT/$1_ref"
        for split in $SPLITS; do echo -n " $OUT/$1_s$split"; done
        ;;
    esac
}

if [[ $COMPARE != 1 ]]; then
    # shellcheck source=/dev/null
    source "$PETSC_ENV"
    if [[ -z $SLIDESIM ]]; then
        (cd "$ROOT" && taskset -c "$PROC_CPUS" nice -n 10 cargo build --quiet --release -j 10 -p slidesim)
        SLIDESIM=$ROOT/target/release/slidesim
    fi
    printf '%-18s %5s %3s %6s %8s %9s %8s %7s %6s %6s %11s %7s %6s %6s\n' run split dt setup \
        pressure transport exchange cells output count cell_steps wall_s load0 load1
    for study in $STUDIES; do
        setting "$study" >/dev/null
        case $study in
        lobules7) splits=$SPLITS7 ;;
        *) splits=$SPLITS ;;
        esac
        for split in $splits; do
            run "$study" "${study}_s$split" "$split" "$((split < DT_MAX ? split : DT_MAX))"
        done
        [[ $study == lobules7 || $REF_RUN == 0 ]] || run "$study" "${study}_ref" "$REF_SPLIT" "$REF_DT"
    done
fi

for study in $STUDIES; do
    echo
    echo "## $study"
    echo
    # shellcheck disable=SC2046 # a list of run folders
    python3 "$ROOT/scripts/bench/split_step.py" $(runs "$study")
done
echo "runs: $OUT"
