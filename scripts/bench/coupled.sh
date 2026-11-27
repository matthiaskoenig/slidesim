#!/usr/bin/env bash
# Coupled benchmark: the SPT studies of cells and fields on the batched path, shortened to END
# seconds, with PETSc's -log_view, and the load of the machine before and after each run. The
# default is the quick case (one coarse lobule, 1 h, a few seconds); the Krylov iterations per
# solve are the deterministic measure, the wall times depend on the load of the machine.
#
# - quick: the generated lobule on the scene's mesh, unrefined, 1 h,
#   testdata/studies/lobules/spt_quick.study.json, on 1 process;
# - lobule: the generated lobule, testdata/studies/lobules/spt_batched.study.json, on 1 process;
# - lobules7: the 7 lobules of testdata/studies/lobules/ring1_spt_batched.study.json, 1 process;
# - field: the real-slide field 281mg_m1, testdata/studies/slide_fields/281mg_m1, on NP
#   processes;
# - slide: the 281 mg/kg slide of milestone 3 (41,252 cells, 165,008 triangles after one
#   refinement; `spt_batched.study.json` beside `scene/` in SLIDE_DATA, made with the commands of
#   docs/results/2026-11-m3-wall-classes.md, "Reproduce") on NP processes.
#
# Every run goes through `run_sim` of scripts/lib/proc.sh: confined to the CPUs PROC_CPUS (0-9)
# at nice 10, with a wall-clock limit, and taken down with the script however it ends.
#
# Prints per run the phases of summary.json (s), the linear solves of the transport (KSPSolve
# events), their Krylov iterations (the sum over the solves of `-ksp_converged_reason`), the
# preconditioner setups (PCSetUp events of the outer solvers: the multigrid), the wall time of
# the run and the load average before and after (uptime). The run folders keep summary.json,
# log_view.txt and run.log.
#
# Settings (environment variables, defaults in brackets):
#   STUDIES        the studies to run: quick, lobule, lobules7, field, slide [quick]
#   SLIDE_DATA     folder of the 281 mg/kg study and scene [.superpowers/data/m3-baseline]
#   OUT            output folder of the runs [target/bench/coupled]
#   TAG            name of this variant, the prefix of the run folders [run]
#   NP             processes of the field and slide studies [4]
#   END            simulated time in s, the outputs hourly up to it [7200; quick: 3600, the
#                  outputs half-hourly]
#   LIMIT          wall-clock limit of each run, a `timeout` duration [5m]
#   PROC_CPUS      the CPUs of the runs and the build, for taskset [0-9]
#   EDIT           jq filter applied to the studies, e.g. '.solver.linear_rtol = 1e-8' [.]
#   PETSC_ARGS     extra PETSc options on the command line, e.g. '-S_ksp_rtol 1e-8' []
#   MPIEXEC_FLAGS  extra mpiexec flags, e.g. --oversubscribe []
#   PETSC_ENV      env.sh of the PETSc/libCEED install [deps/install/native-opt/env.sh]
#   SLIDESIM       the slidesim binary, e.g. of an earlier commit [target/release/slidesim, built
#                  by the script]
# shellcheck source=scripts/lib/proc.sh
source "$(dirname "${BASH_SOURCE[0]}")/../lib/proc.sh"
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
STUDIES=${STUDIES:-quick}
SLIDE_DATA=$(realpath -m "${SLIDE_DATA:-$ROOT/.superpowers/data/m3-baseline}")
OUT=$(realpath -m "${OUT:-$ROOT/target/bench/coupled}")
TAG=${TAG:-run}
NP=${NP:-4}
END_SET=${END:-}
END=${END:-7200}
LIMIT=${LIMIT:-5m}
EDIT=${EDIT:-.}
PETSC_ARGS=${PETSC_ARGS:-}
MPIEXEC_FLAGS=${MPIEXEC_FLAGS:-}
PETSC_ENV=${PETSC_ENV:-$ROOT/deps/install/native-opt/env.sh}
SLIDESIM=${SLIDESIM:-}

# shellcheck source=/dev/null
source "$PETSC_ENV"
if [[ -z $SLIDESIM ]]; then
    (cd "$ROOT" && taskset -c "$PROC_CPUS" nice -n 10 cargo build --quiet --release -j 10 -p slidesim)
    SLIDESIM=$ROOT/target/release/slidesim
fi
mkdir -p "$OUT"
model=$ROOT/testdata/models/spt_liver_cell.xml

# study <name>: writes the study of <name> into OUT and prints its path and process count.
study() {
    local source scene np
    case $1 in
    slide)
        source=$SLIDE_DATA/spt_batched.study.json
        scene=$SLIDE_DATA/scene/scene.json
        np=$NP
        ;;
    quick)
        source=$ROOT/testdata/studies/lobules/spt_quick.study.json
        scene=$ROOT/testdata/studies/lobules/scene.json
        np=1
        ;;
    lobule)
        source=$ROOT/testdata/studies/lobules/spt_batched.study.json
        scene=$ROOT/testdata/studies/lobules/scene.json
        np=1
        ;;
    lobules7)
        source=$ROOT/testdata/studies/lobules/ring1_spt_batched.study.json
        scene=$ROOT/testdata/studies/lobules/ring1.scene.json
        np=1
        ;;
    field)
        source=$ROOT/testdata/studies/slide_fields/281mg_m1/spt_batched.study.json
        scene=$ROOT/testdata/studies/slide_fields/281mg_m1/scene.json
        np=$NP
        ;;
    *)
        echo "unknown study $1 (quick, lobule, lobules7, field, slide)" >&2
        exit 1
        ;;
    esac
    [[ -f $source && -f $scene ]] || { echo "no $source or $scene" >&2; exit 1; }
    local path=$OUT/${TAG}_$1.study.json end=$END every=3600
    if [[ $1 == quick ]]; then
        end=${END_SET:-3600}
        every=1800
    fi
    jq --arg scene "$scene" --arg model "$model" --argjson end "$end" --argjson every "$every" \
        '.scene = $scene | .cells.model = $model | .time.end = $end
         | .time.outputs = [range(0; $end + 1; $every)]
         | if (.time.outputs | last) != $end then .time.outputs += [$end] else . end' \
        "$source" | jq "$EDIT" >"$path"
    echo "$path $np"
}

printf '%-16s %8s %8s %9s %9s %8s %8s %7s %8s %9s %7s %7s  %s\n' run setup pressure transport \
    exchange cells output solves iters/sol pc_setups wall_s load0 load1
for name in $STUDIES; do
    read -r path np < <(study "$name")
    dir=$OUT/${TAG}_$name
    rm -rf "$dir"
    mkdir -p "$dir"
    load0=$(uptime | sed 's/.*load average: //; s/,.*//')
    start=$(date +%s.%N)
    # The Krylov iterations of every solve of every field: -<field>_ksp_converged_reason.
    reasons=$(jq -r '[.fields[].id | "-\(.)_ksp_converged_reason"] | join(" ")' "$path")
    # shellcheck disable=SC2086 # MPIEXEC_FLAGS, PETSC_ARGS and reasons are lists
    run_sim "$LIMIT" mpiexec -n "$np" --bind-to none $MPIEXEC_FLAGS "$SLIDESIM" run "$path" \
        --out "$dir" -- -log_view ":$dir/log_view.txt" $reasons $PETSC_ARGS >"$dir/run.log" 2>&1 ||
        { echo "run $name failed (status $?, 124: over $LIMIT)" >&2; tail -20 "$dir/run.log" >&2; exit 1; }
    wall=$(echo "$(date +%s.%N) - $start" | bc)
    load1=$(uptime | sed 's/.*load average: //; s/,.*//')
    read -r solves iterations < <(awk '/solve converged due to/ { n++; s += $NF }
        END { print n + 0, s + 0 }' "$dir/run.log")
    # The outer solvers' setups: one PCSetUp event per setup of the multigrid of a field, and
    # nested setups of the levels; the count of the outermost is that of the KSPSetUp of the
    # TS's solvers, which -log_view does not separate, so the PCSetUp count is printed whole.
    pc_setups=$(awk '$1 == "PCSetUp" { print $2; exit }' "$dir/log_view.txt")
    jq -r '.phases | [.setup, .pressure, .transport, .exchange, .cells, .output]
        | map(tostring) | join(" ")' "$dir/summary.json" |
        awk -v run="${TAG}_$name" -v solves="$solves" -v iterations="$iterations" \
            -v setups="$pc_setups" -v wall="$wall" -v l0="$load0" -v l1="$load1" '{
            printf "%-16s %8.1f %8.1f %9.1f %9.1f %8.1f %8.1f %7d %8.2f %9d %7.0f %7s %7s\n",
                run, $1, $2, $3, $4, $5, $6, solves, (solves ? iterations / solves : 0),
                setups, wall, l0, l1 }'
done
echo "logs: $OUT"
