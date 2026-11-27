#!/usr/bin/env bash
# Milestone 2 end to end on a real slide (not part of CI: the slides are not in the repository).
#
# 1. slideviz: the zonation layers and the CYP2E1 readout of one H&E/CYP2E1 pair, the mesh inputs
#    of a square field, its polygons, the mesh (with portal and central wall tags) and the
#    slidesim scene (scene.json with the zonation map as Zarr).
# 2. slidesim: the studies of testdata/studies/m2_real_slide on that scene, on NP processes; the
#    baseline also at every level of BASELINE_REFINEMENTS.
# 3. Checks of each run's summary.json: every portal_wall flow > 0, every central_wall flow < 0,
#    their sum within 1e-8 of the largest flow.
# 4. Analysis (scripts/e2e/m2_real_slide/, in the uv environment of the slideviz checkout): the numbers
#    of every run in OUT/runs as OUT/analysis/<run>.json and the tables of the results document as
#    OUT/analysis/report.md (flows per wall piece, mass balance with the outflow through walls
#    without flow, velocities, tracer front), the window search OUT/analysis/windows.json, and the
#    figures into FIGURES. The figures take the H&E slide of SLIDEVIZ_DATA and the runs of the
#    studies at their refinements; the velocity distribution and the cases figure all four studies.
#
# Nothing is written into SLIDEVIZ_DATA. Settings (environment variables, defaults in brackets):
#   SLIDEVIZ_DATA        slides with their sidecars and predictions/ (required)
#   SLIDEVIZ_REPO        the slideviz checkout [~/git/slideviz]; one with the gmsh mesher
#                        build_mesh.py, retired in slideviz a5d287b (e.g. 98b30a7)
#   SLIDE                slide stem without the stain [mouse_apap_000mg_m1]
#   ANIMAL               prefix of the prediction files [SLIDE without "mouse_apap_"]
#   LUMEN, NECROSIS      prediction files [SLIDEVIZ_DATA/predictions/ANIMAL_{lumen,necrosis}.npz]
#   CENTRE_ROW, CENTRE_COL  field centre, H&E level 0 pixels [10114 16388]
#   FIELD_UM             field edge in um [1500]
#   MESH_SIZE_UM         element sizes SMALL:LARGE of build_mesh.py [5:20]
#   OUT                  output folder [target/e2e/m2_real_slide]
#   STUDIES              studies to run, may be empty [baseline calibrated whole_blood radial10]
#   BASELINE_REFINEMENTS extra refinement levels of the baseline, may be empty [0 1]
#   NP                   MPI processes [4]
#   MPIEXEC_FLAGS        extra mpiexec flags, e.g. --oversubscribe []
#   PETSC_ENV            env.sh of the PETSc/libCEED install [deps/install/native-opt/env.sh]
#   STEPS                steps to run, of: scene runs analysis [scene runs analysis]
#   FIGURES              folder of the figures [OUT/figures]; docs/results/2026-11-m2-real-slide
#                        for the results document
#   WINDOW_CENTRES       further window centres (row col ...) for the window search [14000 12000]
#   PIXI                 pixi for the mesh environment of slideviz [~/.pixi/bin/pixi]
set -euo pipefail
shopt -s nullglob

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
: "${SLIDEVIZ_DATA:?set SLIDEVIZ_DATA to the folder of the slides}"
SLIDEVIZ_REPO=${SLIDEVIZ_REPO:-$HOME/git/slideviz}
SLIDE=${SLIDE:-mouse_apap_000mg_m1}
ANIMAL=${ANIMAL:-${SLIDE#mouse_apap_}}
LUMEN=${LUMEN:-$SLIDEVIZ_DATA/predictions/${ANIMAL}_lumen.npz}
NECROSIS=${NECROSIS:-$SLIDEVIZ_DATA/predictions/${ANIMAL}_necrosis.npz}
CENTRE_ROW=${CENTRE_ROW:-10114}
CENTRE_COL=${CENTRE_COL:-16388}
FIELD_UM=${FIELD_UM:-1500}
MESH_SIZE_UM=${MESH_SIZE_UM:-5:20}
OUT=$(realpath -m "${OUT:-$ROOT/target/e2e/m2_real_slide}")
STUDIES=${STUDIES-baseline calibrated whole_blood radial10}
BASELINE_REFINEMENTS=${BASELINE_REFINEMENTS-0 1}
NP=${NP:-4}
MPIEXEC_FLAGS=${MPIEXEC_FLAGS:-}
PETSC_ENV=${PETSC_ENV:-$ROOT/deps/install/native-opt/env.sh}
STEPS=${STEPS:-scene runs analysis}
FIGURES=$(realpath -m "${FIGURES:-$OUT/figures}")
WINDOW_CENTRES=${WINDOW_CENTRES-14000 12000}
PIXI=${PIXI:-$HOME/.pixi/bin/pixi}

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

# The one existing path of the arguments (globs, brace expansions), an error otherwise. Call it
# in an assignment, `x=$(one ...)`, so that its error stops the script.
one() {
    local matches=() path
    for path in "$@"; do
        [[ -e $path ]] && matches+=("$path")
    done
    if [[ ${#matches[@]} -ne 1 ]]; then
        echo "expected one file, found: ${matches[*]}" >&2
        exit 1
    fi
    printf '%s\n' "${matches[0]}"
}

# A slideviz script in the uv environment of the checkout.
slideviz() { (cd "$SLIDEVIZ_REPO" && uv run --quiet python "$@"); }
# A slideviz mesh script in the pixi environment (gmsh, DOLFINx).
slideviz_mesh() { (cd "$SLIDEVIZ_REPO" && "$PIXI" run --manifest-path mesh/pixi.toml python "$@"); }

scene() {
    for file in "$LUMEN" "$NECROSIS"; do
        [[ -f $file ]] || { echo "missing $file" >&2; exit 1; }
    done
    # the slideviz scripts want new or empty folders
    rm -rf "${OUT:?}"/{zonation,readout,inputs,polygons,mesh,scene}
    mkdir -p "$OUT"

    log "zonation layers of ${SLIDE}_cyp2e1"
    slideviz scripts/zonation_layers.py --slides "$SLIDEVIZ_DATA" --out "$OUT/zonation" \
        --blocks "${SLIDE}_" --workers 1
    local lobules readout he inputs polygons mesh
    lobules=$(one "$OUT/zonation/${SLIDE}"_cyp2e1_lobules.npz)

    log "CYP2E1 readout"
    slideviz scripts/cyp2e1_readout.py --slides "$SLIDEVIZ_DATA" --out "$OUT/readout" \
        --blocks "${SLIDE}_" --workers 1
    readout=$(one "$OUT/readout/${SLIDE}"_cyp2e1_readout.npz)

    log "mesh inputs: centre (row, col) = ($CENTRE_ROW, $CENTRE_COL), field $FIELD_UM um"
    he=$(one "$SLIDEVIZ_DATA/${SLIDE}"_he.{zarr,ome.tiff})
    slideviz scripts/mesh_inputs.py --he "$he" --lumen "$LUMEN" --necrosis "$NECROSIS" \
        --cyp2e1 "$readout" --lobules "$lobules" \
        --centre "$CENTRE_ROW" "$CENTRE_COL" --size-um "$FIELD_UM" --out "$OUT/inputs"
    inputs=$(one "$OUT"/inputs/*_mesh_inputs.npz)

    log "polygons"
    slideviz_mesh mesh/polygons.py --inputs "$inputs" --tolerance-um 2 --out "$OUT/polygons"
    polygons=$(one "$OUT"/polygons/*.geojson)

    log "mesh ($MESH_SIZE_UM um)"
    slideviz_mesh mesh/build_mesh.py --polygons "$polygons" --variant plain \
        --size-um "$MESH_SIZE_UM" --out "$OUT/mesh"
    mesh=$(one "$OUT"/mesh/*.json)

    log "scene"
    slideviz scripts/write_scene.py --mesh "$mesh" --lobules "$lobules" --out "$OUT/scene"
}

# Checks the tag flows of a run: portal_wall > 0, central_wall < 0, sum ~ 0.
check_flows() {
    jq -e '.perfusion.tag_flows as $f
        | ([$f[]] | add) as $sum | ([$f[] | fabs] | max) as $max
        | $f.portal_wall > 0 and $f.central_wall < 0 and ($sum | fabs) <= 1e-8 * $max' \
        "$1/summary.json" >/dev/null || {
        echo "flow check failed for $1: $(jq -c .perfusion.tag_flows "$1/summary.json")" >&2
        exit 1
    }
}

# Runs one study at a refinement level: OUT/<name>_r<level>.study.json beside the scene,
# results in OUT/runs/<name>_r<level>/.
run() {
    local name=$1 level=$2
    local study="$OUT/${name}_r${level}.study.json" dir="$OUT/runs/${name}_r${level}"
    jq --argjson level "$level" '.mesh.refinements = $level' \
        "$ROOT/testdata/studies/m2_real_slide/$name.study.json" >"$study"
    mkdir -p "$dir"
    log "slidesim $name, $level refinements, $NP processes"
    local start=$SECONDS
    # shellcheck disable=SC2086 # MPIEXEC_FLAGS is a list of flags
    mpiexec -n "$NP" $MPIEXEC_FLAGS "$ROOT/target/release/slidesim" run "$study" --out "$dir" \
        >"$dir/run.log" 2>&1 || { tail -20 "$dir/run.log" >&2; exit 1; }
    echo "$((SECONDS - start))" >"$dir/seconds"
    check_flows "$dir"
    log "  $((SECONDS - start)) s; tag flows $(jq -c .perfusion.tag_flows "$dir/summary.json")"
}

runs() {
    [[ -f $OUT/scene/scene.json ]] || { echo "no scene in $OUT/scene; run the scene step" >&2; exit 1; }
    # shellcheck source=/dev/null
    source "$PETSC_ENV"
    (cd "$ROOT" && cargo build --quiet --release -p slidesim)
    local name level
    for name in $STUDIES; do
        level=$(jq .mesh.refinements "$ROOT/testdata/studies/m2_real_slide/$name.study.json")
        run "$name" "$level"
    done
    for level in $BASELINE_REFINEMENTS; do
        run baseline "$level"
    done
}

# A script of scripts/e2e/m2_real_slide in the uv environment of the slideviz checkout.
analysis_py() {
    local script=$1
    shift
    uv run --quiet --project "$SLIDEVIZ_REPO" python "$ROOT/scripts/e2e/m2_real_slide/$script" "$@"
}

# The output folder of a study at the refinements of its study file, and its study copy, as
# LABEL=RUN_DIR:STUDY.
study_run() {
    local level
    level=$(jq .mesh.refinements "$ROOT/testdata/studies/m2_real_slide/$1.study.json")
    printf '%s=%s:%s\n' "$1" "$OUT/runs/$1_r$level" "$OUT/$1_r$level.study.json"
}

analysis() {
    local mesh he dir name spec label rest centres=() index
    mesh=$(one "$OUT"/mesh/*.json)
    mkdir -p "$OUT/analysis" "$FIGURES"
    for dir in "$OUT"/runs/*/; do
        name=$(basename "$dir")
        log "analysis of $name"
        analysis_py analyze.py --run "$dir" --study "$OUT/$name.study.json" --mesh "$mesh" \
            --scene "$OUT/scene/scene.json" --out "$OUT/analysis/$name.json"
    done
    analysis_py report.py "$OUT"/analysis/*_r*.json >"$OUT/analysis/report.md"
    log "window search"
    # shellcheck disable=SC2206 # WINDOW_CENTRES is a list of numbers
    local extra=($WINDOW_CENTRES)
    for ((index = 0; index + 1 < ${#extra[@]}; index += 2)); do
        centres+=(--centre "${extra[index]}" "${extra[index + 1]}")
    done
    analysis_py windows.py --slides "$SLIDEVIZ_DATA" --slide "${SLIDE}_he" \
        --lobules "$(one "$OUT/zonation/${SLIDE}"_cyp2e1_lobules.npz)" --lumen "$LUMEN" \
        --centre "$CENTRE_ROW" "$CENTRE_COL" "${centres[@]}" --out "$OUT/analysis/windows.json"
    log "figures into $FIGURES"
    he="$OUT/analysis/he.npz"
    analysis_py he_crop.py --slides "$SLIDEVIZ_DATA" --slide "${SLIDE}_he" --mesh "$mesh" --out "$he"
    local runs=()
    for name in baseline calibrated whole_blood radial10; do
        spec=$(study_run "$name")
        rest=${spec#*=}
        [[ -f ${rest%%:*}/summary.json ]] && runs+=("$spec")
    done
    [[ ${#runs[@]} -gt 0 ]] || { echo "no runs of the studies for the figures" >&2; exit 1; }
    for spec in "${runs[@]}"; do
        label=${spec%%=*}
        rest=${spec#*=}
        case $label in
            baseline) analysis_py render.py --run "${rest%%:*}" --study "${rest#*:}" --mesh "$mesh" \
                --he "$he" --label baseline --out "$FIGURES" ;;
            radial10) analysis_py render.py --run "${rest%%:*}" --study "${rest#*:}" --mesh "$mesh" \
                --he "$he" --label radial10 --only-velocity --suffix _radial10 --out "$FIGURES" ;;
        esac
    done
    # the output index of t = 3 s
    index=$(jq '.time.outputs | index(3)' "$ROOT/testdata/studies/m2_real_slide/baseline.study.json")
    analysis_py cases.py --mesh "$mesh" --index "$index" --out "$FIGURES/sucrose_cases_3s.png" \
        "${runs[@]}"
    analysis_py velocity_distribution.py --mesh "$mesh" --out "$FIGURES/velocity_distribution.png" \
        "${runs[@]}"
}

for step in $STEPS; do
    case $step in
        scene) scene ;;
        runs) runs ;;
        analysis) analysis ;;
        *) echo "unknown step $step (scene, runs, analysis)" >&2; exit 1 ;;
    esac
done
log "done: $OUT"
