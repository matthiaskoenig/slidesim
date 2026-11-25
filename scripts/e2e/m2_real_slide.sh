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
#
# Nothing is written into SLIDEVIZ_DATA. Settings (environment variables, defaults in brackets):
#   SLIDEVIZ_DATA        slides with their sidecars and predictions/ (required)
#   SLIDEVIZ_REPO        the slideviz checkout [~/git/slideviz]
#   SLIDE                slide stem without the stain [mouse_apap_000mg_m1]
#   ANIMAL               prefix of the prediction files [SLIDE without "mouse_apap_"]
#   LUMEN, NECROSIS      prediction files [SLIDEVIZ_DATA/predictions/ANIMAL_{lumen,necrosis}.npz]
#   CENTRE_ROW, CENTRE_COL  field centre, H&E level 0 pixels [10114 16388]
#   FIELD_UM             field edge in um [1500]
#   MESH_SIZE_UM         element sizes SMALL:LARGE of build_mesh.py [5:20]
#   OUT                  output folder [.superpowers/sdd/2026-11-25-slidesim-m2-perfusion/task-9-real]
#   STUDIES              studies to run, may be empty [baseline calibrated whole_blood radial10]
#   BASELINE_REFINEMENTS extra refinement levels of the baseline, may be empty [0 1]
#   NP                   MPI processes [4]
#   MPIEXEC_FLAGS        extra mpiexec flags, e.g. --oversubscribe []
#   PETSC_ENV            env.sh of the PETSc/libCEED install [deps/install/native-opt/env.sh]
#   STEPS                "scene runs" or one of them [scene runs]
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
OUT=${OUT:-$ROOT/.superpowers/sdd/2026-11-25-slidesim-m2-perfusion/task-9-real}
STUDIES=${STUDIES-baseline calibrated whole_blood radial10}
BASELINE_REFINEMENTS=${BASELINE_REFINEMENTS-0 1}
NP=${NP:-4}
MPIEXEC_FLAGS=${MPIEXEC_FLAGS:-}
PETSC_ENV=${PETSC_ENV:-$ROOT/deps/install/native-opt/env.sh}
STEPS=${STEPS:-scene runs}
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

for step in $STEPS; do
    case $step in
        scene) scene ;;
        runs) runs ;;
        *) echo "unknown step $step (scene, runs)" >&2; exit 1 ;;
    esac
done
log "done: $OUT"
