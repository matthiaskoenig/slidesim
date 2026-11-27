#!/usr/bin/env bash
# Milestone 4 meshing on real slides (not part of CI: the slides are not in the repository).
#
# 1. fields: the real-slide example fields of testdata/studies/slide_fields. Per field the mesh
#    inputs of a 500 um square of slideviz (mesh_inputs.py), its tagged polygons (polygons.py,
#    variant follow) and the polygons scene with the zonation, cyp2e1 and necrosis maps
#    (write_scene.py --polygons), copied into OUT/slide_fields/<animal> in the layout of the
#    examples (scene.json, polygons.geojson, <map>.zarr).
# 2. compare: the 1500 um fields of milestones 2 (000mg_m1) and 3 (281mg_m1), meshed by gmsh
#    (slideviz build_mesh.py, variant plain, 5:20 um) and by slidesim from the polygons of
#    variant plain (5:20 um); `slidesim mesh` of both scenes, and the criteria of the comparison
#    as Markdown tables (m4_meshing/compare.py) in OUT/compare.md.
#
# Nothing is written into SLIDEVIZ_DATA. Settings (environment variables, defaults in brackets):
#   SLIDEVIZ_DATA   slides with their sidecars and predictions/ (required)
#   SLIDEVIZ_REPO   the slideviz checkout [~/git/slideviz]
#   OUT             output folder [target/e2e/m4_meshing]
#   STEPS           steps to run, of: fields compare [fields compare]
#   PIXI            pixi for the mesh environment of slideviz [~/.pixi/bin/pixi]
#   PETSC_ENV       env.sh of the PETSc/libCEED install [deps/install/native-opt/env.sh]
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
: "${SLIDEVIZ_DATA:?set SLIDEVIZ_DATA to the folder of the slides}"
SLIDEVIZ_REPO=${SLIDEVIZ_REPO:-$HOME/git/slideviz}
OUT=$(realpath -m "${OUT:-$ROOT/target/e2e/m4_meshing}")
STEPS=${STEPS:-fields compare}
PIXI=${PIXI:-$HOME/.pixi/bin/pixi}
PETSC_ENV=${PETSC_ENV:-$ROOT/deps/install/native-opt/env.sh}
P=$SLIDEVIZ_DATA/predictions
SLIDESIM=$ROOT/target/release/slidesim

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
slideviz() { (cd "$SLIDEVIZ_REPO" && uv run --quiet python "$@"); }
slideviz_mesh() { (cd "$SLIDEVIZ_REPO" && "$PIXI" run --manifest-path mesh/pixi.toml python "$@"); }

# The zonation layers of an animal (position and lumen classes on the working grid), once.
lobules() {
    local slide=mouse_apap_$1 file
    file=$OUT/zonation/${slide}_cyp2e1_lobules.npz
    if [[ ! -f $file ]]; then
        rm -rf "$OUT/zonation_$1"
        slideviz scripts/zonation_layers.py --slides "$SLIDEVIZ_DATA" --out "$OUT/zonation_$1" \
            --blocks "${slide}_" --workers 1 >&2
        mkdir -p "$OUT/zonation"
        mv "$OUT/zonation_$1/${slide}_cyp2e1_lobules.npz" "$file"
        rm -rf "$OUT/zonation_$1"
    fi
    printf '%s\n' "$file"
}

# One field of an animal: mesh inputs, polygons of a variant, the polygons scene with the maps.
# usage: field ANIMAL ROW COL SIZE_UM VARIANT DIR
field() {
    local animal=$1 row=$2 col=$3 size=$4 variant=$5 dir=$6 slide=mouse_apap_$1 lob
    lob=$(lobules "$animal")
    rm -rf "$dir"
    slideviz scripts/mesh_inputs.py --he "$SLIDEVIZ_DATA/${slide}_he.zarr" \
        --lumen "$P/${animal}_lumen.npz" --necrosis "$P/${slide}_he_necrosis.npz" \
        --cyp2e1 "$P/${slide}_cyp2e1_readout.npz" --lobules "$lob" \
        --centre "$row" "$col" --size-um "$size" --out "$dir/inputs" >/dev/null
    slideviz_mesh mesh/polygons.py --inputs "$dir/inputs/${animal}_mesh_inputs.npz" \
        --tolerance-um 2 --variant "$variant" --out "$dir/polygons" >/dev/null
    local polygons
    polygons=$(ls "$dir"/polygons/*.geojson)
    slideviz scripts/write_scene.py --polygons "$polygons" --size-um 5:20 --lobules "$lob" \
        --cyp2e1 "$P/${slide}_cyp2e1_readout.npz" --necrosis "$P/${slide}_he_necrosis.npz" \
        --out "$dir/scene" >/dev/null
}

fields() {
    local spec animal row col dst
    for spec in "000mg_m1 14120 15016" "281mg_m1 11656 28904" "500mg_m1 8072 15016"; do
        read -r animal row col <<<"$spec"
        log "field $animal at ($row, $col), 500 um"
        field "$animal" "$row" "$col" 500 follow "$OUT/fields/$animal"
        dst=$OUT/slide_fields/$animal
        rm -rf "$dst"
        mkdir -p "$dst"
        cp "$OUT/fields/$animal"/polygons/*.geojson "$dst/polygons.geojson"
        cp -r "$OUT/fields/$animal"/scene/maps/*.zarr "$dst/"
        jq '.mesh.path = "polygons.geojson" | .maps |= map(.path = (.id + ".zarr"))' \
            "$OUT/fields/$animal/scene/scene.json" >"$dst/scene.json"
        "$SLIDESIM" mesh "$dst/scene.json" --out "$OUT/fields/$animal/mesh"
    done
}

compare() {
    local spec animal row col dir record msh minsicn
    : >"$OUT/compare.md"
    for spec in "000mg_m1 10114 16388" "281mg_m1 9336 30608"; do
        read -r animal row col <<<"$spec"
        dir=$OUT/compare/$animal
        log "comparison on $animal at ($row, $col), 1500 um"
        field "$animal" "$row" "$col" 1500 follow "$dir/follow"
        field "$animal" "$row" "$col" 1500 plain "$dir/plain"
        # gmsh meshes the follow polygons, joined into one area (variant plain)
        slideviz_mesh mesh/build_mesh.py --polygons "$(ls "$dir"/follow/polygons/*.geojson)" \
            --variant plain --size-um 5:20 --out "$dir/gmsh" >/dev/null
        record=$(ls "$dir"/gmsh/*.json)
        slideviz scripts/write_scene.py --mesh "$record" --lobules "$(lobules "$animal")" \
            --out "$dir/gmsh_scene" >/dev/null
        "$SLIDESIM" mesh "$dir/gmsh_scene/scene.json" --out "$dir/gmsh_mesh"
        "$SLIDESIM" mesh "$dir/plain/scene/scene.json" --out "$dir/rust_mesh"
        msh=$(ls "$dir"/gmsh/*.msh)
        minsicn=$(slideviz_mesh "$ROOT/scripts/e2e/m4_meshing/gmsh_sicn.py" "$msh")
        uv run --quiet --project "$SLIDEVIZ_REPO" python "$ROOT/scripts/e2e/m4_meshing/compare.py" \
            "$animal" "$dir/gmsh_mesh" "$dir/rust_mesh" "$(ls "$dir"/plain/polygons/*.geojson)" \
            "$record" "$minsicn" >>"$OUT/compare.md"
    done
    cat "$OUT/compare.md"
}

mkdir -p "$OUT"
# shellcheck source=/dev/null
source "$PETSC_ENV"
(cd "$ROOT" && cargo build --quiet --release -p slidesim)
for step in $STEPS; do
    "$step"
done
