# Milestone 3: portal and central classification on dosed slides

Status: 2026-11-26. slideviz branch `feat/cell-maps` (task 5 of the milestone 3 plan), slidesim
branch `feat/m3-cells`. Decision: the CYP2E1-based classification is used on all three dosed
slides (user's decision, below).

## Question

The portal and central walls of a slide's mesh come from slideviz's lumen classes
(`zonation_layers.py`, `lumen_class`):

- a lumen beside a CYP2E1 valley (a periportal minimum) is portal;
- a lumen beside a peak (a pericentral maximum) is central;
- a lumen beside both or neither is other.

On a dosed slide, pericentral necrosis could remove the CYP2E1 around a central vein and make it
look portal. The spec's check (portal and central check on dosed slides) says the classification
holds on a dosed slide when:

1. at most 1 in 10 lumina inside or within 50 um of the measured necrosis is portal, and
2. its portal/central ratio lies within the range of the controls.

## Method

`scripts/wall_class_check.py` of slideviz does the following per animal:

- **Lumina:** the connected pieces (4-neighbours) of `lumen_class` on the H&E working grid (about
  1.769 um/px). Each piece takes the class of its pixels. Every piece is at least 500 um^2
  (`LUMEN_MIN_UM2` of `tissue.py`), so a size filter would change nothing.
- **Necrosis:** `call_E` of `predictions/mouse_apap_<animal>_he_necrosis.npz`, the slide-named
  calls of `necrosis_slides.py` from 2026-11-22 (level 1, 0.442 um/px). The cutoff is 0.75 for
  281mg_m1 and 0.775 for the other five slides. Each working pixel takes the majority of its
  4 x 4 calls, as `mesh_inputs.py` and the scene's `necrosis` map do.
- **At the necrosis:** a lumen counts as at the necrosis when one of its pixel centres lies within
  50 um of a necrotic pixel centre, or on one.
- **Gate:** the controls are the 000mg animals. "1 in 10" is portal lumina over all lumina at the
  necrosis (portal, central and other). The table also gives portal over portal plus central.

**Inputs:** `SLIDEVIZ_DATA=/data/koeniglab/image.analysis/APAP/APAP_zarr`, read only. The lobules
files in `predictions/` predate `lumen_class`: they have no `position`, no `lumen_class` and no
working grid. So `zonation_layers.py` was run again for the six slides, with the milestone 2 code
(slideviz `7ddd651`), as `scripts/e2e/m2_real_slide.sh` does for one slide. The check ran at
slideviz `75f9a8d`.

## Results

| animal | necrotic % of measured | portal | central | other | portal/central | at necrosis: portal | central | other | portal share at necrosis | portal / (portal + central) at necrosis |
|---|---|---|---|---|---|---|---|---|---|---|
| 000mg_m1 | 0.01 | 59 | 29 | 83 | 2.03 | 0 | 0 | 2 | 0.000 | - |
| 000mg_m2 | 0.06 | 49 | 37 | 160 | 1.32 | 0 | 1 | 6 | 0.000 | 0.00 |
| 000mg_m3 | 0.02 | 37 | 6 | 78 | 6.17 | 1 | 0 | 2 | 0.333 | 1.00 |
| 281mg_m1 | 30.5 | 54 | 26 | 128 | 2.08 | 7 | 26 | 46 | 0.089 | 0.21 |
| 375mg_m1 | 51.8 | 20 | 23 | 111 | 0.87 | 7 | 23 | 81 | 0.063 | 0.23 |
| 500mg_m1 | 22.0 | 43 | 39 | 103 | 1.10 | 3 | 39 | 53 | 0.032 | 0.07 |

The controls' portal/central ratio runs from 1.32 (000mg_m2) to 6.17 (000mg_m3). In the controls
the necrosis is false calls covering under 0.1 % of the tissue, so their "at necrosis" counts are
a handful of lumina.

| dosed slide | ratio in the controls' range (spec, two-sided) | ratio not above the controls' highest (one-sided) | portal share at necrosis at most 0.1 |
|---|---|---|---|
| 281mg_m1 | yes (2.08) | yes | yes (7 of 79, 0.089) |
| 375mg_m1 | no (0.87 < 1.32) | yes | yes (7 of 111, 0.063) |
| 500mg_m1 | no (1.10 < 1.32) | yes | yes (3 of 95, 0.032) |

By the spec's two-sided reading, criterion 2 fails on 375mg_m1 and 500mg_m1. Milestone 3 stopped
at this point for the user's decision.

## Decision

The user accepts the CYP2E1-based classification on all three dosed slides. Criterion 2 is read
one-sided: a central vein flipped to portal raises the portal/central ratio, so only a ratio above
the controls' range is a sign of a flip. `wall_class_check.py` applies this reading:
`ratio_not_above` decides, and `ratio_in_range` reports the two-sided reading next to it. Under
this reading all three dosed slides pass. The evidence:

- **All central lumina sit at the necrosis:** 26 of 26 (281mg_m1), 23 of 23 (375mg_m1) and
  39 of 39 (500mg_m1), as pericentral necrosis around central veins should give.
- **Portal lumina sit at the necrosis less often than chance:**

    | dosed slide | share of the section within 50 um of necrotic pixels (chance) | portal lumina at the necrosis |
    |---|---|---|
    | 281mg_m1 | 57.0 % | 13 % (7 of 54) |
    | 375mg_m1 | 79.3 % | 35 % (7 of 20) |
    | 500mg_m1 | 46.8 % | 7 % (3 of 43) |

- **The ratios fail low, not high:** 375mg_m1 and 500mg_m1 have fewer portal lumina per central
  lumen than any control. The user's explanation is that fewer valleys are found in necrotic
  tissue. The valley counts measured here only partly fit it:

    | animal | valleys per mm^2 of section | peaks per mm^2 | valleys in necrosis | peaks in necrosis | section necrotic |
    |---|---|---|---|---|---|
    | 000mg_m1 | 6.15 | 6.57 | 0.0 % | 0.0 % | 0.0 % |
    | 000mg_m2 | 6.46 | 6.67 | 0.0 % | 0.0 % | 0.1 % |
    | 000mg_m3 | 6.89 | 7.08 | 0.6 % | 0.0 % | 0.0 % |
    | 281mg_m1 | 8.92 | 9.49 | 22.0 % | 38.7 % | 29.6 % |
    | 375mg_m1 | 7.32 | 7.55 | 36.4 % | 66.2 % | 50.3 % |
    | 500mg_m1 | 7.32 | 8.59 | 11.6 % | 52.0 % | 21.4 % |

    - Over the whole section, the dosed slides have more valleys per mm^2 than the controls, not
      fewer.
    - Inside the necrosis there are fewer valleys than its area would hold: 36 % of the valleys
      against 50 % of the section on 375mg_m1, and 12 % against 21 % on 500mg_m1.
    - Most of the valleys that do lie inside the necrosis are probably not periportal minima,
      because necrosis is pericentral.

    So the valley set changes where the tissue is necrotic. This check does not show how that
    lowers the number of portal lumina, and one slide per dose cannot settle it.

- **The ratio criterion is weak with three controls:** the controls' range is wide (1.32 to
  6.17), and 000mg_m3 has only 6 central lumina.
- **The stricter denominator:** with portal over portal plus central as the "1 in 10" (other
  lumina left out), 281mg_m1 (0.21) and 375mg_m1 (0.23) would fail criterion 1. The spec's
  wording counts all lumina at the necrosis, and the gate follows it.

### Options considered

The spec names a classification from the H&E as the candidate when the check fails. The options
put to the user were:

1. Accept the classification on the dosed slides on the evidence above, with the ratio criterion
   read as one-sided. **Chosen.**
2. Use 281mg_m1 only for the end-to-end study, since it passes both criteria in either reading.
3. Classify portal and central lumina from the H&E (portal triads: bile duct and artery beside
   the vein), independent of CYP2E1.

## Scene maps

`write_scene.py --cyp2e1 <readout.npz> --necrosis <calls.npz>` writes the `cyp2e1` and
`necrosis` maps beside `zonation`, on its grid and crop.

One 1500 um field was built per slide, each with a plain mesh at 5-20 um and the three-map
scene. Each field is the window at 97 % tissue or more with the most portal and central lumen
pixels, with two exceptions:

- For 000mg_m3, the field is centred on a central lumen, because the windows with the most
  lumen pixels held no central wall.
- For 375mg_m1, the second-ranked window by min(portal, central) pixels was used (see below).

| animal | centre (H&E level 0 px, row column) | triangles | portal / central wall pieces | cyp2e1 filled % | necrosis filled % | necrotic % of the necrosis map | `slidesim run` |
|---|---|---|---|---|---|---|---|
| 000mg_m1 | 10024 16568 | 42629 | 205 / 461 | 2.3 | 1.83 | 0.0 | runs |
| 000mg_m2 | 5480 18160 | 48463 | 210 / 490 | 5.08 | 3.37 | 0.0 | runs |
| 000mg_m3 | 6464 13000 | 26903 | 41 / 84 | 2.92 | 0.91 | 0.06 | runs |
| 281mg_m1 | 9336 30608 | 41252 | 140 / 594 | 4.25 | 3.72 | 34.35 | runs |
| 375mg_m1 | 14392 27728 | 40043 | 64 / 221 | 3.3 | 1.87 | 70.05 | runs |
| 500mg_m1 | 9384 13112 | 47546 | 280 / 515 | 4.56 | 4.66 | 26.45 | runs |

**Filled share:** the share of map pixels without a value of their own (outside the section,
lumina, no readout or no calls), filled from the nearest valid pixel. It is stored in the
`provenance` attribute of each map's Zarr array, together with the source file and its sha256.
`scene/1` has no field for it, so it is not in scene.json.

**Agreement with mesh_inputs:** on the pixels both cover, the two match:

- the `necrosis` map equals the `necrotic` layer of `mesh_inputs.py` (H&E level 3,
  `calls_on_level`) wherever that layer is measured;
- the `cyp2e1` map equals mesh_inputs' `cyp2e1` layer / 250 on the tissue, to 3e-8 (float32).

**slidesim run:** the milestone 2 baseline study (`testdata/studies/m2_real_slide/
baseline.study.json`, 0 refinements, 0.1 s, release binary at slidesim `c0dadc5`, 1 process)
loads and runs on all six three-map scenes. The portal and central wall flows balance in each run.

**Clockwise meshes:** before slideviz `8258575`, every triangle of five of these meshes came out
clockwise, and slidesim stopped with an inverted element. gmsh orients a surface's triangles like
its outer loop, and `polygons.py` gives the rings in either direction. `build_mesh.py` now orients
every polygon counterclockwise and stops if any triangle is still clockwise.

**Floating islands:** the first-ranked 375mg_m1 window (centre 10432 24128) has three pieces of
tissue with only `vessel_wall` edges and no pressure wall. On that field, slidesim crashes with a
segmentation fault in `Flow::build` (`KSPDestroy` / `PCReset_MG` / `VecDestroy_Seq`) instead of
reporting the ill-posed pressure problem. That is a slidesim error path, reported with the task.
The window used above has no such island.

## Reproduce

```bash
export SLIDEVIZ_DATA=/data/koeniglab/image.analysis/APAP/APAP_zarr  # read only
cd ~/git/slideviz
OUT=/tmp/wall-classes
uv run python scripts/zonation_layers.py --slides "$SLIDEVIZ_DATA" --out "$OUT/zonation" --workers 3 \
    --blocks mouse_apap_000mg_m1_ mouse_apap_000mg_m2_ mouse_apap_000mg_m3_ \
             mouse_apap_281mg_m1_ mouse_apap_375mg_m1_ mouse_apap_500mg_m1_
# reads predictions/<H&E stem>_necrosis.npz; JSON to stdout, the table to stderr
uv run python scripts/wall_class_check.py 000mg_m1 000mg_m2 000mg_m3 281mg_m1 375mg_m1 500mg_m1 \
    --lobules "$OUT/zonation" > "$OUT/check.json"
# a field scene with the three maps, e.g. 281mg_m1
P=$SLIDEVIZ_DATA/predictions; A=281mg_m1; S=mouse_apap_$A
uv run python scripts/mesh_inputs.py --he "$SLIDEVIZ_DATA/${S}_he.zarr" --lumen "$P/${A}_lumen.npz" \
    --necrosis "$P/${S}_he_necrosis.npz" --cyp2e1 "$P/${S}_cyp2e1_readout.npz" \
    --lobules "$OUT/zonation/${S}_cyp2e1_lobules.npz" --centre 9336 30608 --size-um 1500 --out "$OUT/$A/inputs"
~/.pixi/bin/pixi run --manifest-path mesh/pixi.toml python mesh/polygons.py \
    --inputs "$OUT/$A"/inputs/*_mesh_inputs.npz --tolerance-um 2 --out "$OUT/$A/polygons"
~/.pixi/bin/pixi run --manifest-path mesh/pixi.toml python mesh/build_mesh.py \
    --polygons "$OUT/$A"/polygons/*.geojson --variant plain --size-um 5:20 --out "$OUT/$A/mesh"
uv run python scripts/write_scene.py --mesh "$OUT/$A"/mesh/*.json \
    --lobules "$OUT/zonation/${S}_cyp2e1_lobules.npz" \
    --cyp2e1 "$P/${S}_cyp2e1_readout.npz" --necrosis "$P/${S}_he_necrosis.npz" --out "$OUT/$A/scene"
```
