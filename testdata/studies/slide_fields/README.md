Three fields of real mouse liver slides, as slidesim meshes and simulates them: the tagged
polygons of slideviz around a central vein with portal fields, with the zonation, CYP2E1 and
necrosis maps of the field. One control and two dosed animals of the APAP study, run end to end
by the tests of `crates/slidesim` (`slide_field_examples_mesh`, `slide_field_spt_runs`).

## License and credit

The data in this folder (the polygons and the maps) is licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Credit: Matthias König lab, data from
the APAP mouse liver study.

## Fields

Each field is a square of 500 um edge (282 pixels of 1.7685 um on H&E level 3) centred on a
point of the H&E slide (level-0 pixels, row and column), cut by slideviz `mesh_inputs.py`:

| Field | Slide | Dose | Centre (row, col) | Lumina | Necrotic share | Size on disk |
| --- | --- | --- | --- | --- | --- | --- |
| `000mg_m1` | `mouse_apap_000mg_m1_he` | control | (14120, 15016) | 3 | 0 % | 436 KB |
| `281mg_m1` | `mouse_apap_281mg_m1_he` | 281 mg/kg | (11656, 28904) | 4 | 52 % | 516 KB |
| `500mg_m1` | `mouse_apap_500mg_m1_he` | 500 mg/kg | (8072, 15016) | 5 | 43 % | 528 KB |

Each folder holds:

- `polygons.geojson`: the tagged polygons (`polygons/1`) of slideviz `mesh/polygons.py`, variant
  `follow`: the areas `surviving` and `necrotic` (the necrosis map, 2 um tolerance) and the edge
  chains `portal_wall`, `central_wall`, `vessel_wall`, `necrosis_border`, `section_edge` and
  `field_cut` (those the field has). Coordinates in um on the H&E level-0 frame, x along the
  columns, y along the rows.
- `scene.json`: the `scene/1` of slideviz `write_scene.py --polygons --size-um 5:20` (5 um
  triangles at the walls, 20 um far from them), with the paths of this folder.
- `zonation.zarr`, `cyp2e1.zarr`, `necrosis.zarr`: the maps, cropped to the field: the zonation
  position (0 periportal, 1 pericentral), the CYP2E1 readout (0 to 1) and the necrosis calls
  (1 necrotic, 0 not), on the working grid of `zonation_layers.py`.
- `spt.study.json`, `spt_batched.study.json`: the SPT cell model for 24 h on the engine and on
  the batched path: `../m3_real_slide/spt.study.json`, adapted by hand (the scene and model
  paths; `"solver": "batched"` in the second).

The provenance attributes of the polygons (`meta`) and of the maps name the files they were made
from, with their SHA-256: `$SLIDEVIZ_DATA/...` relative to the slide folder (below),
`$OUT/...` relative to the output folder of `scripts/e2e/m4_meshing.sh`.

## Provenance

Made with slideviz commit `98b30a7` (branch `feat/tagged-polygons`: `zonation_layers.py`,
`mesh_inputs.py`, `mesh/polygons.py`, `write_scene.py`) and slidesim's
`scripts/e2e/m4_meshing.sh` as of the commit which last changed this folder (branch
`feat/m4-lobules`, after `b0c84fa`), from the slides, the lumen masks
`predictions/<animal>_lumen.npz`, the necrosis calls
`predictions/mouse_apap_<animal>_he_necrosis.npz` and the CYP2E1 readouts
`predictions/mouse_apap_<animal>_cyp2e1_readout.npz`, by:

```bash
export SLIDEVIZ_DATA=/data/koeniglab/image.analysis/APAP/APAP_zarr  # read only
STEPS=fields scripts/e2e/m4_meshing.sh  # into target/e2e/m4_meshing/slide_fields/
```

which runs per field `zonation_layers.py`, `mesh_inputs.py --centre ROW COL --size-um 500`,
`mesh/polygons.py --tolerance-um 2` and `write_scene.py --polygons ... --size-um 5:20` of slideviz,
then moves the maps beside the scene. The fields were picked as squares fully in the tissue
holding a central lumen of at least 500 um^2 near the centre and as many portal lumina as found
(the portal lumina of the lumen masks are few: 20 to 31 per section).

## Meshes

`slidesim mesh scene.json --out <dir>` (one process):

| Field | Triangles | Min angle | Min SICN | Mean SICN | Edge length / size |
| --- | --- | --- | --- | --- | --- |
| `000mg_m1` | 6106 | 23.24 deg | 0.547 | 0.955 | 0.46 to 1.53 |
| `281mg_m1` | 13791 | 23.03 deg | 0.498 | 0.948 | 0.43 to 1.63 |
| `500mg_m1` | 17175 | 22.44 deg | 0.541 | 0.944 | 0.41 to 1.67 |

All three meet the quality bounds of the mesher (20 deg, 0.4, [0.4, 2.0]): the polygons keep
edges of one pixel (1.77 um) of the masks, and the size field is capped at 1.5 times the length
of such a segment around its ends, see
[the milestone 4 meshing results](../../../docs/results/2026-11-m4-meshing.md).
