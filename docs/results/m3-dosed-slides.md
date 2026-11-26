# Milestone 3: cell models on dosed slides

<!--
Placeholder of the results document `2026-11-m3-dosed-slides.md`, which the end-to-end task of
milestone 3 writes. When it lands: point the nav entry "M3: cells on dosed slides" of
`zensical.toml` at `results/2026-11-m3-dosed-slides.md`, change the links to this page
(`docs/index.md`, `docs/results/index.md`, `docs/methods/cells.md`) to that file, and delete
this page.
-->

!!! info "Coming with the end of milestone 3"

    This page will hold the end-to-end results of milestone 3: 24 h of the SPT cell model on the
    dosed slides (281, 375 and 500 mg/kg), coupled to the flow of milestone 2, with the portal
    concentration chosen per slide so that the simulated necrotic share equals the measured one.

What the results document will contain, per the design of milestone 3:

- the overlap of simulated and measured necrosis (Dice coefficient), the necrotic fraction along
  the zonation position against the measured one, and the necrosis per lobule;
- both CYP2E1 inputs: the measured readout with the replacement rule in the necrotic areas, and
  pattern 6 everywhere;
- the agreement of the engine and the batched path on the necrosis map of one slide;
- rendered images of the fields and the necrosis;
- the performance per phase and the strong scaling series on 1 to 12 processes, both paths
  (completing the [performance results](2026-11-m3-performance.md)).

Until then: the [wall classes](2026-11-m3-wall-classes.md) of the dosed slides, on which the
study rests, and the coupled 24 h baseline on the 281 mg/kg slide in the
[performance results](2026-11-m3-performance.md#coupled-baseline-cells-and-fields-on-a-real-slide).
