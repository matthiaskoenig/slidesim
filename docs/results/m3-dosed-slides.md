# Cells on dosed slides (deferred)

<!--
Placeholder of the results document of the runs on the dosed slides, deferred after milestone 3
to larger hardware. When it lands: point the nav entry "Cells on dosed slides (deferred)" of
`zensical.toml` at the new document, change the links to this page (`docs/index.md`,
`docs/results/index.md`, `docs/methods/cells.md`) to that file, and delete this page.
-->

!!! info "Deferred to later, larger hardware"

    The runs on the dosed slides are not part of milestone 3: 24 h of the SPT cell model on the
    dosed slides (281, 375 and 500 mg/kg), coupled to the flow of milestone 2, with the portal
    concentration chosen per slide so that the simulated necrotic share equals the measured one.
    The calibration repeats the 24 h run many times per slide, so it waits for larger hardware.
    Milestone 3 delivers the cell models, both cell paths and their performance, which these
    runs build on.

What the results document will contain, per the design of milestone 3:

- the overlap of simulated and measured necrosis (Dice coefficient), the necrotic fraction along
  the zonation position against the measured one, and the necrosis per lobule;
- both CYP2E1 inputs: the measured readout with the replacement rule in the necrotic areas, and
  pattern 6 everywhere;
- the agreement of the engine and the batched path on the necrosis map of one slide;
- rendered images of the fields and the necrosis;
- the performance per phase and the strong scaling series on 1 to 12 processes, both paths
  (completing the [performance results](2026-11-m3-performance.md)).

Until then, from milestone 3: the [wall classes](2026-11-m3-wall-classes.md) of the dosed slides, on which the
study rests, and the coupled 24 h baseline on the 281 mg/kg slide in the
[performance results](2026-11-m3-performance.md#coupled-baseline-cells-and-fields-on-a-real-slide).
