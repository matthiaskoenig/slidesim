# slidesim milestone 3 - cell models coupled to perfusion - design

Status: draft for review, 2026-11-25.

Parent design: `docs/superpowers/specs/2026-11-24-slidesim-design.md` (called "the spec" below).
Milestone 2: `docs/superpowers/specs/2026-11-25-slidesim-m2-perfusion-design.md` and its results
`docs/results/2026-11-m2-real-slide.md`. This document details milestone 3 of the spec. Where both
say something, this document is the more specific one. It changes one decision of the spec (the
ODE integrator, below).

## Intent

Milestone 2 carries species by the blood flow through a real slide. Milestone 3 puts a cell model
into every element of the tissue: an SBML model, solved as one ODE system per element, coupled to
the transport of the species in the blood by operator splitting.

The first model is the SPT liver cell model of pkdb_models (`pkdb_models/tpm/spt`, `spt_liver`,
version 0.9.0): a substrate S from the blood is taken up and converted by the zonated enzyme
CYP2E1 into a product P and a toxic product T; when T stays above a threshold the cell dies
(necrosis). Acetaminophen (APAP) is the case behind it, and slideviz has the APAP mouse slides:
doses 0, 89, 158, 281, 375 and 500 mg/kg, livers taken 24 h after the dose, with CYP2E1 and H&E
sections and a necrosis map per mouse.

The question milestone 3 answers on a real slide: with CYP2E1 from the slide and the flow of
milestone 2, does the model put the necrosis where the slide shows it? The spatial pattern is the
acceptance, not the dose response.

Two decisions of the user frame the how:

- **Performance is the main concern of this milestone.** The coupled run is measured and made
  fast, not only made correct.
- **The cells are solved by sbmlsim-core's own solvers** (CVODE, and IDA for algebraic rules), one
  engine instance per element, because they handle every model sbmlsim-core handles (events,
  algebraic rules, delays). The work starts with the model code of sbmlsim-core as it is
  (WebAssembly run by wasmtime) and then replaces it by native C generated per model, behind the
  same interface.

## Goals

1. A conservative Darcy velocity: element-wise mass conservation of the velocity which carries the
   species, and no normal velocity on walls without flow (the follow-up of milestone 2).
2. sbmlsim-core: an engine instance which a caller advances from `t0` to `t1` from a state it
   gives, with the solver kept warm between calls; then a native C target for the model code which
   implements the engine's model interface.
3. Coupling in slidesim: one engine instance of the cell model per element of the ODE level,
   Strang splitting with the transport of the blood species, parameters per element from the maps
   of the scene, the volumes per element from the porosity.
4. The SPT model in a form sbmlsim-core takes: plain SBML (no `comp`, no `distrib`), with a pattern
   which takes the protein (CYP2E1) as a parameter.
5. slideviz exports the CYP2E1 map and the measured necrosis map of a slide into the scene, and
   checks the portal and central classification of the dosed slides.
6. End to end: 24 h of the SPT model on dosed real slides, the simulated necrosis map compared with
   the measured one; performance measured and reported.

## Non-goals

- Dose response across the six doses, and a calibration of the SPT kinetics to APAP. Milestone 3
  calibrates one number per slide (below) so that the spatial pattern can be compared.
- A time-dependent portal concentration from a pharmacokinetic model. The portal concentration is
  constant in milestone 3; coupling to the pkdb_models whole-body models comes later.
- A batched (vectorized across instances) model kernel and an own integrator. The spec planned
  them (a batched right-hand side and a per-instance Rosenbrock or SDIRK integrator, "not one
  CVODE instance per element"); the user decided for sbmlsim-core's solvers instead. Milestone 6
  (GPU) revisits this: a GPU needs a batched kernel.
- GPU and browser builds of the new pieces (milestones 5 and 6).
- `result/1` output (milestone 4). Output stays VTU, with the ODE states added as cell data.
- Recomputing the pressure during a run. The SPT model does not change the permeability; necrosis
  does not feed back into the flow in milestone 3.
- Several models per element, or cell types other than hepatocytes. Every tissue element runs the
  same model; lumen and elements without tissue run none.

## Anti-goals

- Cell results which disagree with sbmlsim-core's engine run on its own. An instance advanced in
  split steps by slidesim, with the blood held fixed, gives what `simulate` gives for the same
  model and parameters, within the tolerances of the integration.
- Results which depend on the number of processes. Parity within 1e-10 relative, as in milestones
  1 and 2 (with fixed transport steps, below).
- A necrosis map which only matches because the measured necrosis was used as an input. The
  measured map may choose where the CYP2E1 of the slide is replaced (below), never the value, and
  never the geometry or the flow.

## Constraints

- Machine of record for performance: Intel i7-12700 (8 performance and 4 efficiency cores, 20
  threads, AVX2), 62 GB. CI runners are slower and only check correctness.
- Toolchain as in milestones 0-2: Rust, PETSc 3.26.1, libCEED v1.1.2, Open MPI locally, MPICH in
  CI. The system C compiler (gcc 13 here, clang where present) compiles the generated model code.
- sbmlsim-core (`~/git/sbmlsim-core`, v0.26.0, MIT): only `sbmlsim-sbml` is on crates.io; slidesim
  depends on its other crates (engine, host with the feature `native`, compiler, ir) by git tag.
  sbmlsim-core gets a release with the new pieces before slidesim's milestone 3 merges. Its
  native engine links CVODE, IDA and KLU in C (`csolvers`); slidesim's build must link them
  without clashing with PETSc's own SUNDIALS-free build (PETSc 3.26.1 here has no SUNDIALS).
- pkdb_models follows the user's rule for own libraries: slidesim uses its latest release; the SPT
  variant is released there before slidesim's end-to-end study depends on it.

## Context: what exists

- **slidesim** (milestones 0-2, merged): P1 triangles on DMPlex, libCEED operators, BDF2 time
  stepping in PETSc `TS`, geometric multigrid over the refinement levels, maps from Zarr sampled at
  quadrature points, the Darcy flow, advection with SUPG, a mass balance per output time. A
  `Problem` steps one field. The real field of milestone 2 has about 42k coarse triangles and
  about 680k at 2 refinements; one transport step took about 2.2 s on 4 processes there, with flat
  multigrid iteration counts, not profiled.
- **sbmlsim-core**: an IR of the model (`ir::Model`), a symbolic, sparse analytic Jacobian
  (`ir::derive::jacobian`), the WebAssembly code target (`model.wasm`, its math from sbmlsim-core's
  `mathlib`), and the engine (`sbmlsim-engine`: CVODE and IDA with KLU, events, delays, steady
  state). The engine talks to the model through the trait `ModelFns` (`defaults`, `init`, `rhs`,
  `jac`, `roots`, `trigger`, events). The native engine (`host/native.rs`, feature `native`) is
  the engine compiled for the host with the model as `model.wasm` in wasmtime; every call copies
  the state into the model's memory and the results back. Its public call is `simulate(request)`:
  a whole simulation from the initial values, with outputs.
- **SPT model**: states `S_ext`, `P_ext` (plasma, `Vext`), `S`, `P`, `T` (liver, `Vli`),
  `necrosis` and a time counter `tsim` (rate rules); `protein` is an assignment rule over
  `position` and `pattern` (pattern 6 is the CYP2E1 profile measured in control mice). Time in
  min, concentrations in mM. The rates scale with `Vli` and the states divide by their compartment,
  so only the ratio `Vli/Vext` matters. The file requires `comp` (ports) and `distrib` (a uniform
  draw for pattern 5); sbmlsim-core refuses it today ("required package is not supported (comp)").
- **slideviz**: per mouse, the CYP2E1 readout registered onto the H&E grid (relative DAB, 0 to 1,
  at 1.77 um per pixel), the necrosis call map (0.44 um per pixel, about 7 % area error against
  drawn outlines), the zonation position and the lumen classes of milestone 2, both derived from
  the CYP2E1 landscape (peaks pericentral, valleys periportal; a lumen within 15 um of a peak is
  central, of a valley portal).

## Approaches considered

### How the cell models are solved (decided by the user: sbmlsim-core's engine, per element)

1. **One engine instance per element, model as `model.wasm`**: works with what exists (plus the
   advance call), every feature of the engine, per-element step sizes (so no dependence on the
   process count). Cost: every model call crosses into wasm and copies the state.
2. **One engine instance per element, model as native C** (chosen as the target, reached from
   option 1): the same engine; a new code target of sbmlsim-core emits C which implements
   `ModelFns` directly, compiled per study and cached. No wasm crossing, no copies.
3. **One CVODE system per process over all instances** (block diagonal, vectorized right-hand
   side): the cheapest right-hand side, but the stiffest cell sets the step of all, the events of
   all cells meet in one root search, and the results depend on the process count. Rejected.
4. **A batched kernel with an own Rosenbrock integrator** (the spec's plan): the fastest per
   instance, but it handles plain ODEs only. Rejected by the user for now (non-goals).

### Conservative velocity

1. **Flux recovery from the P1 pressure** (chosen, call): edge fluxes per element which are
   consistent with the residual of the discrete Darcy equations and conserve mass in every element
   exactly (equilibrated fluxes, as in Larson and Niklasson 2004), giving a lowest-order
   Raviart-Thomas velocity. It keeps the primal pressure the spec decided and the solver of
   milestone 2, and adds a local postprocessing step.
2. **A mixed Raviart-Thomas formulation**: conservative by construction, but a saddle point system,
   new libCEED bases and a new preconditioner. Rejected: much larger, and the pressure is already
   good.
3. **Refinement only**: the defect falls at order 0.65 (milestone 2); halving it costs about three
   refinements. Rejected.

### CYP2E1 per element (decided by the user: measured on the slide)

CYP2E1 comes from the readout of the slide. Necrotic pericentral cells lose their CYP2E1 staining,
so in the measured necrotic areas of a dosed slide the readout is replaced by the value of pattern
6 at the element's zonation position (the CYP2E1 profile measured on control mice). Every
comparison also runs with pattern 6 everywhere, so that the gain of the measured CYP2E1 is shown.

### Portal and central walls on dosed slides (decided by the user: check first)

The classification of milestone 2 comes from the CYP2E1 landscape, which necrosis disturbs. The
alternatives were a classification from the H&E (portal tracts with bile duct and artery, central
veins alone; new slideviz work) and a correction inside the measured necrosis (rejected: it puts
the measured necrosis into the geometry). Decided: measure the classification on the dosed slides
first, and use it as it is if it holds (below).

## Decided design

### Conservative velocity

- After the pressure solve, per level: the flux through every edge of the mesh from the residual
  of the discrete Darcy equations (equilibrated fluxes), so that for every element the sum of its
  edge fluxes is exactly zero (no sources in the Darcy problem), and the flux through every edge of
  a wall without a pressure condition is exactly zero.
- The velocity at the quadrature points is the lowest-order Raviart-Thomas field of these fluxes.
  It replaces the primal velocity in the transport; the primal velocity stays as an output for
  comparison.
- The wall flows by tag stay those of milestone 2 (consistent fluxes); they now equal the sums of
  the edge fluxes of the tag.

### Cell models on sbmlsim-core's engine

Stage 1 (wasm model):

- sbmlsim-core gains an advance call on an engine instance (call: `Engine::advance` or a
  `Stepper` object; the name is sbmlsim-core's): given the instance's parameters once, and then per
  call a state and an interval `[t0, t1]`, it integrates with the engine's solver (CVODE, or IDA
  for a model with algebraic rules), handles the events and discontinuities in the interval as
  `simulate` does, and returns the state at `t1`. The solver memory stays between calls and is
  reinitialized from the given state (the blood species change between calls); the last step size
  is the first guess of the next call.
- Per-instance setup reads the defaults and the initial values once (`defaults`, `init` with the
  parameters set from the maps fixed).
- Errors per instance (solver failure, a non-finite value) name the element and stop the run
  cleanly on every process.
- slidesim holds one such instance per coarse element on each process.

Stage 2 (native C model):

- A new code target of sbmlsim-core emits, from `ir::Model`, C which implements every function of
  `ModelFns` (the rates, the analytic Jacobian in the engine's sparse pattern, the roots,
  triggers, priorities and event assignments, the initial evaluations), with the same meaning as
  the wasm target. A Rust `ModelFns` wraps the loaded shared library, so the engine and the
  advance call do not change.
- Compilation: the Rust side writes the C source, compiles it with the system compiler into a
  shared library (`-O3 -march=native -ffp-contract=off -fno-fast-math`, the flags in the cache
  key), caches it by the hash of the source and the flags, and loads it. A missing compiler is a
  clear error.
- Math functions (call): `exp`, `log` and `pow` of the wasm target come from sbmlsim-core's
  `mathlib` (the same bits on every host). The C target links the same routines by default, so
  that stage 2 reproduces stage 1 closely; a faster math library is allowed if measured faster
  and within the verification tolerances. The choice and its measurement are recorded.
- A model feature the C target does not cover is an error naming the feature at study start
  (and the study can still run the wasm model); the goal is the whole of `ModelFns`.
- Many instances, one model: the generated code and the model's constants are shared; an instance
  holds only its parameters, states and solver memory.

### Verification of the cells

1. The advance call against `simulate`, per instance: the SPT model, parameters drawn over their
   ranges (protein 0 to 1, position, the threshold crossed and not crossed), advanced in split
   steps of several sizes with nothing changed between calls, against one `simulate` over the
   whole time, within the integration tolerances.
2. Stage 2 against stage 1, the same instances: within the integration tolerances (bitwise where
   the C uses `mathlib` and the engine's step sequence is the same).
3. The C target on the cases of the SBML Test Suite that sbmlsim-core passes, against the
   suite's baseline with the suite's tolerances, the cases with events, delays and algebraic rules
   included.
4. The same results for any number of processes (each instance is independent).

### The SPT model for slidesim (pkdb_models)

- The factory of `pkdb_models/tpm/spt` writes a variant (call: `spt_liver_cell.xml`) without
  `comp` ports and without the `distrib` draw (pattern 5 keeps a constant `protein_random`), with
  a new pattern 7: `protein` is a parameter `protein_ext` in [0, 1] set from outside. Everything
  else is the model of `spt_liver` 0.9.0. Its markdown and its tests come with it, as for the
  other models of the package.
- `tsim` stays in the model; it costs one trivial state.

### Coupling in slidesim

- The study names the cell model (an SBML file, compiled by sbmlsim-core at study start) and maps
  it:
  - species of the model to fields of the study: `S_ext` to the field `S`, `P_ext` to the field
    `P`, both concentrations in the blood space;
  - parameters of the model to maps of the scene, through the coefficient tables of milestone 2:
    `position` from `zonation`, `protein_ext` from `cyp2e1` (with the replacement rule below);
  - fixed parameters (for example `pattern` 7);
  - the time unit of the model (min) against the study's seconds;
  - where the model runs: elements whose tissue class is tissue; none in lumen.
- ODE level: the coarse mesh (the spec). One instance per coarse element. The PDE runs on the finest
  level.
- Volumes: per element `Vext` is the porosity and `Vli` is one minus the porosity (only the ratio
  matters, the model's own volume units are kept).
- Exchange: before an ODE step, `S_ext` and `P_ext` of an instance are the means of the fields
  over its element (L2 projection to the coarse elements); after the step, the change of each is
  given back to the fields as an increment (conservative transfer). Call: the total amount is
  conserved exactly (to round-off), not the amount of every coarse element: P1 fields on the
  refined mesh share every node with neighbouring coarse elements, so exact per-element transfer
  needs a global constrained solve with no unique answer, and at 0 refinements it is impossible.
  The increment is distributed with the element's P1 test functions (each element's amount goes
  to its own nodes, the sum is exact); the builder adds a positivity safeguard if steep uptake
  drives nodal values below zero, and states it.
- Strang splitting per split step dt: ODE dt/2, transport dt, ODE dt/2. The transport keeps its
  BDF2 steps inside a split step; the split step and the transport step are set in the study.
- The mass balance of milestone 2 gains the exchange with the cells as a term, per output time.
- Output: the ODE states per coarse element as cell data in VTU (`necrosis`, `T`, `S`, `P`,
  `protein`), with the fields as before.

### Performance

The target is the coupled 24 h run on a real field. The design makes these choices for speed;
the plan measures every one:

- Production runs use the optimized native dependencies (`native-opt`), not the `-O0` debug build
  of the tests. The test suite also moves to an optimized build with PETSc's debug checks kept
  (the follow-up of milestone 2), so that the coupled tests run in reasonable time.
- The transport reuses its operators and the multigrid preconditioner across steps when the
  coefficients do not change (they do not in milestone 3: the flow is steady and the cells change
  only the fields). Milestone 2 rebuilds them per call.
- Adaptive time steps for the transport inside a split step: the blood equilibrates in seconds,
  the cells change over minutes to hours.
- The cells of a process run one after another on that process; processes are the only
  parallelism in milestone 3 (no threads beside MPI, call). Stage 1 is the baseline; stage 2 is
  measured against it on the same run.
- The engine's linear solver for a small model: the engine chooses (KLU today); if a dense solver
  is clearly faster for a model of a few states, sbmlsim-core may choose it by size.
- Profiling with PETSc `-log_view` and a per-phase timer (pressure, transport, exchange, cells,
  output) in `summary.json`, so that every run reports where the time goes.
- Target (call, to be confirmed by the first profile): 24 h simulated on the milestone 2 field at
  1 refinement in at most 10 minutes on the machine of record with 8 processes, with stage 2. The
  end-to-end report gives the measured times per phase for both stages.

### slideviz export

- `cyp2e1` map: the relative DAB readout registered onto the H&E grid, 0 to 1, the no-value
  outside the measured tissue.
- `necrosis` map: the measured necrosis calls (1 necrotic, 0 not, the no-value not measured),
  resampled to the working grid of the scene.
- Both are written by the scene writer of milestone 2 (`scripts/write_scene.py`), with their
  provenance in `scene.json`.
- The mesh of a dosed slide is the plain variant (necrosis does not shape the mesh).

### Portal and central check on dosed slides

- On each dosed slide of the end-to-end study: the share of lumina classified portal, central and
  other, against the controls; and among the lumina inside or within 50 um of the measured
  necrosis, how many are classified portal. Pericentral necrosis around a vein classified portal
  is the sign of a flipped classification.
- If the classification holds (call: at most 1 in 10 lumina inside or next to the necrosis
  classified portal, and a portal/central ratio within the range of the controls), the dosed
  slide's classification and zonation position are used as they are, and the report shows the
  check.
- If it does not hold, milestone 3 stops at this point for a decision by the user (a
  classification from the H&E is then the candidate); the rest of the milestone does not depend
  on it until the end-to-end study.

### CYP2E1 replacement rule

- `protein_ext` of an element is the mean CYP2E1 readout over the element's tissue where the
  measured necrosis map says not necrotic, and the value of pattern 6 at the element's zonation
  position where it says necrotic (more than half of the element's measured area).
- Scale (call): the readout is the same local readout pattern 6 was fitted to (palest tenth 0,
  darkest tenth 1, per 500 um window), and pattern 6 is that fit divided by its perivenous value
  0.7405. So `protein_ext` is the readout divided by 0.7405, capped at 1; otherwise the replaced
  (necrotic) elements would get about 35 % more enzyme than measured ones and necrose first by
  construction. The end-to-end report states the readout used (raw or flattened) and checks the
  scale on the non-necrotic tissue of each slide (readout against pattern 6 by position).
- The rule is a coupling option of the study (a map for the value, a map for where to replace, a
  replacement table over position), not SPT-specific code.
- Every end-to-end comparison runs twice: the rule above, and pattern 6 everywhere.

### Verification and acceptance

1. Conservative velocity: zero element divergence and zero flux through walls without flow, to
   round-off; `advection_defect` zero; velocity order 1 on the manufactured solutions of
   milestone 2; parity.
2. Cells: the verification above (advance against `simulate`, stage 2 against stage 1, the C
   target on the SBML Test Suite).
3. Coupling, manufactured: a linear cell model with a known analytic coupled solution (uptake
   proportional to the blood concentration in a closed domain), with the order 2 of Strang
   splitting in the split step.
4. Conservation of the coupled system: the amount of S and P in blood and cells together changes
   by the boundary flows only, to 1e-8 relative.
5. Parity across 1, 2 and 3 processes within 1e-10 relative, the ODE states included, with fixed
   transport steps (call). The cells' step control is per instance and does not depend on the
   process count. Adaptive transport steps take their error norm from parallel reductions, whose
   round-off depends on the process count, so with adaptive transport the runs agree within the
   time-step tolerance instead; a test states that bound.
6. End to end on dosed real slides (one mouse each at 281, 375 and 500 mg/kg):
   - the portal and central check above;
   - 24 h with a constant S at the portal walls;
   - per slide, the portal concentration chosen (bisection over runs) so that the simulated
     necrotic share of the field equals the measured one;
   - the spatial comparison: the overlap of simulated and measured necrosis (Dice coefficient),
     the necrotic fraction along the zonation position against the measured one, and the
     necrosis per lobule;
   - both CYP2E1 inputs (measured with replacement, pattern 6 everywhere);
   - no control slide check (call): a control liver has the same CYP2E1 before a dose, so at a
     dosed portal concentration it would necrose in the same pattern, and at zero it is trivial;
   - rendered images inspected;
   - the performance per phase reported, stage 1 and stage 2, against the target.

### Order of work

1. slidesim: conservative velocity.
2. slidesim: optimized test build, operator and preconditioner reuse, adaptive transport steps,
   per-phase timers (the transport made fast before the cells come).
3. pkdb_models: the `spt_liver_cell` variant, released.
4. slideviz: `cyp2e1` and `necrosis` maps in the scene, and the portal and central check on the
   dosed slides (early, because its outcome can stop the end-to-end study).
5. sbmlsim-core: the advance call (stage 1), verified, released.
6. slidesim: the coupling on stage 1 (study, exchange, Strang, output, mass balance), verified;
   a first end-to-end run as the performance baseline.
7. sbmlsim-core: the native C target (stage 2), verified, released.
8. End to end on the dosed slides with stage 2, with the performance report.

## Left to the builder

- The equilibrated flux algorithm (patch-wise or global), as long as the conservation and
  convergence above hold.
- The name and shape of the advance call in sbmlsim-core, and how it reinitializes the solver.
- The cache location of compiled model code.
- The study syntax of the cell model and its mappings, within `study/1` and its schema.
- The split step and transport step of the end-to-end study, as long as the result is converged in
  the split step (checked by halving it once).
- The metrics' exact definitions (Dice threshold on `necrosis`, the position bins).
