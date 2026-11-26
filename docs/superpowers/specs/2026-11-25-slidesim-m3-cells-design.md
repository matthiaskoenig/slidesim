# slidesim milestone 3 - cell models coupled to perfusion - design

Status: draft for review, 2026-11-25.

Parent design: `docs/superpowers/specs/2026-11-24-slidesim-design.md` (called "the spec" below).
Milestone 2: `docs/superpowers/specs/2026-11-25-slidesim-m2-perfusion-design.md` and its results
`docs/results/2026-11-m2-real-slide.md`. This document details milestone 3 of the spec. Where both
say something, this document is the more specific one.

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

Decisions of the user which frame the how:

- **Performance is the main concern of this milestone**, on many cores now and on GPUs later
  (milestone 6), up to the very large meshes of milestones 6 and 7. The coupled run is measured and
  made fast, not only made correct.
- **Two cell solvers behind one interface:**
  - the **engine path**: sbmlsim-core's own solvers (CVODE, IDA for algebraic rules), one engine
    instance per element, native throughout (the native Rust engine with native model code, no
    WebAssembly), because they handle every model sbmlsim-core handles (events, algebraic rules,
    delays). It is the reference and the path for models the batched path does not take;
  - the **batched path**: the model and an adaptive Rosenbrock integrator generated as a libCEED
    QFunction, one point per element, so that libCEED runs it vectorized on the CPU now and on
    CUDA and HIP later, with the data in PETSc vectors beside the transport. It is the fast path
    and the one which scales to GPUs and very large meshes.

## Goals

1. A conservative Darcy velocity: element-wise mass conservation of the velocity which carries the
   species, and no normal velocity on walls without flow (the follow-up of milestone 2).
2. sbmlsim-core: a C code target for the model functions, emitted from its IR; an engine instance
   which a caller advances from `t0` to `t1` from a state it gives, running native model code; and
   a QFunction target which emits the model with a Rosenbrock integrator for libCEED.
3. Coupling in slidesim: a cell solver interface with the two paths, one cell per element of the
   ODE level, Strang splitting with the transport of the blood species, parameters per element
   from the maps of the scene, the volumes per element from the porosity.
4. The SPT model in a form sbmlsim-core takes: plain SBML (no `comp`, no `distrib`), with a pattern
   which takes the protein (CYP2E1) as a parameter.
5. slideviz exports the CYP2E1 map and the measured necrosis map of a slide into the scene, and
   checks the portal and central classification of the dosed slides.
6. End to end: 24 h of the SPT model on dosed real slides, the simulated necrosis map compared with
   the measured one; performance of both paths measured and reported.

## Non-goals

- Dose response across the six doses, and a calibration of the SPT kinetics to APAP. Milestone 3
  calibrates one number per slide (below) so that the spatial pattern can be compared.
- A time-dependent portal concentration from a pharmacokinetic model. The portal concentration is
  constant in milestone 3; coupling to the pkdb_models whole-body models comes later.
- Running on a GPU. The batched path is built so that libCEED's GPU backends can run it (no host
  calls inside the QFunction, data in PETSc vectors), but GPU runs, device vectors and their
  verification are milestone 6.
- Events, delays and algebraic rules on the batched path. The spec decided this subset for the
  batched kernel; a model with them runs on the engine path, and the batched path refuses it with
  an error naming the feature.
- `result/1` output (milestone 4). Output stays VTU, with the cell states added as cell data.
- Recomputing the pressure during a run. The SPT model does not change the permeability; necrosis
  does not feed back into the flow in milestone 3.
- Several models per element, or cell types other than hepatocytes. Every tissue element runs the
  same model; lumen and elements without tissue run none.
- Threads beside MPI. Processes are the parallelism on the CPU in milestone 3 (call).

## Anti-goals

- Cell results which disagree with sbmlsim-core run on its own. An engine instance advanced in
  split steps by slidesim, with the blood held fixed, gives what `simulate` gives; the batched path
  gives what the engine path gives; both within the tolerances of the integration.
- Results which depend on the number of processes, the libCEED backend or the blocking of points.
  Parity within 1e-10 relative, as in milestones 1 and 2 (with fixed transport steps, below).
- WebAssembly on the engine path of a run. The WebAssembly model of sbmlsim-core stays a reference
  in tests only.
- A necrosis map which only matches because the measured necrosis was used as an input. The
  measured map may choose where the CYP2E1 of the slide is replaced (below), never the value, and
  never the geometry or the flow.

## Constraints

- Machine of record for performance: Intel i7-12700 (8 performance and 4 efficiency cores, 20
  threads, AVX2), 62 GB. CI runners are slower and only check correctness.
- Toolchain as in milestones 0-2: Rust, PETSc 3.26.1, libCEED v1.1.2, Open MPI locally, MPICH in
  CI. The system C compiler (gcc 13 here, clang where present) compiles the generated code at
  study start.
- libCEED QFunctions: one C function per QFunction, the source readable by libCEED's JIT for GPU
  backends, no calls outside the C math functions. slidesim already builds its operators this way
  (`crates/pde/qfunctions`, compiled for the CPU backends, read by the JIT backends).
- sbmlsim-core (`~/git/sbmlsim-core`, v0.26.0, MIT): only `sbmlsim-sbml` is on crates.io; slidesim
  depends on its other crates (ir, engine, the C targets) by git tag. sbmlsim-core gets a release
  with the new pieces before slidesim's milestone 3 merges. Its native engine links CVODE, IDA and
  KLU in C (`csolvers`); PETSc 3.26.1 here is built without SUNDIALS, so they do not clash.
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
  `jac`, `roots`, `trigger`, events). The native engine (`host/native.rs`) is the engine compiled
  for the host, but its model is still `model.wasm` in wasmtime: every call copies the state into
  the model's memory and the results back. The public call is `simulate(request)`: a whole
  simulation from the initial values, with outputs. There is no C target and no batched
  evaluation.
- **SPT model**: states `S_ext`, `P_ext` (plasma, `Vext`), `S`, `P`, `T` (liver, `Vli`),
  `necrosis` and a time counter `tsim` (rate rules); `protein` is an assignment rule over
  `position` and `pattern` (pattern 6 is the CYP2E1 profile measured in control mice); the necrosis
  trigger is a `piecewise`. Time in min, concentrations in mM. The rates scale with `Vli` and the
  states divide by their compartment, so only the ratio `Vli/Vext` matters. The file requires
  `comp` (ports) and `distrib` (a uniform draw for pattern 5); sbmlsim-core refuses it today
  ("required package is not supported (comp)").
- **slideviz**: per mouse, the CYP2E1 readout registered onto the H&E grid (relative DAB, 0 to 1,
  at 1.77 um per pixel), the necrosis call map (0.44 um per pixel, about 7 % area error against
  drawn outlines), the zonation position and the lumen classes of milestone 2, both derived from
  the CYP2E1 landscape (peaks pericentral, valleys periportal; a lumen within 15 um of a peak is
  central, of a valley portal).

## Approaches considered

### How the cell models are solved (decided by the user: both paths)

1. **Engine per element, native**: sbmlsim-core's engine, one instance per element, model code
   native. Every feature of the engine; per-element step sizes. CPU only; several KB of solver
   memory per instance, which limits it to about 10^5-10^6 cells per process and rules out GPUs.
2. **Batched libCEED QFunction with Rosenbrock**: one point per element, states and parameters in
   PETSc vectors, about 20 numbers per cell, vectorized by libCEED's CPU backends and compiled for
   GPUs by its JIT. Plain ODE models only.
3. **One CVODE system per process over all cells** (block diagonal): the stiffest cell sets the
   step of all, the events of all cells meet in one root search, and the results depend on the
   process count. Rejected.
4. **The engine with the WebAssembly model**: works today, but every model call crosses into wasm
   and copies the state. Rejected by the user for speed; kept as a test reference.

Decided: 1 and 2 behind one interface; 2 is the fast path, 1 the reference and the fallback.

### Native model code for the engine

The engine is Rust; its model code must be native too. Emitted C, compiled at study start by the
system compiler, is chosen (call) over emitted Rust: compiling Rust at study time needs `rustc` and
a crate around it at run time and takes much longer, and the C emitter is shared with the batched
path, whose QFunction must be C for libCEED.

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

### The C target of sbmlsim-core

- A new crate (call: `sbmlsim-c`) emits C from `ir::Model`: the point functions of a model, each
  for one instance, with the states, parameters and outputs passed as arguments (no globals):
  initial evaluations, the rates, the analytic Jacobian, the roots, triggers, priorities and event
  assignments. The batched path uses the subset for plain ODE models (initial evaluations, rates,
  Jacobian); the engine path all of them.
- The meaning of every expression is that of the WebAssembly target (operators, `piecewise`, the
  relational and logical operators).
- Math functions (call): the C math functions of the platform (the C library on the CPU, the CUDA
  or HIP math library in libCEED's JIT later), not sbmlsim-core's `mathlib`, which is Rust and not
  available inside a GPU QFunction. Agreement with the WebAssembly model is therefore within the
  integration tolerances, not bit for bit; the tests state the bound.
- Compilation: the Rust side writes the C source, compiles it with the system compiler into a
  shared library (`-O3 -march=native -ffp-contract=off -fno-fast-math`, the flags in the cache
  key), caches it by the hash of the source and the flags, and loads it. A missing compiler is a
  clear error. The same source is what libCEED's JIT reads on GPU backends.
- A model feature the target does not cover is an error naming the feature at study start.

### Engine path

- sbmlsim-core gains an advance call on an engine instance (call: `Engine::advance` or a
  `Stepper`; the name is sbmlsim-core's): given the instance's parameters once, and then per call
  a state and an interval `[t0, t1]`, it integrates with the engine's solver (CVODE, or IDA for a
  model with algebraic rules), handles the events and discontinuities in the interval as `simulate`
  does, and returns the state at `t1`. The solver memory stays between calls and is reinitialized
  from the given state (the blood species change between calls); the last step size is the first
  guess of the next call.
- The model behind it is a `ModelFns` over the loaded C library of the C target, native; the
  engine is the native Rust engine. No WebAssembly in a run.
- Per-instance setup reads the defaults and the initial values once (`defaults`, `init` with the
  parameters set from the maps fixed).
- Errors per instance (solver failure, a non-finite value) name the element and stop the run
  cleanly on every process.
- slidesim holds one instance per coarse element on each process; the generated code and the
  model's constants are shared, an instance holds its parameters, states and solver memory.
- The engine's linear solver for a small model: the engine chooses (KLU today); if a dense solver
  is clearly faster for a model of a few states, sbmlsim-core may choose it by size.

### Batched path

- A second emitter of the C target writes one libCEED QFunction per model: for each point, the
  model's point functions and an adaptive Rosenbrock integrator over the split step.
- The integrator: a Rosenbrock method of order 3 or 4 with an embedded error estimate and
  L-stability, per-point step size control and tolerances, a dense LU of the small iteration matrix
  per point, the analytic Jacobian. The method and its coefficients are the builder's choice,
  written once.
- The QFunction body must vectorize across points on libCEED's blocked CPU backend and must not
  diverge into host calls on a GPU. The sequence of steps of a point must not depend on the other
  points (for example: one step attempt per point per QFunction application, with the driver
  applying the operator until every point has reached the end of the split step, and finished
  points unchanged). The builder chooses the scheme within these rules.
- Data: the states, the parameters and the per-point integrator state (time reached, last step
  size) are fields in PETSc vectors over the coarse elements, one point per element, read and
  written by the QFunction through strided element restrictions without a basis.
- The QFunction context holds `t0`, `t1`, the tolerances and the time unit factor.
- A per-point failure (step size underflow, a non-finite value) is flagged in an output field,
  reduced over the processes, and stops the run naming an element.
- Models with events, delays or algebraic rules are refused with an error naming the feature; the
  study then uses the engine path.

### Cell solver interface in slidesim

- One trait for both paths: set up the cells of a process from the parameter arrays, advance all
  of them from `t0` to `t1` with given blood values, read the states, report steps, rejected steps
  and the time spent.
- The study chooses the path per cell model (`"solver": "batched"` or `"engine"`, call); the
  default is the batched path when the model allows it.

### Verification of the cells

1. Engine path: the advance call against `simulate` per instance, on the SPT model with parameters
   drawn over their ranges (protein 0 to 1, position, the threshold crossed and not crossed),
   advanced in split steps of several sizes with nothing changed between calls, against one
   `simulate` over the whole time, within the integration tolerances.
2. The C target against the WebAssembly model in the engine, on the SPT model and on the cases of
   the SBML Test Suite that sbmlsim-core passes (events, delays and algebraic rules included),
   with the suite's tolerances.
3. Batched path against engine path, per element, on the same drawn SPT instances and on the plain
   ODE cases of the SBML Test Suite, within the integration tolerances.
4. Batched path invariance: the same states to round-off (1e-12 relative) on the libCEED backends
   `/cpu/self/ref/serial` and `/cpu/self/opt/blocked`, for any blocking, and on 1, 2 and 3
   processes.
5. A benchmark of both paths alone: cell-steps per second per core on the SPT model, recorded on
   the machine of record.

### The SPT model for slidesim (pkdb_models)

- The factory of `pkdb_models/tpm/spt` writes a variant (call: `spt_liver_cell.xml`) without
  `comp` ports and without the `distrib` draw (pattern 5 keeps a constant `protein_random`), with
  a new pattern 7: `protein` is a parameter `protein_ext` in [0, 1] set from outside. Everything
  else is the model of `spt_liver` 0.9.0. Its markdown and its tests come with it, as for the
  other models of the package.
- `tsim` stays in the model; it costs one trivial state.

### Coupling in slidesim

- The study names the cell model (an SBML file, compiled at study start) and maps it:
  - species of the model to fields of the study: `S_ext` to the field `S`, `P_ext` to the field
    `P`, both concentrations in the blood space;
  - parameters of the model to maps of the scene, through the coefficient tables of milestone 2:
    `position` from `zonation`, `protein_ext` from `cyp2e1` (with the replacement rule below);
  - fixed parameters (for example `pattern` 7);
  - the time unit of the model (min) against the study's seconds;
  - where the model runs: elements whose tissue class is tissue; none in lumen.
- ODE level: the coarse mesh (the spec). One cell per coarse element. The PDE runs on the finest
  level.
- Volumes: per element `Vext` is the porosity and `Vli` is one minus the porosity (only the ratio
  matters, the model's own volume units are kept).
- Exchange: before a cell step, `S_ext` and `P_ext` of a cell are the means of the fields over its
  element (L2 projection to the coarse elements); after the step, the change of each is given back
  to the fields as an increment (conservative transfer). Call: the total amount is conserved
  exactly (to round-off), not the amount of every coarse element: P1 fields on the refined mesh
  share every node with neighbouring coarse elements, so exact per-element transfer needs a global
  constrained solve with no unique answer, and at 0 refinements it is impossible. The increment is
  distributed with the element's P1 test functions (each element's amount goes to its own nodes,
  the sum is exact); the builder adds a positivity safeguard if steep uptake drives nodal values
  below zero, and states it. The projection and the increment are libCEED operators too, so that
  the whole coupling can move to a GPU in milestone 6.
- Strang splitting per split step dt: cells dt/2, transport dt, cells dt/2. The transport keeps its
  BDF2 steps inside a split step; the split step and the transport step are set in the study.
- The mass balance of milestone 2 gains the exchange with the cells as a term, per output time.
- Output: the cell states per coarse element as cell data in VTU (`necrosis`, `T`, `S`, `P`,
  `protein`), with the fields as before.

### Performance

The target is the coupled 24 h run on a real field. The design makes these choices for speed; the
plan measures every one:

- Production runs use the optimized native dependencies (`native-opt`), not the `-O0` debug build
  of the tests. The test suite also moves to an optimized build with PETSc's debug checks kept
  (the follow-up of milestone 2), so that the coupled tests run in reasonable time.
- The transport reuses its operators and the multigrid preconditioner across steps when the
  coefficients do not change (they do not in milestone 3: the flow is steady and the cells change
  only the fields). Milestone 2 rebuilds them per call.
- Adaptive time steps for the transport inside a split step: the blood equilibrates in seconds,
  the cells change over minutes to hours.
- Profiling with PETSc `-log_view` and a per-phase timer (pressure, transport, exchange, cells,
  output) in `summary.json`, so that every run reports where the time goes.
- A strong scaling series on the machine of record (1, 2, 4, 8 and 12 processes) for both paths,
  reported, as the start of the scaling work of milestones 6 and 7.
- Target (call, to be confirmed by the first profile): 24 h simulated on the milestone 2 field at
  1 refinement in at most 10 minutes on the machine of record with 8 processes, with the batched
  path. The end-to-end report gives the measured times per phase for both paths.

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
2. Cells: the verification above (engine advance against `simulate`, the C target against the
   WebAssembly model and the SBML Test Suite, batched against engine, batched invariance).
3. Coupling, manufactured: a linear cell model with a known analytic coupled solution (uptake
   proportional to the blood concentration in a closed domain), with the order 2 of Strang
   splitting in the split step, on both paths.
4. Conservation of the coupled system: the amount of S and P in blood and cells together changes
   by the boundary flows only, to 1e-8 relative.
5. Parity across 1, 2 and 3 processes within 1e-10 relative, the cell states included, with fixed
   transport steps (call), on both paths. The cells' step control is per cell and does not depend
   on the process count. Adaptive transport steps take their error norm from parallel reductions,
   whose round-off depends on the process count, so with adaptive transport the runs agree within
   the time-step tolerance instead; a test states that bound.
6. End to end on dosed real slides (one mouse each at 281, 375 and 500 mg/kg):
   - the portal and central check above;
   - 24 h with a constant S at the portal walls;
   - per slide, the portal concentration chosen (bisection over runs) so that the simulated
     necrotic share of the field equals the measured one;
   - the spatial comparison: the overlap of simulated and measured necrosis (Dice coefficient),
     the necrotic fraction along the zonation position against the measured one, and the
     necrosis per lobule;
   - both CYP2E1 inputs (measured with replacement, pattern 6 everywhere);
   - both paths agree on the necrosis map of one slide (the same cells necrotic, the states within
     the integration tolerances);
   - no control slide check (call): a control liver has the same CYP2E1 before a dose, so at a
     dosed portal concentration it would necrose in the same pattern, and at zero it is trivial;
   - rendered images inspected;
   - the performance per phase and the strong scaling series reported, both paths, against the
     target.

### Order of work

1. slidesim: conservative velocity.
2. slidesim: optimized test build, operator and preconditioner reuse, adaptive transport steps,
   per-phase timers (the transport made fast before the cells come).
3. pkdb_models: the `spt_liver_cell` variant, released.
4. slideviz: `cyp2e1` and `necrosis` maps in the scene, and the portal and central check on the
   dosed slides (early, because its outcome can stop the end-to-end study).
5. sbmlsim-core: the C target (point functions, compilation, cache) and the advance call of the
   engine on native model code, verified against the WebAssembly model and the SBML Test Suite,
   released.
6. slidesim: the cell solver interface and the coupling on the engine path (study, exchange,
   Strang, output, mass balance), verified; a first end-to-end run as the performance baseline.
7. sbmlsim-core: the QFunction emitter with the Rosenbrock integrator, released; slidesim: the
   batched path, verified against the engine path.
8. End to end on the dosed slides with both paths, with the performance and scaling report.

## Left to the builder

- The equilibrated flux algorithm (patch-wise or global), as long as the conservation and
  convergence above hold.
- The Rosenbrock method, its coefficients and step size controller, and the stepping scheme of the
  QFunction within the rules above.
- The name and shape of the advance call in sbmlsim-core, and how it reinitializes the solver.
- The cache location of compiled model code.
- The study syntax of the cell model, its mappings and the path choice, within `study/1` and its
  schema.
- The split step and transport step of the end-to-end study, as long as the result is converged in
  the split step (checked by halving it once).
- The metrics' exact definitions (Dice threshold on `necrosis`, the position bins).
