# Milestone 3 performance

Status: 2026-11-26, slidesim branch `feat/m3-cells`. Started in task 3 of the milestone 3 plan
(transport performance and phase timers); tasks 11 to 13 add the cell phases.

## Setup

- Machine: Intel Core i7-12700 (8 performance and 4 efficiency cores, 20 threads), Open MPI
  4.1.6.
- Build: `deps/install/native-opt` (optimized PETSc and libCEED), `cargo build --release`.
- Field: the milestone 2 baseline study on the real slide
  (`testdata/studies/m2_real_slide/baseline.study.json`) at 1 refinement: 170,244 triangles,
  86,655 vertices, two mesh levels. One field (`sucrose`) with advection by the Darcy flow, SUPG,
  BDF2, 200 fixed steps of 0.05 s to 10 s and 6 output times. The adaptive variant sets
  `time.adaptive` (103 steps).
- Benchmark: `scripts/bench/transport.sh` (inputs from `scripts/e2e/m2_real_slide.sh`, see the
  script). It runs the study on 1, 4 and 8 processes with `-log_view` and prints the seconds per
  step: the `transport` phase of `summary.json` divided by the `TSStep` events of `-log_view`.
  `ADAPTIVE=1` runs the adaptive variant, `SLIDESIM` takes the binary of another state.

`summary.json` now has `"phases"`: the wall time of process 0 in `setup` (from the start of the
process: PETSc, study, mesh, maps, problems), `pressure` (the flow), `transport` (the steps of
the fields) and `output` (files and summary values), with a barrier at each phase boundary.

## Profile before the changes

`-log_view` on 4 processes, milestone 2 code. Fixed steps: 80.7 s in total, 77.4 s in `TSStep`.
The top five events by time, leaving out the solver levels which contain all others (`TSStep`,
`SNESSolve`, `KSPSolve`):

| Event | Calls | Time (s) | Share of run |
| --- | ---: | ---: | ---: |
| `MatMult` | 37,484 | 72.9 | 90 % |
| `PCApply` | 2,213 | 60.6 | 75 % |
| `MatResidual` | 8,821 | 14.8 | 18 % |
| `SNESFunctionEval` | 201 | 3.8 | 5 % |
| `SFReduceEnd` | 11,478 | 2.6 | 3 % |

Adaptive steps: 53.7 s in total, 50.4 s in `TSStep`:

| Event | Calls | Time (s) | Share of run |
| --- | ---: | ---: | ---: |
| `MatMult` | 24,015 | 44.6 | 83 % |
| `PCApply` | 1,159 | 31.0 | 58 % |
| `PCSetUp` | 114 | 10.6 | 20 % |
| `MatResidual` | 4,605 | 7.6 | 14 % |
| `SNESFunctionEval` | 112 | 2.2 | 4 % |

Nearly all the time went into applying the matrix-free libCEED operators: the outer Krylov
method, the Chebyshev smoothers and the residuals of the finest multigrid level all applied the
shell operator, about 5 ms per application on 4 processes (42,000 triangles each), while the
iteration counts were low (11 GMRES iterations per step). `PCSetUp` of the adaptive run renewed
the Chebyshev eigenvalue estimates on the shell operator after every change of the shift. A
callgrind profile on one process (two steps) split the operator application: the tensor
contraction of libCEED's non-tensor basis (`CeedTensorContractApply_Opt`, 24 % of all
instructions), the QFunction `Transport2D` (10 %), the element restriction (4 %), and about
15 % for the bookkeeping of the blocked backend per block of 8 elements (getting and restoring
vector arrays, lengths, basis queries). For P1 triangles with 3 nodes the matrix-free
application costs many times a sparse matrix-vector product with about 7 entries per row.

The reuse of operators the plan asked for was already in place: `Problem::new` builds the
operators of every level and the multigrid once, and a step only changes the shift. Task 3 pins
it with tests (`operators_are_built_once`, `same_result_with_reuse`) and makes the work at a
change of the shift cheap (change 4 below).

## Changes

Seconds per step, transport phase over the steps, fixed steps (200):

| State | 1 process | 4 processes | 8 processes |
| --- | ---: | ---: | ---: |
| Milestone 2 code, with the phase timers | 1.3988 | 0.3863 | 0.2631 |
| 1. Multigrid levels assembled | 0.3584 | 0.0982 | 0.0614 |
| 2. Jacobian and residual assembled | 0.0686 | 0.0195 | 0.0122 |
| 3. Source integrated only when it changes | 0.0500 | 0.0139 | 0.0092 |
| 4. Jacobian reassembled from its steady and mass terms | 0.0486 | 0.0134 | 0.0082 |

Adaptive steps (103):

| State | 1 process | 4 processes | 8 processes |
| --- | ---: | ---: | ---: |
| Milestone 2 code, with the phase timers | 1.8140 | 0.4810 | 0.3255 |
| After changes 1 to 3 | 0.1996 | 0.0552 | 0.0353 |
| After change 4 | 0.0760 | 0.0217 | 0.0143 |

In total the transport step is 29 times faster on 1 and 4 processes and 32 times on 8 with fixed
steps, 22 to 24 times with adaptive steps. On 4 processes the
transport phase of the fixed run went from 77.3 s to 2.7 s.

1. **Multigrid levels assembled** (`crates/pde/src/multigrid.rs`). `configure_multigrid`
   assembles the operator of every level, the finest included, with libCEED
   (`Transport::assemble`, as the coarse level already was), and gives the assembled matrices to
   the smoothers, the residuals of the levels and the coarse solver. `Multigrid::set_shift`
   assembles them again. The flow solve benefits too: the `pressure` phase fell from 1.67 s to
   0.88 s on 4 processes.
2. **Jacobian and residual assembled** (`crates/pde/src/problem.rs`, `crates/petsc/src/ts.rs`).
   The Krylov method applies the assembled finest level of the multigrid (`Multigrid::finest`,
   set as the Jacobian of the `TS` by `Ts::set_jacobian_matrices`) instead of the shell
   operator. The residual applies the assembled steady terms and mass matrix, plus the steady
   terms of the Dirichlet values computed once (a lifting vector), instead of three libCEED
   operators. With options which replace the multigrid preconditioner the Jacobian stays the
   matrix-free operator.
3. **Source integrated only when it changes** (`crates/pde/src/problem.rs`). The integral of the
   source (with SUPG test functions) and of the boundary fluxes is kept, and integrated again
   only after `Problem::source_mut`. Before, every residual applied the libCEED source operator.
4. **Jacobian reassembled from its steady and mass terms** (`crates/pde/src/operators.rs`). The
   Jacobian is affine in the shift. At its first assembly a Jacobian `Transport` keeps the
   coordinate values of its steady and of its mass terms; each change of the shift combines them
   (`steady + shift mass`, one pass over the entries) instead of a libCEED assembly. This matters
   for adaptive steps, where the shift changes every step: `TSJacobianEval` fell from 3.97 s to
   0.31 s on 4 processes.

Results: the L2 norms and total masses at the output times agree with those of the milestone 2
code within 1.2e-15 relative on 1, 4 and 8 processes, fixed and adaptive. Between process
counts they agree within 3.4e-12 (1 against 4, fixed) after the changes and 7.8e-13 (1 against
8) before and after. The assembled matrices equal the libCEED operators up to rounding (tests
`assembled_matches_shell`, `reassembly_follows_the_shift`).

## Profile after the changes

4 processes, fixed steps: 5.4 s in total, 2.7 s in `TSStep`. The steps are now the linear
algebra of the solver: `KSPSolve` 2.75 s, of which `PCApply` 2.11 s and `MatMult` (sparse) 1.70
s; `MatMultTranspose` (restriction) 0.31 s, `MatResidual` 0.30 s, `KSPOrthogonalize` 0.18 s;
`SNESFunctionEval` 0.15 s. Adaptive steps: `PCSetUp` 0.55 s (114 setups) and `TSJacobianEval`
0.31 s besides the solves.

The other phases of a run, 4 processes: `setup` 0.9 s (0.7 s before; the assembly of the levels
moved into it), `pressure` 0.9 s, `output` 1.1 s. On one process `output` (3.7 s for 6 output
times) was a third of the transport phase (9.7 s): the mass balance built new libCEED operators
at every output time. The next section fixes this.

## Setup and output phases

Two more changes, measured with `scripts/bench/transport.sh` against the state above in the same
session (the machine was busier than in the measurements above: the same binary took 0.0579 s
per step on 1 process instead of 0.0486). Fixed steps, seconds of each phase:

| Phase | 1 process | 4 processes | 8 processes |
| --- | ---: | ---: | ---: |
| `setup` before / after | 2.28 / 1.88 | 1.05 / 0.88 | 0.84 / 0.78 |
| `output` before / after | 4.05 / 3.08 | 1.29 / 0.95 | 0.91 / 0.62 |
| `transport` s per step before / after | 0.0579 / 0.0562 | 0.0162 / 0.0161 | 0.0111 / 0.0103 |

The adaptive runs show the same `setup` and `output` changes (1 process: 2.26 / 1.88 s and
3.93 / 3.04 s).

5. **Mass balance built once** (`crates/pde/src/balance.rs`, `crates/pde/src/problem.rs`). The
   operators of the mass balance (steady terms, mass term, source and advection on the finest
   level without essential boundaries), the loads of the flux conditions and the edges of the
   boundary sorted to their entries are built at the first `Problem::mass_balance` and kept;
   each later call applies them. On one process a balance took about 350 ms before, and the
   output phase falls by 1 s. The balances are bitwise the same.
6. **Steady and mass matrices from the Jacobian** (`crates/pde/src/operators.rs`,
   `crates/petsc/src/matrix.rs`). The residual's steady and mass matrices of the finest level
   are the two terms the Jacobian keeps for its reassembly (`Transport::assemble_terms`), in one
   coordinate pattern shared by both (`Matrix::duplicate`), instead of two more operators, two
   more libCEED assemblies and two more patterns. The lifting vector of the Dirichlet values is
   the Jacobian applied at shift 0. On one process an assembly of the finest level took about
   110 ms for the values and 85 ms for the pattern; `setup` falls by 0.4 s.

What remains of `output` is the VTU files of PETSc's viewer (`VecView` 1.6 s of the 3.1 s on one
process, 8 files) and the reductions of the summary. The rest of `setup` on one process: PETSc
and the study 0.3 s, the mesh 0.3 s, the discretizations of the levels 0.2 s, the coefficients
0.1 s, the assembly of the levels and the multigrid 0.6 s.

## Coupled baseline: cells and fields on a real slide

Task 11, first run of cell models coupled to the blood fields on a real slide (Strang splitting,
`crates/slidesim/src/coupled.rs`).

- Study: `testdata/studies/m3_real_slide/spt.study.json`: the SPT cell model
  (`testdata/models/spt_liver_cell.xml`) on the engine path (sbmlsim-core's `Stepper`, the model
  compiled to C), one cell per coarse element; fields `S` (5 mM at the portal wall) and `P`,
  carried by the Darcy flow; 24 h, split step 60 s, fixed transport steps of 10 s (BDF2), 25
  output times (hourly), 1 refinement.
- Scene: the 281 mg/kg slide `281mg_m1`, regenerated with the commands of
  `docs/results/2026-11-m3-wall-classes.md` ("Reproduce"; `zonation_layers.py` for this slide
  alone): 41,252 coarse triangles (= cells; 140 portal and 594 central wall pieces), 165,008
  triangles after the refinement. The study is copied next to the scene (`scene/scene.json`
  relative to it) with the model path made absolute.
- Run: `mpiexec -n 8 target/release/slidesim run spt.study.json --out run -- -log_view
  :run/log_view.txt`, `native-opt`, the generated C in the compiler cache from an earlier run.
  The machine ran other jobs at the same time (load average about 8).

Phases (s, `summary.json`), with the 2 h start of the same study for comparison:

| Phase | 2 h | 24 h |
| --- | ---: | ---: |
| `setup` | 1.2 | 1.1 |
| `pressure` | 0.6 | 0.6 |
| `transport` | 33.7 | 464.9 |
| `exchange` | 2.5 | 34.6 |
| `cells` | 27.4 | 166.4 |
| `output` | 0.7 | 5.6 |
| wall time of the run | 66 | 673 |

The run finishes in 11.2 min for 24 h of simulated time. Transport is 69 % of it, the cells
25 %, the exchange (the means of the fields over the cells and the give-back) 5 %.

**Cells.** 1,464 cell steps per cell (61 per hour between two outputs: the second half of a
split step merged with the first half of the next, the half steps at the output times apart),
1.63e9 integrator steps over all cells and 8.1e7 rejected (5 %): 27 integrator steps per cell
step on average, 89 in the first 2 h (the transient), about 1.2 million integrator steps per second
and process. The `cells` phase includes the wait for the slowest process at the barrier which
ends it (the cells are not balanced by their work).

**Transport and the restarts.** Every give-back changes the field between two steps and
restarts BDF2 (`Problem::modify`): the first step after it is a first-order half step and a
step of the BDF2 start, so a split step of 6 transport steps solves 7 times and sets up the
multigrid for 3 shifts. `-log_view` of the 24 h run: 17,280 `TSStep` (2 fields, 1,440 split
steps, 6 steps each), 20,160 `SNESSolve` (+17 %), 8,642 `PCSetUp` (31.8 s), 20,160
`TSJacobianEval` (20.5 s). The extra solves (pro rata 64 s of the 448 s of `SNESSolve`), the
multigrid setups and the Jacobian evaluations cost roughly 115 s, a quarter of the transport
phase; with one shift for the whole run they would nearly vanish. Without the merging of the
half cell steps there would be twice the restarts.

**Results.** The mass balances of both fields close at every output within 9.2e-11 of their
largest term (the interval balance, exchange and resupply included). Necrosis (cell state
`necrosis`, the cells of the necrosis map start with pattern 6) covers 26,848 of the 41,252
cells after 6 h and 33,787 after 24 h; in the quarter of the cells with the highest `protein`
its mean is 0.95 from 6 h on, in the quarter with the lowest 0 at 6 h and 0.35 at 24 h.
