# Cell models

A study with `cells` puts an SBML model into every element of the coarse mesh (the mesh before
its refinements) whose area tag is in `cells.areas`: one ODE system per element, a "cell". The
linked species of the model exchange with the fields of the blood, and the fields step on the
finest level, coupled to the cells by Strang splitting. The model is data: slidesim knows its
species, parameters and compartments only by the ids the study maps.

## Parameters and volumes per cell

- Every parameter of `cells.parameters` is the mean of its coefficient (a number or a table over
  a map) over the element's tissue: the quadrature points of its finest children in the areas of
  the cells. With `replace`, the parameter takes the replacement value where the element mean of
  the replacement map is above its threshold. A map without a value in a cell's tissue is an
  error naming the element.
- The compartment `volumes.blood` of the model is the porosity of the element, `volumes.cells`
  one minus the porosity; the model keeps its own volume units, only their ratio matters.
- Time: the model's time unit is `cells.time_unit_s` seconds.

## Two paths behind one interface

A cell solver sets up the cells of a process from their parameters, advances all of them from
\(t_0\) to \(t_1\) with given blood concentrations, and reports their states, steps and rejected
steps. The study chooses the path with `cells.solver`.

### Engine path

One instance of sbmlsim-core's engine per cell (its `Stepper`): CVODE, or IDA for a model with
algebraic rules, with KLU, the events and discontinuities of an interval handled as sbmlsim-core's
`simulate` handles them. The model is translated by sbmlsim-core's C target into point functions
(initial values, rates, the analytic sparse Jacobian, roots, triggers, event assignments),
compiled with the system C compiler at the start of the run, cached by the hash of the source and
the flags, and loaded: native model code, no WebAssembly in a run. The solver memory of a cell
stays between calls and is reinitialized from the state it is given. This path takes every model
sbmlsim-core takes (events, delays, algebraic rules) and is the reference of the batched path.

### Batched path

The cells of a process as one libCEED operator: one point per cell, the states, the parameters
and the step control of every cell in libCEED vectors, read and written through strided element
restrictions without a basis. Its QFunction comes from sbmlsim-core's QFunction target: the
model's point functions and the Rosenbrock integrator RODAS4 (an embedded error estimate,
L-stable, a dense LU of the small iteration matrix per point, the analytic Jacobian), which makes
step attempts per point until the point reaches the end of the advance or fails, at most
\(10^6\) per application. An advance applies the operator once: each cell steps to the end on
its own, so a process reads and writes its cells once per advance and a finished cell waits for
no other.

Each cell takes its own steps with the tolerances of the study (`rtol` 1e-6, `atol` 1e-9 by
default) and keeps its step size from one advance to the next. So a cell depends on no other
cell, on the number of processes, on the libCEED backend or on its blocking. The QFunction calls
nothing but the C math functions, which is what libCEED's JIT needs to compile it for CUDA and
HIP later. Models with events, delays or algebraic rules are refused with an error naming the
feature; they run on the engine path. A cell fails, and the run stops naming it on every process,
when rejected steps (an error estimate above the tolerances, values which are not finite, a
singular iteration matrix) make its next step smaller than 1e-10 time units of the model or too
small to advance the time, or after \(10^6\) step attempts in one advance.

## Strang splitting

The time from one output time to the next is split into equal split steps of at most
`cells.split_step`, \([a_k, b_k]\) with midpoint \(m_k\). Each is

1. the cells over the first half, \([a_k, m_k]\);
2. every field over the whole step to \(b_k\), with its BDF2 steps of `time.dt` (or adaptive,
   the last step shortened to end at \(b_k\));
3. the cells over the second half, \([m_k, b_k]\).

The second half of one split step and the first half of the next, with no transport between
them, are one cell step from \(m_k\) to \(m_{k+1}\): the same scheme with half the cell steps and
half the restarts of the time integration of the fields. At an output time the half steps stay
apart, so the cells reach it before the outputs are written. The splitting is of order 2 in the
split step once the split step is short against the time in which the transport replenishes the
blood next to a cell. Over longer split steps a cell depletes its blood within a cell step with
no supply, and the error is of first order: with the perfusion of a lobule (the blood replenished
within seconds) and split steps of minutes, the amount the linear model takes up on the generated
lobule in 1 h differs by 3 % between split steps of 120 and 60 s, the differences of successive
halvings falling by a factor 1.9. With the transport 100 times slower, the same split steps show
order 2.

The SPT studies of the lobules are in this first order regime over all split steps that matter
([split-step results](../results/2026-11-m4-split-step.md)): against a reference at 5 s, the
substrate uptake of the generated lobule in 1 h is off by 4.6, 1.7, 0.6, 0.2 and 0.1 % at split
steps of 120, 60, 30, 15 and 10 s (observed order 1.3 to 1.7), and on a coarser mesh of the
lobule over 4 h the necrosis front lags (necrotic share 0.015 and 0.005 below the reference at
120 and 60 s, Dice down to 0.946 and 0.984, within 0.002 and above 0.997 from 30 s on). The cost
of the cells grows almost in proportion to the number of split steps, each of which starts the
cells with a transient of their blood. The largest split step which keeps the substrate (S)
uptake within 1 % and the necrotic share and Dice within 0.01 is 30 s, at about 1.7 times the
cost of 60 s (the 24 h cost an estimate); the example studies keep 60 s until that choice is
made. It is a choice for the substrate and the necrosis, not for every quantity: at 30 s the
product P given back is off by up to 4.2 %, the S uptake of the 7 lobules in their first 15 min
by up to 13 %, and the necrosis was measured on a coarser mesh over 4 h, whose larger cells
refill more slowly, not on the production mesh over 24 h. At an output time a cell has just made a
half step without transport, so a species the cells give to a quickly washed out blood (the
product P of the SPT model) piles up in the blood: the states P_ext and P at the output times
measure the splitting at every practical split step. A coupling in which the cell step sees the
supply and washout of its blood would remove both errors.

## Exchange with the blood

A cell step from \(t_0\) to \(t_1\) reads the means of the linked fields over each cell, the
concentrations in its blood at \(t_0\), advances the cells with them, and gives the change of
each concentration back to its field. Both are libCEED operators on the quadrature points of the
fine level, so that the whole coupling can move to a GPU.

**Means.** Cell \(i\) is the coarse element \(e_i\) with its blood volume
\(V_i = \int_{e_i} \phi\). The concentration in its blood is the porosity-weighted mean of the
field, the L2 projection to the coarse elements:

\[
u_i = \frac{1}{V_i} \int_{e_i} \phi\, u
\]

**Give-back.** The change \(\delta_i\) a cell made is given back as the amount
\(A_i = \delta_i V_i\), distributed over the nodes of the element with the weights
\(w_{ij} = \int_{e_i} \phi\, v_j\) of the P1 functions \(v_j\):

- an increase in proportion to \(w_{ij}\), the transpose of the mean;
- a decrease in proportion to the substance at each node, \(w_{ij}\, u_j^+\): the positivity
  safeguard. A cell whose blood concentration stays nonnegative takes at most \(u_i V_i\), and
  no nonnegative nodal value becomes negative, however steep the uptake.

The share of a node on a Dirichlet boundary is exchanged with the fixed value there, which the
boundary resupplies (or takes away), and booked as `resupply` of its tag in the mass balance. The
total amount of substance of the field changes by exactly \(\sum_i A_i\) plus the resupply, to
rounding. The amount of every single coarse element is not conserved: P1 fields share every node
with the neighbouring elements, so an exact per-element transfer has no unique answer.

After a give-back the time integration of the field starts afresh, with the start of BDF2 as at
t = 0. A restart builds no new preconditioner (the multigrid is kept while the shift stays within
a factor of 2 of the one it was built at); its cost is the extra solve of the half step, 17 % more
solves than steps in the 24 h run on a real slide
([performance results](../results/2026-11-m3-performance.md#coupled-runs-faster-restarts-the-interval-balance-the-batched-cells)).

## Verification

- Engine path: the advance call of sbmlsim-core against one `simulate` over the whole time, on
  the SPT model with parameters drawn over their ranges, in split steps of several sizes.
- Batched against engine path, per cell, on the same drawn SPT instances
  (`batched_matches_engine`), and the same states to round-off on the libCEED backends
  `/cpu/self/ref/serial` and `/cpu/self/opt/blocked` and on 1, 2 and 3 processes
  (`same_on_backends`, `parallel_on_2_processes`, `parallel_on_3_processes`).
- Coupling, manufactured: a linear cell model in a closed domain with a known coupled solution,
  order 2 of the Strang splitting in the split step (`strang_order_two`,
  `testdata/studies/cells_manufactured`).
- Coupling on the generated lobule (`testdata/studies/lobules/linear.study.json`): order 2 of
  the splitting with split steps of 240, 120 and 60 s and the transport slowed
  (`strang_order_two_on_generated_lobule`), conservation
  (`coupled_conservation_on_generated_lobule`), and 1, 2 and 3 processes within 1e-10
  (`cells_parallel_matches_one_on_generated_lobule`, with the linear solves and the cells to
  1e-12: with the linear solves to 1e-10 the fields of other partitions differ by 5e-11, which
  the cells carry to 1.4e-10, and a cell whose step control decides otherwise differs by its
  integration error).
- Conservation: the amount of substance in blood and cells together changes by the boundary
  flows only (`coupled_conservation`, also with adaptive steps).
- End to end on a real slide, 24 h on 41,252 cells: the two paths agree on every cell's
  `necrosis` within 3.8e-6, and the mass balances close within 9.2e-11
  ([performance results](../results/2026-11-m3-performance.md#coupled-baseline-on-the-batched-path)).

## The SPT liver cell model

The first model is the SPT model of pkdb_models (`testdata/models/spt_liver_cell.xml`): a
substrate S from the blood is taken up and converted by the zonated enzyme CYP2E1 into a product
P and a toxic product T; when T stays above a threshold the cell dies (`necrosis`). Its variant
for slidesim is plain SBML (no `comp`, no `distrib`) with a pattern 7, in which the protein is a
parameter `protein_ext` set from outside: from the slide's CYP2E1 readout, or by pattern 6, the
CYP2E1 profile of control mice by zonation position, where the measured necrosis has removed the
staining. Acetaminophen is the case behind it; comparing its necrosis with the one measured on
dosed slides is deferred to a later milestone on larger hardware
([dosed slides](../results/m3-dosed-slides.md)).
