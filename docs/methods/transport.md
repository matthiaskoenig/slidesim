# Transport

Every field \(u\) of a study (a concentration in mM) solves an advection-diffusion-reaction
equation on the finest level of the mesh:

\[
\phi\, \partial_t u + \mathbf{w} \cdot \nabla u = \nabla \cdot (\phi\, d\, \nabla u) - \phi\, k\, u + \phi\, f
\]

with the porosity \(\phi\) (the fraction of the volume the field lives in, the blood space), the
Darcy velocity \(\mathbf{w}\) (um/s) if the field has `advection` (else zero), the diffusion
coefficient \(d\) (um^2/s), the decay rate \(k\) (1/s) and the source \(f\) (mM/s). Each
coefficient is a number or a table over a map of the scene, evaluated at the quadrature points of
every mesh level. The velocity is the conservative velocity of the flow
([Perfusion](perfusion.md)), with \(\nabla \cdot \mathbf{w} = 0\) in every element and a
continuous normal component, so the advective form above equals the conservative one.

## Discretization

P1 elements on triangles, in the weak form

\[
M_\phi\, \dot u + \left(A_{\mathbf{w}} + K_{\phi d} + M_{\phi k}\right) u = \int \phi f\, v + \sum_i \int_{\Gamma_i} q_i\, v
\]

with \(M_c\) the mass matrix weighted with \(c\), \(A_{\mathbf{w}}\) the advection
\(\int (\mathbf{w} \cdot \nabla u)\, v\), \(K_c\) the stiffness matrix of \(c\) and \(q_i\) the
inward fluxes of the `flux` conditions. The operators are libCEED QFunctions
(`crates/pde/qfunctions/transport.h`), applied matrix-free or assembled into sparse matrices
from the same QFunctions.

**Boundary conditions.** A `dirichlet` value is an essential condition on its tag (where tags
meet, the tag listed last holds); a `flux` is an inward diffusive flux; every other boundary has
zero diffusive flux. With advection, a tag through which the flow enters needs a Dirichlet
condition, and where the flow leaves without a condition the outflow is free. At each Gauss point
of a boundary edge without a Dirichlet condition where \(\mathbf{w} \cdot \mathbf{n} < 0\), the
term \(\int (\mathbf{w} \cdot \mathbf{n})^-\, u\, v\) keeps advective flux from entering.

## SUPG

With advection, every term but diffusion is tested with the streamline upwind Petrov-Galerkin
test function \(v + \tau\, \mathbf{w} \cdot \nabla v\), the transient term included. The
parameter \(\tau\) per quadrature point follows Shakib, Hughes and Johan (1991) for the steady
terms, with the reaction as in Codina (2000):

\[
\tau = \left( \left(\frac{2 |\mathbf{w}|}{h}\right)^2 + 9 \left(\frac{4\, d_w}{h^2}\right)^2 + r^2 \right)^{-1/2}
\]

with \(d_w = \mathbf{w} \cdot D \mathbf{w} / |\mathbf{w}|^2\) the diffusion along the flow, \(r\)
the reaction, and \(h = 2 |\mathbf{w}| / \sum_i |\mathbf{w} \cdot \nabla N_i|\) the length of the
element along the flow (Tezduyar's \(h_{UGN}\), for a triangle the longest chord parallel to
\(\mathbf{w}\)). Its limits are \(h / (2 |\mathbf{w}|)\) for pure advection and
\(h^2 / (12\, d_w)\) where diffusion dominates. \(\tau\) leaves out the time step: with an
adaptive integrator it would change with every step and vanish for small steps, which brings
back the oscillations SUPG removes.

## Time stepping and solvers

- **BDF2** of PETSc `TS`, with fixed steps `time.dt` or adaptive steps (`time.adaptive`). With
  cells, each field steps to the end of a split step, whose last step is shortened to end there.
- **Linear solves**: CG for diffusion-reaction (symmetric positive definite), GMRES with
  advection, the residual reduced by `solver.linear_rtol` of the study. Its default, 1e-10, keeps
  results of different process counts within about the tolerance (on the generated lobule the
  fields of 1 and 2 processes differ by 5e-11; 1e-10 everywhere, the cells included, takes
  linear solves to 1e-12, see [Running a study](../usage.md)). With fixed steps the time
  integration has no error estimate to derive the tolerance from, so the study sets it: the SPT
  examples solve to 1e-6, where the fields differ from those of 1e-12 by 6e-8 of their largest
  value on the generated lobule, below a thousandth of the change from halving the time step,
  and the residuals of the mass balances grow with the tolerance ([performance](../results/2026-11-m4-performance.md)). A
  solver which does not converge is an error, not a silently wrong step.
- **Geometric multigrid** (`PCMG`, V-cycles) over the refinement levels of the mesh. The coarser
  levels discretize the same equation again (coefficients and velocity at their own quadrature
  points), not the Galerkin product. Smoothers: 2 iterations of Chebyshev preconditioned by
  Jacobi, on \([0.1, 1.1]\, \lambda_{max}\) for symmetric operators and \([0.5, 1.2]\,
  \lambda_{max}\) with advection, whose Jacobi preconditioned spectrum is complex. Coarse level:
  one application of algebraic multigrid (`PCGAMG`), coarsened until a level has at most 5000
  equations, which it solves directly (PETSc's default stops at 50: the larger direct solve
  halves the Krylov iterations on the generated lobule,
  [performance](../results/2026-11-m4-performance.md)). Every level, the finest included, works on
  the operator assembled from libCEED, several times cheaper to apply than the matrix-free one
  for P1 triangles ([performance](../results/2026-11-m3-performance.md)).
- **Reuse**: the operators and the multigrid are built once per field; a step changes only the
  shift of the Jacobian, which is reassembled as the steady terms plus the shift times the mass
  terms, one pass over the entries.

The solvers of a field take the PETSc options prefix `<field id>_`, so every choice above can be
changed per field from the study or the command line ([Running a study](../usage.md#petsc-options)).

## Mass balance

`summary.json` has, per field and output interval, the balance of the amount of substance
\(\int \phi\, u\) from the discrete equations, each term its mean rate over the interval:

\[
\begin{aligned}
\text{storage rate} = {} & \text{source} + \text{exchange} - \text{advection defect} \\
& + \sum_{\text{tags}} \left(\text{advective} + \text{diffusive} + \text{resupply}\right)
\end{aligned}
\]

The diffusive inflow at a Dirichlet tag is the consistent flux, the residual of the discrete
equations there; the advective part is the flux of the velocity times the concentration at the
Gauss points of the edges. The terms are integrated by the weights of the BDF steps, the changes
between steps (the exchange with the cells) included, and add up to the change of the amount
exactly, up to the residuals of the linear solves. The `advection_defect` measures the
non-conservation of the velocity: up to 21 % of the portal inflow with the primal P1 Darcy
velocity of milestone 2 ([results](../results/2026-11-m2-real-slide.md#advection_defect)), zero to
rounding with the conservative velocity of milestone 3.

## Verification

- Manufactured solutions on the unit square and on tagged meshes, at 4 refinement levels: the
  convergence rates of P1, from a cell Peclet number below 1 to an advection-dominated case
  (Peclet about 100), and a front test without oscillations.
- Conservation: the mass balance closes to 1e-8 relative or better (3e-13 on the real slide of
  milestone 2).
- Parallel: the same results on 1, 2 and 3 processes within 1e-10 relative, with fixed steps
  (and, on the generated lobule with cells, the linear solves and the cells to 1e-12).
