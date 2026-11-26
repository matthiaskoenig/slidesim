# Perfusion

The sinusoids are a homogenized porous medium. A steady Darcy pressure solve gives the blood
flow from the walls with a high pressure (the portal walls of a liver slide) to those with a low
one (the central walls), and the species are carried by that flow
([Transport](transport.md)). The pressure is solved once, before the first time step, as a step
of its own so that a later milestone can repeat it when the cells change the permeability.

## Darcy pressure

The pressure \(p\) (Pa) solves

\[
-\nabla \cdot \left( \frac{K}{\mu} \nabla p \right) = 0
\]

with the viscosity \(\mu\) (Pa*s), fixed pressures on the tags of `perfusion.boundaries` and no
flow through every other boundary. P1 elements, CG with the geometric multigrid of the transport,
and one step of iterative refinement, so that the residual at the free nodes is at rounding.
At least one pressure tag is required.

**Anisotropic permeability.** \(K\) is anisotropic along the gradient \(\mathbf{g}\) of the
direction map (the zonation map on a liver slide), and blends smoothly to isotropic where the
gradient vanishes (at the peaks and valleys of the zonation):

\[
K = k_{\text{across}}\, I + w\, (k_{\text{along}} - k_{\text{across}})\, \mathbf{t}\, \mathbf{t}^T,
\qquad \mathbf{t} = \frac{\mathbf{g}}{|\mathbf{g}|},
\qquad w = \frac{|\mathbf{g}|^2}{|\mathbf{g}|^2 + g_0^2}
\]

\(K\) is symmetric positive definite whenever \(k_{\text{along}}, k_{\text{across}} > 0\), which is checked.
The gradient of a map is taken by central differences on the raster and then sampled bilinearly,
continuous across pixel borders. The literature finds the in-plane permeability of the lobule
close to isotropic ([perfusion parameters](../parameters/m2-perfusion-mouse.md)); strong radial
anisotropy is a sensitivity case (`radial10` of the
[milestone 2 results](../results/2026-11-m2-real-slide.md#studies)).

**Pressure islands.** A connected part of the mesh which no pressure tag reaches (a piece of
tissue bounded by walls alone) has a pressure determined only up to a constant. Its pressure is
fixed at 0, so it has no flow, and the rest of the mesh has the flow it has without it.
`summary.json` reports the islands (`perfusion.pressure_islands`: their count and area).

## Conservative velocity

The primal P1 velocity \(\mathbf{u}_h = -K \nabla p_h / \mu\) is not conservative: its normal
component jumps between the elements, and it has a normal component on walls without flow. On
the real slide of milestone 2 this showed as an `advection_defect` of 8.7 to 21.2 % of the portal
inflow and a leak of 0.3 to 2.3 % through walls without flow
([results](../results/2026-11-m2-real-slide.md#conservation-of-the-primal-darcy-velocity)).

Since milestone 3 the fields are carried by a conservative velocity, recovered from the pressure:
edge fluxes consistent with the residual of the discrete Darcy equations, equilibrated vertex
patch by vertex patch (as in the equilibrations of Ladeveze, of Ainsworth and Oden, and the
conservative flux of Larson and Niklasson 2004). For an element \(T\) and a vertex \(z\) of it,
with \(v_z\) the P1 function of \(z\), the residual is

\[
r_T(z) = \int_T f\, v_z - \int_T \frac{K}{\mu}\, \nabla p_h \cdot \nabla v_z
\]

and the flux through an edge is the sum of two moments, one per end: at each vertex, the moments
out of an element through its two edges there add up to \(r_T(z)\). Summed over the three
vertices of an element, its outgoing fluxes add up to \(\int_T f\), zero for the Darcy problem,
which has no sources. A sweep around each vertex solves its equations;
what is left free is chosen closest, in the least squares sense, to the moments of the primal
velocity. The result:

- the fluxes out of every element add up to zero;
- walls without a pressure condition have exactly zero flux;
- the fluxes through each pressure tag add up to its consistent flow (`tag_flows`).

The velocity at the quadrature points is the lowest-order Raviart-Thomas field of these fluxes,
on every mesh level. The primal velocity stays an output (`primal_velocity` in `velocity.vtu`)
for comparison. Patches across processes are solved on the mesh distributed with an overlap of
one element, and a vertex where the boundary touches itself (two holes meeting at a point) is
split into fans, each a patch of its own.

## Verification

- Manufactured solutions, isotropic and anisotropic \(K\) (direction from an analytic map), on
  the unit square and on a tagged mesh, at 4 refinement levels: pressure of order 2 in L2,
  velocity of order 1.
- Conservative velocity: zero divergence per element and zero flux through walls without flow,
  to rounding; `advection_defect` zero.
- Parallel: the same flow on 1, 2 and 3 processes within 1e-10 relative.
- End to end on a real slide: net inflow through each of the 5 portal wall pieces, net outflow
  through each of the 13 central ones, net flow below 3e-13 of the wall flow
  ([milestone 2 results](../results/2026-11-m2-real-slide.md#flow-checks)).
