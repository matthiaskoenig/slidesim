//! The mass balance of a [`crate::Problem`] at the end of a time step, from its discrete
//! equations.

use std::collections::BTreeSet;
use std::rc::Rc;

use petsc::{Point, Vector};

use crate::discretization::EDGE_GAUSS;
use crate::operators::PointValues;
use crate::{Discretization, PointData, Result, SourceField, Terms, Transport};

/// The name of the entry of [`MassBalance::boundary`] for the edges of the boundary which no tag
/// marks.
pub const UNTAGGED: &str = "(untagged)";

/// The balance of the amount of substance `int phi c` of a field `c` (porosity `phi`) at the end
/// of a time step, each term in mM*um^2/s, from the discrete equations of [`crate::Problem`]:
///
/// `storage_rate = source + sum(boundary) - advection_defect`,
///
/// up to the residuals of the linear solves.
///
/// Derivation: the discrete equations, tested with the SUPG test functions `v + tau u . grad v`
/// of the P1 functions `v`, hold for the `v` of every free dof. Their sum over all dofs, the
/// constrained (Dirichlet) ones included, is the equation tested with `v = 1`, where
/// `grad v = 0` takes the stabilization and diffusion out:
///
/// `int phi c_t + int phi k c + int u . grad c + int_Gamma_N (u . n)^- c - int phi f
///   - int_Gamma_q q = sum_D R_i`,
///
/// with `R_i` the residual of the equation of dof `i`, zero at the free dofs and the consistent
/// flux (the diffusive inflow, as tested with `v_i`) at the Dirichlet dofs `D`,
/// `int_Gamma_q q` the fluxes of the conditions, and `int_Gamma_N (u . n)^- c` the backflow
/// term over the edges of the boundary `Gamma_N` without Dirichlet condition
/// ([`PointData::with_backflow`], `(u . n)^- = max(0, -u . n)`). Advection is in the
/// non-conservative form, so its boundary term is apart: with `int_Gamma (u . n) c` over the
/// boundary of the domain,
///
/// `int u . grad c = int_Gamma (u . n) c + advection_defect`,
///
/// which defines `advection_defect`. On `Gamma_N` the backflow term cancels the inflowing part
/// of `int (u . n) c`, which leaves the outflow `int_Gamma_N (u . n)^+ c`; on the Dirichlet
/// edges `Gamma_D` the advective flux `int (u . n) c` is whole, in and out. For a velocity which conserves mass (`div u = 0` with a
/// continuous normal component) it is zero; the Darcy velocity of P1 pressures has jumps of
/// `u . n` between the cells (and within a cell a divergence where the permeability varies),
/// and `advection_defect` is their effect, `sum_edges int [u . n] c`: the non-conservation of
/// the primal velocity, reported, not corrected. Hence the terms:
///
/// - `storage_rate`: `int phi c_t`, with `c_t` the time derivative of the BDF formula at the end
///   of the step (Dirichlet values are constant);
/// - `source`: `int phi (f - k c)`, the sources less the decay;
/// - `boundary`: the inflow through each tag (positive into the domain): at a Dirichlet tag
///   `sum R_i` over its dofs (a dof of several Dirichlet tags counts for the one listed last,
///   as its value does) minus the advective flux; at a flux tag `int q` over its edges minus
///   the advective flux; at the other tags of edges of the boundary, minus the advective flux
///   alone (free outflow: no diffusive flux). The advective flux of an edge is
///   `int (u . n) c` on an edge a Dirichlet tag marks, else the outflow `int (u . n)^+ c`
///   (backflow brings nothing in); both by the Gauss points of the edge. An edge of
///   the boundary counts for one tag: of the tags of the conditions of the field the one listed
///   last which marks it, else the first tag of edges of the mesh which marks it, else
///   [`UNTAGGED`]. The entries: the tags of the conditions in the order listed, then the other
///   tags of edges of the mesh which mark edges of the boundary, in the order of the mesh, then
///   [`UNTAGGED`] if there are edges of the boundary without tag;
/// - `advection_defect`: `int u . grad c - int_Gamma (u . n) c`.
///
/// The velocity on an edge of the boundary is that of the flow at its Gauss points
/// ([`crate::Flow::edge_velocity`]): the pressure gradient of its cell with the permeability
/// at the points.
#[derive(Clone, Debug, PartialEq)]
pub struct MassBalance {
    pub storage_rate: f64,
    pub source: f64,
    pub boundary: Vec<(String, f64)>,
    pub advection_defect: f64,
}

/// What the mass balance of a problem is computed from.
pub(crate) struct Parts<'a> {
    /// The finest level, with the Dirichlet tags essential.
    pub(crate) disc: &'a Discretization,
    /// The finest level without essential boundaries, with the local vectors of `disc`.
    pub(crate) free: &'a Discretization,
    /// The coefficients of the equation at the points of `disc`.
    pub(crate) data: &'a Rc<PointData>,
    /// The source `f` and the porosity `phi` at the points.
    pub(crate) f: &'a [f64],
    pub(crate) porosity: &'a [f64],
    /// The solution with its Dirichlet values: a local vector of `disc`.
    pub(crate) c: &'a Vector,
    /// Its time derivative, zero at the Dirichlet dofs: a local vector of `disc`.
    pub(crate) c_t: &'a Vector,
    /// The tags of the conditions of the field, in the order listed.
    pub(crate) conditions: &'a [String],
    /// The Dirichlet tags, in the order listed.
    pub(crate) dirichlet: &'a [&'a str],
    /// The flux tags and their fluxes, in the order listed.
    pub(crate) fluxes: &'a [(&'a str, f64)],
    /// The tags of edges of the mesh, in its order.
    pub(crate) edge_tags: &'a [String],
    /// The velocity at the Gauss points of the edges of the boundary of `disc`, sorted by edge;
    /// none without advection.
    pub(crate) edge_velocity: &'a [(Point, [[f64; 2]; 2])],
}

/// The mass balance of `parts` (see [`MassBalance`]). Collective.
pub(crate) fn mass_balance(parts: &Parts<'_>) -> Result<MassBalance> {
    let Parts {
        disc,
        free,
        data,
        f,
        porosity,
        c,
        c_t,
        conditions,
        dirichlet,
        fluxes,
        edge_tags,
        edge_velocity,
    } = *parts;
    let comm = disc.dm().comm();
    let weights = disc.quadrature_weights();
    let values = PointValues::new(disc)?;
    let mut c_q = vec![0.0; disc.num_quadrature_points()];
    values.apply(c, &mut c_q)?;
    let mut c_t_q = vec![0.0; disc.num_quadrature_points()];
    values.apply(c_t, &mut c_t_q)?;

    let storage: f64 = (weights.iter().zip(data.mass()).zip(&c_t_q))
        .map(|((w, m), c_t)| w * m * c_t)
        .sum();
    let source: f64 = (weights.iter().zip(f).zip(porosity))
        .zip(data.reaction().iter().zip(&c_q))
        .map(|(((w, f), phi), (r, c))| w * (phi * f - r * c))
        .sum();

    // The residual of the equations of every dof: R = S c + M c_t - int phi f (v + SUPG) - load.
    let mut residual = free.global_vector()?;
    let mut work = free.global_vector()?;
    Transport::new(free, Rc::clone(data), Terms::Steady)?.apply_local(c, &mut residual)?;
    Transport::new(free, Rc::clone(data), Terms::Mass)?.apply_local(c_t, &mut work)?;
    residual.axpy(1.0, &work)?;
    let mut sources = SourceField::stabilized(free, Rc::clone(data))?;
    for ((s, f), phi) in sources.values_mut().iter_mut().zip(f).zip(porosity) {
        *s = phi * f;
    }
    sources.apply(&mut work)?;
    residual.axpy(-1.0, &work)?;
    let mut ones = free.global_vector()?;
    ones.set(1.0)?;
    let mut flux_inflows = Vec::with_capacity(fluxes.len());
    if !fluxes.is_empty() {
        residual.axpy(-1.0, &free.boundary_load(fluxes)?)?;
        for &flux in fluxes {
            flux_inflows.push(free.boundary_load(&[flux])?.dot(&ones)?);
        }
    }
    let dirichlet_inflows = disc.essential_sums(free, &residual, dirichlet)?;

    // int u . grad c: the advection of the Galerkin form, tested with v = 1.
    let n = disc.num_quadrature_points();
    let velocity = Rc::new(
        PointData::new(vec![0.0; n], vec![0.0; n], vec![[0.0; 3]; n])
            .with_velocity(data.velocity().to_vec()),
    );
    Transport::new(free, velocity, Terms::Steady)?.apply_local(c, &mut work)?;
    let advection = work.dot(&ones)?;

    // The advective outflow through each edge of the boundary, by entry.
    let mut names: Vec<String> = Vec::new();
    for tag in conditions {
        if !names.contains(tag) {
            names.push(tag.clone());
        }
    }
    let edges = disc.boundary_edges()?;
    let dm = disc.dm();
    let marked = |tag: &str| -> Result<BTreeSet<Point>> {
        let mut points = BTreeSet::new();
        for value in dm.label_values(tag)? {
            points.extend(dm.stratum_points(tag, value)?);
        }
        Ok(points)
    };
    let condition_points = conditions
        .iter()
        .map(|tag| marked(tag))
        .collect::<Result<Vec<_>>>()?;
    let mut fixed = BTreeSet::new();
    for tag in dirichlet {
        fixed.extend(marked(tag)?);
    }
    let others: Vec<&String> = edge_tags.iter().filter(|t| !names.contains(t)).collect();
    let other_points = others
        .iter()
        .map(|tag| marked(tag))
        .collect::<Result<Vec<_>>>()?;
    // Entries: the conditions, then the other tags, then the untagged edges.
    let (num_conditions, num_others) = (names.len(), others.len());
    let mut outflow = vec![0.0; num_conditions + num_others + 1];
    let mut marks = vec![0; num_others + 1];
    // int_Gamma (u . n) c, in and out.
    let mut through = 0.0;
    c.with_array(|c| {
        for edge in &edges {
            let entry = match conditions
                .iter()
                .zip(&condition_points)
                .rev()
                .find(|(_, points)| points.contains(&edge.point))
            {
                Some((tag, _)) => names.iter().position(|n| n == tag).expect("a condition"),
                None => {
                    let other = other_points
                        .iter()
                        .position(|points| points.contains(&edge.point))
                        .unwrap_or(num_others);
                    marks[other] = 1;
                    num_conditions + other
                }
            };
            let Ok(k) = edge_velocity.binary_search_by_key(&edge.point, |(p, _)| *p) else {
                // Without advection.
                continue;
            };
            let dirichlet = fixed.contains(&edge.point);
            let [a, b] = edge.ends;
            for (s, u) in EDGE_GAUSS.iter().zip(edge_velocity[k].1) {
                // `normal` has the length of the edge, the Gauss weight is 1/2.
                let u_n = u[0] * edge.normal[0] + u[1] * edge.normal[1];
                let flux = 0.5 * u_n * ((1.0 - s) * c[a] + s * c[b]);
                through += flux;
                outflow[entry] += if dirichlet || u_n > 0.0 { flux } else { 0.0 };
            }
        }
    })?;

    let storage_rate = comm.sum_real(storage)?;
    let source = comm.sum_real(source)?;
    let outflow = outflow
        .iter()
        .map(|&o| comm.sum_real(o))
        .collect::<petsc::Result<Vec<f64>>>()?;
    let marks = marks
        .iter()
        .map(|&m| comm.max_int(m))
        .collect::<petsc::Result<Vec<_>>>()?;
    let mut boundary: Vec<(String, f64)> = names
        .iter()
        .zip(&outflow)
        .map(|(tag, o)| (tag.clone(), -o))
        .collect();
    for ((tag, _), inflow) in fluxes.iter().zip(&flux_inflows) {
        let entry = names.iter().position(|n| n == tag).expect("a condition");
        boundary[entry].1 += inflow;
    }
    for (tag, inflow) in dirichlet.iter().zip(&dirichlet_inflows) {
        let entry = names.iter().position(|n| n == tag).expect("a condition");
        boundary[entry].1 += inflow;
    }
    for (i, tag) in others
        .iter()
        .map(|t| t.as_str())
        .chain([UNTAGGED])
        .enumerate()
    {
        if marks[i] > 0 {
            boundary.push((tag.to_string(), -outflow[num_conditions + i]));
        }
    }
    Ok(MassBalance {
        storage_rate,
        source,
        boundary,
        advection_defect: advection - comm.sum_real(through)?,
    })
}
