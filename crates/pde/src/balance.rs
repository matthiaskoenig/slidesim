//! The mass balance of a [`crate::Problem`] at the end of a time step, from its discrete
//! equations.

use std::cell::RefCell;
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
/// `storage_rate = source + sum(boundary totals) - advection_defect`,
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
/// edges `Gamma_D` the advective flux `int (u . n) c` is whole, in and out. For a velocity
/// which conserves mass (`div u = 0` with a continuous normal component), as the conservative
/// velocity of a [`crate::Flow`] ([`crate::Flow::velocity`]), it is zero up to rounding; for
/// another velocity, as the primal Darcy velocity of P1 pressures with its jumps of `u . n`
/// between the cells, `advection_defect` is their effect, `sum_edges int [u . n] c`, reported,
/// not corrected. Hence the terms:
///
/// - `storage_rate`: `int phi c_t`, with `c_t` the time derivative of the BDF formula at the end
///   of the step (Dirichlet values are constant);
/// - `source`: `int phi (f - k c)`, the sources less the decay;
/// - `boundary`: the inflow through each tag, positive into the domain, as a
///   [`BoundaryInflow`] of an advective and a diffusive part, whose sum is the term of the
///   balance. An edge of the boundary counts for one tag: of the tags of the conditions of the
///   field the one listed last which marks it, else the first tag of edges of the mesh which
///   marks it, else [`UNTAGGED`]. The entries: the tags of the conditions in the order listed,
///   then the other tags of edges of the mesh which mark edges of the boundary, in the order of
///   the mesh, then [`UNTAGGED`] if there are edges of the boundary without tag;
/// - `advection_defect`: `int u . grad c - int_Gamma (u . n) c`.
///
/// The velocity on an edge of the boundary is that of the flow at its Gauss points
/// ([`crate::Flow::edge_velocity`]): the conservative velocity of its cell, whose normal
/// component is zero on a wall without flow, so no substance leaves there.
#[derive(Clone, Debug, PartialEq)]
pub struct MassBalance {
    pub storage_rate: f64,
    pub source: f64,
    pub boundary: Vec<BoundaryInflow>,
    pub advection_defect: f64,
}

/// The inflow through the edges of the boundary one tag counts for (see [`MassBalance`]), in
/// mM*um^2/s, positive into the domain, as two parts:
///
/// - `advective`: minus the advective flux by the Gauss points of the edges, `-int (u . n) c`
///   on an edge a Dirichlet tag marks (in and out), else `-int (u . n)^+ c` (the outflow; the
///   backflow term lets nothing in); zero without advection;
/// - `diffusive`: at a Dirichlet tag the consistent flux `sum R_i` over its dofs (a dof of
///   several Dirichlet tags counts for the one listed last, as its value does); at a flux tag
///   `int q` over its edges; zero elsewhere (free outflow: no diffusive flux).
#[derive(Clone, Debug, PartialEq)]
pub struct BoundaryInflow {
    pub tag: String,
    pub advective: f64,
    pub diffusive: f64,
}

impl BoundaryInflow {
    /// The inflow, `advective + diffusive`: the term of the balance.
    pub fn total(&self) -> f64 {
        self.advective + self.diffusive
    }
}

/// What the mass balance of a problem is built for: the parts which stay from step to step.
pub(crate) struct Setup<'a> {
    /// The finest level, with the Dirichlet tags essential.
    pub(crate) disc: &'a Discretization,
    /// The finest level without essential boundaries, with the local vectors of `disc`.
    pub(crate) free: &'a Discretization,
    /// The coefficients of the equation at the points of `disc`.
    pub(crate) data: &'a Rc<PointData>,
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

/// The state of a problem at the end of a step, of which [`Balance::compute`] computes the mass
/// balance.
pub(crate) struct State<'a> {
    /// The source `f` and the porosity `phi` at the points.
    pub(crate) f: &'a [f64],
    pub(crate) porosity: &'a [f64],
    /// The solution with its Dirichlet values: a local vector of the finest level.
    pub(crate) c: &'a Vector,
    /// Its time derivative, zero at the Dirichlet dofs: a local vector of the finest level.
    pub(crate) c_t: &'a Vector,
    /// The values of local vectors of the finest level at its quadrature points.
    pub(crate) values: &'a PointValues,
}

/// The mass balance of a problem (see [`MassBalance`]) with what stays from step to step built
/// once: the operators on the finest level without essential boundaries, the loads of the flux
/// conditions, and the edges of the boundary by entry of [`MassBalance::boundary`].
#[derive(Debug)]
pub(crate) struct Balance {
    disc: Discretization,
    free: Discretization,
    data: Rc<PointData>,
    /// The steady terms and the mass term on `free`.
    steady: Transport,
    mass: Transport,
    /// `int u . grad c`: the advection of the Galerkin form, without the other terms.
    advection: Transport,
    /// `int phi f (v + SUPG)` on `free`.
    sources: RefCell<SourceField>,
    /// The load of the flux conditions on `free`, if any.
    load: Option<Vector>,
    /// The global vector of ones of `free`.
    ones: Vector,
    /// The Dirichlet tags, in the order listed.
    dirichlet: Vec<String>,
    /// The entries of [`MassBalance::boundary`]: the tags of the conditions, then the other
    /// tags of edges of the mesh, then [`UNTAGGED`].
    entries: Vec<String>,
    /// The number of entries of the conditions.
    num_conditions: usize,
    /// Whether an entry after those of the conditions marks an edge of the boundary (on any
    /// process).
    marked: Vec<bool>,
    /// The diffusive inflow of each flux condition, by entry.
    flux_inflows: Vec<(usize, f64)>,
    /// The edges of the boundary with a velocity (none without advection).
    edges: Vec<EdgeFlux>,
}

/// An edge of the boundary with the flow through it.
#[derive(Debug)]
struct EdgeFlux {
    /// The entry of [`MassBalance::boundary`] the edge counts for.
    entry: usize,
    /// Whether a Dirichlet tag marks it: its advective flux counts in and out.
    dirichlet: bool,
    /// The entries of local vectors of its two vertices.
    ends: [usize; 2],
    /// The outward normal, of the length of the edge.
    normal: [f64; 2],
    /// The velocity at its two Gauss points.
    velocity: [[f64; 2]; 2],
}

impl Balance {
    /// The mass balance of the problem of `setup`. Collective.
    pub(crate) fn new(setup: &Setup<'_>) -> Result<Balance> {
        let Setup {
            disc,
            free,
            data,
            conditions,
            dirichlet,
            fluxes,
            edge_tags,
            edge_velocity,
        } = *setup;
        let comm = disc.dm().comm();
        let mut ones = free.global_vector()?;
        ones.set(1.0)?;
        let load = if fluxes.is_empty() {
            None
        } else {
            Some(free.boundary_load(fluxes)?)
        };

        let mut entries: Vec<String> = Vec::new();
        for tag in conditions {
            if !entries.contains(tag) {
                entries.push(tag.clone());
            }
        }
        let num_conditions = entries.len();
        let mut flux_inflows = Vec::with_capacity(fluxes.len());
        for &flux in fluxes {
            let entry = entries
                .iter()
                .position(|n| n == flux.0)
                .expect("a condition");
            flux_inflows.push((entry, free.boundary_load(&[flux])?.dot(&ones)?));
        }
        let others: Vec<&String> = edge_tags.iter().filter(|t| !entries.contains(t)).collect();
        let num_others = others.len();
        entries.extend(others.iter().map(|t| t.to_string()));
        entries.push(UNTAGGED.to_string());

        let dm = disc.dm();
        let marked_points = |tag: &str| -> Result<BTreeSet<Point>> {
            let mut points = BTreeSet::new();
            for value in dm.label_values(tag)? {
                points.extend(dm.stratum_points(tag, value)?);
            }
            Ok(points)
        };
        let condition_points = conditions
            .iter()
            .map(|tag| marked_points(tag))
            .collect::<Result<Vec<_>>>()?;
        let mut fixed = BTreeSet::new();
        for tag in dirichlet {
            fixed.extend(marked_points(tag)?);
        }
        let other_points = others
            .iter()
            .map(|tag| marked_points(tag))
            .collect::<Result<Vec<_>>>()?;
        let mut marks = vec![0; num_others + 1];
        let mut edges = Vec::new();
        for edge in disc.boundary_edges()? {
            let entry = match conditions
                .iter()
                .zip(&condition_points)
                .rev()
                .find(|(_, points)| points.contains(&edge.point))
            {
                Some((tag, _)) => entries.iter().position(|n| n == tag).expect("a condition"),
                None => {
                    let other = other_points
                        .iter()
                        .position(|points| points.contains(&edge.point))
                        .unwrap_or(num_others);
                    marks[other] = 1;
                    num_conditions + other
                }
            };
            // Without advection, no edge has a velocity.
            if let Ok(k) = edge_velocity.binary_search_by_key(&edge.point, |(p, _)| *p) {
                edges.push(EdgeFlux {
                    entry,
                    dirichlet: fixed.contains(&edge.point),
                    ends: edge.ends,
                    normal: edge.normal,
                    velocity: edge_velocity[k].1,
                });
            }
        }
        let marked = marks
            .iter()
            .map(|&m| Ok(comm.max_int(m)? > 0))
            .collect::<Result<Vec<_>>>()?;

        // The velocity alone, for the advection of the Galerkin form.
        let n = disc.num_quadrature_points();
        let velocity = Rc::new(
            PointData::new(vec![0.0; n], vec![0.0; n], vec![[0.0; 3]; n])
                .with_velocity(data.velocity().to_vec()),
        );
        Ok(Balance {
            disc: disc.clone(),
            free: free.clone(),
            data: Rc::clone(data),
            steady: Transport::new(free, Rc::clone(data), Terms::Steady)?,
            mass: Transport::new(free, Rc::clone(data), Terms::Mass)?,
            advection: Transport::new(free, velocity, Terms::Steady)?,
            sources: RefCell::new(SourceField::stabilized(free, Rc::clone(data))?),
            load,
            ones,
            dirichlet: dirichlet.iter().map(|t| t.to_string()).collect(),
            entries,
            num_conditions,
            marked,
            flux_inflows,
            edges,
        })
    }

    /// The mass balance at `state` (see [`MassBalance`]). Collective.
    pub(crate) fn compute(&self, state: &State<'_>) -> Result<MassBalance> {
        let State {
            f,
            porosity,
            c,
            c_t,
            values,
        } = *state;
        let (disc, free, data) = (&self.disc, &self.free, &self.data);
        let comm = disc.dm().comm();
        let weights = disc.quadrature_weights();
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

        // The residual of the equations of every dof:
        // R = S c + M c_t - int phi f (v + SUPG) - load.
        let mut residual = free.global_vector()?;
        let mut work = free.global_vector()?;
        self.steady.apply_local(c, &mut residual)?;
        self.mass.apply_local(c_t, &mut work)?;
        residual.axpy(1.0, &work)?;
        {
            let mut sources = self.sources.borrow_mut();
            for ((s, f), phi) in sources.values_mut().iter_mut().zip(f).zip(porosity) {
                *s = phi * f;
            }
            sources.apply(&mut work)?;
        }
        residual.axpy(-1.0, &work)?;
        if let Some(load) = &self.load {
            residual.axpy(-1.0, load)?;
        }
        let dirichlet: Vec<&str> = self.dirichlet.iter().map(String::as_str).collect();
        let dirichlet_inflows = disc.essential_sums(free, &residual, &dirichlet)?;

        // int u . grad c: the advection of the Galerkin form, tested with v = 1.
        self.advection.apply_local(c, &mut work)?;
        let advection = work.dot(&self.ones)?;

        // The advective outflow through each edge of the boundary, by entry, and
        // int_Gamma (u . n) c, in and out.
        let mut outflow = vec![0.0; self.entries.len()];
        let mut through = 0.0;
        c.with_array(|c| {
            for edge in &self.edges {
                let [a, b] = edge.ends;
                for (s, u) in EDGE_GAUSS.iter().zip(edge.velocity) {
                    // `normal` has the length of the edge, the Gauss weight is 1/2.
                    let u_n = u[0] * edge.normal[0] + u[1] * edge.normal[1];
                    let flux = 0.5 * u_n * ((1.0 - s) * c[a] + s * c[b]);
                    through += flux;
                    outflow[edge.entry] += if edge.dirichlet || u_n > 0.0 {
                        flux
                    } else {
                        0.0
                    };
                }
            }
        })?;

        let storage_rate = comm.sum_real(storage)?;
        let source = comm.sum_real(source)?;
        let outflow = outflow
            .iter()
            .map(|&o| comm.sum_real(o))
            .collect::<petsc::Result<Vec<f64>>>()?;
        let mut boundary: Vec<BoundaryInflow> = self.entries[..self.num_conditions]
            .iter()
            .zip(&outflow)
            .map(|(tag, o)| BoundaryInflow {
                tag: tag.clone(),
                advective: -o,
                diffusive: 0.0,
            })
            .collect();
        for &(entry, inflow) in &self.flux_inflows {
            boundary[entry].diffusive += inflow;
        }
        for (tag, inflow) in self.dirichlet.iter().zip(&dirichlet_inflows) {
            let entry = self
                .entries
                .iter()
                .position(|n| n == tag)
                .expect("a condition");
            boundary[entry].diffusive += inflow;
        }
        for (entry, tag) in self.entries.iter().enumerate().skip(self.num_conditions) {
            if self.marked[entry - self.num_conditions] {
                boundary.push(BoundaryInflow {
                    tag: tag.clone(),
                    advective: -outflow[entry],
                    diffusive: 0.0,
                });
            }
        }
        Ok(MassBalance {
            storage_rate,
            source,
            boundary,
            advection_defect: advection - comm.sum_real(through)?,
        })
    }
}
