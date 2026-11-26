//! The exchange of a field with cell models on the coarse elements of the mesh: the mean of the
//! field over an element, and the change the cell model makes to it given back to the field.

use std::cell::RefCell;
use std::rc::Rc;

use contracts::Coefficient;
use maps::Maps;
use mesh::Mesh;
use petsc::Vector;

use crate::operators::PointValues;
use crate::{Discretization, PdeError, Problem, Result, SourceField, coefficient_values};

/// The free share of an element (the part of its blood volume at nodes off the Dirichlet
/// boundaries, relative to its blood volume) below which it counts as without free nodes.
const NO_FREE_NODE: f64 = 1e-12;

/// The exchange of the field of a [`Problem`] with cell models on coarse elements of the mesh
/// ("cells", one model each).
///
/// Cell `i` is the coarse element `cells[i]` (a local cell of level 0 of the mesh) with the
/// fine cells of the problem's level in it ([`Mesh::coarse_cell`]). Its blood volume is
/// `V_i = int_(e_i) phi` (um^2 in 2D), with the porosity `phi` of the field.
///
/// - [`Exchange::means`]: the porosity-weighted mean of the field over each cell,
///   `u_i = int_(e_i) phi u / V_i`, the concentration in the cell's blood (the L2 projection to
///   the coarse elements, with the field's Dirichlet values).
/// - [`Exchange::give_back`]: the change `delta_i` of that concentration (mM), which a cell
///   model made, given back to the field as the amount `A_i = delta_i V_i`: the amount of
///   substance of the field, `int phi u` ([`Problem::total_mass`]), changes by exactly
///   `sum_i A_i`, up to rounding.
///
/// Both are libCEED operators on the quadrature points of the problem's level: the values of
/// the field at the points, and the integral of values at the points against the test
/// functions `v_j` ([`SourceField`]).
///
/// How an amount is distributed: the field has P1 nodal values `u_j`, of which the nodes off the
/// Dirichlet boundaries ("free") change. With `w_ij = int_(e_i) phi v_j` and the lumped mass
/// `m_j = int phi v_j` (over the whole domain), `int phi u = sum_j m_j u_j`, so a change
/// `du_j = a_j / m_j` of free node `j` changes the amount by `a_j`. Each cell gives its amount
/// to the free nodes of its element:
///
/// - an increase (`A_i > 0`) in proportion to `w_ij`: `a_ij = A_i w_ij / F_i` with the free
///   blood volume `F_i = sum_(j free) w_ij`. A uniform increase of a field without Dirichlet
///   nodes in the cells' elements is a uniform increase of its nodal values.
/// - a decrease (`A_i < 0`), the positivity safeguard: in proportion to the substance at each
///   free node, `w_ij u_j^+` (`u^+ = max(u, 0)`): `a_ij = -r_i w_ij u_j^+` with
///   `r_i = |A_i| / G_i` and `G_i = sum_(j free) w_ij u_j^+`, the substance at the free nodes of
///   the element. A node loses at most `sum_i r_i w_ij u_j / m_j <= u_j` while `r_i <= 1`, so no
///   nodal value which is nonnegative becomes negative, however steep the uptake, as long as
///   no cell takes more than the substance at the free nodes of its element (`|A_i| <= G_i`).
///   A cell can take more only next to Dirichlet nodes, whose values count in its mean but do
///   not change, or where the field has negative values: then `r_i = 1` (its free nodes are
///   emptied) and the rest, `|A_i| - G_i`, is taken in proportion to `w_ij` as an increase is
///   given, so that the amount stays exact (and the values of those nodes go below zero).
///
/// Then `sum_j a_j = sum_i A_i`: the change of the amount is exact.
///
/// The time integration of the problem starts afresh after a change ([`Problem::modify`]).
#[derive(Debug)]
pub struct Exchange {
    disc: Discretization,
    /// For each local cell of the level, the cell of the exchange its coarse element is.
    owner: Vec<Option<usize>>,
    /// The porosity at the quadrature points of the level.
    porosity: Vec<f64>,
    /// `V_i`.
    volumes: Vec<f64>,
    /// `F_i`.
    free_volumes: Vec<f64>,
    /// Of the local entries of the level: 1 for a free dof, 0 for an essential one.
    free: Vec<f64>,
    /// `m_j`, a global vector.
    lumped: Vector,
    values: PointValues,
    /// The integral of values at the points against the test functions.
    source: RefCell<SourceField>,
}

impl Exchange {
    /// The exchange of the cells on the coarse elements `cells` (local cells of level 0 of
    /// `mesh`, each once) with a field on `problem_disc`, the discretization of a [`Problem`]
    /// ([`Problem::discretization`]) on the finest level of `mesh`, of the porosity `porosity`
    /// (the field's, a coefficient over `maps`). Collective.
    ///
    /// Errors (on every process): [`PdeError::Unsupported`] if `problem_disc` is not on the
    /// finest level of `mesh`, a cell is not a local coarse element or is listed twice, or
    /// every node of an element is on a Dirichlet boundary (its amount could not be given
    /// back; possible without refinements); [`PdeError::Map`] if the porosity cannot be
    /// evaluated.
    pub fn new(
        problem_disc: &Discretization,
        mesh: &Mesh,
        cells: &[usize],
        porosity: &Coefficient,
        maps: &Maps,
    ) -> Result<Exchange> {
        let disc = problem_disc.clone();
        let comm = disc.dm().comm();
        let level = mesh.levels().len() - 1;
        let q = disc.points_per_cell();
        let fine_cells = disc.num_quadrature_points() / q;
        let coarse_cells = mesh.levels()[0].height_stratum(0)?.len();
        let finest = mesh.finest().height_stratum(0)?.len();
        let mut index = vec![None; coarse_cells];
        let mut problem = None;
        if fine_cells != finest {
            problem = Some(format!(
                "a discretization of {fine_cells} local cells on a finest level of {finest}"
            ));
        }
        for (i, &cell) in cells.iter().enumerate() {
            match index.get_mut(cell) {
                None => {
                    problem.get_or_insert_with(|| {
                        format!("cell {cell} of {coarse_cells} local coarse cells")
                    });
                }
                Some(Some(_)) => {
                    problem.get_or_insert_with(|| format!("coarse cell {cell} listed twice"));
                }
                Some(slot) => *slot = Some(i),
            }
        }
        if comm.max_int(problem.is_some().into())? > 0 {
            return Err(PdeError::Unsupported(format!(
                "exchange with cells: {}",
                problem.unwrap_or_else(|| "invalid cells on another process".into())
            )));
        }
        let owner: Vec<Option<usize>> = (0..fine_cells)
            .map(|e| index[mesh.coarse_cell(level, e)])
            .collect();

        let porosity = coefficient_values(&disc, porosity, maps)?;
        let per_cell = |values: &[f64]| cell_sums(&disc, &owner, &porosity, cells.len(), values);
        let volumes = per_cell(&vec![1.0; porosity.len()]);
        let free: Vec<f64> = disc
            .shared()
            .local_to_global
            .iter()
            .map(|&g| if g >= 0 { 1.0 } else { 0.0 })
            .collect();
        let values = PointValues::new(&disc)?;
        let mut local = disc.local_vector()?;
        local.with_array_mut(|x| x.copy_from_slice(&free))?;
        let mut fraction = vec![0.0; porosity.len()];
        values.apply(&local, &mut fraction)?;
        let free_volumes = per_cell(&fraction);
        // An element without free nodes, by its centroid.
        let pinned = free_volumes
            .iter()
            .zip(&volumes)
            .position(|(f, v)| *f <= NO_FREE_NODE * v);
        if comm.max_int(pinned.is_some().into())? > 0 {
            let message = match pinned {
                Some(i) => {
                    let centroid = centroid(&disc, &owner, i);
                    format!(
                        "the element at ({}, {}) um has every node on a Dirichlet boundary: \
                         cells there cannot exchange with the field (refine the mesh)",
                        centroid[0], centroid[1]
                    )
                }
                None => "an element of another process has every node on a Dirichlet boundary \
                         (refine the mesh)"
                    .to_string(),
            };
            return Err(PdeError::Unsupported(message));
        }

        let mut source = SourceField::new(&disc)?;
        source.values_mut().copy_from_slice(&porosity);
        let mut lumped = disc.global_vector()?;
        source.apply(&mut lumped)?;
        Ok(Exchange {
            disc,
            owner,
            porosity,
            volumes,
            free_volumes,
            free,
            lumped,
            values,
            source: RefCell::new(source),
        })
    }

    /// The number of cells.
    pub fn len(&self) -> usize {
        self.volumes.len()
    }

    /// Whether there are no cells (on this process).
    pub fn is_empty(&self) -> bool {
        self.volumes.is_empty()
    }

    /// The blood volume `V_i = int_(e_i) phi` of each cell (um^2 in 2D).
    pub fn volumes(&self) -> &[f64] {
        &self.volumes
    }

    /// The porosity-weighted mean of the field of `problem` over each cell, its Dirichlet values
    /// included, into `out` (one value per cell, mM). Collective.
    ///
    /// Errors: [`PdeError::Unsupported`] if `problem` is not the problem of the exchange (its
    /// discretization and porosity), or `out` is not one value per cell.
    pub fn means(&self, problem: &Problem, out: &mut [f64]) -> Result<()> {
        self.check(problem, out.len())?;
        let u = self.point_values(&problem.local_solution()?)?;
        let amounts = self.per_cell(&u);
        for ((m, a), v) in out.iter_mut().zip(amounts).zip(&self.volumes) {
            *m = a / v;
        }
        Ok(())
    }

    /// Gives the change `delta[i]` of the concentration of each cell back to the field of
    /// `problem`: the amount `delta[i] V_i`, distributed to the free nodes of the cell's element
    /// as [`Exchange`] describes, so that `int phi u` changes by exactly `sum_i delta[i] V_i`
    /// (over every process). The time integration of `problem` starts afresh
    /// ([`Problem::modify`]). Collective.
    ///
    /// Errors (on every process): [`PdeError::Unsupported`] if `problem` is not the problem of
    /// the exchange, `delta` is not one value per cell, or a value is not finite;
    /// [`PdeError::Poisoned`] after a failed step of `problem`.
    pub fn give_back(&self, problem: &mut Problem, delta: &[f64]) -> Result<()> {
        self.check(problem, delta.len())?;
        let comm = self.disc.dm().comm();
        let infinite = delta.iter().position(|d| !d.is_finite());
        if comm.max_int(infinite.is_some().into())? > 0 {
            return Err(PdeError::Unsupported(match infinite {
                Some(i) => format!("a change of {} for cell {i} to give back", delta[i]),
                None => "a change which is not finite on another process".into(),
            }));
        }
        // The substance at the free nodes: u^+ there, 0 at the essential ones.
        let mut positive = problem.local_solution()?;
        positive.with_array_mut(|u| {
            for (u, free) in u.iter_mut().zip(&self.free) {
                *u = u.max(0.0) * free;
            }
        })?;
        let substance = self.per_cell(&self.point_values(&positive)?);
        // Per cell: the increase per unit of `w_ij` (`c_i`) and the share of the substance taken
        // (`r_i`).
        let mut increase = vec![0.0; self.len()];
        let mut share = vec![0.0; self.len()];
        for i in 0..self.len() {
            let amount = delta[i] * self.volumes[i];
            if amount >= 0.0 {
                increase[i] = amount / self.free_volumes[i];
            } else {
                let taken = (-amount).min(substance[i]);
                if taken > 0.0 {
                    share[i] = taken / substance[i];
                }
                increase[i] = -(-amount - taken) / self.free_volumes[i];
            }
        }
        let increase = self.integrate(&increase)?;
        let share = self.integrate(&share)?;
        problem.modify(|u| {
            increase.with_array(|increase| {
                share.with_array(|share| {
                    self.lumped.with_array(|lumped| {
                        u.with_array_mut(|u| {
                            for (j, u) in u.iter_mut().enumerate() {
                                *u += (increase[j] - u.max(0.0) * share[j]) / lumped[j];
                            }
                        })
                    })
                })
            })????;
            Ok(())
        })
    }

    /// [`PdeError::Unsupported`] unless `problem` is the problem of the exchange and `n` is the
    /// number of cells.
    fn check(&self, problem: &Problem, n: usize) -> Result<()> {
        if !Rc::ptr_eq(problem.discretization().shared(), self.disc.shared()) {
            return Err(PdeError::Unsupported(
                "an exchange with the field of a problem on another discretization".into(),
            ));
        }
        if problem.porosity() != self.porosity.as_slice() {
            return Err(PdeError::Unsupported(
                "an exchange with a field of another porosity".into(),
            ));
        }
        if n != self.len() {
            return Err(PdeError::Unsupported(format!(
                "{n} values for an exchange of {} cells",
                self.len()
            )));
        }
        Ok(())
    }

    /// The values of the local vector `local` at the quadrature points.
    fn point_values(&self, local: &Vector) -> Result<Vec<f64>> {
        let mut values = vec![0.0; self.porosity.len()];
        self.values.apply(local, &mut values)?;
        Ok(values)
    }

    /// `sum_(points of e_i) w phi f` of the values `f` at the quadrature points, per cell.
    fn per_cell(&self, values: &[f64]) -> Vec<f64> {
        cell_sums(&self.disc, &self.owner, &self.porosity, self.len(), values)
    }

    /// `sum_i x_i w_ij` for the free dofs `j`, a global vector: `int phi x v_j` of the values
    /// `x_i` per cell. Collective.
    fn integrate(&self, x: &[f64]) -> Result<Vector> {
        let q = self.disc.points_per_cell();
        let mut source = self.source.borrow_mut();
        for (e, (values, owner)) in source
            .values_mut()
            .chunks_exact_mut(q)
            .zip(&self.owner)
            .enumerate()
        {
            let porosity = &self.porosity[e * q..(e + 1) * q];
            let x = owner.map_or(0.0, |i| x[i]);
            for (value, phi) in values.iter_mut().zip(porosity) {
                *value = phi * x;
            }
        }
        let mut b = self.disc.global_vector()?;
        source.apply(&mut b)?;
        Ok(b)
    }
}

/// `sum_(points of e_i) w phi f` of the values `f` at the quadrature points of `disc`, for each
/// of the `n` cells: `owner` gives the cell of each local cell of `disc`, `porosity` is `phi` at
/// the points.
fn cell_sums(
    disc: &Discretization,
    owner: &[Option<usize>],
    porosity: &[f64],
    n: usize,
    values: &[f64],
) -> Vec<f64> {
    let q = disc.points_per_cell();
    let weights = disc.quadrature_weights();
    let mut sums = vec![0.0; n];
    for (e, owner) in owner.iter().enumerate() {
        if let Some(i) = owner {
            for k in e * q..(e + 1) * q {
                sums[*i] += weights[k] * porosity[k] * values[k];
            }
        }
    }
    sums
}

/// The centroid of cell `i`: the mean of the quadrature points of its fine cells, weighted.
fn centroid(disc: &Discretization, owner: &[Option<usize>], i: usize) -> [f64; 2] {
    let q = disc.points_per_cell();
    let (points, weights) = (disc.quadrature_points(), disc.quadrature_weights());
    let mut sum = [0.0; 2];
    let mut total = 0.0;
    for (e, _) in owner.iter().enumerate().filter(|(_, o)| **o == Some(i)) {
        for k in e * q..(e + 1) * q {
            total += weights[k];
            sum[0] += weights[k] * points[k][0];
            sum[1] += weights[k] * points[k][1];
        }
    }
    sum.map(|s| s / total)
}
