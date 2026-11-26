//! The exchange of a field with cell models on the coarse elements of the mesh: the mean of the
//! field over an element, and the change the cell model makes to it given back to the field.

use std::cell::RefCell;
use std::rc::Rc;

use contracts::Coefficient;
use maps::Maps;
use mesh::Mesh;
use petsc::Vector;

use crate::operators::PointValues;
use crate::{
    Discretization, Exchanged, PdeError, Problem, Result, SourceField, UNTAGGED, coefficient_values,
};

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
///   model made, given back as the amount `A_i = delta_i V_i`, distributed over the nodes of
///   the element. The free nodes (off the Dirichlet boundaries) change; the share of a Dirichlet
///   node is exchanged with the Dirichlet value, which the boundary holds fixed (a reservoir):
///   substance it resupplies to the field, or takes away. The amount of substance of the field,
///   `int phi u` ([`Problem::total_mass`]), changes by exactly `sum_i A_i` plus the resupply,
///   which `give_back` returns per Dirichlet tag (inflow through the tag).
///
/// Both are libCEED operators on the quadrature points of the problem's level: the values of
/// the field at the points, and the integral of values at the points against the test
/// functions `v_j` ([`SourceField`]).
///
/// How an amount is distributed: the field has P1 nodal values `u_j`. With
/// `w_ij = int_(e_i) phi v_j` (`sum_j w_ij = V_i`, over every node of the element) and the
/// lumped mass `m_j = int phi v_j` (over the whole domain), `int phi u = sum_j m_j u_j`, so a
/// change `du_j = a_j / m_j` of free node `j` changes the amount by `a_j`. Each cell distributes
/// its amount over every node of its element:
///
/// - an increase (`A_i > 0`) in proportion to `w_ij`: `a_ij = A_i w_ij / V_i`, the transpose of
///   the mean, so that a uniform change of concentration is a uniform change of the nodal
///   values of the free nodes (whose cells all change alike). The share of the Dirichlet nodes,
///   `A_i D'_i / V_i` with `D'_i = sum_(j Dirichlet) w_ij`, leaves through their tags.
/// - a decrease (`A_i < 0`), the positivity safeguard: in proportion to the substance at each
///   node, `w_ij u_j^+` (`u^+ = max(u, 0)`): `a_ij = -r_i w_ij u_j^+` with
///   `r_i = |A_i| / G_i` and `G_i = sum_j w_ij u_j^+ = int_(e_i) phi u^+ >= u_i V_i`. A cell
///   model whose blood concentration stays nonnegative takes at most `u_i V_i`, so `r_i <= 1`,
///   and a free node loses at most `u_j sum_i r_i w_ij / m_j <= u_j`: no nodal value which is
///   nonnegative becomes negative, however steep the uptake, next to Dirichlet nodes as well.
///   The share of the Dirichlet nodes, `r_i D_i` with `D_i = sum_(j Dirichlet) w_ij u_j^+`, is
///   resupplied by their tags.
/// - only if a cell model ends with a negative blood concentration (`|A_i| > G_i`, within its
///   integration tolerance for a model which keeps its concentrations nonnegative) is
///   `r_i = 1` and the rest `|A_i| - G_i` taken as an increase is given: exact, and the field
///   takes on the negative value the model reached. Not an error, so that a run does not end on
///   the rounding of an integrator at a concentration of zero.
///
/// Then the change of `int phi u` is `sum_i A_i` plus the resupply, exactly up to rounding. A
/// Dirichlet node of several tags counts for the tag listed last, as its value does.
///
/// The time integration of the problem starts afresh after a change ([`Problem::modify`]).
#[derive(Debug)]
pub struct Exchange {
    disc: Discretization,
    /// The number of cells.
    n: usize,
    /// For each local cell of the level, the cell of the exchange its coarse element is.
    owner: Vec<Option<usize>>,
    /// The porosity at the quadrature points of the level.
    porosity: Vec<f64>,
    /// `V_i`.
    volumes: Vec<f64>,
    /// The share of `V_i` at the free nodes, `F_i = sum_(j free) w_ij`.
    free_volumes: Vec<f64>,
    /// The Dirichlet tags (the essential tags of the discretization, in their order, then
    /// [`UNTAGGED`] if a Dirichlet dof is of none).
    tags: Vec<String>,
    /// Of each local entry of the level: `None` for a free dof, else the index of its tag.
    tag_of: Vec<Option<usize>>,
    /// The share of `V_i` at the Dirichlet nodes of each tag: `[tag][cell]`.
    tag_volumes: Vec<Vec<f64>>,
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
    /// finest level of `mesh`, or a cell is not a local coarse element or is listed twice;
    /// [`PdeError::Map`] if the porosity cannot be evaluated.
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
        let (tags, tag_of) = dirichlet_tags(&disc)?;
        let values = PointValues::new(&disc)?;
        let mut exchange = Exchange {
            n: cells.len(),
            owner,
            porosity,
            volumes: Vec::new(),
            free_volumes: Vec::new(),
            tags,
            tag_of,
            tag_volumes: Vec::new(),
            lumped: disc.global_vector()?,
            values,
            source: RefCell::new(SourceField::new(&disc)?),
            disc,
        };
        exchange.volumes = exchange.per_cell(&vec![1.0; exchange.porosity.len()]);
        let ones = vec![1.0; exchange.tag_of.len()];
        let (free, tagged) = exchange.split(&ones)?;
        exchange.free_volumes = free;
        exchange.tag_volumes = tagged;
        {
            let mut source = exchange.source.borrow_mut();
            source.values_mut().copy_from_slice(&exchange.porosity);
            source.apply(&mut exchange.lumped)?;
        }
        Ok(exchange)
    }

    /// The number of cells.
    pub fn len(&self) -> usize {
        self.n
    }

    /// Whether there are no cells (on this process).
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// The blood volume `V_i = int_(e_i) phi` of each cell (um^2 in 2D).
    pub fn volumes(&self) -> &[f64] {
        &self.volumes
    }

    /// The Dirichlet tags of the field, in the order of [`Exchange::give_back`]'s resupply:
    /// the tags of its Dirichlet conditions in the order listed, then [`UNTAGGED`] if a
    /// Dirichlet dof is of none. The same on every process.
    pub fn dirichlet_tags(&self) -> &[String] {
        &self.tags
    }

    /// The porosity-weighted mean of the field of `problem` over each cell, its Dirichlet values
    /// included, into `out` (one value per cell, mM). Collective.
    ///
    /// Errors (on every process): [`PdeError::Unsupported`] if `problem` is not the problem of
    /// the exchange (its discretization and porosity), or `out` is not one value per cell.
    pub fn means(&self, problem: &Problem, out: &mut [f64]) -> Result<()> {
        self.collective(self.check(problem, out.len()))?;
        let u = self.point_values(&problem.local_solution()?)?;
        let amounts = self.per_cell(&u);
        for ((m, a), v) in out.iter_mut().zip(amounts).zip(&self.volumes) {
            *m = a / v;
        }
        Ok(())
    }

    /// Gives the change `delta[i]` of the concentration of each cell back to the field of
    /// `problem`: the amount `delta[i] V_i`, distributed over the nodes of the cell's element as
    /// [`Exchange`] describes. Returns what it exchanged ([`Exchanged`], summed over every
    /// process): the amount the cells gave, `sum_i delta[i] V_i` (mM*um^2), and the resupply,
    /// per Dirichlet tag ([`Exchange::dirichlet_tags`], in that order) the amount its Dirichlet
    /// values supplied to the field (negative where they took it away), so that `int phi u`
    /// changes by exactly the amount plus the sum of the resupply. The time integration of
    /// `problem` starts afresh ([`Problem::modify`]), which books the amount as the exchange of
    /// the mass balance over an interval and the resupply as inflow through the tags
    /// ([`Problem::interval_balance`]). Collective.
    ///
    /// Errors (on every process): [`PdeError::Unsupported`] if `problem` is not the problem of
    /// the exchange, `delta` is not one value per cell, or a value is not finite;
    /// [`PdeError::Poisoned`] after a failed step of `problem`.
    pub fn give_back(&self, problem: &mut Problem, delta: &[f64]) -> Result<Exchanged> {
        self.collective(self.check(problem, delta.len()).and_then(|()| {
            match delta.iter().position(|d| !d.is_finite()) {
                Some(i) => Err(PdeError::Unsupported(format!(
                    "a change of {} for cell {i} to give back",
                    delta[i]
                ))),
                None => Ok(()),
            }
        }))?;
        // The substance at the nodes, u^+, split into the free nodes and each Dirichlet tag.
        let mut positive = problem.local_solution()?;
        let positive = positive.with_array_mut(|u| {
            u.iter_mut().for_each(|u| *u = u.max(0.0));
            u.to_vec()
        })?;
        let (free, tagged) = self.split(&positive)?;
        // Per cell: the increase per unit of `w_ij` (`c_i`) and the share of the substance taken
        // (`r_i`); per tag the resupply.
        let mut increase = vec![0.0; self.len()];
        let mut share = vec![0.0; self.len()];
        let mut resupply = vec![0.0; self.tags.len()];
        for i in 0..self.len() {
            let amount = delta[i] * self.volumes[i];
            let substance = free[i] + tagged.iter().map(|t| t[i]).sum::<f64>();
            let volume = self.free_volumes[i] + self.tag_volumes.iter().map(|t| t[i]).sum::<f64>();
            let mut added = amount;
            if amount < 0.0 {
                let taken = (-amount).min(substance);
                if taken > 0.0 {
                    share[i] = taken / substance;
                }
                added = amount + taken;
            }
            increase[i] = added / volume;
            for (t, r) in resupply.iter_mut().enumerate() {
                *r += share[i] * tagged[t][i] - increase[i] * self.tag_volumes[t][i];
            }
        }
        let b_increase = self.integrate(&increase)?;
        let b_share = self.integrate(&share)?;
        let comm = self.disc.dm().comm();
        let amount: f64 = delta.iter().zip(&self.volumes).map(|(d, v)| d * v).sum();
        let resupply = self
            .tags
            .iter()
            .zip(resupply)
            .map(|(tag, r)| Ok((tag.clone(), comm.sum_real(r)?)))
            .collect::<Result<Vec<_>>>()?;
        let exchanged = Exchanged {
            exchange: comm.sum_real(amount)?,
            resupply,
        };
        problem.modify(
            |u| {
                b_increase.with_array(|increase| {
                    b_share.with_array(|share| {
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
            },
            &exchanged,
        )?;
        Ok(exchanged)
    }

    /// `local` on every process if no process has an error, else an error on every process:
    /// this process's, or one naming another process. Collective.
    fn collective(&self, local: Result<()>) -> Result<()> {
        let comm = self.disc.dm().comm();
        if comm.max_int(local.is_err().into())? > 0 {
            return Err(local.err().unwrap_or_else(|| {
                PdeError::Unsupported("an invalid exchange on another process".into())
            }));
        }
        Ok(())
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

    /// The nodal values `x` (of the local entries) split by node: `int_(e_i) phi x` per cell of
    /// the interpolant of `x` at the free nodes only (and zero elsewhere), and of `x` at the
    /// Dirichlet nodes of each tag only (`[tag][cell]`).
    fn split(&self, x: &[f64]) -> Result<(Vec<f64>, Vec<Vec<f64>>)> {
        let mut local = self.disc.local_vector()?;
        let mut part = |tag: Option<usize>| -> Result<Vec<f64>> {
            local.with_array_mut(|values| {
                for ((v, x), t) in values.iter_mut().zip(x).zip(&self.tag_of) {
                    *v = if *t == tag { *x } else { 0.0 };
                }
            })?;
            Ok(self.per_cell(&self.point_values(&local)?))
        };
        let free = part(None)?;
        let tagged = (0..self.tags.len())
            .map(|t| part(Some(t)))
            .collect::<Result<Vec<_>>>()?;
        Ok((free, tagged))
    }

    /// The values of the local vector `local` at the quadrature points.
    fn point_values(&self, local: &Vector) -> Result<Vec<f64>> {
        let mut values = vec![0.0; self.porosity.len()];
        self.values.apply(local, &mut values)?;
        Ok(values)
    }

    /// `sum_(points of e_i) w phi f` of the values `f` at the quadrature points, per cell.
    fn per_cell(&self, values: &[f64]) -> Vec<f64> {
        let q = self.disc.points_per_cell();
        let weights = self.disc.quadrature_weights();
        let mut sums = vec![0.0; self.n];
        for (e, owner) in self.owner.iter().enumerate() {
            if let Some(i) = owner {
                for k in e * q..(e + 1) * q {
                    sums[*i] += weights[k] * self.porosity[k] * values[k];
                }
            }
        }
        sums
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

/// The Dirichlet tags of `disc` and the tag of each local entry (`None` for a free dof): the
/// essential tags in their order, a dof of several counting for the one listed last, then
/// [`UNTAGGED`] if an essential dof is of none on any process. Collective.
fn dirichlet_tags(disc: &Discretization) -> Result<(Vec<String>, Vec<Option<usize>>)> {
    let essential = disc.essential();
    let mut tags: Vec<String> = essential.iter().map(|(tag, _)| tag.clone()).collect();
    let local_to_global = &disc.shared().local_to_global;
    let mut tag_of: Vec<Option<usize>> = vec![None; local_to_global.len()];
    for (t, (_, dofs)) in essential.iter().enumerate() {
        for &i in dofs {
            tag_of[i] = Some(t);
        }
    }
    let untagged = tags.len();
    let mut any = false;
    for (t, &g) in tag_of.iter_mut().zip(local_to_global) {
        if g < 0 && t.is_none() {
            *t = Some(untagged);
            any = true;
        }
    }
    if disc.dm().comm().max_int(any.into())? > 0 {
        tags.push(UNTAGGED.to_string());
    }
    Ok((tags, tag_of))
}
