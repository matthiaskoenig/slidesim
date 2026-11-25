//! Steady Darcy perfusion: the pressure of `-div(K grad p / mu) = 0` with fixed pressures on
//! boundary tags, and the velocity `u = -K grad p / mu` at the quadrature points.

use std::cell::RefCell;
use std::rc::Rc;

use ceed::{Ceed, ElemRestriction, EvalMode, FieldVector, Operator, QFunction};
use contracts::Perfusion;
use maps::Maps;
use mesh::Mesh;
use petsc::{Fe, InsertMode, Ksp, OptionsScope, Vector, sys};

use crate::coefficients::map_gradients;
use crate::operators::PointValues;
use crate::qfunctions::{self as qf, GEOMETRY_2D_SIZE, TRANSPORT_DATA_SIZE};
use crate::{
    Discretization, PdeError, PointData, Result, SharedOperator, SourceField, Terms, Transport,
    coefficient_values, configure_multigrid,
};

/// The options prefix of the pressure solves.
const PREFIX: &str = "perfusion_";

/// The options every pressure solve sets before those of the caller, who may override them: a
/// solver which does not converge is an error ([`PdeError::SolveFailed`]), not a wrong
/// pressure; and the residual reduced by `1e-12`.
///
/// The tolerance is tighter than the `1e-10` of [`crate::Problem`]: the flows through the
/// pressure tags come from the residual at the Dirichlet dofs, which is more sensitive to the
/// error of the solve than the pressure itself. With `1e-10`, the flows of a manufactured
/// anisotropic solution on 2 and 3 processes differed from those on one by up to `2.5e-10`
/// relative (the L2 norms of pressure and velocity by `4e-12`), and the velocity of a uniform
/// flow, which P1 represents exactly, was off by `1.1e-10`; with `1e-12`, by below `2e-12` and
/// `1e-11`. The pressure is solved once per level, so the few more iterations cost little.
const DEFAULTS: &str = "-ksp_error_if_not_converged -ksp_rtol 1e-12";

/// The unit of pressures (of the contracts): the name of the component of the pressure, for
/// viewers.
const PRESSURE_UNIT: &str = "Pa";

/// The unit of velocities (of the contracts): the name of each component of the velocity, for
/// viewers.
const VELOCITY_UNIT: &str = "um/s";

/// The steady Darcy perfusion of a [`Perfusion`] on a [`Mesh`]: the pressure `p` (Pa) of
///
/// `-div(K grad p / mu) = 0`, `p` fixed on the tags of `Perfusion::boundaries`,
///
/// no flow through the rest of the boundary (the natural boundary condition), `mu` the
/// viscosity (Pa*s) and `K` the permeability (um^2), a tensor at each point:
///
/// `K = k_across I + w (k_along - k_across) t t^T`, `t = g / |g|`, `w = |g|^2 / (|g|^2 + g0^2)`,
///
/// with `g` the gradient (1/um) of the direction map ([`maps::Map::gradient`]) and `g0`
/// (1/um) of `Perfusion::direction`, and `k_along`, `k_across` coefficients (numbers or tables
/// over maps, [`coefficient_values`]). Where the gradient vanishes `K` blends smoothly to
/// `k_across I`. As computed, `w t t^T = g g^T / (|g|^2 + g0^2)`, which needs no direction where
/// `g = 0`. `K` is symmetric positive definite for positive `k_along`, `k_across` (checked by
/// the contracts).
///
/// The velocity (um/s) is `u = -K grad p / mu` at the quadrature points
/// ([`Flow::velocity`]), not projected onto a finite element space.
///
/// Discretization: P1 on every level of the mesh, with the pressure tags essential, and the
/// operator `-div(D grad p)`, `D = K / mu`, of [`Transport`] ([`Terms::Steady`]). The pressure
/// is solved on every level, each from the same equation discretized on the level: the finest
/// gives the pressure, and each level the velocity at its own quadrature points, consistent
/// with its own mesh (for the operators of the coarser levels of a multigrid preconditioner of
/// a transport equation carried by the flow).
///
/// Solver: CG on the shell operator of the level, preconditioned by geometric multigrid over
/// the levels up to it ([`configure_multigrid`]; level 0 by its coarse solver, algebraic
/// multigrid, alone). PETSc options: the solvers have the options prefix `perfusion_`, and
/// read, lowest precedence first: the defaults `-ksp_error_if_not_converged -ksp_rtol 1e-12`
/// (a tighter tolerance than that of [`crate::Problem`], for the flows through the tags); the
/// options of the caller (`petsc_options` of [`Flow::solve`], written without the prefix);
/// options under the prefix already in the global options database (the command line, for
/// example `-perfusion_ksp_monitor`). The solves of every level read them. The options are set
/// while the solve runs only ([`OptionsScope`]).
///
/// Boundary flows ([`Flow::tag_flows`]): the net flow into the domain through each pressure
/// tag (um^2/s in 2D, positive for inflow), from the residual of the full discrete equation at
/// the dofs the tag constrains (the consistent flux): testing the equation with the function of
/// a constrained dof `i` gives `R_i = a(p, v_i) - int f v_i = int_boundary (D grad p . n) v_i`,
/// and `D grad p . n = -u . n` is the inflow. A dof several tags constrain counts for the tag
/// listed last, whose pressure it has. The flows of the tags add up to the inflow through the
/// whole boundary, `-int f` (zero for [`Flow::solve`]), to the tolerance of the solve.
///
/// Processes: as for [`crate::Problem`], the results depend on the number of processes up to
/// the tolerance of the linear solves: with the defaults, L2 norms and flows through the tags
/// on 1, 2 and 3 processes agree within about `2e-12` relative on the unit square. Every error
/// is the same on every process.
#[derive(Debug)]
pub struct Flow {
    /// The finest level, with the pressure tags essential.
    disc: Discretization,
    /// The finest level without essential boundaries: its local vectors are those of `disc`.
    free: Discretization,
    /// The pressure on the finest level: a global vector of `disc`.
    pressure: Vector,
    /// The pressure on the finest level with its Dirichlet values: a local vector of `disc`.
    pressure_local: Vector,
    /// The velocity at the quadrature points of each level, coarse first.
    velocity: Vec<Vec<[f64; 2]>>,
    tag_flows: Vec<(String, f64)>,
    values: PointValues,
}

/// The Dirichlet values and the source of a manufactured solution.
#[derive(Clone, Copy)]
struct Manufactured<'a> {
    exact: &'a dyn Fn([f64; 2]) -> f64,
    source: &'a dyn Fn([f64; 2]) -> f64,
}

impl Flow {
    /// The flow of `perfusion` on `mesh`, with the maps of `maps` (read for the cells of this
    /// process, see [`Mesh::local_bounds`]: the direction map and those of tables of `k_along`
    /// and `k_across`), and with the PETSc options `petsc_options`, which override the
    /// defaults and give way to options under the prefix `perfusion_` already in the database
    /// (see [`Flow`]). Collective.
    ///
    /// `perfusion` is valid (`contracts::Study::validate`): positive viscosity, permeabilities
    /// and `g0`.
    ///
    /// Errors: [`PdeError::Unsupported`] without pressure boundaries (the pressure is not
    /// determined), or for a viscosity or `g0` which is not positive; [`PdeError::Mesh`] for a
    /// pressure tag the mesh does not have, listing its tags; [`PdeError::Map`] (on every
    /// process) if a map cannot be evaluated at a quadrature point of a level, naming the
    /// coefficient (`k_along`, `k_across` or `direction`); [`PdeError::SolveFailed`] if a
    /// pressure solve does not converge, naming the level; [`PdeError::Petsc`] if a
    /// pressure solve is running (it owns the options of the prefix) or PETSc rejects the
    /// options.
    pub fn solve(
        mesh: &Mesh,
        perfusion: &Perfusion,
        maps: &Maps,
        ceed: &Ceed,
        petsc_options: &str,
    ) -> Result<Flow> {
        Flow::build(mesh, perfusion, maps, ceed, petsc_options, None)
    }

    /// As [`Flow::solve`], for convergence tests: the pressure fixed to `exact` at the nodes of
    /// the pressure tags (whose pressures are ignored), and `-div(K grad p / mu) = source`.
    #[doc(hidden)]
    pub fn solve_manufactured(
        mesh: &Mesh,
        perfusion: &Perfusion,
        maps: &Maps,
        ceed: &Ceed,
        petsc_options: &str,
        exact: &dyn Fn([f64; 2]) -> f64,
        source: &dyn Fn([f64; 2]) -> f64,
    ) -> Result<Flow> {
        let manufactured = Manufactured { exact, source };
        Flow::build(
            mesh,
            perfusion,
            maps,
            ceed,
            petsc_options,
            Some(manufactured),
        )
    }

    fn build(
        mesh: &Mesh,
        perfusion: &Perfusion,
        maps: &Maps,
        ceed: &Ceed,
        petsc_options: &str,
        manufactured: Option<Manufactured<'_>>,
    ) -> Result<Flow> {
        if perfusion.boundaries.is_empty() {
            return Err(PdeError::Unsupported(
                "perfusion without pressure boundaries: the pressure is not determined".into(),
            ));
        }
        for (what, value) in [
            ("viscosity", perfusion.viscosity),
            ("direction g0", perfusion.direction.g0),
        ] {
            if !(value > 0.0 && value.is_finite()) {
                return Err(PdeError::Unsupported(format!(
                    "perfusion {what} {value} (not positive and finite)"
                )));
            }
        }
        let mut boundaries = Vec::with_capacity(perfusion.boundaries.len());
        for boundary in &perfusion.boundaries {
            boundaries.push((mesh.tag(&boundary.tag)?.name, boundary.pressure));
        }
        let _options = OptionsScope::new(PREFIX, &[DEFAULTS, petsc_options])?;

        let essential: Vec<&str> = boundaries.iter().map(|(tag, _)| tag.as_str()).collect();
        let levels = Discretization::hierarchy(mesh, ceed, 1, &essential)?;
        let finest = levels.len() - 1;
        // The coefficients of the finest level first: a map error names a point of the level of
        // the pressure, where there is one.
        let mut data = levels
            .iter()
            .rev()
            .map(|level| Ok(Rc::new(permeability(level, perfusion, maps)?)))
            .collect::<Result<Vec<_>>>()?;
        data.reverse();
        let operators = levels
            .iter()
            .zip(&data)
            .map(|(level, data)| {
                let op = Transport::new(level, Rc::clone(data), Terms::Steady)?;
                Ok(SharedOperator(Rc::new(RefCell::new(op))))
            })
            .collect::<Result<Vec<_>>>()?;

        let mut velocity = Vec::with_capacity(levels.len());
        let mut solution = None;
        for (l, disc) in levels.iter().enumerate() {
            let (pressure, local) =
                solve_level(mesh, disc, &operators[..=l], &boundaries, manufactured)?;
            velocity.push(point_velocity(disc, &data[l], &local)?);
            solution = Some((pressure, local));
        }
        let (pressure, pressure_local) = solution.expect("a mesh has a level");
        let disc = levels[finest].clone();
        // For viewers of the pressure (see `Flow::local_pressure`); the clone is a handle of the
        // same DM.
        disc.dm()
            .clone()
            .set_field_names(0, "pressure", &[PRESSURE_UNIT])?;

        let free = Discretization::new(mesh, finest, ceed, 1)?;
        let same_layout = free.node_coordinates() == disc.node_coordinates();
        if disc.dm().comm().max_int((!same_layout).into())? > 0 {
            return Err(PdeError::Unsupported(
                "local vectors of a level which differ with and without essential boundaries"
                    .into(),
            ));
        }
        let tag_flows = tag_flows(
            &disc,
            &free,
            &data[finest],
            &pressure_local,
            &essential,
            manufactured,
        )?;
        Ok(Flow {
            values: PointValues::new(&disc)?,
            disc,
            free,
            pressure,
            pressure_local,
            velocity,
            tag_flows,
        })
    }

    /// The velocity `u = -K grad p / mu` (um/s) at the quadrature points of the local cells of
    /// level `level` of the mesh (coarse first), in the layout of
    /// [`Discretization::quadrature_points`] of any discretization of the level, from the
    /// pressure solved on that level.
    ///
    /// Panics if the mesh has no level `level`.
    pub fn velocity(&self, level: usize) -> &[[f64; 2]] {
        &self.velocity[level]
    }

    /// The net flow into the domain through each pressure tag (um^2/s in 2D, positive for
    /// inflow), in the order of `Perfusion::boundaries`: from the residual of the discrete
    /// equation at the dofs the tag constrains (see [`Flow`]). The same on every process.
    pub fn tag_flows(&self) -> &[(String, f64)] {
        &self.tag_flows
    }

    /// The pressure (Pa) on the finest level: a global vector of [`Flow::discretization`],
    /// without the essential dofs.
    pub fn pressure(&self) -> &Vector {
        &self.pressure
    }

    /// The discretization of the pressure: the finest level, with the pressure tags essential.
    pub fn discretization(&self) -> &Discretization {
        &self.disc
    }

    /// The pressure as a new local vector of [`Flow::discretization`]: the entries of the local
    /// cells, ghosts and the pressures of the tags included. For viewers: the field of the DM
    /// is named `pressure` and its component after the unit, `Pa`, and the vector has an empty
    /// name, so that a VTK viewer names its point data `pressure.Pa`. Collective.
    pub fn local_pressure(&self) -> Result<Vector> {
        let mut local = self.disc.local_vector()?;
        self.pressure_local
            .with_array(|values| local.with_array_mut(|copy| copy.copy_from_slice(values)))??;
        local.set_name("")?;
        Ok(local)
    }

    /// The L2 norm of the error of the pressure against `exact` on the finest level, by the
    /// quadrature of the discretization. Collective.
    pub fn pressure_l2_error(&self, exact: impl Fn([f64; 2]) -> f64) -> Result<f64> {
        let mut values = vec![0.0; self.disc.num_quadrature_points()];
        self.values.apply(&self.pressure_local, &mut values)?;
        let local: f64 = values
            .iter()
            .zip(self.disc.quadrature_points())
            .zip(self.disc.quadrature_weights())
            .map(|((p, &x), w)| w * (p - exact(x)).powi(2))
            .sum();
        Ok(self.disc.dm().comm().sum_real(local)?.sqrt())
    }

    /// The velocity of the finest level at the nodes, for output only: its lumped L2
    /// projection, `u_i = int u v_i / int v_i` for the P1 function `v_i` of node `i`. A new local
    /// vector (2 components per node, x then y) of a DM of the finest level with one P1 field
    /// of 2 components, named `velocity`, each component after the unit, `um/s`; the vector has
    /// an empty name, so that a VTK viewer writes the point vector `velocity`. Collective.
    pub fn nodal_velocity(&self) -> Result<Vector> {
        let free = &self.free;
        let velocity = self.velocity.last().expect("a mesh has a level");
        let mut source = SourceField::new(free)?;
        // int v_i, int u_x v_i, int u_y v_i at the entries of local vectors of `free`.
        let mut integrals: [Vec<f64>; 3] = Default::default();
        let mut global = free.global_vector()?;
        let mut local = free.local_vector()?;
        for (component, integral) in integrals.iter_mut().enumerate() {
            for (s, u) in source.values_mut().iter_mut().zip(velocity) {
                *s = if component == 0 {
                    1.0
                } else {
                    u[component - 1]
                };
            }
            source.apply(&mut global)?;
            free.dm().global_to_local(&global, &mut local)?;
            *integral = local.with_array(<[f64]>::to_vec)?;
        }
        let [mass, ux, uy] = integrals;

        let mut dm = free.dm().clone_mesh()?;
        let dim = 2;
        let fe = Fe::lagrange(dim, dim, true, 1, 2)?;
        dm.add_field(&fe)?;
        dm.create_ds()?;
        dm.set_field_names(0, "velocity", &[VELOCITY_UNIT; 2])?;
        let mut nodal = Vector::create_local(&dm)?;
        let mut entries = Vec::new();
        for point in free.dm().depth_stratum(0)? {
            let i = free.dm().local_dofs(point)?.start;
            let entry = dm.local_dofs(point)?.start;
            entries.push((entry, [ux[i] / mass[i], uy[i] / mass[i]]));
        }
        nodal.with_array_mut(|values| {
            values.fill(0.0);
            for (entry, u) in entries {
                values[entry..entry + 2].copy_from_slice(&u);
            }
        })?;
        nodal.set_name("")?;
        Ok(nodal)
    }
}

/// The data of the operator `-div(D grad p)` on `disc`: `D = K / mu` at the quadrature points
/// (see [`Flow`]), mass and reaction zero. Collective.
///
/// Errors (on every process): [`PdeError::Map`], naming the coefficient.
fn permeability(disc: &Discretization, perfusion: &Perfusion, maps: &Maps) -> Result<PointData> {
    let named = |what: &str, e: PdeError| match e {
        PdeError::Map { message } => PdeError::Map {
            message: format!("perfusion, {what}: {message}"),
        },
        e => e,
    };
    let k_along =
        coefficient_values(disc, &perfusion.k_along, maps).map_err(|e| named("k_along", e))?;
    let k_across =
        coefficient_values(disc, &perfusion.k_across, maps).map_err(|e| named("k_across", e))?;
    let gradients =
        map_gradients(disc, &perfusion.direction.map, maps).map_err(|e| named("direction", e))?;
    let (g0, mu) = (perfusion.direction.g0, perfusion.viscosity);
    let diffusion = gradients
        .iter()
        .zip(k_along.iter().zip(&k_across))
        .map(|(&[gx, gy], (&along, &across))| {
            // w (k_along - k_across) t t^T = (k_along - k_across) g g^T / (|g|^2 + g0^2).
            let s = (along - across) / (gx * gx + gy * gy + g0 * g0);
            [
                (across + s * gx * gx) / mu,
                s * gx * gy / mu,
                (across + s * gy * gy) / mu,
            ]
        })
        .collect();
    let n = gradients.len();
    Ok(PointData::new(vec![0.0; n], vec![0.0; n], diffusion))
}

/// The pressure on the level of `disc`, the last of the levels of `operators` (the operators
/// of the levels up to it): a global vector, and a local vector with the Dirichlet values
/// (the pressures of `boundaries`, in order, or the manufactured values). Collective.
fn solve_level(
    mesh: &Mesh,
    disc: &Discretization,
    operators: &[SharedOperator],
    boundaries: &[(String, f64)],
    manufactured: Option<Manufactured<'_>>,
) -> Result<(Vector, Vector)> {
    let level = operators.len() - 1;
    let mut local = disc.local_vector()?;
    local.set(0.0)?;
    for (tag, pressure) in boundaries {
        match manufactured {
            None => disc.set_essential_values(&mut local, tag, *pressure)?,
            Some(Manufactured { exact, .. }) => {
                let dofs = disc.essential_dofs(tag)?;
                let nodes = disc.node_coordinates();
                local.with_array_mut(|values| {
                    for &i in dofs {
                        values[i] = exact(nodes[i]);
                    }
                })?;
            }
        }
    }
    // A p = b - A g, for the Dirichlet values g and the load b of the source.
    let op = &operators[level];
    let mut rhs = load(disc, manufactured)?;
    let mut lifted = disc.global_vector()?;
    op.0.borrow().apply_local(&local, &mut lifted)?;
    rhs.axpy(-1.0, &lifted)?;

    let mut a = disc.shell_matrix(op.clone())?;
    a.set_spd()?;
    let mut ksp = Ksp::create(&disc.dm().comm())?;
    ksp.set_options_prefix(PREFIX)?;
    ksp.set_operators(&a, &a)?;
    ksp.set_type("cg")?;
    let _multigrid = configure_multigrid(&mut ksp, mesh, operators)?;
    ksp.set_from_options()?;
    let mut pressure = disc.global_vector()?;
    pressure.set(0.0)?;
    let diverged = |reason: String| PdeError::SolveFailed {
        solve: "perfusion".to_string(),
        reason: format!("{reason} (pressure on level {level})"),
    };
    match ksp.solve(&rhs, &mut pressure) {
        Err(e) if e.code == sys::PETSC_ERR_NOT_CONVERGED => {
            let reason = ksp.converged_reason()?;
            return Err(diverged(if reason.diverged() {
                format!("KSP {reason}")
            } else {
                e.message.lines().next().unwrap_or_default().to_string()
            }));
        }
        Err(e) => return Err(e.into()),
        Ok(()) => {
            let reason = ksp.converged_reason()?;
            if reason.diverged() {
                return Err(diverged(format!("KSP {reason}")));
            }
        }
    }
    disc.dm().global_to_local(&pressure, &mut local)?;
    Ok((pressure, local))
}

/// The load of the source of `manufactured` on `disc`, `int f v` for the test functions of
/// the global dofs; zero without one. Collective.
fn load(disc: &Discretization, manufactured: Option<Manufactured<'_>>) -> Result<Vector> {
    let mut b = disc.global_vector()?;
    match manufactured {
        None => b.set(0.0)?,
        Some(Manufactured { source: f, .. }) => {
            let mut source = SourceField::new(disc)?;
            for (s, &x) in source.values_mut().iter_mut().zip(disc.quadrature_points()) {
                *s = f(x);
            }
            source.apply(&mut b)?;
        }
    }
    Ok(b)
}

/// The velocity `-D grad p` at the quadrature points of `disc`, with `D` of `data`, for the
/// pressure of the local vector `pressure` (with its Dirichlet values), in the layout of
/// [`Discretization::quadrature_points`].
fn point_velocity(
    disc: &Discretization,
    data: &PointData,
    pressure: &Vector,
) -> Result<Vec<[f64; 2]>> {
    let shared = disc.shared();
    let ceed = &shared.ceed;
    let dim = shared.basis.dim();
    let q = disc.points_per_cell();
    let n = disc.num_quadrature_points();
    // SAFETY: the fields match what Velocity2D reads and writes; no context.
    let mut qf = unsafe {
        QFunction::interior(ceed, 1, Some(qf::slidesim_pde_Velocity2D), qf::VELOCITY_2D)
    }?;
    qf.add_input("dp", dim, EvalMode::Grad)?;
    qf.add_input("q_data", GEOMETRY_2D_SIZE, EvalMode::None)?;
    qf.add_input("p_data", TRANSPORT_DATA_SIZE, EvalMode::None)?;
    qf.add_output("u", dim, EvalMode::None)?;
    let mut op = Operator::new(ceed, &qf)?;
    op.set_field(
        "dp",
        Some(&shared.restriction),
        Some(&shared.basis),
        FieldVector::Active,
    )?;
    op.set_field(
        "q_data",
        Some(&shared.q_data_restriction),
        None,
        FieldVector::Passive(&shared.q_data),
    )?;
    data.with_packed(ceed, q, |p_data| {
        op.set_field(
            "p_data",
            Some(&shared.p_data_restriction),
            None,
            FieldVector::Passive(p_data),
        )?;
        Ok(())
    })?;
    // Cell e holds u_x then u_y of its points: [e][c][q].
    let cells = n / q;
    let restriction = ElemRestriction::strided_with(ceed, cells, q, dim, n * dim, [1, q, q * dim])?;
    op.set_field("u", Some(&restriction), None, FieldVector::Active)?;
    let mut u = vec![0.0; n * dim];
    pressure.with_array(|p| op.apply_slices(p, &mut u))??;
    Ok(u.chunks_exact(dim * q)
        .flat_map(|cell| (0..q).map(move |i| [cell[i], cell[q + i]]))
        .collect())
}

/// The flows into the domain through the tags `tags` (see [`Flow::tag_flows`]), of the
/// pressure `pressure` (a local vector of `disc`, with its Dirichlet values) with the
/// coefficients `data` of `disc`: the residual of the equation on `free` (the level of `disc`
/// without essential boundaries, with the same local vectors) summed over the dofs each tag
/// constrains in `disc`. Collective.
fn tag_flows(
    disc: &Discretization,
    free: &Discretization,
    data: &Rc<PointData>,
    pressure: &Vector,
    tags: &[&str],
    manufactured: Option<Manufactured<'_>>,
) -> Result<Vec<(String, f64)>> {
    // R = A p - b on every dof.
    let steady = Transport::new(free, Rc::clone(data), Terms::Steady)?;
    let mut residual = free.global_vector()?;
    steady.apply_local(pressure, &mut residual)?;
    let b = load(free, manufactured)?;
    residual.axpy(-1.0, &b)?;

    // The indicator of the dofs of each tag, as a global vector: 1 where any process marks the
    // dof.
    let mut indicators = Vec::with_capacity(tags.len());
    let mut local = free.local_vector()?;
    for tag in tags {
        let dofs = disc.essential_dofs(tag)?;
        local.with_array_mut(|values| {
            values.fill(0.0);
            for &i in dofs {
                values[i] = 1.0;
            }
        })?;
        let mut indicator = free.global_vector()?;
        indicator.set(0.0)?;
        free.dm()
            .local_to_global(&local, InsertMode::Add, &mut indicator)?;
        indicator.with_array_mut(|values| {
            for v in values {
                *v = if *v > 0.0 { 1.0 } else { 0.0 };
            }
        })?;
        indicators.push(indicator);
    }
    // A dof of several tags counts for the one listed last.
    let mut taken = free.global_vector()?;
    taken.set(0.0)?;
    let mut flows = vec![0.0; tags.len()];
    for (i, indicator) in indicators.iter_mut().enumerate().rev() {
        taken.with_array_mut(|taken| {
            indicator.with_array_mut(|values| {
                for (v, t) in values.iter_mut().zip(taken.iter_mut()) {
                    if *t > 0.0 {
                        *v = 0.0;
                    } else if *v > 0.0 {
                        *t = 1.0;
                    }
                }
            })
        })??;
        flows[i] = residual.dot(indicator)?;
    }
    Ok(tags.iter().map(|tag| tag.to_string()).zip(flows).collect())
}
