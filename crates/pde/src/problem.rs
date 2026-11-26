//! One field stepped in time: BDF2 of PETSc's `TS` on the P1 discretization of the finest mesh
//! level, with the coefficients of the field at the quadrature points, and advection by the
//! velocity of a [`Flow`] stabilized by SUPG.

use std::cell::{OnceCell, RefCell};
use std::rc::Rc;

use ceed::Ceed;
use contracts::{Coefficient, Condition, FieldSpec, TimeSpec};
use maps::Maps;
use mesh::Mesh;
use petsc::{
    ExactFinalTime, Implicit, Matrix, OptionsScope, Point, ProblemType, Ts, VecMut, VecRef, Vector,
    sys,
};

use crate::balance::{Balance, Setup, State};
use crate::integral::Integral;
use crate::operators::PointValues;
use crate::{
    BoundaryInflow, Discretization, Flow, MassBalance, Multigrid, PdeError, PointData, Result,
    SharedOperator, SourceField, Terms, Transport, coefficient_values, configure_multigrid,
};

/// The options every problem sets before those of the caller, who may override them: a solver
/// which does not converge is an error ([`PdeError::Diverged`]), not a silently wrong step.
const ERRORS_IF_NOT_CONVERGED: &str =
    "-ksp_error_if_not_converged -snes_error_if_not_converged -ts_error_if_step_fails";

/// The tolerance every problem sets for its linear solves, before the options of the caller, who
/// may override it: the residual reduced by `1e-10`. The solution depends on the number of
/// processes up to the tolerance (the multigrid preconditioner does, through the partition), so
/// the tolerance keeps results of different process counts within about `1e-10` of each other,
/// where PETSc's default (`1e-5`) would not.
const TOLERANCE: &str = "-ksp_rtol 1e-10";

/// The unit of concentrations (of the contracts): the name of the component of the field, for
/// viewers.
const CONCENTRATION_UNIT: &str = "mM";

/// A change of the shift of the Jacobian, relative to the shift, beyond which its matrices are
/// assembled again and the preconditioner is rebuilt. Smaller changes are rounding (BDF2 with a
/// fixed step computes shifts which differ in the last bits).
const SHIFT_TOLERANCE: f64 = 1e-12;

/// The net flow through a tag, relative to the largest net flow through a tag, above which the
/// tag has inflow ([`PdeError::Inflow`]).
const INFLOW_THRESHOLD: f64 = 1e-10;

/// What a [`Problem`] takes besides its field: the maps its coefficient tables refer to, and
/// the flow which carries a field with advection.
#[derive(Debug, Clone, Copy)]
pub struct Inputs<'a> {
    /// The maps, read for the cells of this process (see [`mesh::Mesh::local_bounds`]).
    pub maps: &'a Maps,
    /// The Darcy flow of the study on the same mesh, for fields with advection (others ignore
    /// it).
    pub flow: Option<&'a Flow>,
}

/// What a change of the solution between steps ([`Problem::modify`]) exchanged with what is
/// outside the field, for the mass balance over an interval ([`Problem::interval_balance`]):
/// amounts in mM*um^2, positive into the field. The change of `int phi u`
/// ([`Problem::total_mass`]) is `exchange` plus the sum of `resupply`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Exchanged {
    /// What cell models gave to the field ([`MassBalance::exchange`]).
    pub exchange: f64,
    /// What the Dirichlet values of each tag resupplied ([`BoundaryInflow::resupply`]).
    pub resupply: Vec<(String, f64)>,
}

/// The field `u` of a [`FieldSpec`] on a [`Mesh`], stepped in time:
///
/// `phi u_t + w . grad u = div(phi d grad u) - phi k u + phi f`, with `phi` the porosity
/// (dimensionless), `w` the Darcy velocity of the flow (um/s) if the field has advection (else
/// zero), `d` the diffusion coefficient (um^2/s), `k` the decay rate (1/s) and `f` the source
/// (mM/s), on the finest level of the mesh, in the weak form of P1 elements:
///
/// `M_phi u_t + (A_w + K_(phi d) + M_(phi k)) u = int phi f v + sum_i int_(Gamma_i) q_i v`,
///
/// `M_c` the mass matrix weighted with `c`, `A_w` advection (`int (w . grad u) v`) and `K_c` the
/// stiffness matrix of `c`; with advection, every term but diffusion is tested with the SUPG
/// test function `v + tau w . grad v` instead of `v` (streamline upwind Petrov-Galerkin, `tau`
/// per point from [`PointData::stabilized`]; see [`Terms`]). The velocity of a flow is its
/// conservative velocity, with `div w = 0` in each cell and a continuous normal component
/// (Darcy without sources), so the advective form equals the conservative one. The coefficients
/// are numbers or tables over maps ([`Coefficient`]), evaluated at the quadrature points of every
/// level of the mesh ([`coefficient_values`]), with the velocity of the flow on that level
/// ([`Flow::velocity`], solved on every level): the coarser levels of the multigrid
/// preconditioner discretize the same equation again.
///
/// Boundary conditions, by tag ([`FieldSpec::boundaries`]): a Dirichlet value is fixed on the
/// tag (an essential boundary; where tags meet, the value of the tag listed last holds); a flux
/// `q` is an inward diffusive flux (`phi d grad u . n`) on the boundary edges of the tag
/// (mM*um/s); everywhere else the diffusive flux is zero (the natural boundary condition). With
/// advection, every tag through which the flow enters the domain (positive [`Flow::tag_flows`])
/// needs a Dirichlet condition ([`PdeError::Inflow`] otherwise); where the flow leaves, without
/// a condition, the outflow is free: the advective form needs no boundary term, and the
/// diffusive flux is zero. The flow may come back in locally through a tag of net outflow (the
/// conservative velocity has no normal component on walls without flow); at each Gauss point of
/// an edge of the boundary without Dirichlet condition where `w . n < 0`, the term
/// `int (w . n)^- u v`
/// (`(w . n)^- = max(0, -w . n)`) keeps the advective flux from entering
/// ([`PointData::with_backflow`], with the velocity at the edge, [`Flow::edge_velocity`]), in
/// the residual, the Jacobian and the operators of every multigrid level.
///
/// Conservation: [`Problem::total_mass`] is the amount of substance `int phi u`, and
/// [`Problem::mass_balance`] its balance at the end of a step from the discrete equations (the
/// consistent fluxes at the Dirichlet tags, the advective outflow at the others, and the
/// advection defect, zero to rounding for the conservative velocity, see [`MassBalance`]).
/// [`Problem::interval_balance`] is the balance over the steps since a time
/// ([`Problem::start_interval`]), the changes of [`Problem::modify`] included, each term its
/// mean rate: the terms integrated by the weights of the BDF steps ([`Integral`]), so that they
/// add up to the change of the amount exactly (up to the residuals of the linear solves).
///
/// Time integration: `TSBDF` of order 2 with the time step [`TimeSpec::dt`]: fixed, or the
/// initial step of `TSAdapt` `basic` if [`TimeSpec::adaptive`]. The equation is linear, so each
/// step is one linear solve (`SNESKSPONLY`) with the Jacobian
/// `shift M_phi + A_w + K_(phi d) + M_(phi k)`: CG without advection (the Jacobian is symmetric
/// positive definite), GMRES with it; preconditioned by geometric multigrid over the levels of
/// the mesh ([`configure_multigrid`]; a mesh without refinements is solved by its coarse solver,
/// algebraic multigrid, alone). The operators of libCEED ([`Transport`]) are built once and
/// assembled into sparse matrices, which the steps apply: an assembled matrix is several times
/// cheaper to apply than its operator, and equals it up to rounding. The Jacobian is the
/// assembled operator of the finest level of the multigrid ([`Multigrid::finest`]); if the
/// options replace the multigrid preconditioner, the matrix-free operator. The residual applies
/// the assembled steady terms and mass matrix, with the steady terms of the Dirichlet values
/// computed once, and the source integrated matrix-free when it changed
/// ([`Problem::source_mut`]). A step changes the shift only, on every level; when the shift changes
/// the operators of the levels are assembled again and the preconditioner is rebuilt (the
/// smoothers and the coarse solver set up again). The operators, their matrices and the
/// multigrid are kept from step to step and from one [`Problem::step_to`] to the next.
///
/// PETSc options: the `TS` and its solvers have the options prefix `<field id>_` (`u_` for the
/// field `u`, so its linear solver reads `-u_ksp_type`). Their options, lowest precedence first:
/// 1. the defaults of the problem: `-ksp_error_if_not_converged -snes_error_if_not_converged
///    -ts_error_if_step_fails` and `-ksp_rtol 1e-10`;
/// 2. the options of the caller (`petsc_options` of [`Problem::new`], for example
///    `solver.petsc_options` of a study), written without the prefix;
/// 3. options under the prefix already in the global options database, from the command line
///    (or set by the caller of PETSc), for example `-u_ksp_rtol 1e-6`.
///
/// The problem sets the options of 1 and 2 which are not in the database already under the
/// prefix while it lives ([`OptionsScope`]), and removes them when dropped; it leaves those of
/// 3 as they are. The options override the multigrid preconditioner too: for example
/// `-pc_type jacobi` replaces it, `-mg_levels_ksp_max_it 3` changes its smoothers and
/// `-mg_coarse_pc_type lu` its coarse solver (on one process).
///
/// Processes: the solution depends on the number of processes up to the tolerance of the linear
/// solves, since the preconditioner does (algebraic multigrid aggregates the coarse level by
/// partition, and the eigenvalue estimates of the smoothers start from a vector of the global
/// numbering of the dofs, which follows the partition). With the default `-ksp_rtol 1e-10`, runs
/// on different numbers of processes agree to about the tolerance (integrals of the solution
/// within `1e-11` relative on `tissue_small`); a looser tolerance loosens this too.
///
/// Limit: one problem per field id lives at a time in a process, since it owns the options under
/// its prefix; [`Problem::new`] fails for a second one until the first is dropped. Field ids
/// match `^[A-Za-z][A-Za-z0-9_]*$` (the contracts enforce it; [`Problem::new`] checks it again).
#[derive(Debug)]
pub struct Problem {
    ts: Ts<Equation>,
    /// The solution: a global vector, without the essential dofs.
    u: Vector,
    /// The time of `u`, in s.
    t: f64,
    /// The size of the next step: fixed, or the latest choice of the adaptor.
    dt: f64,
    adaptive: bool,
    /// Why the time integration failed, if it did: then the problem does not step again.
    failure: Option<String>,
    disc: Discretization,
    /// The finest level without essential boundaries, for the mass balance.
    free: Discretization,
    dirichlet: Vec<(String, f64)>,
    fluxes: Vec<(String, f64)>,
    /// The tags of the conditions of the field, in the order listed.
    conditions: Vec<String>,
    /// The tags of edges of the mesh, in its order.
    edge_tags: Vec<String>,
    /// The velocity at the Gauss points of the edges of the boundary of the finest level (none
    /// without advection).
    edge_velocity: Vec<(Point, [[f64; 2]; 2])>,
    values: PointValues,
    /// The operators of the mass balance, built at its first computation.
    balance: OnceCell<Balance>,
    /// The start of the interval of [`Problem::interval_balance`].
    mark: Mark,
    /// The discretizations of the levels, coarse first, for tests of the reuse of operators.
    levels: Vec<Discretization>,
    /// Dropped after the `TS`, which may read options until it is destroyed.
    _options: OptionsScope,
}

impl Problem {
    /// The problem of the field `field` on the finest level of `mesh`, at t = 0 with the uniform
    /// initial value `field.initial` and the source `field.source`, with its coefficients from
    /// the maps of `inputs` and, with advection, the velocity of the flow of `inputs` (solved on
    /// `mesh`), stepped as `time` says (its time step and adaptivity), and with the PETSc
    /// options `petsc_options` (for example `-ksp_rtol 1e-8`), which override the defaults of
    /// the problem and give way to options under the prefix already in the database (see
    /// [`Problem`]). Collective.
    ///
    /// Errors: [`PdeError::Unsupported`] if the field id cannot be an options prefix (it starts
    /// with an ASCII letter and has ASCII letters, digits and `_` only), a flux is on a tag
    /// which is not of boundary edges, or the field has advection without a flow (or with the
    /// flow of another mesh); [`PdeError::Inflow`] if the field has advection and a tag with
    /// inflow has no Dirichlet condition; [`PdeError::Mesh`] for a tag the mesh does not have;
    /// [`PdeError::Map`] (on every process) if a coefficient table cannot be evaluated at a
    /// quadrature point of a level, naming the field and the coefficient; [`PdeError::Petsc`]
    /// if a problem of the same field id is alive, or PETSc rejects the options.
    pub fn new(
        mesh: &Mesh,
        field: &FieldSpec,
        time: &TimeSpec,
        ceed: &Ceed,
        petsc_options: &str,
        inputs: &Inputs,
    ) -> Result<Problem> {
        let prefix = options_prefix(&field.id)?;
        let flow = match (field.advection, inputs.flow) {
            (false, _) => None,
            (true, Some(flow)) => Some(flow),
            (true, None) => {
                return Err(PdeError::Unsupported(format!(
                    "advection of field '{}' without a flow (a perfusion)",
                    field.id
                )));
            }
        };
        let options = OptionsScope::new(
            &prefix,
            &[ERRORS_IF_NOT_CONVERGED, TOLERANCE, petsc_options],
        )?;

        let dim = mesh.finest().dimension()?;
        let mut dirichlet = Vec::new();
        let mut fluxes = Vec::new();
        for boundary in &field.boundaries {
            let tag = mesh.tag(&boundary.tag)?;
            match boundary.condition {
                Condition::Dirichlet { value } => dirichlet.push((tag.name, value)),
                Condition::Flux { value } => {
                    if usize::from(tag.dim) + 1 != dim {
                        return Err(PdeError::Unsupported(format!(
                            "a flux on tag '{}' of dimension {}: fluxes are on boundary edges \
                             (dimension {})",
                            tag.name,
                            tag.dim,
                            dim - 1
                        )));
                    }
                    fluxes.push((tag.name, value));
                }
            }
        }
        // Before the first step: the flow enters only where the field is fixed. The flows are the
        // same on every process.
        if let Some(flow) = flow {
            let fixed: Vec<&str> = dirichlet.iter().map(|(t, _)| t.as_str()).collect();
            if let Some(tag) = inflow_without_dirichlet(flow.tag_flows(), &fixed) {
                return Err(PdeError::Inflow {
                    tag: tag.to_string(),
                    field: field.id.clone(),
                });
            }
        }
        let conditions = field
            .boundaries
            .iter()
            .map(|b| Ok(mesh.tag(&b.tag)?.name))
            .collect::<Result<Vec<_>>>()?;
        let edge_tags = mesh
            .tags()
            .iter()
            .filter(|tag| usize::from(tag.dim) + 1 == dim)
            .map(|tag| tag.name.clone())
            .collect();
        let essential: Vec<&str> = dirichlet.iter().map(|(tag, _)| tag.as_str()).collect();
        let levels = Discretization::hierarchy(mesh, ceed, 1, &essential)?;
        let finest = levels.len() - 1;
        let disc = levels.last().expect("a mesh has a level").clone();
        // For viewers of the solution (see `Problem::local_solution`); the clone is a handle of
        // the same DM.
        disc.dm()
            .clone()
            .set_field_names(0, &field.id, &[CONCENTRATION_UNIT])?;

        let load = if fluxes.is_empty() {
            None
        } else {
            let fluxes: Vec<(&str, f64)> = fluxes.iter().map(|(t, q)| (t.as_str(), *q)).collect();
            Some(disc.boundary_load(&fluxes)?)
        };
        // The velocity of the flow on each level, with advection.
        let velocity = |level: usize, disc: &Discretization| -> Result<Option<Advection<'_>>> {
            let Some(flow) = flow else {
                return Ok(None);
            };
            let other = flow.levels() != mesh.levels().len()
                || flow.velocity(level).len() != disc.num_quadrature_points();
            if disc.dm().comm().max_int(other.into())? > 0 {
                return Err(PdeError::Unsupported(format!(
                    "advection of field '{}' by a flow of another mesh (velocities for other \
                     quadrature points)",
                    field.id
                )));
            }
            Ok(Some(Advection {
                velocity: flow.velocity(level),
                edges: flow.edge_velocity(level),
                dirichlet: &essential,
            }))
        };
        // The coefficients of the finest level first: a map error names a point of the level
        // of the solution, where there is one.
        let fine = Coefficients::new(&disc, field, inputs.maps, true)?;
        let data = Rc::new(fine.point_data(&disc, velocity(finest, &disc)?)?);
        let jacobian = Rc::new(RefCell::new(Transport::new(
            &disc,
            Rc::clone(&data),
            Terms::Jacobian,
        )?));
        let symmetric = flow.is_none();
        let mut shell = disc.shell_matrix(SharedOperator(Rc::clone(&jacobian)))?;
        if symmetric {
            shell.set_spd()?;
        }
        // The operators of the coarser levels, then the Jacobian on the finest.
        let level_operators = levels[..finest]
            .iter()
            .enumerate()
            .map(|(l, level)| {
                let coefficients = Coefficients::new(level, field, inputs.maps, false)?;
                let data = Rc::new(coefficients.point_data(level, velocity(l, level)?)?);
                let op = Transport::new(level, data, Terms::Jacobian)?;
                Ok(SharedOperator(Rc::new(RefCell::new(op))))
            })
            .chain(std::iter::once(Ok(SharedOperator(Rc::clone(&jacobian)))))
            .collect::<Result<Vec<_>>>()?;
        // The steady terms and the mass term from the Jacobian, whose shift is still 0: there it
        // is the steady terms.
        let [steady, mass] = jacobian.borrow().assemble_terms()?;
        let mut lift = disc.global_vector()?;
        jacobian
            .borrow()
            .apply_local(&local_with_dirichlet(&disc, &dirichlet)?, &mut lift)?;
        let equation = Equation {
            steady,
            lift,
            mass,
            data: Rc::clone(&data),
            jacobian,
            built_shift: 0.0,
            shifts: Vec::new(),
            multigrid: None,
            source: SourceField::stabilized(&disc, data)?,
            f: fine.source,
            porosity: fine.porosity,
            load,
            forcing: disc.global_vector()?,
            forcing_stale: true,
            work: disc.global_vector()?,
            rate: disc.global_vector()?,
            rate_shift: None,
            integral: Integral::new(0.0, zeros(&disc)?, disc.num_quadrature_points()),
        };

        let mut ts = Ts::implicit(&disc.dm().comm(), equation, &shell, &shell)?;
        ts.set_options_prefix(&prefix)?;
        ts.set_type("bdf")?;
        ts.set_bdf_order(2)?;
        ts.set_problem_type(ProblemType::Linear)?;
        ts.set_adapt_type(if time.adaptive { "basic" } else { "none" })?;
        ts.set_exact_final_time(ExactFinalTime::MatchStep)?;
        ts.set_time(0.0)?;
        ts.set_time_step(time.dt)?;
        let mut ksp = ts.snes()?.ksp()?;
        ksp.set_type(if symmetric { "cg" } else { "gmres" })?;
        // The SNES sets the same operators when it solves; set here, configure_multigrid checks
        // the levels against them.
        ksp.set_operators(&shell, &shell)?;
        let multigrid = configure_multigrid(&mut ksp, mesh, &level_operators, symmetric)?;
        ts.set_from_options()?;
        // Unless the options replaced it.
        if ksp.pc()?.type_name()? == "mg" {
            ts.set_jacobian_matrices(multigrid.finest(), multigrid.finest())?;
            ts.context_mut().multigrid = Some(multigrid);
        }

        let mut u = disc.global_vector()?;
        u.set(field.initial)?;
        let integral = &ts.context().integral;
        let mark = Mark {
            t: 0.0,
            integral: integral.current().c.duplicate()?,
            source: integral.current_source(&ts.context().f),
            c: u.duplicate()?,
            mass: 0.0,
            jumps: zeros(&disc)?,
            exchanged: Exchanged::default(),
        };
        let mut problem = Problem {
            ts,
            u,
            t: 0.0,
            dt: time.dt,
            adaptive: time.adaptive,
            failure: None,
            values: PointValues::new(&disc)?,
            free: disc.without_essential(mesh, finest)?,
            disc,
            dirichlet,
            fluxes,
            conditions,
            edge_tags,
            edge_velocity: flow.map_or_else(Vec::new, |flow| flow.edge_velocity(finest).to_vec()),
            balance: OnceCell::new(),
            mark,
            levels,
            _options: options,
        };
        problem.mark.mass = problem.total_mass()?;
        Ok(problem)
    }

    /// Sets the solution to the interpolant of `f` (its values at the nodes), as the initial
    /// condition instead of the uniform `FieldSpec::initial`. The Dirichlet values stay on their
    /// boundaries. Set later, it is the solution at the current time, from which the time
    /// integration starts afresh, and the interval of [`Problem::interval_balance`] starts
    /// there. Collective.
    ///
    /// Errors: [`PdeError::Poisoned`] after a failed [`Problem::step_to`].
    pub fn set_initial_condition(&mut self, f: impl Fn([f64; 2]) -> f64) -> Result<()> {
        self.ensure_usable()?;
        let set = self.disc.interpolate(f, &mut self.u);
        self.restart()?;
        set?;
        self.start_interval()
    }

    /// Changes the solution between two steps: `f` changes the global vector of the solution
    /// ([`Problem::solution`], without the essential dofs) in place, at the current time, and the
    /// time integration starts afresh from it, as after [`Problem::set_initial_condition`]: BDF2
    /// starts again with its first step, without the history of the steps before, and the rate of
    /// change is unknown until the next step ([`Problem::mass_balance`]). The operators, their
    /// matrices and the multigrid do not depend on the solution and are kept. The change is
    /// booked in the balance over the interval ([`Problem::interval_balance`]) as what
    /// `exchanged` says it exchanged: its amount, `int phi` of the change, is `exchanged.exchange`
    /// plus the resupply, which the caller makes sure of. Collective if `f` is.
    ///
    /// Errors: [`PdeError::Poisoned`] after a failed [`Problem::step_to`] (then `f` is not
    /// called), and the error of `f` (the solution may be changed in part, and the time
    /// integration starts afresh from it all the same, but nothing is booked: the balance over
    /// the interval is then off by the change).
    pub fn modify(
        &mut self,
        f: impl FnOnce(&mut Vector) -> Result<()>,
        exchanged: &Exchanged,
    ) -> Result<()> {
        self.ensure_usable()?;
        self.mark.jumps.axpy(-1.0, &self.u)?;
        let changed = f(&mut self.u);
        self.mark.jumps.axpy(1.0, &self.u)?;
        self.restart()?;
        changed?;
        let booked = &mut self.mark.exchanged;
        booked.exchange += exchanged.exchange;
        for (tag, amount) in &exchanged.resupply {
            match booked.resupply.iter_mut().find(|(t, _)| t == tag) {
                Some((_, sum)) => *sum += amount,
                None => booked.resupply.push((tag.clone(), *amount)),
            }
        }
        Ok(())
    }

    /// Starts the time integration afresh from the solution: without the history of the steps
    /// before, and with the rate of change unknown until the next step.
    fn restart(&mut self) -> Result<()> {
        self.ts.restart_step()?;
        let equation = self.ts.context_mut();
        equation.rate_shift = None;
        equation.integral.restart();
        Ok(())
    }

    /// The porosity of the field at the quadrature points of the local cells, in the layout of
    /// [`Discretization::quadrature_points`].
    pub(crate) fn porosity(&self) -> &[f64] {
        &self.ts.context().porosity
    }

    /// The source `f` (mM/s) at the quadrature points of the local cells, in the layout of
    /// [`Discretization::quadrature_points`], for writing: the steps use the values it holds,
    /// integrated with the porosity (`int phi f v`). Initially the values of `FieldSpec::source`.
    pub fn source_mut(&mut self) -> &mut [f64] {
        let equation = self.ts.context_mut();
        // Integrated again at the next residual.
        equation.forcing_stale = true;
        equation.integral.source_changes(&equation.f);
        &mut equation.f
    }

    /// Steps to the time `t` (in s), the last steps shortened to end there (`TSAdapt` may split
    /// the remainder in two, rather than take a tiny last step). Fixed steps go on with
    /// `TimeSpec::dt` after it; adaptive steps with the adaptor's choice. Collective.
    ///
    /// Errors: [`PdeError::InvalidTime`] if `t` is before the current time or not finite;
    /// [`PdeError::Diverged`] if a solver does not converge or a step fails (then
    /// [`Problem::time`] is the time of the last step which succeeded), and [`PdeError::Petsc`]
    /// for other failures of PETSc. Either leaves the solvers in an unknown state: from then on
    /// the problem is poisoned, and [`Problem::step_to`] and [`Problem::set_initial_condition`]
    /// return [`PdeError::Poisoned`]; what it reached stays readable ([`Problem::time`],
    /// [`Problem::solution`], [`Problem::l2_error`], [`Problem::total_mass`]).
    pub fn step_to(&mut self, t: f64) -> Result<()> {
        self.ensure_usable()?;
        if !t.is_finite() || t < self.t {
            return Err(PdeError::InvalidTime { t, time: self.t });
        }
        if t == self.t {
            return Ok(());
        }
        let stepped = self.solve_to(t);
        if let Err(e) = &stepped {
            self.failure = Some(e.to_string());
        }
        stepped
    }

    /// [`PdeError::Poisoned`] if the time integration failed earlier.
    fn ensure_usable(&self) -> Result<()> {
        match &self.failure {
            Some(failure) => Err(PdeError::Poisoned {
                failure: failure.clone(),
                t: self.t,
            }),
            None => Ok(()),
        }
    }

    fn solve_to(&mut self, t: f64) -> Result<()> {
        self.ts.set_max_time(t)?;
        if !self.adaptive {
            // TSAdapt `none` keeps the step it last took, which may have been shortened to end
            // at the time before.
            self.ts.set_time_step(self.dt)?;
        }
        let solved = self.ts.solve(&mut self.u);
        self.t = self.ts.time()?;
        if self.adaptive {
            // Chosen after the last step, which ended at `t`: not shortened.
            self.dt = self.ts.time_step()?;
        }
        match solved {
            Err(e) if e.code == sys::PETSC_ERR_NOT_CONVERGED => Err(self.diverged(&e.message)?),
            Err(e) => Err(e.into()),
            Ok(()) => {
                let reason = self.ts.converged_reason()?;
                if reason.diverged() {
                    Err(self.diverged(&reason.name)?)
                } else {
                    Ok(())
                }
            }
        }
    }

    /// The error of a failed time integration: the reason of the innermost solver which
    /// diverged, else `message`.
    fn diverged(&self, message: &str) -> Result<PdeError> {
        let snes = self.ts.snes()?;
        let reasons = [
            ("KSP", snes.ksp()?.converged_reason()?),
            ("SNES", snes.converged_reason()?),
            ("TS", self.ts.converged_reason()?),
        ];
        let reason = reasons
            .iter()
            .find(|(_, reason)| reason.diverged())
            .map_or_else(
                || message.lines().next().unwrap_or_default().to_string(),
                |(solver, reason)| format!("{solver} {reason}"),
            );
        Ok(PdeError::Diverged { reason, t: self.t })
    }

    /// The time of the solution, in s.
    pub fn time(&self) -> f64 {
        self.t
    }

    /// The number of steps taken. Collective.
    pub fn steps(&self) -> Result<usize> {
        Ok(self.ts.step_number()?)
    }

    /// The number of linear (Krylov) iterations of all steps so far, failed ones included.
    /// Collective.
    pub fn linear_iterations(&self) -> Result<usize> {
        Ok(self.ts.linear_iterations()?)
    }

    /// The number of Jacobians built on the levels of the problem: one per level of the mesh,
    /// built once and kept for every step. For tests.
    #[doc(hidden)]
    pub fn operator_builds(&self) -> usize {
        self.levels
            .iter()
            .map(|level| level.shared().counts.jacobians.get())
            .sum()
    }

    /// The number of multigrid preconditioners configured over the levels of the problem: one,
    /// kept for every step (none if the options replace it). For tests.
    #[doc(hidden)]
    pub fn multigrid_builds(&self) -> usize {
        self.disc.shared().counts.multigrids.get()
    }

    /// The number of assemblies of the Jacobian of the finest level into its matrix: one when
    /// the multigrid is configured (at shift 0), then one per change of the shift
    /// ([`Problem::jacobian_shifts`]). For tests.
    #[doc(hidden)]
    pub fn jacobian_assemblies(&self) -> usize {
        self.disc.shared().counts.jacobian_assemblies.get()
    }

    /// The shifts the time integration evaluated the Jacobian at, in order. For tests.
    #[doc(hidden)]
    pub fn jacobian_shifts(&self) -> &[f64] {
        &self.ts.context().shifts
    }

    /// The size of the next step (in s), before it is shortened to end at a requested time: the
    /// fixed `TimeSpec::dt`, or the latest choice of the adaptor.
    pub fn time_step(&self) -> f64 {
        self.dt
    }

    /// The solution: a global vector of [`Problem::discretization`], without the essential dofs.
    pub fn solution(&self) -> &Vector {
        &self.u
    }

    /// The discretization of the field: the finest level, with the Dirichlet tags essential.
    pub fn discretization(&self) -> &Discretization {
        &self.disc
    }

    /// The L2 norm of the error of the solution against `exact(x, t)` at the current time, by
    /// the quadrature of the discretization. Collective.
    pub fn l2_error(&self, exact: impl Fn([f64; 2], f64) -> f64) -> Result<f64> {
        let values = self.point_values()?;
        let local: f64 = values
            .iter()
            .zip(self.disc.quadrature_points())
            .zip(self.disc.quadrature_weights())
            .map(|((u, &x), w)| w * (u - exact(x, self.t)).powi(2))
            .sum();
        Ok(self.disc.dm().comm().sum_real(local)?.sqrt())
    }

    /// The L2 norm of the solution at the current time (in mM*um), by the quadrature of the
    /// discretization: [`Problem::l2_error`] against zero. Collective.
    pub fn l2_norm(&self) -> Result<f64> {
        self.l2_error(|_, _| 0.0)
    }

    /// The solution as a new local vector of [`Problem::discretization`]: the entries of the
    /// local cells, ghosts and Dirichlet values included. For viewers: the field of the DM is
    /// named after the field id and its component after the unit, `mM`, and the vector has an
    /// empty name, so that a VTK viewer names its point data `<field id>.mM`. Collective.
    pub fn local_solution(&self) -> Result<Vector> {
        let mut local = local_with_dirichlet(&self.disc, &self.dirichlet)?;
        self.disc.dm().global_to_local(&self.u, &mut local)?;
        local.set_name("")?;
        Ok(local)
    }

    /// The amount of substance in the domain, `int phi u`: the integral of the solution weighed
    /// by the porosity (in mM*um^2), whose rate of change is [`MassBalance::storage_rate`].
    /// Collective.
    pub fn total_mass(&self) -> Result<f64> {
        let values = self.point_values()?;
        let local: f64 = values
            .iter()
            .zip(&self.ts.context().porosity)
            .zip(self.disc.quadrature_weights())
            .map(|((u, phi), w)| w * phi * u)
            .sum();
        Ok(self.disc.dm().comm().sum_real(local)?)
    }

    /// The mass balance at the end of the last step, from the discrete equations: the rate of
    /// change of [`Problem::total_mass`], the sources, the inflow through each tag of the
    /// boundary and the defect of the advecting velocity (see [`MassBalance`]). Its operators and
    /// the edges of the boundary by tag are built at the first call and kept. Collective.
    ///
    /// Errors: [`PdeError::NotStepped`] before the first step, and before the first step after
    /// [`Problem::set_initial_condition`]; [`PdeError::Poisoned`] after a failed
    /// [`Problem::step_to`].
    pub fn mass_balance(&self) -> Result<MassBalance> {
        self.ensure_usable()?;
        let equation = self.ts.context();
        let Some(shift) = equation.rate_shift else {
            return Err(PdeError::NotStepped { t: self.t });
        };
        // The BDF derivative is affine in the solution: u_t = shift u + (u_t - shift u) of the
        // last Jacobian.
        let mut rate = self.disc.global_vector()?;
        rate.set(0.0)?;
        rate.axpy(1.0, &equation.rate)?;
        rate.axpy(shift, &self.u)?;
        let mut c_t = self.disc.local_vector()?;
        c_t.set(0.0)?;
        self.disc.dm().global_to_local(&rate, &mut c_t)?;
        let c = self.local_solution()?;
        self.balance()?.compute(&State {
            f: &equation.f,
            porosity: &equation.porosity,
            c: &c,
            c_t: &c_t,
            values: &self.values,
            duration: 1.0,
        })
    }

    /// The time the interval of [`Problem::interval_balance`] starts at (s).
    pub fn interval_start(&self) -> f64 {
        self.mark.t
    }

    /// Starts the interval of [`Problem::interval_balance`] at the current time. Collective.
    ///
    /// Errors: [`PdeError::Poisoned`] after a failed [`Problem::step_to`].
    pub fn start_interval(&mut self) -> Result<()> {
        self.ensure_usable()?;
        let mass = self.total_mass()?;
        let jumps = zeros(&self.disc)?;
        let equation = self.ts.context();
        let now = equation.integral.current();
        self.mark = Mark {
            t: now.t,
            integral: now.c.duplicate()?,
            source: equation.integral.current_source(&equation.f),
            c: self.u.duplicate()?,
            mass,
            jumps,
            exchanged: Exchanged::default(),
        };
        Ok(())
    }

    /// The mass balance over the interval from [`Problem::interval_start`] to the current time,
    /// each term its mean rate there (the amount over the length of the interval, in
    /// mM*um^2/s): `storage_rate` the change of [`Problem::total_mass`]; `source`, the inflows
    /// and `advection_defect` the terms of the steps, integrated by the weights of the BDF
    /// steps ([`Integral`]); `exchange` and [`BoundaryInflow::resupply`] what the changes of
    /// [`Problem::modify`] booked. The interval starts at t = 0, then at every
    /// [`Problem::start_interval`] and [`Problem::set_initial_condition`]. Collective.
    ///
    /// Errors: [`PdeError::NotStepped`] for an interval of no length; [`PdeError::Poisoned`]
    /// after a failed [`Problem::step_to`]; [`PdeError::Unsupported`] if the steps were not of
    /// `TSBDF` of order 2 as [`Integral`] follows them (other options of the time integration).
    pub fn interval_balance(&self) -> Result<MassBalance> {
        self.ensure_usable()?;
        let equation = self.ts.context();
        equation.integral.check()?;
        let now = equation.integral.current();
        let duration = now.t - self.mark.t;
        if duration <= 0.0 || duration.is_nan() {
            return Err(PdeError::NotStepped { t: self.t });
        }
        // The integrals of the solution, the Dirichlet values included, and of the source.
        let mut integral = now.c.duplicate()?;
        integral.axpy(-1.0, &self.mark.integral)?;
        let scaled: Vec<(String, f64)> = self
            .dirichlet
            .iter()
            .map(|(tag, value)| (tag.clone(), value * duration))
            .collect();
        let mut c = local_with_dirichlet(&self.disc, &scaled)?;
        self.disc.dm().global_to_local(&integral, &mut c)?;
        let mut f = equation.integral.current_source(&equation.f);
        for (f, start) in f.iter_mut().zip(&self.mark.source) {
            *f -= start;
        }
        // The change of the steps: of the solution, but the changes between them.
        let mut change = self.u.duplicate()?;
        change.axpy(-1.0, &self.mark.c)?;
        change.axpy(-1.0, &self.mark.jumps)?;
        let mut c_t = self.disc.local_vector()?;
        c_t.set(0.0)?;
        self.disc.dm().global_to_local(&change, &mut c_t)?;
        let mut balance = self.balance()?.compute(&State {
            f: &f,
            porosity: &equation.porosity,
            c: &c,
            c_t: &c_t,
            values: &self.values,
            duration,
        })?;
        balance.storage_rate = self.total_mass()? - self.mark.mass;
        balance.exchange = self.mark.exchanged.exchange;
        for (tag, amount) in &self.mark.exchanged.resupply {
            match balance.boundary.iter_mut().find(|b| &b.tag == tag) {
                Some(inflow) => inflow.resupply += amount,
                None => balance.boundary.push(BoundaryInflow {
                    tag: tag.clone(),
                    advective: 0.0,
                    diffusive: 0.0,
                    resupply: *amount,
                }),
            }
        }
        balance.storage_rate /= duration;
        balance.source /= duration;
        balance.advection_defect /= duration;
        balance.exchange /= duration;
        for inflow in &mut balance.boundary {
            inflow.advective /= duration;
            inflow.diffusive /= duration;
            inflow.resupply /= duration;
        }
        Ok(balance)
    }

    /// The operators of the mass balance, built at the first call and kept. Collective at the
    /// first call.
    fn balance(&self) -> Result<&Balance> {
        let equation = self.ts.context();
        if self.balance.get().is_none() {
            let dirichlet: Vec<&str> = self.dirichlet.iter().map(|(t, _)| t.as_str()).collect();
            let fluxes: Vec<(&str, f64)> =
                self.fluxes.iter().map(|(t, q)| (t.as_str(), *q)).collect();
            let balance = Balance::new(&Setup {
                disc: &self.disc,
                free: &self.free,
                data: &equation.data,
                conditions: &self.conditions,
                dirichlet: &dirichlet,
                fluxes: &fluxes,
                edge_tags: &self.edge_tags,
                edge_velocity: &self.edge_velocity,
            })?;
            self.balance.get_or_init(|| balance);
        }
        Ok(self.balance.get().expect("built above"))
    }

    /// The solution at the quadrature points of the local cells, Dirichlet values included.
    fn point_values(&self) -> Result<Vec<f64>> {
        let local = self.local_solution()?;
        let mut values = vec![0.0; self.disc.num_quadrature_points()];
        self.values.apply(&local, &mut values)?;
        Ok(values)
    }
}

/// The first tag of `flows` (net flows into the domain by tag) with inflow which `fixed` does not
/// list. A tag whose flow is at most `INFLOW_THRESHOLD` of the largest one counts as without
/// flow: its sign is rounding, which may differ between process counts and builds.
fn inflow_without_dirichlet<'a>(flows: &'a [(String, f64)], fixed: &[&str]) -> Option<&'a str> {
    let largest = flows.iter().fold(0.0_f64, |m, (_, q)| m.max(q.abs()));
    flows
        .iter()
        .find(|(tag, q)| *q > INFLOW_THRESHOLD * largest && !fixed.contains(&tag.as_str()))
        .map(|(tag, _)| tag.as_str())
}

/// The options prefix of the field `id`: `<id>_`.
fn options_prefix(id: &str) -> Result<String> {
    let valid = id.starts_with(|c: char| c.is_ascii_alphabetic())
        && id.chars().all(|c| c.is_ascii_alphanumeric() || c == '_');
    if valid {
        Ok(format!("{id}_"))
    } else {
        Err(PdeError::Unsupported(format!(
            "field id '{id}': it names the PETSc options prefix of the field, so it starts with \
             an ASCII letter and has ASCII letters, digits and '_' only"
        )))
    }
}

/// A global vector of `disc` of zeros.
fn zeros(disc: &Discretization) -> Result<Vector> {
    let mut zeros = disc.global_vector()?;
    zeros.set(0.0)?;
    Ok(zeros)
}

/// A local vector of `disc` with the Dirichlet values `dirichlet` (in order) and zeros.
fn local_with_dirichlet(disc: &Discretization, dirichlet: &[(String, f64)]) -> Result<Vector> {
    let mut local = disc.local_vector()?;
    local.set(0.0)?;
    for (tag, value) in dirichlet {
        disc.set_essential_values(&mut local, tag, *value)?;
    }
    Ok(local)
}

/// The coefficients of a field at the quadrature points of a discretization (see [`Problem`]).
struct Coefficients {
    porosity: Vec<f64>,
    diffusion: Vec<f64>,
    decay: Vec<f64>,
    /// Of the finest level only (empty on the others).
    source: Vec<f64>,
}

impl Coefficients {
    /// The coefficients of `field` on `disc` from `maps`, with the source if `source`.
    /// Collective.
    ///
    /// Errors (on every process): [`PdeError::Map`], naming the field and the coefficient.
    fn new(disc: &Discretization, field: &FieldSpec, maps: &Maps, source: bool) -> Result<Self> {
        let values = |name: &str, c: &Coefficient| {
            coefficient_values(disc, c, maps).map_err(|e| match e {
                PdeError::Map { message } => PdeError::Map {
                    message: format!("field '{}', {name}: {message}", field.id),
                },
                e => e,
            })
        };
        Ok(Coefficients {
            porosity: values("porosity", &field.porosity)?,
            diffusion: values("diffusion", &field.diffusion)?,
            decay: values("decay", &field.decay)?,
            source: if source {
                values("source", &field.source)?
            } else {
                Vec::new()
            },
        })
    }

    /// The data of the operator: mass `phi`, reaction `phi k` and diffusion `phi d I`; and,
    /// with `advection`, its velocity with SUPG and the backflow term on the boundary. Collective.
    fn point_data(
        &self,
        disc: &Discretization,
        advection: Option<Advection<'_>>,
    ) -> Result<PointData> {
        let scaled = |c: &[f64]| -> Vec<f64> {
            c.iter()
                .zip(&self.porosity)
                .map(|(c, phi)| phi * c)
                .collect()
        };
        let data = PointData::isotropic(
            disc,
            &self.porosity,
            &scaled(&self.decay),
            &scaled(&self.diffusion),
        );
        Ok(match advection {
            Some(Advection {
                velocity,
                edges,
                dirichlet,
            }) => data
                .with_velocity(velocity.to_vec())
                .stabilized(disc)
                .with_backflow(disc, edges, dirichlet)?,
            None => data,
        })
    }
}

/// The flow of a level, for the advection of a field.
#[derive(Clone, Copy)]
struct Advection<'a> {
    /// At the quadrature points.
    velocity: &'a [[f64; 2]],
    /// At the Gauss points of the edges of the boundary ([`Flow::edge_velocity`]).
    edges: &'a [(Point, [[f64; 2]; 2])],
    /// The Dirichlet tags of the field.
    dirichlet: &'a [&'a str],
}

/// The start of the interval of [`Problem::interval_balance`], and what changed the solution
/// since then besides the steps.
#[derive(Debug)]
struct Mark {
    /// The time of the start (s).
    t: f64,
    /// The integrals of the steps ([`Integral`]) at the start: of the solution (a global
    /// vector), and of the source at the quadrature points.
    integral: Vector,
    source: Vec<f64>,
    /// The solution at the start: a global vector.
    c: Vector,
    /// `int phi u` at the start.
    mass: f64,
    /// The sum of the changes of [`Problem::modify`] since the start: a global vector.
    jumps: Vector,
    /// What they exchanged, the resupply summed by tag.
    exchanged: Exchanged,
}

/// The equation of a [`Problem`] for its `TS`:
/// `F(u, u_t) = M_phi u_t + (A_w + K_(phi d) + M_(phi k)) u - int phi f v - int_Gamma q v`
/// (with SUPG test functions for advection, see [`Problem`]).
#[derive(Debug)]
struct Equation {
    /// `A_w + K_(phi d) + M_(phi k)` assembled, on the dofs without the essential ones.
    steady: Matrix,
    /// The steady terms of the Dirichlet values: [`Transport::apply_local`] of the steady
    /// terms to the local vector of the Dirichlet values and zeros.
    lift: Vector,
    /// `M_phi` assembled.
    mass: Matrix,
    /// The coefficients of the finest level.
    data: Rc<PointData>,
    /// The Jacobian `shift M_phi + A_w + K_(phi d) + M_(phi k)`, shared with its shell matrix.
    jacobian: Rc<RefCell<Transport>>,
    /// The shift the preconditioner was built at, and with multigrid the matrices of the levels
    /// assembled at, the Jacobian the Krylov method applies among them: within
    /// [`SHIFT_TOLERANCE`] of the shift of the time integration.
    built_shift: f64,
    /// The shifts of the Jacobians of the time integration, in order (see
    /// [`Problem::jacobian_shifts`]).
    shifts: Vec<f64>,
    /// The multigrid preconditioner of the Jacobian, unless the options chose another one.
    multigrid: Option<Multigrid>,
    /// `phi f` at the points, integrated.
    source: SourceField,
    /// The source `f` at the points (see [`Problem::source_mut`]).
    f: Vec<f64>,
    /// The porosity `phi` at the points.
    porosity: Vec<f64>,
    /// The load of the boundary fluxes, if any.
    load: Option<Vector>,
    /// The forcing of the equation, `int phi f v + int_Gamma q v`: the source integrated, and
    /// the load. Integrated again only when the source changed.
    forcing: Vector,
    /// Whether the source changed since the forcing was integrated.
    forcing_stale: bool,
    work: Vector,
    /// `u_t - shift u` of the last Jacobian: the time derivative of an implicit step of `TS` is
    /// affine in the solution (`u_t = shift u + rate` within a step, for `TSBDF` the
    /// derivative of its interpolating polynomial), so `shift u + rate` is the derivative at
    /// the solution of the last step.
    rate: Vector,
    /// The shift of the last Jacobian; `None` before the first, and after a new initial
    /// condition.
    rate_shift: Option<f64>,
    /// The integral of the solution and the source over the steps.
    integral: Integral,
}

impl Equation {
    fn residual(&mut self, u: &VecRef<'_>, u_t: &VecRef<'_>, f: &mut VecMut<'_>) -> Result<()> {
        // The steady terms of u with its Dirichlet values: of the dofs u holds, and of the
        // Dirichlet values.
        self.steady.mult(u, f)?;
        f.axpy(1.0, &self.lift)?;
        // The essential dofs are constant: u_t holds the others.
        self.mass.mult(u_t, &mut self.work)?;
        f.axpy(1.0, &self.work)?;
        if self.forcing_stale {
            for ((s, f), phi) in self
                .source
                .values_mut()
                .iter_mut()
                .zip(&self.f)
                .zip(&self.porosity)
            {
                *s = phi * f;
            }
            self.source.apply(&mut self.forcing)?;
            if let Some(load) = &self.load {
                self.forcing.axpy(1.0, load)?;
            }
            self.forcing_stale = false;
        }
        f.axpy(-1.0, &self.forcing)?;
        Ok(())
    }
}

impl Implicit for Equation {
    fn residual(
        &mut self,
        _t: f64,
        u: &VecRef<'_>,
        u_t: &VecRef<'_>,
        f: &mut VecMut<'_>,
    ) -> petsc::Result<()> {
        Equation::residual(self, u, u_t, f).map_err(PdeError::into_petsc)
    }

    fn jacobian(
        &mut self,
        _t: f64,
        u: &VecRef<'_>,
        u_t: &VecRef<'_>,
        shift: f64,
    ) -> petsc::Result<bool> {
        self.rate.set(0.0)?;
        self.rate.axpy(1.0, u_t)?;
        self.rate.axpy(-shift, u)?;
        self.rate_shift = Some(shift);
        self.shifts.push(shift);
        // The matrix-free operator applies the exact shift. The preconditioner, and with
        // multigrid the assembled Jacobian the Krylov method applies, follow changes beyond
        // rounding from the shift they were built at; comparing with the shift of the last call
        // instead would let small changes add up unseen.
        self.jacobian.borrow_mut().shift = shift;
        let rebuild = (shift - self.built_shift).abs() > SHIFT_TOLERANCE * shift.abs();
        if rebuild {
            self.built_shift = shift;
            if let Some(multigrid) = &mut self.multigrid {
                multigrid.set_shift(shift).map_err(PdeError::into_petsc)?;
            }
        }
        Ok(rebuild)
    }

    fn post_stage(&mut self, t: f64, y: &VecRef<'_>) -> petsc::Result<()> {
        let shift = self.rate_shift.unwrap_or(f64::NAN);
        self.integral
            .stage(t, y, shift)
            .map_err(PdeError::into_petsc)
    }

    fn post_step(&mut self, t: f64, _u: &VecRef<'_>) -> petsc::Result<()> {
        self.integral.step(t);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::inflow_without_dirichlet;

    fn flows(values: &[(&str, f64)]) -> Vec<(String, f64)> {
        values.iter().map(|&(t, q)| (t.to_string(), q)).collect()
    }

    /// Rounding of either sign is no inflow; a flow above the threshold is, fixed or not.
    #[test]
    fn rounding_is_not_inflow() {
        let rounding = flows(&[
            ("left", 1.0),
            ("right", -1.0),
            ("top", 2e-12),
            ("bottom", -5e-12),
        ]);
        assert_eq!(inflow_without_dirichlet(&rounding, &["left"]), None);
        assert_eq!(inflow_without_dirichlet(&rounding, &[]), Some("left"));
        // Relative to the largest flow, whatever its scale.
        let scaled = flows(&[("left", 1e6), ("right", -1e6), ("top", 1e-5)]);
        assert_eq!(inflow_without_dirichlet(&scaled, &["left"]), None);
        // Above the threshold: inflow.
        let small = flows(&[("left", 1.0), ("right", -1.0), ("top", 1e-9)]);
        assert_eq!(inflow_without_dirichlet(&small, &["left"]), Some("top"));
        assert_eq!(inflow_without_dirichlet(&small, &["left", "top"]), None);
        // No flow at all: no inflow.
        let none = flows(&[("left", 0.0), ("right", 0.0)]);
        assert_eq!(inflow_without_dirichlet(&none, &[]), None);
    }
}
