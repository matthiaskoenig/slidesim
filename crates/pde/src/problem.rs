//! One diffusion-reaction field stepped in time: BDF2 of PETSc's `TS` on the P1 discretization
//! of the finest mesh level.

use std::cell::RefCell;
use std::rc::Rc;

use ceed::Ceed;
use contracts::{Condition, FieldSpec, TimeSpec};
use mesh::Mesh;
use petsc::{ExactFinalTime, Implicit, OptionsScope, ProblemType, Ts, VecMut, VecRef, Vector, sys};

use crate::operators::PointValues;
use crate::{
    DiffusionReaction, Discretization, Multigrid, PdeError, Result, SharedOperator, SourceField,
    configure_multigrid,
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

/// A change of the shift of the Jacobian, relative to the shift, beyond which the
/// preconditioner is rebuilt. Smaller changes are rounding (BDF2 with a fixed step computes
/// shifts which differ in the last bits).
const SHIFT_TOLERANCE: f64 = 1e-12;

/// The field `u` of a [`FieldSpec`] on a [`Mesh`], stepped in time:
///
/// `u_t = div(d grad u) - k u + f`, with `d` the diffusion coefficient (um^2/s), `k` the decay
/// rate (1/s) and `f` the source (mM/s), on the finest level of the mesh, in the weak form of P1
/// elements:
///
/// `M u_t + (d K + k M) u = int f v + sum_i int_(Gamma_i) q_i v`.
///
/// Boundary conditions, by tag ([`FieldSpec::boundaries`]): a Dirichlet value is fixed on the
/// tag (an essential boundary; where tags meet, the value of the tag listed last holds); a flux
/// `q` is an inward flux on the boundary edges of the tag (mM*um/s); everywhere else the flux
/// is zero (the natural boundary condition).
///
/// Time integration: `TSBDF` of order 2 with the time step [`TimeSpec::dt`]: fixed, or the
/// initial step of `TSAdapt` `basic` if [`TimeSpec::adaptive`]. The equation is linear, so each
/// step is one linear solve (`SNESKSPONLY`): CG with the Jacobian `shift M + d K + k M` as a
/// matrix-free shell operator, preconditioned by geometric multigrid over the levels of the mesh
/// ([`configure_multigrid`]; a mesh without refinements is solved by its coarse solver, algebraic
/// multigrid, alone). The operators are built once; a step changes the shift only, on every
/// level, and when the shift changes the preconditioner is rebuilt (the smoothers set up again,
/// the coarse operator assembled again).
///
/// PETSc options: the `TS` and its solvers have the options prefix `<field id>_` (`u_` for the
/// field `u`, so its linear solver reads `-u_ksp_type`). The problem sets its options in the
/// global options database under that prefix while it lives, and removes them when dropped:
/// first `-ksp_error_if_not_converged -snes_error_if_not_converged -ts_error_if_step_fails` and
/// `-ksp_rtol 1e-10`, then the options of the caller (`petsc_options` of [`Problem::new`], written without the
/// prefix), which may override them. Options under the prefix from the command line hold too,
/// unless the caller's override them. They override the multigrid preconditioner too: for
/// example `-pc_type jacobi` replaces it, `-mg_levels_ksp_max_it 3` changes its smoothers and
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
    dirichlet: Vec<(String, f64)>,
    values: PointValues,
    /// Dropped after the `TS`, which may read options until it is destroyed.
    _options: OptionsScope,
}

impl Problem {
    /// The problem of the field `field` on the finest level of `mesh`, at t = 0 with the uniform
    /// initial value `field.initial` and the uniform source `field.source`, stepped as `time`
    /// says (its time step and adaptivity), and with the PETSc options `petsc_options` (for
    /// example `-ksp_rtol 1e-8`). Collective.
    ///
    /// Errors: [`PdeError::Unsupported`] if the field id cannot be an options prefix (it starts
    /// with an ASCII letter and has ASCII letters, digits and `_` only), or a flux is on a tag
    /// which is not of boundary edges; [`PdeError::Mesh`] for a tag the mesh does not have;
    /// [`PdeError::Petsc`] if a problem of the same field id is alive, or PETSc rejects the
    /// options.
    pub fn new(
        mesh: &Mesh,
        field: &FieldSpec,
        time: &TimeSpec,
        ceed: &Ceed,
        petsc_options: &str,
    ) -> Result<Problem> {
        let prefix = options_prefix(&field.id)?;
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
        let essential: Vec<&str> = dirichlet.iter().map(|(tag, _)| tag.as_str()).collect();
        let levels = Discretization::hierarchy(mesh, ceed, 1, &essential)?;
        let disc = levels.last().expect("a mesh has a level").clone();
        // For viewers of the solution (see `Problem::local_solution`); the clone is a handle of
        // the same DM.
        disc.dm()
            .clone()
            .set_field_names(0, &field.id, &[CONCENTRATION_UNIT])?;

        let (d, k) = (field.diffusion, field.decay);
        let mut source = SourceField::new(&disc)?;
        source.values_mut().fill(field.source);
        let load = if fluxes.is_empty() {
            None
        } else {
            let fluxes: Vec<(&str, f64)> = fluxes.iter().map(|(t, q)| (t.as_str(), *q)).collect();
            Some(disc.boundary_load(&fluxes)?)
        };
        let jacobian = Rc::new(RefCell::new(DiffusionReaction::new(&disc, d, k, 0.0)?));
        let mut shell = disc.shell_matrix(SharedOperator(Rc::clone(&jacobian)))?;
        shell.set_spd()?;
        // The operators of the coarser levels, then the Jacobian on the finest.
        let level_operators = levels[..levels.len() - 1]
            .iter()
            .map(|level| {
                let op = DiffusionReaction::new(level, d, k, 0.0)?;
                Ok(SharedOperator(Rc::new(RefCell::new(op))))
            })
            .chain(std::iter::once(Ok(SharedOperator(Rc::clone(&jacobian)))))
            .collect::<Result<Vec<_>>>()?;
        let equation = Equation {
            operator: DiffusionReaction::new(&disc, d, k, 0.0)?,
            mass: DiffusionReaction::new(&disc, 0.0, 0.0, 1.0)?,
            jacobian,
            multigrid: None,
            source,
            load,
            u_local: local_with_dirichlet(&disc, &dirichlet)?,
            work: disc.global_vector()?,
            disc: disc.clone(),
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
        ksp.set_type("cg")?;
        let multigrid = configure_multigrid(&mut ksp, mesh, &level_operators)?;
        ts.set_from_options()?;
        // Unless the options replaced it.
        if ksp.pc()?.type_name()? == "mg" {
            ts.context_mut().multigrid = Some(multigrid);
        }

        let mut u = disc.global_vector()?;
        u.set(field.initial)?;
        Ok(Problem {
            ts,
            u,
            t: 0.0,
            dt: time.dt,
            adaptive: time.adaptive,
            failure: None,
            values: PointValues::new(&disc)?,
            disc,
            dirichlet,
            _options: options,
        })
    }

    /// Sets the solution to the interpolant of `f` (its values at the nodes), as the initial
    /// condition instead of the uniform `FieldSpec::initial`. The Dirichlet values stay on their
    /// boundaries. Set later, it is the solution at the current time, from which the time
    /// integration starts afresh. Collective.
    ///
    /// Errors: [`PdeError::Poisoned`] after a failed [`Problem::step_to`].
    pub fn set_initial_condition(&mut self, f: impl Fn([f64; 2]) -> f64) -> Result<()> {
        self.ensure_usable()?;
        self.disc.interpolate(f, &mut self.u)?;
        self.ts.restart_step()?;
        Ok(())
    }

    /// The source `f` (mM/s) at the quadrature points of the local cells, in the layout of
    /// [`Discretization::quadrature_points`], for writing: the steps use the values it holds.
    /// Initially the uniform `FieldSpec::source`.
    pub fn source_mut(&mut self) -> &mut [f64] {
        self.ts.context_mut().source.values_mut()
    }

    /// Steps to the time `t` (in s), the last steps shortened to end there (`TSAdapt` may split
    /// the remainder in two, rather than take a tiny last step). Fixed steps go on with
    /// `TimeSpec::dt` after it; adaptive steps with the adaptor's choice. Collective.
    ///
    /// Errors: [`PdeError::InvalidTime`] if `t` is before the current time;
    /// [`PdeError::Diverged`] if a solver does not converge or a step fails (then
    /// [`Problem::time`] is the time of the last step which succeeded), and [`PdeError::Petsc`]
    /// for other failures of PETSc. Either leaves the solvers in an unknown state: from then on
    /// the problem is poisoned, and [`Problem::step_to`] and [`Problem::set_initial_condition`]
    /// return [`PdeError::Poisoned`]; what it reached stays readable ([`Problem::time`],
    /// [`Problem::solution`], [`Problem::l2_error`], [`Problem::total_mass`]).
    pub fn step_to(&mut self, t: f64) -> Result<()> {
        self.ensure_usable()?;
        if t.is_nan() || t < self.t {
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

    /// The integral of the solution over the domain (in mM*um^2). Collective.
    pub fn total_mass(&self) -> Result<f64> {
        let values = self.point_values()?;
        let local: f64 = values
            .iter()
            .zip(self.disc.quadrature_weights())
            .map(|(u, w)| w * u)
            .sum();
        Ok(self.disc.dm().comm().sum_real(local)?)
    }

    /// The solution at the quadrature points of the local cells, Dirichlet values included.
    fn point_values(&self) -> Result<Vec<f64>> {
        let local = self.local_solution()?;
        let mut values = vec![0.0; self.disc.num_quadrature_points()];
        self.values.apply(&local, &mut values)?;
        Ok(values)
    }
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

/// A local vector of `disc` with the Dirichlet values `dirichlet` (in order) and zeros.
fn local_with_dirichlet(disc: &Discretization, dirichlet: &[(String, f64)]) -> Result<Vector> {
    let mut local = disc.local_vector()?;
    local.set(0.0)?;
    for (tag, value) in dirichlet {
        disc.set_essential_values(&mut local, tag, *value)?;
    }
    Ok(local)
}

/// The equation of a [`Problem`] for its `TS`:
/// `F(u, u_t) = M u_t + (d K + k M) u - int f v - int_Gamma q v`.
#[derive(Debug)]
struct Equation {
    disc: Discretization,
    /// `d K + k M`.
    operator: DiffusionReaction,
    /// `M`.
    mass: DiffusionReaction,
    /// The Jacobian `shift M + d K + k M`, shared with its shell matrix.
    jacobian: Rc<RefCell<DiffusionReaction>>,
    /// The multigrid preconditioner of the Jacobian, unless the options chose another one.
    multigrid: Option<Multigrid>,
    source: SourceField,
    /// The load of the boundary fluxes, if any.
    load: Option<Vector>,
    /// The solution with its Dirichlet values, which stay.
    u_local: Vector,
    work: Vector,
}

impl Equation {
    fn residual(&mut self, u: &VecRef<'_>, u_t: &VecRef<'_>, f: &mut VecMut<'_>) -> Result<()> {
        self.disc.dm().global_to_local(u, &mut self.u_local)?;
        self.operator.apply_local(&self.u_local, f)?;
        // The essential dofs are constant: u_t holds the others.
        self.mass.mult_global(u_t, &mut self.work)?;
        f.axpy(1.0, &self.work)?;
        self.source.apply(&mut self.work)?;
        f.axpy(-1.0, &self.work)?;
        if let Some(load) = &self.load {
            f.axpy(-1.0, load)?;
        }
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
        _u: &VecRef<'_>,
        _u_t: &VecRef<'_>,
        shift: f64,
    ) -> petsc::Result<bool> {
        let (d, k, same) = {
            let mut jacobian = self.jacobian.borrow_mut();
            // The operator applies the exact shift; the preconditioner follows changes beyond
            // rounding.
            let same = (shift - jacobian.shift).abs() <= SHIFT_TOLERANCE * shift.abs();
            jacobian.shift = shift;
            (jacobian.d, jacobian.k, same)
        };
        if !same && let Some(multigrid) = &mut self.multigrid {
            multigrid
                .set_coefficients(d, k, shift)
                .map_err(PdeError::into_petsc)?;
        }
        Ok(!same)
    }
}
