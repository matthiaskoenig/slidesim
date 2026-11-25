//! One diffusion-reaction field stepped in time: BDF2 of PETSc's `TS` on the P1 discretization
//! of the finest mesh level.

use std::cell::RefCell;
use std::rc::Rc;

use ceed::Ceed;
use contracts::{Condition, FieldSpec, TimeSpec};
use mesh::Mesh;
use petsc::{ExactFinalTime, Implicit, OptionsScope, ProblemType, Ts, VecMut, VecRef, Vector, sys};

use crate::operators::PointValues;
use crate::{DiffusionReaction, Discretization, PdeError, Result, SharedOperator, SourceField};

/// The options every problem sets before those of the caller, who may override them: a solver
/// which does not converge is an error ([`PdeError::Diverged`]), not a silently wrong step.
const ERRORS_IF_NOT_CONVERGED: &str =
    "-ksp_error_if_not_converged -snes_error_if_not_converged -ts_error_if_step_fails";

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
/// matrix-free shell operator, preconditioned by Jacobi from its diagonal. The operators are
/// built once; a step changes the shift only.
///
/// PETSc options: the `TS` and its solvers have the options prefix `<field id>_` (`u_` for the
/// field `u`, so its linear solver reads `-u_ksp_type`). The problem sets its options in the
/// global options database under that prefix while it lives, and removes them when dropped:
/// first `-ksp_error_if_not_converged -snes_error_if_not_converged -ts_error_if_step_fails`,
/// then the options of the caller (`petsc_options` of [`Problem::new`], written without the
/// prefix), which may override them. Options under the prefix from the command line hold too,
/// unless the caller's override them.
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
        let options = OptionsScope::new(&prefix, &[ERRORS_IF_NOT_CONVERGED, petsc_options])?;

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
        let finest = mesh.levels().len() - 1;
        let disc = Discretization::with_essential(mesh, finest, ceed, 1, &essential)?;

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
        let shell = disc.shell_matrix(SharedOperator(Rc::clone(&jacobian)))?;
        let equation = Equation {
            operator: DiffusionReaction::new(&disc, d, k, 0.0)?,
            mass: DiffusionReaction::new(&disc, 0.0, 0.0, 1.0)?,
            jacobian,
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
        ksp.set_pc_type("jacobi")?;
        ts.set_from_options()?;

        let mut u = disc.global_vector()?;
        u.set(field.initial)?;
        Ok(Problem {
            ts,
            u,
            t: 0.0,
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
    pub fn set_initial_condition(&mut self, f: impl Fn([f64; 2]) -> f64) -> Result<()> {
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

    /// Steps to the time `t` (in s), the last step shortened to end there. Collective.
    ///
    /// Errors: [`PdeError::InvalidTime`] if `t` is before the current time;
    /// [`PdeError::Diverged`] if a solver does not converge or a step fails (then
    /// [`Problem::time`] is the time of the last step which succeeded).
    pub fn step_to(&mut self, t: f64) -> Result<()> {
        if t.is_nan() || t < self.t {
            return Err(PdeError::InvalidTime { t, time: self.t });
        }
        if t == self.t {
            return Ok(());
        }
        self.ts.set_max_time(t)?;
        let solved = self.ts.solve(&mut self.u);
        self.t = self.ts.time()?;
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
        let mut local = local_with_dirichlet(&self.disc, &self.dirichlet)?;
        self.disc.dm().global_to_local(&self.u, &mut local)?;
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
        let mut jacobian = self.jacobian.borrow_mut();
        // The operator applies the exact shift; the preconditioner follows changes beyond
        // rounding.
        let same = (shift - jacobian.shift).abs() <= SHIFT_TOLERANCE * shift.abs();
        jacobian.shift = shift;
        Ok(!same)
    }
}
