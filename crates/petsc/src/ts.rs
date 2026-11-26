//! `TS`: time steppers, with implicit equations given by Rust code.

use std::ffi::{c_char, c_void};
use std::ptr::null_mut;

use crate::error::{callback, failed_solver};
use crate::{
    Comm, ConvergedReason, Matrix, Result, Snes, VecMut, VecRef, Vector, alive, c_string, chk, sys,
};

/// An implicit equation `F(t, u, u_t) = 0` for a [`Ts`]: its residual and its Jacobian.
///
/// PETSc lends the vectors for the duration of the call ([`VecRef`], [`VecMut`]), and calls one
/// method at a time.
pub trait Implicit {
    /// `f = F(t, u, u_t)`.
    fn residual(
        &mut self,
        t: f64,
        u: &VecRef<'_>,
        u_t: &VecRef<'_>,
        f: &mut VecMut<'_>,
    ) -> Result<()>;

    /// Makes the matrices given to [`Ts::implicit`] the Jacobian `dF/du + shift dF/du_t` at
    /// `(t, u, u_t)` (and the matrix of its preconditioner), for example by changing the
    /// coefficients of a shell operator. Returns whether they changed: then they are marked
    /// changed (`MatAssemblyBegin` and `MatAssemblyEnd`), so that the preconditioner is rebuilt.
    fn jacobian(&mut self, t: f64, u: &VecRef<'_>, u_t: &VecRef<'_>, shift: f64) -> Result<bool>;

    /// Called after the solve of each stage of a step (`TSSetPostStage`), with the time of the
    /// stage and its solution `y`. A step of `TSBDF` has one stage, the first step after a
    /// restart two (a half step of the first order, then the step). An adaptive step control
    /// may reject a stage after this call: then the step is tried again, shorter. Nothing by
    /// default.
    fn post_stage(&mut self, _t: f64, _y: &VecRef<'_>) -> Result<()> {
        Ok(())
    }

    /// Called after each step which is accepted (`TSSetPostStep`), with the time it reached and
    /// the solution `u` there. Nothing by default.
    fn post_step(&mut self, _t: f64, _u: &VecRef<'_>) -> Result<()> {
        Ok(())
    }
}

/// The kind of equation of a [`Ts`] (`TSProblemType`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProblemType {
    /// `F` is linear in `u` and `u_t`: each step solves one linear system (`TS_LINEAR`, with
    /// `SNESKSPONLY`).
    Linear,
    /// `TS_NONLINEAR`.
    Nonlinear,
}

/// How a [`Ts`] ends at its final time (`TSExactFinalTimeOption`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExactFinalTime {
    /// The last step may go beyond it.
    StepOver,
    /// The last step may go beyond it, and the solution is interpolated back to it.
    Interpolate,
    /// The steps are adjusted to end exactly at it.
    MatchStep,
}

/// An owned PETSc time stepper (`TS`), with the context `C` of its callbacks: `()` for none, or
/// an [`Implicit`] equation ([`Ts::implicit`]). The context lives as long as the `TS`.
///
/// A time stepper whose [`Ts::solve`] failed is never destroyed, see there.
#[derive(Debug)]
pub struct Ts<C = ()> {
    raw: sys::TS,
    /// From `Box::into_raw`, freed on drop after the `TS`; only the callbacks and
    /// [`Ts::context_mut`] use it, one at a time.
    context: *mut C,
    /// Whether a [`Ts::solve`] failed.
    failed: bool,
}

impl Ts {
    /// `TSCreate`.
    pub fn create(comm: &Comm) -> Result<Ts> {
        Ts::with_context(comm, ())
    }
}

impl<C: Implicit> Ts<C> {
    /// A time stepper of the implicit equation `equation`: `TSSetIFunction`, and
    /// `TSSetIJacobian` with the Jacobian `jacobian` and the matrix `preconditioner` of its
    /// preconditioner (often `jacobian` again), which the `TS` keeps references to; the
    /// equation is the application context of the `TS`, for [`Implicit::post_stage`]
    /// (`TSSetPostStage`) and [`Implicit::post_step`] (`TSSetPostStep`).
    pub fn implicit(
        comm: &Comm,
        equation: C,
        jacobian: &Matrix,
        preconditioner: &Matrix,
    ) -> Result<Ts<C>> {
        let ts = Ts::with_context(comm, equation)?;
        let ctx: *mut c_void = ts.context.cast();
        // SAFETY: the functions get the context of a `Ts<C>`, which lives as long as the TS.
        chk!(unsafe { sys::TSSetIFunction(ts.raw, null_mut(), Some(ifunction::<C>), ctx) });
        chk!(unsafe {
            sys::TSSetIJacobian(
                ts.raw,
                jacobian.as_raw(),
                preconditioner.as_raw(),
                Some(ijacobian::<C>),
                ctx,
            )
        });
        // SAFETY: as above; the hooks find the context as the application context.
        chk!(unsafe { sys::TSSetApplicationContext(ts.raw, ctx) });
        chk!(unsafe { sys::TSSetPostStage(ts.raw, Some(post_stage::<C>)) });
        chk!(unsafe { sys::TSSetPostStep(ts.raw, Some(post_step::<C>)) });
        Ok(ts)
    }

    /// Replaces the matrices of the Jacobian and of its preconditioner (`TSSetIJacobian`, with
    /// the equation of [`Ts::implicit`]), which the `TS` keeps references to; the matrices
    /// [`Implicit::jacobian`] marks changed.
    pub fn set_jacobian_matrices(
        &mut self,
        jacobian: &Matrix,
        preconditioner: &Matrix,
    ) -> Result<()> {
        self.usable()?;
        let ctx: *mut c_void = self.context.cast();
        // SAFETY: as in `Ts::implicit`.
        chk!(unsafe {
            sys::TSSetIJacobian(
                self.raw,
                jacobian.as_raw(),
                preconditioner.as_raw(),
                Some(ijacobian::<C>),
                ctx,
            )
        });
        Ok(())
    }
}

impl<C> Ts<C> {
    fn with_context(comm: &Comm, context: C) -> Result<Ts<C>> {
        let mut raw = null_mut();
        chk!(unsafe { sys::TSCreate(comm.as_raw(), &mut raw) });
        Ok(Ts {
            raw,
            context: Box::into_raw(Box::new(context)),
            failed: false,
        })
    }

    pub fn as_raw(&self) -> sys::TS {
        self.raw
    }

    /// An error if a solve failed (see [`Ts::solve`]): for every method which changes or runs the
    /// time stepper.
    fn usable(&self) -> Result<()> {
        if self.failed {
            return Err(failed_solver("TS"));
        }
        Ok(())
    }

    /// The context of the callbacks.
    pub fn context(&self) -> &C {
        // SAFETY: valid while `self` lives; callbacks only run inside `&mut self` methods.
        unsafe { &*self.context }
    }

    /// The context of the callbacks, for writing.
    pub fn context_mut(&mut self) -> &mut C {
        // SAFETY: as in `context`, and `&mut self` excludes other uses meanwhile.
        unsafe { &mut *self.context }
    }

    /// `TSSetFromOptions`.
    pub fn set_from_options(&mut self) -> Result<()> {
        self.usable()?;
        chk!(unsafe { sys::TSSetFromOptions(self.raw) });
        Ok(())
    }

    /// Sets the prefix of the options of the `TS` and of its solvers (`TSSetOptionsPrefix`).
    pub fn set_options_prefix(&mut self, prefix: &str) -> Result<()> {
        self.usable()?;
        let prefix = c_string(prefix, "options prefix")?;
        chk!(unsafe { sys::TSSetOptionsPrefix(self.raw, prefix.as_ptr()) });
        Ok(())
    }

    /// Sets the method, for example `bdf` (`TSSetType`).
    pub fn set_type(&mut self, name: &str) -> Result<()> {
        self.usable()?;
        let name = c_string(name, "TS type")?;
        chk!(unsafe { sys::TSSetType(self.raw, name.as_ptr()) });
        Ok(())
    }

    /// Sets the order of a `bdf` method (`TSBDFSetOrder`; nothing for other methods).
    pub fn set_bdf_order(&mut self, order: usize) -> Result<()> {
        self.usable()?;
        let order = crate::to_int(order, "BDF order")?;
        chk!(unsafe { sys::TSBDFSetOrder(self.raw, order) });
        Ok(())
    }

    /// Sets the type of the time step adaptivity, for example `none` or `basic`
    /// (`TSAdaptSetType` of `TSGetAdapt`).
    pub fn set_adapt_type(&mut self, name: &str) -> Result<()> {
        self.usable()?;
        let name = c_string(name, "TSAdapt type")?;
        let mut adapt = null_mut();
        // SAFETY: the adaptor is borrowed from the TS.
        chk!(unsafe { sys::TSGetAdapt(self.raw, &mut adapt) });
        chk!(unsafe { sys::TSAdaptSetType(adapt, name.as_ptr()) });
        Ok(())
    }

    /// `TSSetProblemType`.
    pub fn set_problem_type(&mut self, kind: ProblemType) -> Result<()> {
        self.usable()?;
        let kind = match kind {
            ProblemType::Linear => sys::TS_LINEAR,
            ProblemType::Nonlinear => sys::TS_NONLINEAR,
        };
        chk!(unsafe { sys::TSSetProblemType(self.raw, kind) });
        Ok(())
    }

    /// `TSSetExactFinalTime`.
    pub fn set_exact_final_time(&mut self, option: ExactFinalTime) -> Result<()> {
        self.usable()?;
        let option = match option {
            ExactFinalTime::StepOver => sys::TS_EXACTFINALTIME_STEPOVER,
            ExactFinalTime::Interpolate => sys::TS_EXACTFINALTIME_INTERPOLATE,
            ExactFinalTime::MatchStep => sys::TS_EXACTFINALTIME_MATCHSTEP,
        };
        chk!(unsafe { sys::TSSetExactFinalTime(self.raw, option) });
        Ok(())
    }

    /// The current time (`TSGetTime`).
    pub fn time(&self) -> Result<f64> {
        let mut t = 0.0;
        chk!(unsafe { sys::TSGetTime(self.raw, &mut t) });
        Ok(t)
    }

    /// `TSSetTime`.
    pub fn set_time(&mut self, t: f64) -> Result<()> {
        self.usable()?;
        chk!(unsafe { sys::TSSetTime(self.raw, t) });
        Ok(())
    }

    /// The time step of the next step (`TSGetTimeStep`).
    pub fn time_step(&self) -> Result<f64> {
        let mut dt = 0.0;
        chk!(unsafe { sys::TSGetTimeStep(self.raw, &mut dt) });
        Ok(dt)
    }

    /// `TSSetTimeStep`.
    pub fn set_time_step(&mut self, dt: f64) -> Result<()> {
        self.usable()?;
        chk!(unsafe { sys::TSSetTimeStep(self.raw, dt) });
        Ok(())
    }

    /// Sets the time at which [`Ts::solve`] stops (`TSSetMaxTime`).
    pub fn set_max_time(&mut self, t: f64) -> Result<()> {
        self.usable()?;
        chk!(unsafe { sys::TSSetMaxTime(self.raw, t) });
        Ok(())
    }

    /// The number of steps taken (`TSGetStepNumber`).
    pub fn step_number(&self) -> Result<usize> {
        let mut steps = 0;
        chk!(unsafe { sys::TSGetStepNumber(self.raw, &mut steps) });
        Ok(steps as usize)
    }

    /// The number of linear iterations of all steps so far, failed ones included
    /// (`TSGetKSPIterations`).
    pub fn linear_iterations(&self) -> Result<usize> {
        let mut n = 0;
        // SAFETY: a valid TS; PETSc writes one integer.
        chk!(unsafe { sys::TSGetKSPIterations(self.raw, &mut n) });
        Ok(n as usize)
    }

    /// Starts the next step afresh, without the history of the steps before, as after a change
    /// of the solution (`TSRestartStep`): a multistep method starts with one step again.
    pub fn restart_step(&mut self) -> Result<()> {
        self.usable()?;
        chk!(unsafe { sys::TSRestartStep(self.raw) });
        Ok(())
    }

    /// Steps from the current time and solution `u` to the final time (`TSSolve`); `u` holds the
    /// solution reached, and becomes the solution vector of the `TS`. Collective.
    ///
    /// After an error the time stepper is failed, as a [`crate::Ksp`] after a failed
    /// [`crate::Ksp::solve`], for the same reason (its linear solver may be inconsistent): it is
    /// never destroyed (dropping it leaks it, with the context of its callbacks, and everything it
    /// holds), and every later call of a method which changes or runs it is an error
    /// (`PETSC_ERR_ARG_WRONGSTATE`); the methods taking `&self` still work.
    pub fn solve(&mut self, u: &mut Vector) -> Result<()> {
        self.usable()?;
        let solved = (|| {
            chk!(unsafe { sys::TSSolve(self.raw, u.as_raw()) });
            Ok(())
        })();
        self.failed = solved.is_err();
        solved
    }

    /// Why the last [`Ts::solve`] stopped (`TSGetConvergedReason`).
    pub fn converged_reason(&self) -> Result<ConvergedReason> {
        let mut code = 0;
        chk!(unsafe { sys::TSGetConvergedReason(self.raw, &mut code) });
        // TSConvergedReasons is indexed by the code, from its first to its last value.
        let known = sys::TSADJOINT_DIVERGED_LINEAR_SOLVE..=sys::TS_CONVERGED_PSEUDO_FRTOL;
        let name: *const c_char = if known.contains(&code) {
            // SAFETY: a static array of PETSc with a name for every code in `known`.
            unsafe { *sys::TSConvergedReasons.offset(code as isize) }
        } else {
            std::ptr::null()
        };
        Ok(ConvergedReason::new(code, name))
    }

    /// The nonlinear solver (`TSGetSNES`), shared.
    pub fn snes(&self) -> Result<Snes> {
        let mut raw = null_mut();
        chk!(unsafe { sys::TSGetSNES(self.raw, &mut raw) });
        // SAFETY: a valid SNES, borrowed from the TS.
        unsafe { Snes::shared(raw) }
    }
}

impl<C> Drop for Ts<C> {
    fn drop(&mut self) {
        if self.failed {
            // Leaked, with the context its callbacks may still be given (see `Ts::solve`).
            return;
        }
        let ierr = if alive() {
            unsafe { sys::TSDestroy(&mut self.raw) }
        } else {
            0
        };
        // After the TS, which may call the callbacks until it is destroyed. If PETSc is
        // finalized the TS is leaked, but it is never called again.
        drop(unsafe { Box::from_raw(self.context) });
        crate::error::destroyed(ierr, "TSDestroy");
    }
}

/// The context of a `Ts<C>`.
///
/// # Safety
/// `ctx` is the context of a live `Ts<C>`, used by nothing else meanwhile.
unsafe fn context<'a, C>(ctx: *mut c_void) -> &'a mut C {
    unsafe { &mut *ctx.cast::<C>() }
}

unsafe extern "C" fn ifunction<C: Implicit>(
    _ts: sys::TS,
    t: sys::PetscReal,
    u: sys::Vec,
    u_t: sys::Vec,
    f: sys::Vec,
    ctx: sys::PetscCtx,
) -> sys::PetscErrorCode {
    callback(c"Implicit::residual", || {
        // SAFETY: PETSc calls this for `Ts::implicit` only, from `Ts::solve` (`&mut self`), one
        // callback at a time, with valid vectors for the duration of the call.
        let equation = unsafe { context::<C>(ctx) };
        equation.residual(
            t,
            &unsafe { VecRef::lent(u) },
            &unsafe { VecRef::lent(u_t) },
            &mut unsafe { VecMut::lent(f) },
        )
    })
}

#[allow(clippy::too_many_arguments)]
unsafe extern "C" fn ijacobian<C: Implicit>(
    _ts: sys::TS,
    t: sys::PetscReal,
    u: sys::Vec,
    u_t: sys::Vec,
    shift: sys::PetscReal,
    a: sys::Mat,
    p: sys::Mat,
    ctx: sys::PetscCtx,
) -> sys::PetscErrorCode {
    callback(c"Implicit::jacobian", || {
        // SAFETY: as in `ifunction`.
        let equation = unsafe { context::<C>(ctx) };
        let changed = equation.jacobian(
            t,
            &unsafe { VecRef::lent(u) },
            &unsafe { VecRef::lent(u_t) },
            shift,
        )?;
        if changed {
            let matrices = if a == p { vec![a] } else { vec![a, p] };
            for m in matrices {
                chk!(unsafe { sys::MatAssemblyBegin(m, sys::MAT_FINAL_ASSEMBLY) });
                chk!(unsafe { sys::MatAssemblyEnd(m, sys::MAT_FINAL_ASSEMBLY) });
            }
        }
        Ok(())
    })
}

/// The context of the `Ts<C>` of `ts`, its application context ([`Ts::implicit`]).
///
/// # Safety
/// `ts` is the `TS` of a live `Ts<C>` made by [`Ts::implicit`], whose context nothing else uses
/// meanwhile.
unsafe fn application_context<'a, C>(ts: sys::TS) -> Result<&'a mut C> {
    let mut ctx: *mut c_void = null_mut();
    chk!(unsafe { sys::TSGetApplicationContext(ts, (&raw mut ctx).cast()) });
    Ok(unsafe { context::<C>(ctx) })
}

unsafe extern "C" fn post_stage<C: Implicit>(
    ts: sys::TS,
    t: sys::PetscReal,
    _stage: sys::PetscInt,
    y: *mut sys::Vec,
) -> sys::PetscErrorCode {
    callback(c"Implicit::post_stage", || {
        // SAFETY: PETSc calls this for `Ts::implicit` only, from `Ts::solve`, one callback at a
        // time, with the vector of the stage (`TSBDF` has one per stage) for the call.
        let equation = unsafe { application_context::<C>(ts) }?;
        equation.post_stage(t, &unsafe { VecRef::lent(*y) })
    })
}

unsafe extern "C" fn post_step<C: Implicit>(ts: sys::TS) -> sys::PetscErrorCode {
    callback(c"Implicit::post_step", || {
        let (mut t, mut u) = (0.0, null_mut());
        chk!(unsafe { sys::TSGetTime(ts, &mut t) });
        // SAFETY: the solution is borrowed from the TS for the call.
        chk!(unsafe { sys::TSGetSolution(ts, &mut u) });
        // SAFETY: as in `post_stage`.
        let equation = unsafe { application_context::<C>(ts) }?;
        equation.post_step(t, &unsafe { VecRef::lent(u) })
    })
}
