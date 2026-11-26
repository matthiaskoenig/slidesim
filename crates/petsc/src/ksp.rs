//! `KSP`: Krylov solvers, and the converged reasons of the solvers.

use std::ffi::{CStr, c_char};
use std::fmt;
use std::ptr::{null, null_mut};

use crate::error::failed_solver;
use crate::{
    AsVec, AsVecMut, Comm, Dm, Matrix, Pc, Result, alive, c_string, chk, string_from, sys, to_int,
};

/// Why a solver (`KSP`, `SNES` or `TS`) stopped: PETSc's converged reason, positive if it
/// converged, negative if it diverged, and zero while it is iterating.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConvergedReason {
    /// The value of the PETSc enum (`KSPConvergedReason`, `SNESConvergedReason` or
    /// `TSConvergedReason`).
    pub code: i32,
    /// PETSc's name of the reason, for example `DIVERGED_ITS`.
    pub name: String,
}

impl ConvergedReason {
    pub fn diverged(&self) -> bool {
        self.code < 0
    }

    /// The reason `code` with the name `name` from PETSc (null if PETSc has none).
    pub(crate) fn new(code: i32, name: *const c_char) -> ConvergedReason {
        let name = if name.is_null() {
            format!("unknown reason {code}")
        } else {
            unsafe { CStr::from_ptr(name) }
                .to_string_lossy()
                .into_owned()
        };
        ConvergedReason { code, name }
    }
}

impl fmt::Display for ConvergedReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.name)
    }
}

/// An owned reference to a PETSc linear solver (`KSP`): created here, or shared with the
/// object which holds it (for example [`crate::Snes::ksp`]); PETSc destroys it with its last
/// reference.
///
/// A solver whose [`Ksp::solve`] failed is never destroyed, see there.
#[derive(Debug)]
pub struct Ksp {
    raw: sys::KSP,
    /// Whether a [`Ksp::solve`] of this reference failed.
    failed: bool,
}

impl Ksp {
    /// `KSPCreate`.
    pub fn create(comm: &Comm) -> Result<Ksp> {
        let mut raw = null_mut();
        chk!(unsafe { sys::KSPCreate(comm.as_raw(), &mut raw) });
        Ok(Ksp { raw, failed: false })
    }

    /// Takes a new reference to `raw`.
    ///
    /// # Safety
    /// `raw` is a valid `KSP`.
    pub(crate) unsafe fn shared(raw: sys::KSP) -> Result<Ksp> {
        chk!(unsafe { sys::PetscObjectReference(raw as sys::PetscObject) });
        Ok(Ksp { raw, failed: false })
    }

    pub fn as_raw(&self) -> sys::KSP {
        self.raw
    }

    /// An error if a solve of this reference failed (see [`Ksp::solve`]): for every method
    /// which changes or runs the solver.
    fn usable(&self) -> Result<()> {
        if self.failed {
            return Err(failed_solver("KSP"));
        }
        Ok(())
    }

    /// `KSPSetFromOptions`.
    pub fn set_from_options(&mut self) -> Result<()> {
        self.usable()?;
        chk!(unsafe { sys::KSPSetFromOptions(self.raw) });
        Ok(())
    }

    /// Sets the Krylov method, for example `cg` (`KSPSetType`).
    pub fn set_type(&mut self, name: &str) -> Result<()> {
        self.usable()?;
        let name = c_string(name, "KSP type")?;
        chk!(unsafe { sys::KSPSetType(self.raw, name.as_ptr()) });
        Ok(())
    }

    /// Sets the type of the preconditioner, for example `jacobi` (`PCSetType` of `KSPGetPC`).
    pub fn set_pc_type(&mut self, name: &str) -> Result<()> {
        self.usable()?;
        let name = c_string(name, "PC type")?;
        let mut pc = null_mut();
        // SAFETY: the PC is borrowed from the KSP.
        chk!(unsafe { sys::KSPGetPC(self.raw, &mut pc) });
        chk!(unsafe { sys::PCSetType(pc, name.as_ptr()) });
        Ok(())
    }

    /// The Krylov method (`KSPGetType`); empty while none is set.
    pub fn type_name(&self) -> Result<String> {
        let mut name: sys::KSPType = null();
        // SAFETY: `name` receives a string of the KSP, or stays null.
        chk!(unsafe { sys::KSPGetType(self.raw, &mut name) });
        // SAFETY: null or a NUL terminated string of the KSP, valid for this call.
        Ok(unsafe { string_from(name) }.unwrap_or_default())
    }

    /// The preconditioner (`KSPGetPC`), shared.
    pub fn pc(&self) -> Result<Pc> {
        let mut raw = null_mut();
        // SAFETY: `raw` receives the PC, borrowed from the KSP.
        chk!(unsafe { sys::KSPGetPC(self.raw, &mut raw) });
        // SAFETY: a valid PC.
        unsafe { Pc::shared(raw) }
    }

    /// Sets the prefix of the options of the solver and its preconditioner
    /// (`KSPSetOptionsPrefix`).
    pub fn set_options_prefix(&mut self, prefix: &str) -> Result<()> {
        self.usable()?;
        let prefix = c_string(prefix, "options prefix")?;
        // SAFETY: a valid KSP and a NUL terminated prefix, which PETSc copies.
        chk!(unsafe { sys::KSPSetOptionsPrefix(self.raw, prefix.as_ptr()) });
        Ok(())
    }

    /// Sets the operator `a` and the matrix `p` its preconditioner is built from (often `a`
    /// again; `KSPSetOperators`). The solver keeps references to both.
    pub fn set_operators(&mut self, a: &Matrix, p: &Matrix) -> Result<()> {
        self.usable()?;
        // SAFETY: valid matrices; PETSc takes its own references.
        chk!(unsafe { sys::KSPSetOperators(self.raw, a.as_raw(), p.as_raw()) });
        Ok(())
    }

    /// The sizes of the operator of the solver, if one is set (`KSPGetOperatorsSet`): its rows
    /// on this process and in all (`MatGetLocalSize`, `MatGetSize`); `None` before
    /// [`Ksp::set_operators`] (a `SNES` sets them when it solves).
    pub fn operator_sizes(&self) -> Result<Option<(usize, usize)>> {
        let (mut set, mut pmat_set) = (false, false);
        // SAFETY: a valid KSP; PETSc writes two flags.
        chk!(unsafe { sys::KSPGetOperatorsSet(self.raw, &mut set, &mut pmat_set) });
        if !set {
            return Ok(None);
        }
        let mut a = null_mut();
        // SAFETY: the operator is borrowed from the KSP; it is set, so none is created.
        chk!(unsafe { sys::KSPGetOperators(self.raw, &mut a, null_mut()) });
        let (mut local, mut global, mut columns) = (0, 0, 0);
        // SAFETY: a valid matrix; PETSc writes two integers each.
        chk!(unsafe { sys::MatGetLocalSize(a, &mut local, &mut columns) });
        chk!(unsafe { sys::MatGetSize(a, &mut global, &mut columns) });
        Ok(Some((local as usize, global as usize)))
    }

    /// Gives the solver (and its preconditioner) the DM `dm` (`KSPSetDM`). If not `active`, the
    /// DM only describes the problem (for example a level of multigrid), and does not compute
    /// the operators, right-hand side or initial guess (`KSPSetDMActive` false). The solver
    /// keeps a reference to `dm`.
    pub fn set_dm(&mut self, dm: &Dm, active: bool) -> Result<()> {
        self.usable()?;
        // SAFETY: a valid KSP and DM; PETSc takes its own reference to the DM.
        chk!(unsafe { sys::KSPSetDM(self.raw, dm.as_raw()) });
        // SAFETY: a valid KSP.
        chk!(unsafe { sys::KSPSetDMActive(self.raw, sys::KSP_DMACTIVE_ALL, active) });
        Ok(())
    }

    /// Sets the relative and absolute tolerances of the residual norm and the maximum number of
    /// iterations (`KSPSetTolerances`); `None` keeps the current value.
    pub fn set_tolerances(
        &mut self,
        rtol: Option<f64>,
        atol: Option<f64>,
        max_it: Option<usize>,
    ) -> Result<()> {
        self.usable()?;
        let current = sys::PETSC_CURRENT as sys::PetscReal;
        let max_it = match max_it {
            Some(n) => to_int(n, "maximum iterations")?,
            None => sys::PETSC_CURRENT as sys::PetscInt,
        };
        // SAFETY: a valid KSP; PETSC_CURRENT keeps a value.
        chk!(unsafe {
            sys::KSPSetTolerances(
                self.raw,
                rtol.unwrap_or(current),
                atol.unwrap_or(current),
                current,
                max_it,
            )
        });
        Ok(())
    }

    /// Solves `A x = b` for the operator of [`Ksp::set_operators`], from the initial guess
    /// zero (`KSPSolve`). Collective.
    ///
    /// After an error the solver is failed: PETSc returns an error without restoring the state
    /// of the objects the solve passed through, which may then be inconsistent. For example a
    /// multigrid preconditioner (`PCMG`) holds the vector it is applied to, without a reference,
    /// for the length of the application; when a solver inside it fails, and the solver is to
    /// error if not converged (`-ksp_error_if_not_converged`), the vector stays, and destroying
    /// or resetting the preconditioner releases a reference it never took: the vector, a work
    /// vector of the Krylov method, is freed while the method still holds it. So a failed solver
    /// is never destroyed: dropping this reference leaks it with everything it holds (whose
    /// references it keeps). Every later call of a method taking `&mut self` is an error
    /// (`PETSC_ERR_ARG_WRONGSTATE`); [`Ksp::converged_reason`] and [`Ksp::iterations`] still
    /// tell why the solve stopped. Other references to the solver or its parts (for example
    /// from [`Ksp::pc`] or [`crate::Pc::mg_smoother`]) must not run or change it either.
    pub fn solve(&mut self, b: &impl AsVec, x: &mut impl AsVecMut) -> Result<()> {
        self.usable()?;
        let solved = (|| {
            // SAFETY: a valid KSP and distinct vectors (`&` and `&mut`).
            chk!(unsafe { sys::KSPSolve(self.raw, b.as_raw(), x.as_raw()) });
            Ok(())
        })();
        self.failed = solved.is_err();
        solved
    }

    /// The number of iterations of the last solve (`KSPGetIterationNumber`).
    pub fn iterations(&self) -> Result<usize> {
        let mut n = 0;
        // SAFETY: a valid KSP; PETSc writes one integer.
        chk!(unsafe { sys::KSPGetIterationNumber(self.raw, &mut n) });
        Ok(n as usize)
    }

    /// Makes a Chebyshev solver (type `chebyshev`) estimate the extreme eigenvalues of its
    /// preconditioned operator with a few iterations of another Krylov solver, and target the
    /// interval `[a emin + b emax, c emin + d emax]` for the estimates `emin`, `emax`
    /// (`KSPChebyshevEstEigSet`). The estimator only applies the operator and the
    /// preconditioner, so it works for shell matrices; it is CG when the matrix of the
    /// preconditioner is known to be symmetric positive definite ([`Matrix::set_spd`]), else
    /// GMRES. The estimates are renewed when the operator changes.
    pub fn chebyshev_est_eig_set(&mut self, a: f64, b: f64, c: f64, d: f64) -> Result<()> {
        self.usable()?;
        // SAFETY: a valid KSP; a no-op for other types.
        chk!(unsafe { sys::KSPChebyshevEstEigSet(self.raw, a, b, c, d) });
        Ok(())
    }

    /// The eigenvalue estimator of a Chebyshev solver from [`Ksp::chebyshev_est_eig_set`]
    /// (`KSPChebyshevEstEigGetKSP`), shared; `None` for other solvers, or without estimation.
    pub fn chebyshev_est_eig_ksp(&self) -> Result<Option<Ksp>> {
        let mut raw = null_mut();
        // SAFETY: `raw` receives the estimator, borrowed from the KSP, or stays null.
        chk!(unsafe { sys::KSPChebyshevEstEigGetKSP(self.raw, &mut raw) });
        if raw.is_null() {
            return Ok(None);
        }
        // SAFETY: a valid KSP.
        unsafe { Ksp::shared(raw) }.map(Some)
    }

    /// Why the last solve stopped (`KSPGetConvergedReason`).
    pub fn converged_reason(&self) -> Result<ConvergedReason> {
        let mut code = 0;
        chk!(unsafe { sys::KSPGetConvergedReason(self.raw, &mut code) });
        let mut name: *const c_char = null();
        // SAFETY: a static string of PETSc.
        chk!(unsafe { sys::KSPGetConvergedReasonString(self.raw, &mut name) });
        Ok(ConvergedReason::new(code, name))
    }
}

impl Drop for Ksp {
    fn drop(&mut self) {
        // A failed solver is leaked (see `Ksp::solve`).
        if alive() && !self.failed {
            crate::error::destroyed(unsafe { sys::KSPDestroy(&mut self.raw) }, "KSPDestroy");
        }
    }
}
