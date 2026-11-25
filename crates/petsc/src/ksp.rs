//! `KSP`: Krylov solvers, and the converged reasons of the solvers.

use std::ffi::{CStr, c_char};
use std::fmt;
use std::ptr::{null, null_mut};

use crate::{Comm, Result, alive, c_string, chk, sys};

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
#[derive(Debug)]
pub struct Ksp {
    raw: sys::KSP,
}

impl Ksp {
    /// `KSPCreate`.
    pub fn create(comm: &Comm) -> Result<Ksp> {
        let mut raw = null_mut();
        chk!(unsafe { sys::KSPCreate(comm.as_raw(), &mut raw) });
        Ok(Ksp { raw })
    }

    /// Takes a new reference to `raw`.
    ///
    /// # Safety
    /// `raw` is a valid `KSP`.
    pub(crate) unsafe fn shared(raw: sys::KSP) -> Result<Ksp> {
        chk!(unsafe { sys::PetscObjectReference(raw as sys::PetscObject) });
        Ok(Ksp { raw })
    }

    pub fn as_raw(&self) -> sys::KSP {
        self.raw
    }

    /// `KSPSetFromOptions`.
    pub fn set_from_options(&mut self) -> Result<()> {
        chk!(unsafe { sys::KSPSetFromOptions(self.raw) });
        Ok(())
    }

    /// Sets the Krylov method, for example `cg` (`KSPSetType`).
    pub fn set_type(&mut self, name: &str) -> Result<()> {
        let name = c_string(name, "KSP type")?;
        chk!(unsafe { sys::KSPSetType(self.raw, name.as_ptr()) });
        Ok(())
    }

    /// Sets the type of the preconditioner, for example `jacobi` (`PCSetType` of `KSPGetPC`).
    pub fn set_pc_type(&mut self, name: &str) -> Result<()> {
        let name = c_string(name, "PC type")?;
        let mut pc = null_mut();
        // SAFETY: the PC is borrowed from the KSP.
        chk!(unsafe { sys::KSPGetPC(self.raw, &mut pc) });
        chk!(unsafe { sys::PCSetType(pc, name.as_ptr()) });
        Ok(())
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
        if alive() {
            unsafe { sys::KSPDestroy(&mut self.raw) };
        }
    }
}
