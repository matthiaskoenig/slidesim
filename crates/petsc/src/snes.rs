//! `SNES`: nonlinear solvers.

use std::ffi::c_char;
use std::ptr::{null, null_mut};

use crate::{ConvergedReason, Ksp, Result, alive, chk, sys};

/// An owned reference to a PETSc nonlinear solver (`SNES`), shared with the object which holds
/// it (for example [`crate::Ts::snes`]); PETSc destroys it with its last reference.
#[derive(Debug)]
pub struct Snes {
    raw: sys::SNES,
}

impl Snes {
    /// Takes a new reference to `raw`.
    ///
    /// # Safety
    /// `raw` is a valid `SNES`.
    pub(crate) unsafe fn shared(raw: sys::SNES) -> Result<Snes> {
        chk!(unsafe { sys::PetscObjectReference(raw as sys::PetscObject) });
        Ok(Snes { raw })
    }

    pub fn as_raw(&self) -> sys::SNES {
        self.raw
    }

    /// The linear solver (`SNESGetKSP`), shared.
    pub fn ksp(&self) -> Result<Ksp> {
        let mut raw = null_mut();
        chk!(unsafe { sys::SNESGetKSP(self.raw, &mut raw) });
        // SAFETY: a valid KSP, borrowed from the SNES.
        unsafe { Ksp::shared(raw) }
    }

    /// Why the last solve stopped (`SNESGetConvergedReason`).
    pub fn converged_reason(&self) -> Result<ConvergedReason> {
        let mut code = 0;
        chk!(unsafe { sys::SNESGetConvergedReason(self.raw, &mut code) });
        let mut name: *const c_char = null();
        // SAFETY: a static string of PETSc.
        chk!(unsafe { sys::SNESGetConvergedReasonString(self.raw, &mut name) });
        Ok(ConvergedReason::new(code, name))
    }
}

impl Drop for Snes {
    fn drop(&mut self) {
        if alive() {
            unsafe { sys::SNESDestroy(&mut self.raw) };
        }
    }
}
