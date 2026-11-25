//! `KSP`: Krylov solvers.

use std::ptr::null_mut;

use crate::{Comm, Result, alive, chk, sys};

/// An owned PETSc linear solver (`KSP`).
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

    pub fn as_raw(&self) -> sys::KSP {
        self.raw
    }

    /// `KSPSetFromOptions`.
    pub fn set_from_options(&mut self) -> Result<()> {
        chk!(unsafe { sys::KSPSetFromOptions(self.raw) });
        Ok(())
    }
}

impl Drop for Ksp {
    fn drop(&mut self) {
        if alive() {
            unsafe { sys::KSPDestroy(&mut self.raw) };
        }
    }
}
