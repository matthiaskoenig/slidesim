//! `TS`: time steppers.

use std::ptr::null_mut;

use crate::{Comm, Result, alive, chk, sys};

/// An owned PETSc time stepper (`TS`).
#[derive(Debug)]
pub struct Ts {
    raw: sys::TS,
}

impl Ts {
    /// `TSCreate`.
    pub fn create(comm: &Comm) -> Result<Ts> {
        let mut raw = null_mut();
        chk!(unsafe { sys::TSCreate(comm.as_raw(), &mut raw) });
        Ok(Ts { raw })
    }

    pub fn as_raw(&self) -> sys::TS {
        self.raw
    }

    /// `TSSetFromOptions`.
    pub fn set_from_options(&mut self) -> Result<()> {
        chk!(unsafe { sys::TSSetFromOptions(self.raw) });
        Ok(())
    }
}

impl Drop for Ts {
    fn drop(&mut self) {
        if alive() {
            unsafe { sys::TSDestroy(&mut self.raw) };
        }
    }
}
