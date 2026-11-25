//! `DM`: meshes (DMPlex).

use crate::{Result, alive, chk, sys};

/// An owned reference to a PETSc `DM` (a DMPlex). Cloning shares the `DM` (PETSc reference
/// counting); the `DM` is destroyed with its last reference.
#[derive(Debug)]
pub struct Dm {
    raw: sys::DM,
}

impl Dm {
    /// Takes ownership of one reference to `raw`.
    ///
    /// # Safety
    /// `raw` is a valid `DM`, and the caller gives up its reference.
    pub unsafe fn from_raw(raw: sys::DM) -> Dm {
        Dm { raw }
    }

    pub fn as_raw(&self) -> sys::DM {
        self.raw
    }

    /// `DMSetFromOptions`.
    pub fn set_from_options(&mut self) -> Result<()> {
        chk!(unsafe { sys::DMSetFromOptions(self.raw) });
        Ok(())
    }
}

impl Clone for Dm {
    fn clone(&self) -> Dm {
        // Cannot fail on a valid object. After PetscFinalize nothing is destroyed any more, so
        // the reference count does not matter.
        if alive() {
            unsafe { sys::PetscObjectReference(self.raw as sys::PetscObject) };
        }
        Dm { raw: self.raw }
    }
}

impl Drop for Dm {
    fn drop(&mut self) {
        if alive() {
            unsafe { sys::DMDestroy(&mut self.raw) };
        }
    }
}
