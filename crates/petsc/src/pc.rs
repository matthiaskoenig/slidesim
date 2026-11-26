//! `PC`: preconditioners, and the levels of multigrid (`PCMG`).

use std::ffi::c_char;
use std::ptr::{null, null_mut};

use crate::{Ksp, Matrix, Result, alive, c_string, chk, string_from, sys, to_int};

/// An owned reference to a PETSc preconditioner (`PC`), shared with the solver which holds it
/// ([`Ksp::pc`]); PETSc destroys it with its last reference.
#[derive(Debug)]
pub struct Pc {
    raw: sys::PC,
}

impl Pc {
    /// Takes a new reference to `raw`.
    ///
    /// # Safety
    /// `raw` is a valid `PC`.
    pub(crate) unsafe fn shared(raw: sys::PC) -> Result<Pc> {
        // SAFETY: `raw` is valid (the caller's promise).
        chk!(unsafe { sys::PetscObjectReference(raw as sys::PetscObject) });
        Ok(Pc { raw })
    }

    pub fn as_raw(&self) -> sys::PC {
        self.raw
    }

    /// Sets the type, for example `jacobi` or `mg` (`PCSetType`).
    pub fn set_type(&mut self, name: &str) -> Result<()> {
        let name = c_string(name, "PC type")?;
        // SAFETY: a valid PC and a NUL terminated name, which PETSc copies.
        chk!(unsafe { sys::PCSetType(self.raw, name.as_ptr()) });
        Ok(())
    }

    /// The type (`PCGetType`); empty while none is set.
    pub fn type_name(&self) -> Result<String> {
        let mut name: sys::PCType = null();
        // SAFETY: `name` receives a string of the PC, or stays null.
        chk!(unsafe { sys::PCGetType(self.raw, &mut name) });
        // SAFETY: null or a NUL terminated string of the PC, valid for this call.
        Ok(unsafe { string_from(name) }.unwrap_or_default())
    }

    /// The prefix of its options (`PCGetOptionsPrefix`), empty if none.
    pub fn options_prefix(&self) -> Result<String> {
        let mut prefix: *const c_char = null();
        // SAFETY: `prefix` receives a string of the PC, or stays null.
        chk!(unsafe { sys::PCGetOptionsPrefix(self.raw, &mut prefix) });
        // SAFETY: null or a NUL terminated string of the PC, valid for this call.
        Ok(unsafe { string_from(prefix) }.unwrap_or_default())
    }

    /// Makes a multigrid preconditioner (`PCMG`, which [`Pc::set_type`] `mg` selects) one of
    /// `levels` levels on the communicator of the PC (`PCMGSetLevels`). Level 0 is the coarsest,
    /// `levels - 1` the finest, the level of the operator of the solver. Replaces the levels
    /// set before, with their smoothers and interpolations.
    pub fn mg_set_levels(&mut self, levels: usize) -> Result<()> {
        let levels = to_int(levels, "multigrid levels")?;
        // SAFETY: a valid PC; no communicators, so every level is on the PC's.
        chk!(unsafe { sys::PCMGSetLevels(self.raw, levels, null_mut()) });
        Ok(())
    }

    /// The number of levels of a `PCMG` (`PCMGGetLevels`): zero before [`Pc::mg_set_levels`].
    pub fn mg_levels(&self) -> Result<usize> {
        let mut levels = 0;
        // SAFETY: a valid PC; PETSc writes one integer.
        chk!(unsafe { sys::PCMGGetLevels(self.raw, &mut levels) });
        // -1 before the levels are set.
        Ok(usize::try_from(levels).unwrap_or(0))
    }

    /// Sets the interpolation from level `level - 1` to level `level` (`PCMGSetInterpolation`;
    /// `level` is at least 1). The restriction is its transpose unless set otherwise. The PC
    /// keeps a reference to `interpolation`.
    pub fn mg_set_interpolation(&mut self, level: usize, interpolation: &Matrix) -> Result<()> {
        let level = to_int(level, "multigrid level")?;
        // SAFETY: a valid PC and matrix; PETSc takes its own reference to the matrix.
        chk!(unsafe { sys::PCMGSetInterpolation(self.raw, level, interpolation.as_raw()) });
        Ok(())
    }

    /// The solver of level `level` of a `PCMG` (`PCMGGetSmoother`), shared: the smoother of
    /// the level, or the coarse solver for level 0.
    pub fn mg_smoother(&self, level: usize) -> Result<Ksp> {
        let level = to_int(level, "multigrid level")?;
        let mut raw = null_mut();
        // SAFETY: `raw` receives the KSP of the level, borrowed from the PC.
        chk!(unsafe { sys::PCMGGetSmoother(self.raw, level, &mut raw) });
        // SAFETY: a valid KSP.
        unsafe { Ksp::shared(raw) }
    }
}

impl Drop for Pc {
    fn drop(&mut self) {
        if alive() {
            // SAFETY: our reference.
            crate::error::destroyed(unsafe { sys::PCDestroy(&mut self.raw) }, "PCDestroy");
        }
    }
}
