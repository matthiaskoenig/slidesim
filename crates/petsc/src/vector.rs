//! `Vec`: PETSc vectors.

use std::mem::ManuallyDrop;
use std::ptr::null_mut;

use crate::{Comm, Dm, Petsc, Result, alive, chk, sys, to_int};

/// A PETSc vector (`Vec`): owned, or lent by PETSc to a callback.
#[derive(Debug)]
pub struct Vector {
    raw: sys::Vec,
    /// Whether dropping it destroys `raw` (false for a vector PETSc lends to a callback).
    owned: bool,
}

impl Vector {
    /// A sequential vector of length `n` on `PETSC_COMM_SELF`.
    pub fn create_seq(_petsc: &Petsc, n: usize) -> Result<Vector> {
        let n = to_int(n, "vector length")?;
        let mut raw = null_mut();
        chk!(sys::VecCreateSeq(Comm::self_().as_raw(), n, &mut raw));
        Ok(Vector::owned(raw))
    }

    /// A global vector of `dm` (`DMCreateGlobalVector`).
    pub fn create_global(dm: &Dm) -> Result<Vector> {
        let mut raw = null_mut();
        chk!(sys::DMCreateGlobalVector(dm.as_raw(), &mut raw));
        Ok(Vector::owned(raw))
    }

    /// A local vector of `dm`, with ghost points (`DMCreateLocalVector`).
    pub fn create_local(dm: &Dm) -> Result<Vector> {
        let mut raw = null_mut();
        chk!(sys::DMCreateLocalVector(dm.as_raw(), &mut raw));
        Ok(Vector::owned(raw))
    }

    /// Takes ownership of `raw`.
    fn owned(raw: sys::Vec) -> Vector {
        Vector { raw, owned: true }
    }

    /// A vector PETSc owns, lent to Rust for the duration of a callback: never destroyed, even
    /// if the callback swaps it out.
    pub(crate) fn borrowed(raw: sys::Vec) -> Vector {
        Vector { raw, owned: false }
    }

    pub fn as_raw(&self) -> sys::Vec {
        self.raw
    }

    /// The global length.
    pub fn size(&self) -> Result<usize> {
        let mut n = 0;
        chk!(sys::VecGetSize(self.raw, &mut n));
        Ok(n as usize)
    }

    /// The length of the part on this process.
    pub fn local_size(&self) -> Result<usize> {
        let mut n = 0;
        chk!(sys::VecGetLocalSize(self.raw, &mut n));
        Ok(n as usize)
    }

    /// The 2-norm.
    pub fn norm2(&self) -> Result<f64> {
        let mut norm = 0.0;
        chk!(sys::VecNorm(self.raw, sys::NORM_2, &mut norm));
        Ok(norm)
    }

    /// The dot product with `y`.
    pub fn dot(&self, y: &Vector) -> Result<f64> {
        let mut dot = 0.0;
        chk!(sys::VecDot(self.raw, y.raw, &mut dot));
        Ok(dot)
    }

    /// `self += alpha * x`.
    pub fn axpy(&mut self, alpha: f64, x: &Vector) -> Result<()> {
        chk!(sys::VecAXPY(self.raw, alpha, x.raw));
        Ok(())
    }

    /// Sets every entry to `alpha`.
    pub fn set(&mut self, alpha: f64) -> Result<()> {
        chk!(sys::VecSet(self.raw, alpha));
        Ok(())
    }

    /// Calls `f` with the entries on this process.
    pub fn with_array<R>(&self, f: impl FnOnce(&[f64]) -> R) -> Result<R> {
        let n = self.local_size()?;
        let mut ptr = std::ptr::null();
        chk!(sys::VecGetArrayRead(self.raw, &mut ptr));
        let array = Array {
            vec: self.raw,
            ptr: ptr.cast_mut(),
            write: false,
        };
        let r = f(unsafe { array.slice(n) });
        array.restore()?;
        Ok(r)
    }

    /// Calls `f` with the entries on this process, for reading and writing.
    pub fn with_array_mut<R>(&mut self, f: impl FnOnce(&mut [f64]) -> R) -> Result<R> {
        let n = self.local_size()?;
        let mut ptr = null_mut();
        chk!(sys::VecGetArray(self.raw, &mut ptr));
        let array = Array {
            vec: self.raw,
            ptr,
            write: true,
        };
        let r = f(unsafe { array.slice_mut(n) });
        array.restore()?;
        Ok(r)
    }
}

/// An array taken from a vector: restored by `restore`, or on drop if `f` panics.
struct Array {
    vec: sys::Vec,
    ptr: *mut f64,
    write: bool,
}

impl Array {
    /// # Safety
    /// `n` is the local size of the vector.
    unsafe fn slice<'a>(&self, n: usize) -> &'a [f64] {
        if n == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(self.ptr, n) }
        }
    }

    /// # Safety
    /// `n` is the local size of the vector, and the array was taken for writing.
    unsafe fn slice_mut<'a>(&self, n: usize) -> &'a mut [f64] {
        if n == 0 {
            &mut []
        } else {
            unsafe { std::slice::from_raw_parts_mut(self.ptr, n) }
        }
    }

    fn restore_raw(&mut self) -> sys::PetscErrorCode {
        unsafe {
            if self.write {
                sys::VecRestoreArray(self.vec, &mut self.ptr)
            } else {
                let mut ptr = self.ptr.cast_const();
                sys::VecRestoreArrayRead(self.vec, &mut ptr)
            }
        }
    }

    fn restore(self) -> Result<()> {
        let mut array = ManuallyDrop::new(self);
        chk!(array.restore_raw());
        Ok(())
    }
}

impl Drop for Array {
    fn drop(&mut self) {
        // Only reached when unwinding: the panic is the error to report.
        self.restore_raw();
    }
}

impl Drop for Vector {
    fn drop(&mut self) {
        if self.owned && alive() {
            unsafe { sys::VecDestroy(&mut self.raw) };
        }
    }
}
