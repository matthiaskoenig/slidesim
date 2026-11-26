//! `Vec`: PETSc vectors, owned ([`Vector`]) or lent by PETSc to a callback ([`VecRef`],
//! [`VecMut`]).

use std::ffi::c_char;
use std::marker::PhantomData;
use std::mem::ManuallyDrop;
use std::ptr::null_mut;

use crate::{Comm, Dm, Petsc, Result, Viewer, alive, c_string, chk, string_from, sys, to_int};

mod sealed {
    pub trait Sealed {}
}

/// A PETSc vector of any kind: [`Vector`], [`VecRef`] or [`VecMut`]. Sealed: its handle is
/// always a valid `Vec`.
pub trait AsVec: sealed::Sealed {
    fn as_raw(&self) -> sys::Vec;
}

/// A PETSc vector which may be written: [`Vector`] or [`VecMut`].
pub trait AsVecMut: AsVec {
    /// Sets every entry to zero (`VecZeroEntries`), for code generic over the vector kind.
    fn zero(&mut self) -> Result<()> {
        chk!(unsafe { sys::VecZeroEntries(self.as_raw()) });
        Ok(())
    }
}

/// An owned PETSc vector (`Vec`).
#[derive(Debug)]
pub struct Vector {
    raw: sys::Vec,
}

/// A vector PETSc lends to a callback for reading, valid for the call only: the lifetime ties it
/// to the callback, so it cannot be kept.
#[derive(Debug)]
pub struct VecRef<'a> {
    raw: sys::Vec,
    _call: PhantomData<&'a ()>,
}

/// A vector PETSc lends to a callback for reading and writing, valid for the call only.
///
/// It cannot be kept beyond the call, not even by swapping it into the state of a (`'static`)
/// operator:
///
/// ```compile_fail
/// use std::cell::RefCell;
/// use petsc::VecMut;
///
/// fn keep(y: &mut VecMut<'_>, state: &RefCell<VecMut<'static>>) {
///     std::mem::swap(y, &mut *state.borrow_mut());
/// }
/// ```
#[derive(Debug)]
pub struct VecMut<'a> {
    raw: sys::Vec,
    _call: PhantomData<&'a mut ()>,
}

impl<'a> VecRef<'a> {
    /// # Safety
    /// `raw` is a valid `Vec` for `'a`.
    pub(crate) unsafe fn lent(raw: sys::Vec) -> VecRef<'a> {
        VecRef {
            raw,
            _call: PhantomData,
        }
    }
}

impl<'a> VecMut<'a> {
    /// # Safety
    /// `raw` is a valid `Vec` for `'a`, which nothing else uses meanwhile.
    pub(crate) unsafe fn lent(raw: sys::Vec) -> VecMut<'a> {
        VecMut {
            raw,
            _call: PhantomData,
        }
    }
}

impl Vector {
    /// A sequential vector of length `n` on `PETSC_COMM_SELF`.
    pub fn create_seq(_petsc: &Petsc, n: usize) -> Result<Vector> {
        let n = to_int(n, "vector length")?;
        let mut raw = null_mut();
        chk!(unsafe { sys::VecCreateSeq(Comm::self_().as_raw(), n, &mut raw) });
        Ok(Vector { raw })
    }

    /// A global vector of `dm` (`DMCreateGlobalVector`).
    pub fn create_global(dm: &Dm) -> Result<Vector> {
        let mut raw = null_mut();
        chk!(unsafe { sys::DMCreateGlobalVector(dm.as_raw(), &mut raw) });
        Ok(Vector { raw })
    }

    /// Names the vector (`PetscObjectSetName`), for example for the data a viewer writes.
    pub fn set_name(&mut self, name: &str) -> Result<()> {
        let name = c_string(name, "vector name")?;
        // SAFETY: a valid vector and a NUL terminated name, which PETSc copies.
        chk!(unsafe { sys::PetscObjectSetName(self.raw as sys::PetscObject, name.as_ptr()) });
        Ok(())
    }

    /// A local vector of `dm`, with ghost points (`DMCreateLocalVector`).
    pub fn create_local(dm: &Dm) -> Result<Vector> {
        let mut raw = null_mut();
        chk!(unsafe { sys::DMCreateLocalVector(dm.as_raw(), &mut raw) });
        Ok(Vector { raw })
    }
}

impl Drop for Vector {
    fn drop(&mut self) {
        if alive() {
            crate::error::destroyed(unsafe { sys::VecDestroy(&mut self.raw) }, "VecDestroy");
        }
    }
}

/// The operations which read a vector.
macro_rules! read_ops {
    () => {
        pub fn as_raw(&self) -> sys::Vec {
            self.raw
        }

        /// The global length.
        pub fn size(&self) -> Result<usize> {
            let mut n = 0;
            chk!(unsafe { sys::VecGetSize(self.raw, &mut n) });
            Ok(n as usize)
        }

        /// The length of the part on this process.
        pub fn local_size(&self) -> Result<usize> {
            let mut n = 0;
            chk!(unsafe { sys::VecGetLocalSize(self.raw, &mut n) });
            Ok(n as usize)
        }

        /// The 2-norm.
        pub fn norm2(&self) -> Result<f64> {
            let mut norm = 0.0;
            chk!(unsafe { sys::VecNorm(self.raw, sys::NORM_2, &mut norm) });
            Ok(norm)
        }

        /// The dot product with `y`.
        pub fn dot(&self, y: &impl AsVec) -> Result<f64> {
            let mut dot = 0.0;
            chk!(unsafe { sys::VecDot(self.raw, y.as_raw(), &mut dot) });
            Ok(dot)
        }

        /// The name of the vector (`PetscObjectGetName`); PETSc names an unnamed one `Vec_<n>`.
        pub fn name(&self) -> Result<String> {
            let mut name: *const c_char = std::ptr::null();
            // SAFETY: a valid vector; `name` receives a string of the vector.
            chk!(unsafe { sys::PetscObjectGetName(self.raw as sys::PetscObject, &mut name) });
            // SAFETY: null or a NUL terminated string of the vector, valid for this call.
            Ok(unsafe { string_from(name) }.unwrap_or_default())
        }

        /// Views the vector with `viewer` (`VecView`): for a [`Viewer::vtk`], queues it to be
        /// written by [`Viewer::flush`]. Collective.
        pub fn view(&self, viewer: &mut Viewer) -> Result<()> {
            // SAFETY: a valid vector and viewer.
            chk!(unsafe { sys::VecView(self.raw, viewer.as_raw()) });
            Ok(())
        }

        /// Calls `f` with the entries on this process.
        pub fn with_array<R>(&self, f: impl FnOnce(&[f64]) -> R) -> Result<R> {
            let n = self.local_size()?;
            let mut ptr = std::ptr::null();
            chk!(unsafe { sys::VecGetArrayRead(self.raw, &mut ptr) });
            let array = Array {
                vec: self.raw,
                ptr: ptr.cast_mut(),
                write: false,
            };
            // SAFETY: the array holds the `n` local entries until it is restored, after `f`.
            let r = f(unsafe { array.slice(n) });
            array.restore()?;
            Ok(r)
        }
    };
}

/// The operations which write a vector.
macro_rules! write_ops {
    () => {
        /// `self += alpha * x`.
        pub fn axpy(&mut self, alpha: f64, x: &impl AsVec) -> Result<()> {
            chk!(unsafe { sys::VecAXPY(self.raw, alpha, x.as_raw()) });
            Ok(())
        }

        /// Sets every entry to `alpha`.
        pub fn set(&mut self, alpha: f64) -> Result<()> {
            chk!(unsafe { sys::VecSet(self.raw, alpha) });
            Ok(())
        }

        /// Calls `f` with the entries on this process, for reading and writing.
        pub fn with_array_mut<R>(&mut self, f: impl FnOnce(&mut [f64]) -> R) -> Result<R> {
            let n = self.local_size()?;
            let mut ptr = null_mut();
            chk!(unsafe { sys::VecGetArray(self.raw, &mut ptr) });
            let array = Array {
                vec: self.raw,
                ptr,
                write: true,
            };
            // SAFETY: the array holds the `n` local entries for writing until it is restored,
            // after `f`; `&mut self` excludes other uses of the vector meanwhile.
            let r = f(unsafe { array.slice_mut(n) });
            array.restore()?;
            Ok(r)
        }
    };
}

impl Vector {
    read_ops!();
    write_ops!();
}

impl VecRef<'_> {
    read_ops!();
}

impl VecMut<'_> {
    read_ops!();
    write_ops!();
}

macro_rules! as_vec {
    ($($t:ty),*) => {$(
        impl sealed::Sealed for $t {}
        impl AsVec for $t {
            fn as_raw(&self) -> sys::Vec {
                self.raw
            }
        }
    )*};
}

as_vec!(Vector, VecRef<'_>, VecMut<'_>);
impl AsVecMut for Vector {}
impl AsVecMut for VecMut<'_> {}

/// An array taken from a vector: restored by `restore`, or on drop if `f` panics.
struct Array {
    vec: sys::Vec,
    ptr: *mut f64,
    write: bool,
}

impl Array {
    /// # Safety
    /// `n` is the local size of the vector, and the slice is not used after `restore`.
    unsafe fn slice<'a>(&self, n: usize) -> &'a [f64] {
        if n == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(self.ptr, n) }
        }
    }

    /// # Safety
    /// As [`Array::slice`], and the array was taken for writing.
    unsafe fn slice_mut<'a>(&self, n: usize) -> &'a mut [f64] {
        if n == 0 {
            &mut []
        } else {
            unsafe { std::slice::from_raw_parts_mut(self.ptr, n) }
        }
    }

    fn restore_raw(&mut self) -> sys::PetscErrorCode {
        // SAFETY: `ptr` was taken from `vec` with the matching Get call.
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
