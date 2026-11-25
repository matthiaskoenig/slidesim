//! `CeedVector`: libCEED vectors on the host.

use std::ptr::{null, null_mut};
use std::slice;

use crate::{Ceed, Result, chk, sys, to_size};

/// A libCEED vector (`CeedVector`). It always holds valid values (zeros when created), so its
/// array can be read at any time.
#[derive(Debug)]
pub struct CVector {
    raw: sys::CeedVector,
    ceed: Ceed,
    len: usize,
}

impl CVector {
    /// A vector of `len` zeros.
    pub fn new(ceed: &Ceed, len: usize) -> Result<CVector> {
        let mut raw: sys::CeedVector = null_mut();
        chk!(ceed, unsafe {
            sys::CeedVectorCreate(ceed.as_raw(), to_size(len, "vector length")?, &mut raw)
        });
        let mut v = CVector {
            raw,
            ceed: ceed.clone(),
            len,
        };
        v.set_value(0.0)?;
        Ok(v)
    }

    /// A vector holding a copy of `values`.
    pub fn from_slice(ceed: &Ceed, values: &[f64]) -> Result<CVector> {
        let v = CVector::new(ceed, values.len())?;
        // SAFETY: CEED_COPY_VALUES copies `values.len()` values (the vector length) and keeps no
        // pointer; the array is only read.
        chk!(ceed, unsafe {
            sys::CeedVectorSetArray(
                v.raw,
                sys::CEED_MEM_HOST,
                sys::CEED_COPY_VALUES,
                values.as_ptr().cast_mut(),
            )
        });
        Ok(v)
    }

    /// The length.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the length is zero.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Sets every entry to `value`.
    pub fn set_value(&mut self, value: f64) -> Result<()> {
        chk!(&self.ceed, unsafe {
            sys::CeedVectorSetValue(self.raw, value)
        });
        Ok(())
    }

    /// Calls `f` with the values (`CeedVectorGetArrayRead`).
    pub fn with_array<R>(&self, f: impl FnOnce(&[f64]) -> R) -> Result<R> {
        let mut ptr: *const f64 = null();
        chk!(&self.ceed, unsafe {
            sys::CeedVectorGetArrayRead(self.raw, sys::CEED_MEM_HOST, &mut ptr)
        });
        let mut access = Access {
            vec: self,
            ptr: ptr.cast_mut(),
            write: false,
            restored: false,
        };
        // SAFETY: libCEED lends `len` values until the restore; its access lock refuses writers
        // (every libCEED write to this vector) meanwhile.
        let values = unsafe { slice_or_empty(ptr, self.len) };
        let r = f(values);
        access.restore()?;
        Ok(r)
    }

    /// Calls `f` with the values for reading and writing (`CeedVectorGetArray`).
    pub fn with_array_mut<R>(&mut self, f: impl FnOnce(&mut [f64]) -> R) -> Result<R> {
        let mut ptr: *mut f64 = null_mut();
        chk!(&self.ceed, unsafe {
            sys::CeedVectorGetArray(self.raw, sys::CEED_MEM_HOST, &mut ptr)
        });
        let len = self.len;
        let mut access = Access {
            vec: self,
            ptr,
            write: true,
            restored: false,
        };
        // SAFETY: libCEED lends `len` values exclusively until the restore: its access lock
        // refuses every other access to this vector meanwhile.
        let values = if ptr.is_null() || len == 0 {
            &mut []
        } else {
            unsafe { slice::from_raw_parts_mut(ptr, len) }
        };
        let r = f(values);
        access.restore()?;
        Ok(r)
    }

    /// A copy of the values.
    pub fn to_vec(&self) -> Result<Vec<f64>> {
        self.with_array(<[f64]>::to_vec)
    }

    /// The raw handle, for the libCEED API not wrapped here.
    pub fn as_raw(&self) -> sys::CeedVector {
        self.raw
    }

    pub(crate) fn ceed(&self) -> &Ceed {
        &self.ceed
    }
}

/// # Safety
/// Unless null or `len == 0`, `ptr` points to `len` values valid for `'a`.
unsafe fn slice_or_empty<'a>(ptr: *const f64, len: usize) -> &'a [f64] {
    if ptr.is_null() || len == 0 {
        &[]
    } else {
        unsafe { slice::from_raw_parts(ptr, len) }
    }
}

/// An array lent by libCEED, restored on [`Access::restore`] or, if `f` panics, on drop.
struct Access<'a> {
    vec: &'a CVector,
    ptr: *mut f64,
    write: bool,
    restored: bool,
}

impl Access<'_> {
    fn restore(&mut self) -> Result<()> {
        self.restored = true;
        let (raw, ceed) = (self.vec.raw, &self.vec.ceed);
        if self.write {
            chk!(ceed, unsafe {
                sys::CeedVectorRestoreArray(raw, &mut self.ptr)
            });
        } else {
            let mut ptr = self.ptr.cast_const();
            chk!(ceed, unsafe {
                sys::CeedVectorRestoreArrayRead(raw, &mut ptr)
            });
        }
        Ok(())
    }
}

impl Drop for Access<'_> {
    fn drop(&mut self) {
        if !self.restored {
            // Unwinding from `f`: the panic is reported, not this.
            let _ = self.restore();
        }
    }
}

impl Drop for CVector {
    fn drop(&mut self) {
        // Nothing to report an error to; a stale message is reset by the next call.
        unsafe { sys::CeedVectorDestroy(&mut self.raw) };
    }
}
