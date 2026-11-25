//! Thin safe wrappers around libCEED.
//!
//! A [`Ceed`] is a libCEED context on one backend. Its objects ([`CVector`],
//! [`ElemRestriction`], [`Basis`], [`QFunction`], [`Operator`]) are reference counted by
//! libCEED: an [`Operator`] keeps the QFunction, restrictions, bases and passive vectors of its
//! fields alive, so their wrappers may be dropped before it. Every wrapper destroys its own
//! reference once, on drop.
//!
//! Errors never abort or print: every failing libCEED call returns an [`Error`] carrying the
//! libCEED message (see [`chk!`]). libCEED aborts on errors raised before a `Ceed` exists,
//! notably in `CeedInit` for a resource it cannot resolve, so [`Ceed::new`] only passes it the
//! exact resource of an available backend.
//!
//! The objects of an operator call must belong to one [`Ceed`] (else an error): libCEED stores
//! an error message in the `Ceed` of the object raising it.
//!
//! libCEED objects are not thread safe: no type here is `Send` or `Sync`.
//!
//! ```compile_fail,E0277
//! fn send<T: Send>() {}
//! send::<ceed::Ceed>();
//! ```
//!
//! ```compile_fail,E0277
//! fn sync<T: Sync>() {}
//! sync::<ceed::CVector>();
//! ```
//!
//! The raw bindings are re-exported as [`sys`], and every wrapper gives its raw handle with
//! `as_raw()`, for the libCEED API not wrapped here.

mod basis;
mod error;
mod operator;
mod qfunction;
mod restriction;
mod vector;

use std::ffi::{CStr, CString, c_char, c_void};
use std::fmt;
use std::ptr::{null, null_mut};
use std::rc::Rc;
use std::sync::{Mutex, PoisonError};

pub use ceed_sys as sys;

pub use basis::{Basis, Topology};
pub use error::{Error, Result};
#[doc(hidden)]
pub use error::{check as __check, reset as __reset};
pub use operator::{FieldVector, Operator};
pub use qfunction::{EvalMode, QFunction};
pub use restriction::ElemRestriction;
pub use vector::CVector;

/// The backend slidesim uses on every target: always the full resource, since a prefix such as
/// `/cpu/self` resolves to different backends natively and in wasm.
pub const DEFAULT_RESOURCE: &str = "/cpu/self/opt/blocked";

/// Serializes libCEED's global backend registry (`CeedRegisterAll`, `CeedInit`), which is not
/// thread safe without OpenMP.
static REGISTRY: Mutex<()> = Mutex::new(());

/// `usize` to `CeedInt`, or an error naming `what`.
pub(crate) fn to_int(n: usize, what: &str) -> Result<sys::CeedInt> {
    sys::CeedInt::try_from(n).map_err(|_| {
        Error::new(
            sys::CEED_ERROR_DIMENSION,
            format!("{what} = {n} does not fit in CeedInt"),
        )
    })
}

/// `usize` to `CeedSize`, or an error naming `what`.
pub(crate) fn to_size(n: usize, what: &str) -> Result<sys::CeedSize> {
    sys::CeedSize::try_from(n).map_err(|_| {
        Error::new(
            sys::CEED_ERROR_DIMENSION,
            format!("{what} = {n} does not fit in CeedSize"),
        )
    })
}

/// A string for libCEED, or an error naming `what`.
pub(crate) fn c_string(s: &str, what: &str) -> Result<CString> {
    CString::new(s).map_err(|_| {
        Error::new(
            sys::CEED_ERROR_INCOMPATIBLE,
            format!("{what} {s:?} contains a NUL character"),
        )
    })
}

/// The available backends (resources of `CeedRegistryGetList`). Call with [`REGISTRY`] held.
fn registry() -> Result<Vec<String>> {
    let ierr = unsafe { sys::CeedRegisterAll() };
    if ierr != 0 {
        return Err(Error::new(ierr, "CeedRegisterAll failed"));
    }
    let mut n = 0usize;
    let mut list: *const *mut c_char = null();
    // Fails (with NULL Ceed, so aborting) only if no backend is registered or malloc fails.
    let ierr = unsafe { sys::CeedRegistryGetList(&mut n, &mut list, null_mut()) };
    if ierr != 0 {
        return Err(Error::new(ierr, "CeedRegistryGetList failed"));
    }
    // SAFETY: `list` holds `n` NUL terminated names owned by libCEED's registry; the caller frees
    // the list itself (not its entries) with free().
    let names = (0..n)
        .map(|i| {
            unsafe { CStr::from_ptr(*list.add(i)) }
                .to_string_lossy()
                .into_owned()
        })
        .collect();
    unsafe { sys::free(list.cast_mut().cast::<c_void>()) };
    Ok(names)
}

/// The resources of the available libCEED backends, for example `/cpu/self/opt/blocked`.
pub fn resources() -> Result<Vec<String>> {
    let _registry = REGISTRY.lock().unwrap_or_else(PoisonError::into_inner);
    registry()
}

/// The `Ceed` handle, destroyed once when the last [`Ceed`] clone (and with it the last
/// wrapper) is dropped. libCEED's own references keep it alive for its objects.
struct Handle(sys::Ceed);

impl Drop for Handle {
    fn drop(&mut self) {
        // Nothing to report an error to.
        unsafe { sys::CeedDestroy(&mut self.0) };
    }
}

/// A libCEED context (`Ceed`) on one backend. Cloning it is cheap (a shared reference); every
/// object keeps a clone, for its error messages.
#[derive(Clone)]
pub struct Ceed {
    handle: Rc<Handle>,
}

impl fmt::Debug for Ceed {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Ceed").field(&self.handle.0).finish()
    }
}

impl Ceed {
    /// A libCEED context on the backend `resource`, which must be the full resource of an
    /// available backend (see [`resources`]), for example [`DEFAULT_RESOURCE`]. Prefixes such
    /// as `/cpu/self` and options after `:` are refused: libCEED resolves a prefix to a
    /// backend which depends on the build, and aborts (instead of returning an error) when a
    /// resource does not resolve or the backend refuses it.
    pub fn new(resource: &str) -> Result<Ceed> {
        let _registry = REGISTRY.lock().unwrap_or_else(PoisonError::into_inner);
        let available = registry()?;
        if !available.iter().any(|r| r == resource) {
            return Err(Error::new(
                sys::CEED_ERROR_MAJOR,
                format!(
                    "no libCEED backend {resource:?}: give the full resource of an available \
                     backend ({})",
                    available.join(", ")
                ),
            ));
        }
        let c_resource = c_string(resource, "libCEED resource")?;
        let mut raw: sys::Ceed = null_mut();
        // Not `chk!`: no Ceed exists yet. The resource names a registered backend exactly, so
        // CeedInit resolves it and the backend accepts it.
        let ierr = unsafe { sys::CeedInit(c_resource.as_ptr(), &mut raw) };
        if ierr != 0 {
            if !raw.is_null() {
                unsafe { sys::CeedDestroy(&mut raw) };
            }
            return Err(Error::new(
                ierr,
                format!("CeedInit({resource:?}) failed with error code {ierr}"),
            ));
        }
        // Also sets it on the delegates the backend created (and they pass messages to `raw`).
        unsafe { sys::CeedSetErrorHandler(raw, Some(sys::CeedErrorStore)) };
        Ok(Ceed {
            handle: Rc::new(Handle(raw)),
        })
    }

    /// The resource of the backend (`CeedGetResource`).
    pub fn resource(&self) -> Result<String> {
        let mut text: *const c_char = null();
        chk!(self, unsafe {
            sys::CeedGetResource(self.as_raw(), &mut text)
        });
        // SAFETY: a NUL terminated string owned by the Ceed.
        Ok(unsafe { CStr::from_ptr(text) }
            .to_string_lossy()
            .into_owned())
    }

    /// The raw handle, for the libCEED API not wrapped here.
    pub fn as_raw(&self) -> sys::Ceed {
        self.handle.0
    }

    /// Whether `self` and `other` are the same context.
    pub(crate) fn same(&self, other: &Ceed) -> bool {
        Rc::ptr_eq(&self.handle, &other.handle)
    }

    /// An error unless `other` (the context of `what`) is `self`.
    pub(crate) fn ensure_same(&self, other: &Ceed, what: &str) -> Result<()> {
        if self.same(other) {
            Ok(())
        } else {
            Err(Error::new(
                sys::CEED_ERROR_INCOMPATIBLE,
                format!("{what} belongs to another Ceed"),
            ))
        }
    }
}
