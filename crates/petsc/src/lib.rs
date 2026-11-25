//! Thin safe wrappers around PETSc.
//!
//! [`Petsc::init`] initializes PETSc (once per process) and installs an error handler which
//! neither prints nor aborts: every failing PETSc call returns an [`Error`] carrying the PETSc
//! message and traceback (see [`chk!`]). Dropping [`Petsc`] finalizes PETSc; after that, every
//! wrapper call returns an error (`PETSC_ERR_ORDER`) and dropped objects are leaked instead of
//! destroyed.
//!
//! PETSc is not thread safe: no type here is `Send` or `Sync`.
//!
//! ```compile_fail
//! fn send<T: Send>() {}
//! send::<petsc::Comm>();
//! ```
//!
//! ```compile_fail
//! fn sync<T: Sync>() {}
//! sync::<petsc::Comm>();
//! ```
//!
//! The raw bindings are re-exported as [`sys`], and every wrapper gives its raw handle with
//! `as_raw()`, for the PETSc API not wrapped here.

mod dm;
mod error;
mod ksp;
mod matrix;
mod ts;
mod vector;

use std::ffi::{CString, c_char, c_int};
use std::marker::PhantomData;
use std::ptr::{null, null_mut};
use std::sync::atomic::{AtomicBool, Ordering};

pub use petsc_sys as sys;

pub use dm::Dm;
use error::check;
pub use error::{Error, Result};
#[doc(hidden)]
pub use error::{check as __check, ensure_alive as __ensure_alive};
pub use ksp::Ksp;
pub use matrix::{Matrix, ShellOperator};
pub use ts::Ts;
pub use vector::{AsVec, AsVecMut, VecMut, VecRef, Vector};

static INITIALIZED: AtomicBool = AtomicBool::new(false);

/// Whether PETSc is initialized and not yet finalized: PETSc objects may be destroyed.
pub(crate) fn alive() -> bool {
    unsafe { sys::PetscInitializeCalled && !sys::PetscFinalizeCalled }
}

/// `usize` to `PetscInt`, or an error naming `what`.
pub(crate) fn to_int(n: usize, what: &str) -> Result<sys::PetscInt> {
    sys::PetscInt::try_from(n).map_err(|_| {
        Error::new(
            sys::PETSC_ERR_ARG_OUTOFRANGE,
            format!("{what} = {n} does not fit in PetscInt"),
        )
    })
}

/// An initialized PETSc (and MPI, if PETSc initialized it). Dropping it finalizes PETSc.
pub struct Petsc {
    // PETSc keeps pointers to the arguments: they live until PetscFinalize.
    _args: Vec<CString>,
    _argv: Vec<*mut c_char>,
}

impl Petsc {
    /// Initializes PETSc with the options `args` (for example `["-ksp_type", "cg"]`, without a
    /// program name). PETSc can be initialized once per process, since MPI cannot be
    /// initialized again after it is finalized.
    pub fn init(args: &[&str]) -> Result<Petsc> {
        let args = std::iter::once("slidesim")
            .chain(args.iter().copied())
            .map(|a| {
                CString::new(a).map_err(|_| {
                    Error::new(
                        sys::PETSC_ERR_ARG_WRONG,
                        format!("PETSc option {a:?} contains a NUL character"),
                    )
                })
            })
            .collect::<Result<Vec<_>>>()?;
        if INITIALIZED.swap(true, Ordering::SeqCst) {
            return Err(Error::new(
                sys::PETSC_ERR_ORDER,
                "PETSc can be initialized once per process",
            ));
        }
        let mut argv: Vec<*mut c_char> = args.iter().map(|a| a.as_ptr().cast_mut()).collect();
        argv.push(null_mut());
        let mut argc = args.len() as c_int;
        let mut argv_ptr = argv.as_mut_ptr();
        // Not `chk!`: PETSc is not initialized yet. SAFETY: `argv` is NULL terminated, and it
        // and the strings live in `Petsc` until PetscFinalize (PETSc keeps pointers to them).
        check(
            unsafe { sys::PetscInitialize(&mut argc, &mut argv_ptr, null(), null()) },
            "PetscInitialize",
        )?;
        let petsc = Petsc {
            _args: args,
            _argv: argv,
        };
        chk!(unsafe { sys::PetscPushErrorHandler(Some(error::record), null_mut()) });
        Ok(petsc)
    }

    /// `PETSC_COMM_WORLD`.
    pub fn comm(&self) -> Comm {
        Comm::new(unsafe { sys::PETSC_COMM_WORLD })
    }
}

impl Drop for Petsc {
    fn drop(&mut self) {
        // Nothing to report an error to; PETSc is finalized either way.
        unsafe {
            sys::PetscPopErrorHandler();
            sys::PetscFinalize();
        }
    }
}

/// An MPI communicator. Valid while PETSc is initialized. Not `Send` or `Sync` (on every MPI:
/// MPICH's handle is an integer).
#[derive(Clone, Copy, Debug)]
pub struct Comm(sys::MPI_Comm, PhantomData<*const ()>);

impl Comm {
    fn new(raw: sys::MPI_Comm) -> Comm {
        Comm(raw, PhantomData)
    }

    /// `PETSC_COMM_SELF`.
    pub(crate) fn self_() -> Comm {
        Comm::new(unsafe { sys::petsc_sys_comm_self() })
    }

    /// The rank of this process.
    pub fn rank(&self) -> usize {
        assert!(alive(), "MPI communicator used after PetscFinalize");
        let mut rank: c_int = 0;
        // Cannot fail on a valid communicator (and MPI aborts on errors by default).
        unsafe { sys::MPI_Comm_rank(self.0, &mut rank) };
        rank as usize
    }

    /// The number of processes.
    pub fn size(&self) -> usize {
        assert!(alive(), "MPI communicator used after PetscFinalize");
        let mut size: c_int = 0;
        unsafe { sys::MPI_Comm_size(self.0, &mut size) };
        size as usize
    }

    pub fn as_raw(&self) -> sys::MPI_Comm {
        self.0
    }
}
