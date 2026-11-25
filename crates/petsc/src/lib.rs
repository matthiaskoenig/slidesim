//! Thin safe wrappers around PETSc.
//!
//! [`Petsc::init`] initializes PETSc (once per process) and installs an error handler which
//! neither prints nor aborts: every failing PETSc call returns an [`Error`] carrying the PETSc
//! message and traceback (see [`chk!`]). Dropping [`Petsc`] finalizes PETSc; after that, every
//! wrapper call returns an error (`PETSC_ERR_ORDER`) and dropped objects are leaked instead of
//! destroyed. One exception: if [`Petsc`] is dropped inside a callback called by PETSc (for
//! example by a [`ShellOperator`] owning it, dropped with its matrix or during `mult`), PETSc
//! code is on the stack, so PETSc is not finalized but stays initialized (and usable through
//! the wrappers) for the rest of the process.
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
mod fe;
mod ksp;
mod matrix;
mod options;
mod pc;
mod snes;
mod ts;
mod vector;

use std::ffi::{CStr, CString, c_char, c_int};
use std::marker::PhantomData;
use std::ptr::{null, null_mut};
use std::sync::atomic::{AtomicBool, Ordering};

pub use petsc_sys as sys;

pub use dm::{Dm, InsertMode, LocalOffsets, Point};
use error::check;
pub use error::{Error, Result};
#[doc(hidden)]
pub use error::{check as __check, ensure_alive as __ensure_alive};
pub use fe::{Fe, Quadrature, Tabulation};
pub use ksp::{ConvergedReason, Ksp};
pub use matrix::{Matrix, ShellOperator};
pub use options::OptionsScope;
pub use pc::Pc;
pub use snes::Snes;
pub use ts::{ExactFinalTime, Implicit, ProblemType, Ts};
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

/// `s` as a C string, or an error naming `what` if it contains a NUL character.
pub(crate) fn c_string(s: &str, what: &str) -> Result<CString> {
    CString::new(s).map_err(|_| {
        Error::new(
            sys::PETSC_ERR_ARG_WRONG,
            format!("{what} {s:?} contains a NUL character"),
        )
    })
}

/// A copy of the C string `s`, or `None` if `s` is null.
///
/// # Safety
/// `s` is null or a NUL terminated string, valid for the call.
pub(crate) unsafe fn string_from(s: *const c_char) -> Option<String> {
    if s.is_null() {
        return None;
    }
    // SAFETY: not null, so a valid string (the caller's promise).
    Some(unsafe { CStr::from_ptr(s) }.to_string_lossy().into_owned())
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
            .map(|a| c_string(a, "PETSc option"))
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

    /// `PETSC_COMM_SELF`: this process alone.
    pub fn comm_self(&self) -> Comm {
        Comm::self_()
    }

    /// The value of the option `name` (for example `-ksp_type`) in the global options database:
    /// `None` if it is not set, `Some("")` if it is set without a value (`PetscOptionsFindPair`).
    pub fn option(&self, name: &str) -> Result<Option<String>> {
        options::get(name)
    }

    /// Sets the option `name` (starting with `-`) to `value`, or without a value, in the global
    /// options database (`PetscOptionsSetValue`).
    pub fn set_option(&self, name: &str, value: Option<&str>) -> Result<()> {
        options::set(name, value)
    }

    /// Removes the option `name` from the global options database (`PetscOptionsClearValue`).
    pub fn clear_option(&self, name: &str) -> Result<()> {
        options::clear(name)
    }

    /// Runs `body` with the option `name` set to `value` (or without a value) in the global
    /// options database, then restores the option as it was, also when `body` fails. An error of
    /// `body` takes precedence over one of restoring.
    pub fn with_option<T, E: From<Error>>(
        &self,
        name: &str,
        value: Option<&str>,
        body: impl FnOnce() -> std::result::Result<T, E>,
    ) -> std::result::Result<T, E> {
        let previous = self.option(name)?;
        self.set_option(name, value)?;
        let result = body();
        let restored = options::restore(name, previous.as_deref());
        let value = result?;
        restored?;
        Ok(value)
    }
}

impl Drop for Petsc {
    fn drop(&mut self) {
        if error::in_callback() {
            // Dropped inside a callback (for example by an operator owning `Petsc`): PETSc code
            // is on the stack and must not see PETSc and MPI finalized under it. PETSc stays
            // initialized for the rest of the process, and keeps pointers to the arguments.
            std::mem::forget(std::mem::take(&mut self._args));
            std::mem::forget(std::mem::take(&mut self._argv));
            return;
        }
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

    /// The sum of `x` over all processes (`MPI_Allreduce`).
    pub fn sum_int(&self, x: sys::PetscInt) -> Result<sys::PetscInt> {
        let op = unsafe { sys::petsc_sys_mpi_sum() };
        self.all_reduce(x, unsafe { sys::petsc_sys_mpiu_int() }, op)
    }

    /// The maximum of `x` over all processes (`MPI_Allreduce`).
    pub fn max_int(&self, x: sys::PetscInt) -> Result<sys::PetscInt> {
        let op = unsafe { sys::petsc_sys_mpi_max() };
        self.all_reduce(x, unsafe { sys::petsc_sys_mpiu_int() }, op)
    }

    /// The sum of `x` over all processes (`MPI_Allreduce`).
    pub fn sum_real(&self, x: sys::PetscReal) -> Result<sys::PetscReal> {
        let op = unsafe { sys::petsc_sys_mpi_sum() };
        self.all_reduce(x, unsafe { sys::petsc_sys_mpiu_real() }, op)
    }

    /// `MPI_Allreduce` of one value; `datatype` is the MPI type of `T`.
    fn all_reduce<T: Copy>(&self, x: T, datatype: sys::MPI_Datatype, op: sys::MPI_Op) -> Result<T> {
        error::ensure_alive("MPI_Allreduce")?;
        let mut y = x;
        // SAFETY: one value of `datatype` in and out; MPI aborts on errors by default, and any
        // other return is reported.
        let ierr = unsafe {
            sys::MPI_Allreduce(
                (&raw const x).cast(),
                (&raw mut y).cast(),
                1,
                datatype,
                op,
                self.0,
            )
        };
        if ierr != 0 {
            return Err(Error::new(
                sys::PETSC_ERR_MPI,
                format!("MPI_Allreduce failed with MPI error {ierr}"),
            ));
        }
        Ok(y)
    }

    pub fn as_raw(&self) -> sys::MPI_Comm {
        self.0
    }
}
