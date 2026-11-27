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
//! A solver whose solve failed ([`Ksp::solve`], [`Ts::solve`]) is leaked too when dropped: PETSc
//! returns errors without restoring the state of the objects they passed through, and destroying
//! them can free memory twice. If PETSc reports an error when a vector, matrix, mesh or solver is
//! destroyed, the drop panics with it: the state of PETSc is corrupt.
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
mod metric;
mod options;
mod pc;
mod sf;
mod snes;
mod ts;
mod vector;
mod viewer;

use std::ffi::{CStr, CString, c_char, c_int};
use std::io::Write;
use std::marker::PhantomData;
use std::ptr::{null, null_mut};
use std::sync::atomic::{AtomicBool, Ordering};

pub use petsc_sys as sys;

pub use dm::{Dm, InsertMode, LocalOffsets, Partitioner, Point};
use error::check;
pub use error::{Error, Result};
#[doc(hidden)]
pub use error::{check as __check, ensure_alive as __ensure_alive};
pub use fe::{Fe, Quadrature, Tabulation};
pub use ksp::{ConvergedReason, Ksp};
pub use matrix::{Matrix, ShellOperator};
pub use metric::MetricOptions;
pub use options::OptionsScope;
pub use pc::Pc;
pub use sf::{Leaf, Sf};
pub use snes::Snes;
pub use ts::{ExactFinalTime, Implicit, ProblemType, Ts};
pub use vector::{AsVec, AsVecMut, VecMut, VecRef, Vector};
pub use viewer::Viewer;

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

    /// `PETSC_COMM_WORLD`, borrowed from `self`: valid until PETSc is finalized.
    pub fn comm(&self) -> Comm<'_> {
        Comm::new(unsafe { sys::PETSC_COMM_WORLD })
    }

    /// Ends every process of `MPI_COMM_WORLD` with the exit code `code` (`MPI_Abort`), without
    /// finalizing PETSc. For an error on some processes only, which would otherwise leave the
    /// others waiting in a collective call forever; with one process, return the error instead.
    ///
    /// Flushes this process's buffered output first, Rust's standard output and C's `stdio`
    /// streams (`fflush(NULL)`), so that what it printed is not lost: Rust keeps an unfinished
    /// line, and C an unfinished line, or whole blocks if the output is a pipe (MPICH makes C's
    /// stdout unbuffered and Open MPI's `mpiexec` gives a terminal, other launchers need not).
    /// PETSc flushes after each of its own prints; other C code need not.
    pub fn abort_world(&self, code: i32) -> ! {
        // MPI is initialized while `self` lives.
        abort_world(code)
    }

    /// Makes a panic on this process end every process of `MPI_COMM_WORLD` (as
    /// [`Petsc::abort_world`] with the exit code `code`), after printing
    /// `[rank <rank>] panic: <message> (at <location>)` to stderr, and the backtrace if
    /// `RUST_BACKTRACE` asks for one. Without it, a panicking process would finalize PETSc and MPI
    /// while unwinding (dropping `Petsc`) and wait there for the others, which wait for it in a
    /// collective call: the run hangs. Replaces the panic hook of the process; a panic after
    /// PETSc is finalized only prints.
    pub fn abort_world_on_panic(&self, code: i32) {
        let rank = self.comm().rank();
        std::panic::set_hook(Box::new(move |info| {
            let message = info.payload_as_str().unwrap_or("Box<dyn Any>");
            let location = info
                .location()
                .map_or_else(|| "unknown location".to_string(), ToString::to_string);
            eprintln!("[rank {rank}] panic: {message} (at {location})");
            let backtrace = std::backtrace::Backtrace::capture();
            if backtrace.status() == std::backtrace::BacktraceStatus::Captured {
                eprintln!("{backtrace}");
            }
            if alive() {
                abort_world(code);
            }
        }));
    }

    /// `PETSC_COMM_SELF`: this process alone. Borrowed from `self`, as [`Petsc::comm`].
    pub fn comm_self(&self) -> Comm<'_> {
        Comm::self_()
    }

    /// Whether PETSc was built with the external package `name`, for example `ptscotch`
    /// (`PetscHasExternalPackage`; case insensitive).
    pub fn has_external_package(&self, name: &str) -> Result<bool> {
        let name = c_string(name, "package name")?;
        let mut has = false;
        // SAFETY: PETSc reads the name during the call.
        chk!(unsafe { sys::PetscHasExternalPackage(name.as_ptr(), &mut has) });
        Ok(has)
    }

    /// Resets the random generator of the graph partitioner PT-Scotch to its initial seed
    /// (`SCOTCH_randomReset`), so that the next partitioning gives the same partition as the
    /// first in this process; does nothing if PETSc has no PT-Scotch. The generator is global and
    /// otherwise continues from one partitioning to the next. Call it on every process.
    ///
    /// The reset alone makes partitions reproducible only with a PT-Scotch built without threads
    /// and with its deterministic algorithms (`THREADS=OFF`, `SCOTCH_DETERMINISTIC=FULL`, as
    /// `scripts/deps/build-native.sh` builds it): then the partition depends on the graph and the
    /// number of processes only.
    pub fn reset_partitioner_random(&self) {
        // SAFETY: no arguments; PT-Scotch is initialized lazily.
        unsafe { sys::petsc_sys_ptscotch_random_reset() };
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

/// Flushes the output of this process and ends every process of `MPI_COMM_WORLD` with the exit
/// code `code`: [`Petsc::abort_world`], for a caller which ensures that MPI is initialized.
fn abort_world(code: i32) -> ! {
    // Errors are ignored: the processes end either way.
    let _ = std::io::stdout().flush();
    // SAFETY: PetscFFlush(NULL) is fflush(NULL), which flushes every C output stream.
    let _ = unsafe { sys::PetscFFlush(null_mut()) };
    // SAFETY: MPI is initialized (the caller's contract). MPI_Abort does not return; if an MPI
    // implementation does, this process exits all the same.
    unsafe { sys::MPI_Abort(sys::petsc_sys_comm_world(), code) };
    std::process::exit(code)
}

/// An MPI communicator, borrowed for `'a` from what keeps it alive: [`Petsc`] for
/// `PETSC_COMM_WORLD` and `PETSC_COMM_SELF`, a [`Dm`] for PETSc's communicator of the mesh (freed
/// with the last object on it), an [`OwnedComm`]. To keep a communicator beyond its source, own
/// a reference to it with [`Comm::duplicate`]. Not `Send` or `Sync` (on every MPI: MPICH's
/// handle is an integer).
///
/// ```compile_fail
/// let petsc = petsc::Petsc::init(&[]).unwrap();
/// let comm = petsc.comm();
/// drop(petsc);
/// comm.rank();
/// ```
#[derive(Clone, Copy, Debug)]
pub struct Comm<'a>(sys::MPI_Comm, PhantomData<(&'a (), *const ())>);

impl<'a> Comm<'a> {
    /// The communicator `raw`, valid for `'a` (the caller's contract).
    fn new(raw: sys::MPI_Comm) -> Comm<'a> {
        Comm(raw, PhantomData)
    }

    /// `PETSC_COMM_SELF`, for a call within this crate while PETSc is initialized.
    pub(crate) fn self_() -> Comm<'static> {
        Comm::new(unsafe { sys::petsc_sys_comm_self() })
    }

    /// A reference of its own to PETSc's inner communicator of this one, valid until it is
    /// dropped, whatever happens to the objects it came from (`PetscCommDuplicate`). Collective:
    /// PETSc built with debugging waits for every process of the communicator here.
    pub fn duplicate(&self) -> Result<OwnedComm> {
        error::ensure_alive("PetscCommDuplicate")?;
        let mut inner = self.0;
        chk!(unsafe { sys::PetscCommDuplicate(self.0, &mut inner, null_mut()) });
        Ok(OwnedComm(std::rc::Rc::new(Reference(inner))))
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

    /// The maximum of `x` over all processes (`MPI_Allreduce`).
    pub fn max_real(&self, x: sys::PetscReal) -> Result<sys::PetscReal> {
        let op = unsafe { sys::petsc_sys_mpi_max() };
        self.all_reduce(x, unsafe { sys::petsc_sys_mpiu_real() }, op)
    }

    /// The bytes `data` of the process `root`, on every process (`MPI_Bcast`): the others'
    /// `data` is replaced, whatever its length. For messages of errors which one process found,
    /// so that every process reports the same.
    pub fn broadcast_bytes(&self, root: usize, data: &mut Vec<u8>) -> Result<()> {
        let root_rank = c_int::try_from(root)
            .map_err(|_| Error::new(sys::PETSC_ERR_ARG_OUTOFRANGE, format!("rank {root}")))?;
        let mut length = sys::PetscInt::try_from(data.len()).map_err(|_| {
            Error::new(
                sys::PETSC_ERR_ARG_OUTOFRANGE,
                format!("{} bytes to broadcast", data.len()),
            )
        })?;
        self.broadcast(
            (&raw mut length).cast(),
            1,
            unsafe { sys::petsc_sys_mpiu_int() },
            root_rank,
        )?;
        let length = usize::try_from(length).expect("a length is not negative");
        data.resize(length, 0);
        if length == 0 {
            // MPICH refuses the dangling buffer of an empty vector.
            return Ok(());
        }
        let count = c_int::try_from(length).map_err(|_| {
            Error::new(
                sys::PETSC_ERR_ARG_OUTOFRANGE,
                format!("{length} bytes to broadcast"),
            )
        })?;
        self.broadcast(
            data.as_mut_ptr().cast(),
            count,
            unsafe { sys::petsc_sys_mpi_byte() },
            root_rank,
        )
    }

    /// `MPI_Bcast` of `count` values of `datatype` at `buffer` from `root`.
    fn broadcast(
        &self,
        buffer: *mut std::ffi::c_void,
        count: c_int,
        datatype: sys::MPI_Datatype,
        root: c_int,
    ) -> Result<()> {
        error::ensure_alive("MPI_Bcast")?;
        // SAFETY: `buffer` holds `count` values of `datatype` (the caller's contract); MPI aborts
        // on errors by default, and any other return is reported.
        let ierr = unsafe { sys::MPI_Bcast(buffer, count, datatype, root, self.0) };
        if ierr != 0 {
            return Err(Error::new(
                sys::PETSC_ERR_MPI,
                format!("MPI_Bcast failed with MPI error {ierr}"),
            ));
        }
        Ok(())
    }

    /// Waits until every process has called it (`MPI_Barrier`).
    pub fn barrier(&self) -> Result<()> {
        error::ensure_alive("MPI_Barrier")?;
        // SAFETY: a valid communicator while PETSc is initialized; MPI aborts on errors by
        // default, and any other return is reported.
        let ierr = unsafe { sys::MPI_Barrier(self.0) };
        if ierr != 0 {
            return Err(Error::new(
                sys::PETSC_ERR_MPI,
                format!("MPI_Barrier failed with MPI error {ierr}"),
            ));
        }
        Ok(())
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

/// A communicator owned for as long as it lives, from [`Comm::duplicate`]: a reference to
/// PETSc's inner communicator, released when the last clone is dropped (`PetscCommDestroy`; MPI
/// frees the communicator with its last reference). Clones share the reference, so that cloning
/// is not collective. Not `Send` or `Sync`.
#[derive(Clone, Debug)]
pub struct OwnedComm(std::rc::Rc<Reference>);

impl OwnedComm {
    /// The communicator, borrowed from `self`.
    pub fn comm(&self) -> Comm<'_> {
        Comm::new(self.0.0)
    }
}

/// One reference counted by PETSc to an inner communicator.
#[derive(Debug)]
struct Reference(sys::MPI_Comm);

impl Drop for Reference {
    fn drop(&mut self) {
        if !alive() {
            // MPI is finalized and its communicators with it.
            return;
        }
        crate::error::destroyed(
            unsafe { sys::PetscCommDestroy(&mut self.0) },
            "PetscCommDestroy",
        );
    }
}
