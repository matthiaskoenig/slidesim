//! Errors of PETSc calls, and of Rust callbacks called by PETSc.
//!
//! PETSc reports an error by calling the error handler once for the routine which raised it
//! (`PETSC_ERROR_INITIAL`, with the specific message) and once for every routine it passes on
//! the way up (`PETSC_ERROR_REPEAT`). [`record`] is that handler: it prints nothing, keeps the
//! message and the traceback for [`check`], and returns the error code, as
//! `PetscReturnErrorHandler` does. So an error never aborts and never prints: it becomes an
//! [`Error`] of the Rust call.

use std::any::Any;
use std::cell::{Cell, RefCell};
use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::fmt;
use std::panic::{AssertUnwindSafe, Location, catch_unwind};
use std::ptr::null_mut;

use crate::sys;

/// A PETSc error: the PETSc error code and a message with the traceback.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Error {
    /// The PETSc error code (`PetscErrorCodeEnum`, for example `PETSC_ERR_ARG_SIZ`).
    pub code: i32,
    /// The generic text of the code, the specific message, and the PETSc routines it passed.
    pub message: String,
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

impl Error {
    /// An error raised by the wrappers themselves, with the PETSc error code `code`.
    pub(crate) fn new(code: i32, message: impl Into<String>) -> Error {
        Error {
            code,
            message: message.into(),
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PETSc error {}: {}", self.code, self.message)
    }
}

impl std::error::Error for Error {}

/// Calls a PETSc function and turns a nonzero return code into an [`Error`] (with the message
/// and traceback PETSc reported), which the enclosing function returns with `?`.
///
/// The caller writes the `unsafe` block: `chk!(unsafe { sys::VecAXPY(y, alpha, x) })`. The call
/// is not made unless PETSc is initialized and not yet finalized (else `PETSC_ERR_ORDER`).
#[macro_export]
macro_rules! chk {
    ($call:expr) => {{
        $crate::__ensure_alive(stringify!($call))?;
        let ierr: $crate::sys::PetscErrorCode = $call;
        $crate::__check(ierr, stringify!($call))?
    }};
}

/// The call of `chk!(unsafe { call })`, for messages (`stringify!` of the block spaces paths
/// as `sys :: VecAXPY`).
fn call_text(call: &str) -> String {
    let call = call.trim();
    call.strip_prefix("unsafe")
        .and_then(|c| c.trim_start().strip_prefix('{'))
        .and_then(|c| c.strip_suffix('}'))
        .map_or(call, str::trim)
        .replace(" :: ", "::")
}

/// An error unless PETSc is initialized and not yet finalized: PETSc calls are allowed.
#[doc(hidden)]
pub fn ensure_alive(call: &str) -> Result<()> {
    if crate::alive() {
        Ok(())
    } else {
        Err(Error::new(
            sys::PETSC_ERR_ORDER,
            format!(
                "PETSc is not initialized or already finalized\n  called as {}",
                call_text(call)
            ),
        ))
    }
}

/// The error of a nonzero return code `ierr` of the PETSc call `call` (the text of the call).
#[doc(hidden)]
pub fn check(ierr: sys::PetscErrorCode, call: &str) -> Result<()> {
    if ierr == 0 {
        return Ok(());
    }
    let mut text: *const c_char = std::ptr::null();
    // A table lookup, which cannot fail.
    unsafe { sys::PetscErrorMessage(ierr, &mut text, null_mut()) };
    let mut message = if text.is_null() {
        format!("unknown error code {ierr}")
    } else {
        unsafe { CStr::from_ptr(text) }
            .to_string_lossy()
            .into_owned()
    };
    let trace = TRACE.with(|t| t.take());
    if !trace.message.is_empty() {
        message.push_str(": ");
        message.push_str(&trace.message);
    }
    for frame in &trace.frames {
        message.push_str("\n  ");
        message.push_str(frame);
    }
    message.push_str("\n  called as ");
    message.push_str(&call_text(call));
    Err(Error {
        code: ierr,
        message,
    })
}

/// The error being raised: the message and the routines it passed (innermost first).
#[derive(Default)]
struct Trace {
    message: String,
    frames: Vec<String>,
}

thread_local! {
    static TRACE: RefCell<Trace> = RefCell::default();
}

fn lossy(s: *const c_char) -> String {
    if s.is_null() {
        String::new()
    } else {
        unsafe { CStr::from_ptr(s) }.to_string_lossy().into_owned()
    }
}

/// The PETSc error handler: records the error and returns its code without printing.
pub(crate) unsafe extern "C" fn record(
    _comm: sys::MPI_Comm,
    line: c_int,
    func: *const c_char,
    file: *const c_char,
    n: sys::PetscErrorCode,
    p: sys::PetscErrorType,
    mess: *const c_char,
    _ctx: *mut c_void,
) -> sys::PetscErrorCode {
    // Nothing here may unwind into C.
    let _ = catch_unwind(|| {
        TRACE.with(|t| {
            let Ok(mut t) = t.try_borrow_mut() else {
                return;
            };
            if p == sys::PETSC_ERROR_INITIAL {
                t.message = lossy(mess).trim().to_string();
                t.frames.clear();
            }
            let frame = format!("{}() at {}:{line}", lossy(func), lossy(file));
            t.frames.push(frame);
        })
    });
    n
}

thread_local! {
    /// The number of Rust callbacks called by PETSc on the stack.
    static DEPTH: Cell<usize> = const { Cell::new(0) };
}

/// Whether PETSc code is on the stack (a Rust callback called by PETSc is running).
pub(crate) fn in_callback() -> bool {
    DEPTH.with(Cell::get) > 0
}

/// One level of [`DEPTH`], left on drop (also when unwinding).
struct Depth;

impl Depth {
    fn enter() -> Depth {
        DEPTH.with(|d| d.set(d.get() + 1));
        Depth
    }
}

impl Drop for Depth {
    fn drop(&mut self) {
        DEPTH.with(|d| d.set(d.get() - 1));
    }
}

fn panic_message(payload: &(dyn Any + Send)) -> &str {
    if let Some(s) = payload.downcast_ref::<&str>() {
        s
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s
    } else {
        "(no message)"
    }
}

/// Runs the body of a callback called by PETSc (`func` names it in the traceback) and returns
/// its PETSc error code. A Rust error or a panic is raised as a PETSc error (`PetscError`), so
/// it travels up through PETSc and comes out as the [`Error`] of the outer Rust call.
///
/// `func` must not start with `main`: PetscError aborts the program for errors raised there.
#[track_caller]
pub(crate) fn callback(func: &CStr, body: impl FnOnce() -> Result<()>) -> sys::PetscErrorCode {
    let depth = Depth::enter();
    let result = catch_unwind(AssertUnwindSafe(body));
    drop(depth);
    let (code, message) = match result {
        Ok(Ok(())) => return 0,
        Ok(Err(e)) if e.code > 0 => (e.code, e.message),
        Ok(Err(e)) => (sys::PETSC_ERR_LIB, e.message),
        Err(payload) => (
            sys::PETSC_ERR_LIB,
            format!("panic: {}", panic_message(payload.as_ref())),
        ),
    };
    let location = Location::caller();
    let file = CString::new(location.file()).unwrap_or_default();
    let message = CString::new(message.replace('\0', "\\0")).unwrap_or_default();
    // The format is a literal: the message is data. PetscError returns the code (`record`).
    unsafe {
        sys::PetscError(
            sys::petsc_sys_comm_self(),
            location.line() as c_int,
            func.as_ptr(),
            file.as_ptr(),
            code,
            sys::PETSC_ERROR_INITIAL,
            c"%s".as_ptr(),
            message.as_ptr(),
        )
    };
    code
}
