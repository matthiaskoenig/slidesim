//! Errors of libCEED calls.
//!
//! Every [`Ceed`] uses libCEED's error handler `CeedErrorStore`: a failing libCEED routine
//! stores its message in the `Ceed` (the root of its delegates) and returns the error code,
//! without printing or aborting. [`chk!`] reads that message with `CeedGetErrorMessage` and
//! resets it with `CeedResetErrorMessage`, before and after the call, so a message always
//! belongs to the call it is reported for. A code returned without a message (for example by a
//! QFunction) gets a generic text instead of a stale one.

use std::ffi::{CStr, c_char, c_int};
use std::fmt;
use std::ptr::null;

use crate::{Ceed, sys};

/// A libCEED error: the libCEED error code and a message.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Error {
    /// The libCEED error code (`CeedErrorType`, for example `CEED_ERROR_DIMENSION`).
    pub code: i32,
    /// libCEED's message (`file:line in function(): text`) and the call which failed.
    pub message: String,
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

impl Error {
    /// An error raised by the wrappers themselves, with the libCEED error code `code`.
    pub(crate) fn new(code: sys::CeedErrorType, message: impl Into<String>) -> Error {
        Error {
            code,
            message: message.into(),
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "libCEED error {}: {}", self.code, self.message)
    }
}

impl std::error::Error for Error {}

/// Calls a libCEED function on objects of the [`Ceed`] `ceed` and turns a nonzero return code
/// into an [`Error`] (with the message libCEED stored), which the enclosing function returns
/// with `?`.
///
/// The caller writes the `unsafe` block: `chk!(&ceed, unsafe { sys::CeedVectorSetValue(v, 0.0) })`.
#[macro_export]
macro_rules! chk {
    ($ceed:expr, $call:expr) => {{
        let ceed: &$crate::Ceed = $ceed;
        $crate::__reset(ceed);
        let ierr: ::std::ffi::c_int = $call;
        $crate::__check(ceed, ierr, stringify!($call))?
    }};
}

/// The text libCEED keeps when no message is stored.
const NO_MESSAGE: &str = "No error message stored";

/// The call of `chk!(ceed, unsafe { call })`, for messages (`stringify!` of the block spaces
/// paths as `sys :: CeedVectorSetValue`).
fn call_text(call: &str) -> String {
    let call = call.trim();
    call.strip_prefix("unsafe")
        .and_then(|c| c.trim_start().strip_prefix('{'))
        .and_then(|c| c.strip_suffix('}'))
        .map_or(call, str::trim)
        .replace(" :: ", "::")
}

/// Forgets the message stored in `ceed`, if any (left by a routine whose error was not
/// reported, for example while destroying an object).
#[doc(hidden)]
pub fn reset(ceed: &Ceed) {
    let mut text: *const c_char = null();
    // Cannot fail: it copies the "no message" text into the root Ceed.
    unsafe { sys::CeedResetErrorMessage(ceed.as_raw(), &mut text) };
}

/// The error of a nonzero return code `ierr` of the libCEED call `call` (its text) on objects
/// of `ceed`.
#[doc(hidden)]
pub fn check(ceed: &Ceed, ierr: c_int, call: &str) -> Result<()> {
    if ierr == 0 {
        return Ok(());
    }
    let mut text: *const c_char = null();
    // Cannot fail: it returns the message buffer of the root Ceed.
    unsafe { sys::CeedGetErrorMessage(ceed.as_raw(), &mut text) };
    let stored = if text.is_null() {
        String::new()
    } else {
        // SAFETY: a NUL terminated buffer of the Ceed, copied before the reset overwrites it.
        unsafe { CStr::from_ptr(text) }
            .to_string_lossy()
            .trim()
            .to_string()
    };
    reset(ceed);
    let mut message = if stored.is_empty() || stored == NO_MESSAGE {
        format!(
            "error code {ierr} with no message from libCEED (returned by a QFunction or a backend)"
        )
    } else {
        stored
    };
    message.push_str("\n  called as ");
    message.push_str(&call_text(call));
    Err(Error {
        code: ierr,
        message,
    })
}
