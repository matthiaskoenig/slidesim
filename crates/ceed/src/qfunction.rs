//! `CeedQFunction`: pointwise functions at quadrature points.

use std::ffi::c_void;
use std::ptr::null_mut;

use crate::{Ceed, Error, Result, c_string, chk, sys, to_int};

/// What an operator passes to a QFunction field (`CeedEvalMode`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvalMode {
    /// The E-vector values themselves (no basis).
    None,
    /// Values at quadrature points.
    Interp,
    /// Reference gradients at quadrature points.
    Grad,
    /// Divergence at quadrature points.
    Div,
    /// Curl at quadrature points.
    Curl,
    /// Quadrature weights (no restriction, no vector).
    Weight,
}

impl EvalMode {
    fn raw(self) -> sys::CeedEvalMode {
        match self {
            EvalMode::None => sys::CEED_EVAL_NONE,
            EvalMode::Interp => sys::CEED_EVAL_INTERP,
            EvalMode::Grad => sys::CEED_EVAL_GRAD,
            EvalMode::Div => sys::CEED_EVAL_DIV,
            EvalMode::Curl => sys::CEED_EVAL_CURL,
            EvalMode::Weight => sys::CEED_EVAL_WEIGHT,
        }
    }
}

/// A libCEED QFunction (`CeedQFunction`).
#[derive(Debug)]
pub struct QFunction {
    raw: sys::CeedQFunction,
    ceed: Ceed,
}

impl QFunction {
    /// A QFunction for interior terms (`CeedQFunctionCreateInterior`) calling `f` on blocks of
    /// quadrature points (a multiple of `vlength`), whose source is at `source`, in the form
    /// `/abs/path/file.h:function` (used by backends which compile the source, such as GPU
    /// JIT).
    ///
    /// libCEED calls `f(ctx, Q, in, out)` with the context data ([`QFunction::set_context`],
    /// else NULL), the number of points `Q`, and arrays of `size * Q` values for the fields in
    /// the order added, laid out `[size][Q]`. `f` returns 0, or an error code which makes the
    /// operator call fail. It must not unwind.
    pub fn interior(
        ceed: &Ceed,
        vlength: usize,
        f: sys::CeedQFunctionUser,
        source: &str,
    ) -> Result<QFunction> {
        if f.is_none() {
            return Err(Error::new(
                sys::CEED_ERROR_INCOMPLETE,
                format!("QFunction {source:?} without a function"),
            ));
        }
        if vlength == 0 {
            // libCEED does not check it, and divides by it.
            return Err(Error::new(
                sys::CEED_ERROR_DIMENSION,
                format!("QFunction {source:?} with vector length 0 (need at least 1)"),
            ));
        }
        let c_source = c_string(source, "QFunction source")?;
        let mut raw: sys::CeedQFunction = null_mut();
        chk!(ceed, unsafe {
            sys::CeedQFunctionCreateInterior(
                ceed.as_raw(),
                to_int(vlength, "QFunction vector length")?,
                f,
                c_source.as_ptr(),
                &mut raw,
            )
        });
        Ok(QFunction {
            raw,
            ceed: ceed.clone(),
        })
    }

    /// Adds the input field `name` with `size` values per point (components, times the
    /// dimension for [`EvalMode::Grad`]).
    pub fn add_input(&mut self, name: &str, size: usize, mode: EvalMode) -> Result<()> {
        let c_name = c_string(name, "QFunction field")?;
        chk!(&self.ceed, unsafe {
            sys::CeedQFunctionAddInput(
                self.raw,
                c_name.as_ptr(),
                to_int(size, "field size")?,
                mode.raw(),
            )
        });
        Ok(())
    }

    /// Adds the output field `name` with `size` values per point.
    pub fn add_output(&mut self, name: &str, size: usize, mode: EvalMode) -> Result<()> {
        let c_name = c_string(name, "QFunction field")?;
        chk!(&self.ceed, unsafe {
            sys::CeedQFunctionAddOutput(
                self.raw,
                c_name.as_ptr(),
                to_int(size, "field size")?,
                mode.raw(),
            )
        });
        Ok(())
    }

    /// Sets the context data passed to the function: a libCEED copy of `data`
    /// (`CeedQFunctionContext` with `CEED_COPY_VALUES`), replacing any previous context.
    pub fn set_context(&mut self, data: &[f64]) -> Result<()> {
        let mut ctx = Context(null_mut());
        chk!(&self.ceed, unsafe {
            sys::CeedQFunctionContextCreate(self.ceed.as_raw(), &mut ctx.0)
        });
        // SAFETY: CEED_COPY_VALUES copies `size_of_val(data)` bytes and keeps no pointer.
        chk!(&self.ceed, unsafe {
            sys::CeedQFunctionContextSetData(
                ctx.0,
                sys::CEED_MEM_HOST,
                sys::CEED_COPY_VALUES,
                size_of_val(data),
                data.as_ptr().cast_mut().cast::<c_void>(),
            )
        });
        // The QFunction takes its own reference.
        chk!(&self.ceed, unsafe {
            sys::CeedQFunctionSetContext(self.raw, ctx.0)
        });
        Ok(())
    }

    /// The raw handle, for the libCEED API not wrapped here.
    pub fn as_raw(&self) -> sys::CeedQFunction {
        self.raw
    }

    pub(crate) fn ceed(&self) -> &Ceed {
        &self.ceed
    }
}

impl Drop for QFunction {
    fn drop(&mut self) {
        // Nothing to report an error to; operators using it hold their own reference.
        unsafe { sys::CeedQFunctionDestroy(&mut self.raw) };
    }
}

/// Our reference to a `CeedQFunctionContext`, released on drop (also on errors).
struct Context(sys::CeedQFunctionContext);

impl Drop for Context {
    fn drop(&mut self) {
        unsafe { sys::CeedQFunctionContextDestroy(&mut self.0) };
    }
}
