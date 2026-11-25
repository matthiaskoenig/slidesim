//! `CeedOperator`: a QFunction applied on the elements of restrictions and bases.

use std::ptr::null_mut;
use std::slice;

use crate::{
    Basis, CVector, Ceed, ElemRestriction, Error, QFunction, Result, c_string, chk, sys, to_size,
};

/// The vector of an operator field.
#[derive(Clone, Copy, Debug)]
pub enum FieldVector<'a> {
    /// The input or output of [`Operator::apply`] (`CEED_VECTOR_ACTIVE`).
    Active,
    /// No vector, for [`crate::EvalMode::Weight`] fields (`CEED_VECTOR_NONE`).
    None,
    /// A fixed vector, such as coordinates or stored geometric data. The operator keeps a
    /// reference: later changes of its values are seen by the operator.
    Passive(&'a CVector),
}

/// A libCEED operator (`CeedOperator`) on one QFunction.
#[derive(Debug)]
pub struct Operator {
    raw: sys::CeedOperator,
    ceed: Ceed,
}

impl Operator {
    /// An operator applying `qf` (`CeedOperatorCreate`, without Jacobian QFunctions). The
    /// operator keeps a reference to `qf`.
    pub fn new(ceed: &Ceed, qf: &QFunction) -> Result<Operator> {
        ceed.ensure_same(qf.ceed(), "QFunction")?;
        let mut raw: sys::CeedOperator = null_mut();
        chk!(ceed, unsafe {
            sys::CeedOperatorCreate(
                ceed.as_raw(),
                qf.as_raw(),
                sys::CEED_QFUNCTION_NONE,
                sys::CEED_QFUNCTION_NONE,
                &mut raw,
            )
        });
        Ok(Operator {
            raw,
            ceed: ceed.clone(),
        })
    }

    /// Sets the QFunction field `name` (`CeedOperatorSetField`): its restriction (`None` for
    /// [`crate::EvalMode::Weight`] fields), basis (`None` for [`crate::EvalMode::None`]
    /// fields) and vector. The operator keeps references to all of them.
    pub fn set_field(
        &mut self,
        name: &str,
        restriction: Option<&ElemRestriction>,
        basis: Option<&Basis>,
        vector: FieldVector<'_>,
    ) -> Result<()> {
        let c_name = c_string(name, "operator field")?;
        // SAFETY (all three): reading the handles libCEED exports for "none" and "active".
        let rstr = match restriction {
            Some(r) => {
                self.ceed.ensure_same(r.ceed(), "element restriction")?;
                r.as_raw()
            }
            None => unsafe { sys::CEED_ELEMRESTRICTION_NONE },
        };
        let basis = match basis {
            Some(b) => {
                self.ceed.ensure_same(b.ceed(), "basis")?;
                b.as_raw()
            }
            None => unsafe { sys::CEED_BASIS_NONE },
        };
        let vec = match vector {
            FieldVector::Active => unsafe { sys::CEED_VECTOR_ACTIVE },
            FieldVector::None => unsafe { sys::CEED_VECTOR_NONE },
            FieldVector::Passive(v) => {
                self.ceed.ensure_same(v.ceed(), "field vector")?;
                v.as_raw()
            }
        };
        chk!(&self.ceed, unsafe {
            sys::CeedOperatorSetField(self.raw, c_name.as_ptr(), rstr, basis, vec)
        });
        Ok(())
    }

    /// The lengths of the active input and output vectors.
    pub fn active_lengths(&self) -> Result<(usize, usize)> {
        let (mut n_in, mut n_out): (sys::CeedSize, sys::CeedSize) = (0, 0);
        chk!(&self.ceed, unsafe {
            sys::CeedOperatorGetActiveVectorLengths(self.raw, &mut n_in, &mut n_out)
        });
        let length = |n: sys::CeedSize, what: &str| {
            usize::try_from(n).map_err(|_| {
                Error::new(
                    sys::CEED_ERROR_INCOMPLETE,
                    format!("operator without an active {what} field"),
                )
            })
        };
        Ok((length(n_in, "input")?, length(n_out, "output")?))
    }

    /// An error unless `v` (named `what`) belongs to this context and has length `expected`.
    fn check_vector(&self, v: &CVector, expected: usize, what: &str) -> Result<()> {
        self.ceed.ensure_same(v.ceed(), what)?;
        if v.len() != expected {
            return Err(Error::new(
                sys::CEED_ERROR_DIMENSION,
                format!(
                    "{what} has length {}, the operator's active vector length {expected}",
                    v.len()
                ),
            ));
        }
        Ok(())
    }

    /// `output = A input` (`CeedOperatorApply`).
    pub fn apply(&self, input: &CVector, output: &mut CVector) -> Result<()> {
        let (n_in, n_out) = self.active_lengths()?;
        self.check_vector(input, n_in, "input vector")?;
        self.check_vector(output, n_out, "output vector")?;
        chk!(&self.ceed, unsafe {
            sys::CeedOperatorApply(
                self.raw,
                input.as_raw(),
                output.as_raw(),
                sys::CEED_REQUEST_IMMEDIATE,
            )
        });
        Ok(())
    }

    /// `output = A input` on arrays libCEED uses in place, without copies: vectors without
    /// arrays of their own view them (`CEED_USE_POINTER`) for the call, and give them back
    /// (`CeedVectorTakeArray`) before returning. libCEED only reads `input`.
    pub fn apply_slices(&self, input: &[f64], output: &mut [f64]) -> Result<()> {
        let (n_in, n_out) = self.active_lengths()?;
        for (what, len, expected) in [
            ("input", input.len(), n_in),
            ("output", output.len(), n_out),
        ] {
            if len != expected {
                return Err(Error::new(
                    sys::CEED_ERROR_DIMENSION,
                    format!(
                        "{what} array has length {len}, the operator's active vector length \
                         {expected}"
                    ),
                ));
            }
        }
        // The operator reads its active input and never writes it.
        let x = View::new(&self.ceed, input.as_ptr().cast_mut(), n_in)?;
        let y = View::new(&self.ceed, output.as_mut_ptr(), n_out)?;
        chk!(&self.ceed, unsafe {
            sys::CeedOperatorApply(self.raw, x.raw, y.raw, sys::CEED_REQUEST_IMMEDIATE)
        });
        x.give_back()?;
        y.give_back()
    }

    /// The diagonal of the (square) operator matrix (`CeedOperatorLinearAssembleDiagonal`).
    pub fn assemble_diagonal(&self, diagonal: &mut CVector) -> Result<()> {
        let (_, n_out) = self.active_lengths()?;
        self.check_vector(diagonal, n_out, "diagonal vector")?;
        chk!(&self.ceed, unsafe {
            sys::CeedOperatorLinearAssembleDiagonal(
                self.raw,
                diagonal.as_raw(),
                sys::CEED_REQUEST_IMMEDIATE,
            )
        });
        Ok(())
    }

    /// The operator matrix in coordinate format: rows, columns and values of its entries
    /// (`CeedOperatorLinearAssembleSymbolic`, `CeedOperatorLinearAssemble`). Pairs `(i, j)`
    /// repeat (one entry per element node pair): their values add up.
    pub fn assemble_coo(&self) -> Result<(Vec<i64>, Vec<i64>, CVector)> {
        let mut n: sys::CeedSize = 0;
        let mut rows = Allocated(null_mut());
        let mut cols = Allocated(null_mut());
        chk!(&self.ceed, unsafe {
            sys::CeedOperatorLinearAssembleSymbolic(self.raw, &mut n, &mut rows.0, &mut cols.0)
        });
        let n = usize::try_from(n).map_err(|_| {
            Error::new(
                sys::CEED_ERROR_MAJOR,
                format!("CeedOperatorLinearAssembleSymbolic gave {n} entries"),
            )
        })?;
        // SAFETY: libCEED allocated `n` indices each.
        let rows = unsafe { rows.to_vec(n) };
        let cols = unsafe { cols.to_vec(n) };
        let values = CVector::new(&self.ceed, n)?;
        chk!(&self.ceed, unsafe {
            sys::CeedOperatorLinearAssemble(self.raw, values.as_raw())
        });
        Ok((rows, cols, values))
    }

    /// The raw handle, for the libCEED API not wrapped here.
    pub fn as_raw(&self) -> sys::CeedOperator {
        self.raw
    }
}

impl Drop for Operator {
    fn drop(&mut self) {
        // Nothing to report an error to. Releases the references to the fields' objects.
        unsafe { sys::CeedOperatorDestroy(&mut self.raw) };
    }
}

/// A libCEED vector viewing an array of the caller (`CEED_USE_POINTER`). [`View::give_back`]
/// takes the array back from libCEED, so that libCEED keeps no pointer to it; on drop (after an
/// error) it is taken back too, and the vector destroyed.
struct View<'c> {
    raw: sys::CeedVector,
    ceed: &'c Ceed,
    /// Whether libCEED holds the array.
    lent: bool,
}

impl<'c> View<'c> {
    /// A vector of `len` entries viewing `array`.
    ///
    /// The caller makes sure that `array` holds `len` values valid for the life of the view, and
    /// that libCEED does not write them unless they are writable.
    fn new(ceed: &'c Ceed, array: *mut f64, len: usize) -> Result<View<'c>> {
        let mut view = View {
            raw: null_mut(),
            ceed,
            lent: false,
        };
        chk!(ceed, unsafe {
            sys::CeedVectorCreate(ceed.as_raw(), to_size(len, "vector length")?, &mut view.raw)
        });
        // SAFETY: `array` holds `len` values (the vector length) for the life of the view.
        chk!(ceed, unsafe {
            sys::CeedVectorSetArray(view.raw, sys::CEED_MEM_HOST, sys::CEED_USE_POINTER, array)
        });
        view.lent = true;
        Ok(view)
    }

    /// Takes the array back from libCEED.
    fn take(&mut self) -> Result<()> {
        if self.lent {
            self.lent = false;
            let mut array: *mut f64 = null_mut();
            chk!(self.ceed, unsafe {
                sys::CeedVectorTakeArray(self.raw, sys::CEED_MEM_HOST, &mut array)
            });
        }
        Ok(())
    }

    fn give_back(mut self) -> Result<()> {
        self.take()
    }
}

impl Drop for View<'_> {
    fn drop(&mut self) {
        // After an error: nothing to report it to. A vector which keeps the array is leaked
        // rather than destroyed, so that libCEED can never reach the array again.
        if self.take().is_ok() {
            unsafe { sys::CeedVectorDestroy(&mut self.raw) };
        }
    }
}

/// An index array libCEED allocated (with calloc) for the caller, freed on drop. libCEED's
/// `CeedFree` is internal to it and calls free().
struct Allocated(*mut sys::CeedInt);

impl Allocated {
    /// # Safety
    /// Unless null, the array holds `n` indices.
    unsafe fn to_vec(&self, n: usize) -> Vec<i64> {
        if self.0.is_null() || n == 0 {
            Vec::new()
        } else {
            unsafe { slice::from_raw_parts(self.0, n) }
                .iter()
                .map(|&i| i64::from(i))
                .collect()
        }
    }
}

impl Drop for Allocated {
    fn drop(&mut self) {
        unsafe { sys::free(self.0.cast()) };
    }
}
