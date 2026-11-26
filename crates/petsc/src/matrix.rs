//! `Mat`: shell matrices backed by Rust operators, and AIJ matrices assembled from COO.

use std::ffi::c_void;
use std::ptr::null_mut;

use crate::error::callback;
use crate::error::{check, ensure_alive};
use crate::{AsVec, AsVecMut, Comm, Error, Result, VecMut, VecRef, alive, chk, sys, to_int};

/// A linear operator applied by Rust code: the context of a shell matrix.
///
/// PETSc lends the vectors for the duration of the call ([`VecRef`], [`VecMut`]).
pub trait ShellOperator {
    /// `y = A x`.
    fn mult(&self, x: &VecRef<'_>, y: &mut VecMut<'_>) -> Result<()>;
    /// `d = diag(A)`.
    fn diagonal(&self, d: &mut VecMut<'_>) -> Result<()>;
}

/// An owned PETSc matrix (`Mat`).
#[derive(Debug)]
pub struct Matrix {
    raw: sys::Mat,
    /// The number of COO entries of an AIJ matrix from [`Matrix::aij_coo`].
    coo_len: Option<usize>,
}

type Context = Box<dyn ShellOperator>;

impl Matrix {
    /// A square shell matrix (`MATSHELL`) with `local_rows` rows (and columns) on this process,
    /// `global_rows` in all, applied by `op`. PETSc owns `op` and drops it with the matrix,
    /// which may outlive this wrapper (for example in a `KSP`).
    pub fn shell(
        comm: &Comm,
        local_rows: usize,
        global_rows: usize,
        op: Box<dyn ShellOperator>,
    ) -> Result<Matrix> {
        let m = to_int(local_rows, "local rows")?;
        let n = to_int(global_rows, "global rows")?;
        ensure_alive("MatCreateShell")?;
        // A thin pointer to the trait object, the shell context.
        let ctx: *mut Context = Box::into_raw(Box::new(op));
        let mut raw = null_mut();
        // SAFETY: `ctx` stays valid until `destroy_context` (or the cleanup below) frees it.
        // The destroy function can only be set once the context is (MatShellSetContextDestroy
        // does nothing without one).
        let created = check(
            unsafe { sys::MatCreateShell(comm.as_raw(), m, m, n, n, ctx.cast(), &mut raw) },
            "MatCreateShell",
        )
        .and_then(|()| {
            check(
                unsafe { sys::MatShellSetContextDestroy(raw, Some(destroy_context)) },
                "MatShellSetContextDestroy",
            )
        });
        let matrix = Matrix { raw, coo_len: None };
        if let Err(e) = created {
            drop(matrix);
            drop(unsafe { Box::from_raw(ctx) });
            return Err(e);
        }
        // From here PETSc owns the context.
        let mult = shell_mult as unsafe extern "C" fn(_, _, _) -> _;
        let diagonal = shell_diagonal as unsafe extern "C" fn(_, _) -> _;
        // SAFETY: the functions have the signatures PETSc casts them back to.
        chk!(unsafe { sys::MatShellSetOperation(raw, sys::MATOP_MULT, erase(mult as *const ())) });
        chk!(unsafe {
            sys::MatShellSetOperation(raw, sys::MATOP_GET_DIAGONAL, erase(diagonal as *const ()))
        });
        Ok(matrix)
    }

    /// A square AIJ matrix with `local_rows` rows (and columns) on this process, `global_rows`
    /// in all, and nonzeros at the (global) indices `coo_i[k]`, `coo_j[k]`
    /// (`MatSetPreallocationCOO`). Repeated entries are summed; negative indices are ignored.
    /// The values come from [`Matrix::set_values_coo`].
    pub fn aij_coo(
        comm: &Comm,
        local_rows: usize,
        global_rows: usize,
        coo_i: &[i64],
        coo_j: &[i64],
    ) -> Result<Matrix> {
        if coo_i.len() != coo_j.len() {
            return Err(Error::new(
                sys::PETSC_ERR_ARG_SIZ,
                format!(
                    "COO indices: {} row and {} column indices",
                    coo_i.len(),
                    coo_j.len()
                ),
            ));
        }
        let m = to_int(local_rows, "local rows")?;
        let n = to_int(global_rows, "global rows")?;
        let index = |k: &i64| {
            sys::PetscInt::try_from(*k).map_err(|_| {
                Error::new(
                    sys::PETSC_ERR_ARG_OUTOFRANGE,
                    format!("COO index {k} does not fit in PetscInt"),
                )
            })
        };
        // PETSc may modify the index arrays.
        let mut i = coo_i.iter().map(index).collect::<Result<Vec<_>>>()?;
        let mut j = coo_j.iter().map(index).collect::<Result<Vec<_>>>()?;
        let mut raw = null_mut();
        chk!(unsafe { sys::MatCreate(comm.as_raw(), &mut raw) });
        let matrix = Matrix {
            raw,
            coo_len: Some(i.len()),
        };
        chk!(unsafe { sys::MatSetSizes(raw, m, m, n, n) });
        chk!(unsafe { sys::MatSetType(raw, sys::MATAIJ.as_ptr().cast()) });
        // SAFETY: `i` and `j` hold `i.len()` indices each; PETSc copies them.
        chk!(unsafe {
            sys::MatSetPreallocationCOO(
                raw,
                i.len() as sys::PetscCount,
                i.as_mut_ptr(),
                j.as_mut_ptr(),
            )
        });
        Ok(matrix)
    }

    /// Sets the values of the COO entries of [`Matrix::aij_coo`], in their order, replacing the
    /// previous values, and assembles the matrix.
    pub fn set_values_coo(&mut self, values: &[f64]) -> Result<()> {
        match self.coo_len {
            None => Err(Error::new(
                sys::PETSC_ERR_ARG_WRONGSTATE,
                "set_values_coo needs a matrix from Matrix::aij_coo",
            )),
            Some(n) if n != values.len() => Err(Error::new(
                sys::PETSC_ERR_ARG_SIZ,
                format!("COO values: {} given, {n} expected", values.len()),
            )),
            Some(_) => {
                // SAFETY: `values` holds one value per COO entry (checked above).
                chk!(unsafe {
                    sys::MatSetValuesCOO(self.raw, values.as_ptr(), sys::INSERT_VALUES)
                });
                Ok(())
            }
        }
    }

    /// Takes ownership of one reference to `raw` (for which [`Matrix::set_values_coo`] is an
    /// error).
    ///
    /// # Safety
    /// `raw` is a valid `Mat`, and the caller gives up its reference.
    pub unsafe fn from_raw(raw: sys::Mat) -> Matrix {
        Matrix { raw, coo_len: None }
    }

    pub fn as_raw(&self) -> sys::Mat {
        self.raw
    }

    /// Marks the matrix symmetric positive definite for good, also after its values change
    /// (`MatSetOption` `MAT_SPD` and `MAT_SPD_ETERNAL`): solvers and preconditioners may rely
    /// on it, for example CG to estimate eigenvalues.
    pub fn set_spd(&mut self) -> Result<()> {
        for option in [sys::MAT_SPD, sys::MAT_SPD_ETERNAL] {
            // SAFETY: a valid matrix and option.
            chk!(unsafe { sys::MatSetOption(self.raw, option, true) });
        }
        Ok(())
    }

    /// A new matrix of the layout and the nonzero pattern of this one, its values zero
    /// (`MatDuplicate` with `MAT_DO_NOT_COPY_VALUES`). It shares the coordinate pattern of
    /// [`Matrix::aij_coo`], so [`Matrix::set_values_coo`] sets its values in the same order.
    /// Collective.
    pub fn duplicate(&self) -> Result<Matrix> {
        let mut raw = null_mut();
        // SAFETY: a valid matrix; the new one is owned by the result.
        chk!(unsafe { sys::MatDuplicate(self.raw, sys::MAT_DO_NOT_COPY_VALUES, &mut raw) });
        Ok(Matrix {
            raw,
            coo_len: self.coo_len,
        })
    }

    /// Marks the matrix changed (`MatAssemblyBegin` and `MatAssemblyEnd`), for a change PETSc
    /// cannot see, as of the coefficients of a shell operator: preconditioners built from it
    /// are rebuilt at their next use. Collective.
    pub fn mark_changed(&mut self) -> Result<()> {
        // SAFETY: a valid matrix; the final assembly of an assembled matrix keeps its values.
        chk!(unsafe { sys::MatAssemblyBegin(self.raw, sys::MAT_FINAL_ASSEMBLY) });
        chk!(unsafe { sys::MatAssemblyEnd(self.raw, sys::MAT_FINAL_ASSEMBLY) });
        Ok(())
    }

    /// `y = A x`.
    pub fn mult(&self, x: &impl AsVec, y: &mut impl AsVecMut) -> Result<()> {
        chk!(unsafe { sys::MatMult(self.raw, x.as_raw(), y.as_raw()) });
        Ok(())
    }

    /// `d = diag(A)`.
    pub fn diagonal(&self, d: &mut impl AsVecMut) -> Result<()> {
        chk!(unsafe { sys::MatGetDiagonal(self.raw, d.as_raw()) });
        Ok(())
    }
}

impl Drop for Matrix {
    fn drop(&mut self) {
        if alive() {
            crate::error::destroyed(unsafe { sys::MatDestroy(&mut self.raw) }, "MatDestroy");
        }
    }
}

/// PETSc stores operations type-erased (`PetscErrorCodeFn`) and casts them back to their real
/// type before calling them.
fn erase(f: *const ()) -> sys::PetscErrorCodeFn {
    Some(unsafe { std::mem::transmute::<*const (), unsafe extern "C" fn()>(f) })
}

/// The operator of a shell matrix from [`Matrix::shell`].
///
/// # Safety
/// `a` is a shell matrix from [`Matrix::shell`].
unsafe fn operator<'a>(a: sys::Mat) -> Result<&'a dyn ShellOperator> {
    let mut ctx: *mut c_void = null_mut();
    chk!(unsafe { sys::MatShellGetContext(a, (&mut ctx as *mut *mut c_void).cast()) });
    Ok(unsafe { &**ctx.cast::<Context>() })
}

unsafe extern "C" fn shell_mult(a: sys::Mat, x: sys::Vec, y: sys::Vec) -> sys::PetscErrorCode {
    callback(c"ShellOperator::mult", || {
        // SAFETY: PETSc calls this for shells from `Matrix::shell` only, with valid vectors
        // for the duration of the call (x != y, MatMult checks).
        let op = unsafe { operator(a) }?;
        op.mult(&unsafe { VecRef::lent(x) }, &mut unsafe { VecMut::lent(y) })
    })
}

unsafe extern "C" fn shell_diagonal(a: sys::Mat, d: sys::Vec) -> sys::PetscErrorCode {
    callback(c"ShellOperator::diagonal", || {
        // SAFETY: as in `shell_mult`.
        let op = unsafe { operator(a) }?;
        op.diagonal(&mut unsafe { VecMut::lent(d) })
    })
}

/// Drops the shell context when PETSc destroys the matrix (`PetscCtxDestroyFn`: `ctx` is the
/// address of the context pointer).
unsafe extern "C" fn destroy_context(ctx: *mut c_void) -> sys::PetscErrorCode {
    callback(c"ShellOperator::drop", || {
        let ctx = unsafe { &mut *ctx.cast::<*mut Context>() };
        if !ctx.is_null() {
            drop(unsafe { Box::from_raw(*ctx) });
            *ctx = null_mut();
        }
        Ok(())
    })
}
