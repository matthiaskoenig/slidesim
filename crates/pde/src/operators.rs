//! Operators on a [`Discretization`]: diffusion-reaction and sources at quadrature points.

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use ceed::{CVector, EvalMode, FieldVector, Operator, QFunction};
use petsc::{AsVec, AsVecMut, InsertMode, Matrix, ShellOperator, VecMut, VecRef, Vector};

use crate::discretization::Shared;
use crate::qfunctions::{self as qf, GEOMETRY_2D_SIZE};
use crate::{Discretization, PdeError, Result};

/// The operator `shift M + d K + k M` on global vectors of a [`Discretization`]: `M` is the mass
/// matrix, `K` the stiffness matrix (`d` the diffusion coefficient in um^2/s, `k` the decay rate
/// in 1/s), and `shift` the shift of an implicit time step (in 1/s).
///
/// The coefficients may be changed between applications. As a [`ShellOperator`] it is a
/// Jacobian: essential dofs are zero in its input (and absent from its output); the residual of
/// nonzero essential values comes from [`DiffusionReaction::apply_local`]. Each operator has its
/// own libCEED operator and work vectors, and shares the restrictions, basis and geometry of
/// the discretization.
#[derive(Debug)]
pub struct DiffusionReaction {
    pub d: f64,
    pub k: f64,
    pub shift: f64,
    shared: Rc<Shared>,
    qf: RefCell<QFunction>,
    op: Operator,
    /// The coefficients the QFunction context holds: `[d, shift + k]`.
    context: Cell<[f64; 2]>,
    /// Local vectors: the input of [`ShellOperator::mult`], and results.
    work: RefCell<Work>,
}

#[derive(Debug)]
struct Work {
    x: Vector,
    y: Vector,
}

impl DiffusionReaction {
    /// The operator with the coefficients `d`, `k` and `shift` on `disc`.
    pub fn new(disc: &Discretization, d: f64, k: f64, shift: f64) -> Result<DiffusionReaction> {
        let shared = disc.shared();
        let ceed = &shared.ceed;
        let dim = shared.basis.dim();
        // SAFETY: the fields and the context (two values, `DiffusionReactionContext`) match what
        // DiffusionReaction2D reads and writes.
        let mut qf = unsafe {
            QFunction::interior(
                ceed,
                1,
                Some(qf::slidesim_pde_DiffusionReaction2D),
                qf::DIFFUSION_REACTION_2D,
            )
        }?;
        qf.add_input("u", 1, EvalMode::Interp)?;
        qf.add_input("du", dim, EvalMode::Grad)?;
        qf.add_input("q_data", GEOMETRY_2D_SIZE, EvalMode::None)?;
        qf.add_output("v", 1, EvalMode::Interp)?;
        qf.add_output("dv", dim, EvalMode::Grad)?;
        let context = [d, shift + k];
        qf.set_context(&context)?;
        let mut op = Operator::new(ceed, &qf)?;
        let (r, b) = (Some(&shared.restriction), Some(&shared.basis));
        op.set_field("u", r, b, FieldVector::Active)?;
        op.set_field("du", r, b, FieldVector::Active)?;
        op.set_field(
            "q_data",
            Some(&shared.q_data_restriction),
            None,
            FieldVector::Passive(&shared.q_data),
        )?;
        op.set_field("v", r, b, FieldVector::Active)?;
        op.set_field("dv", r, b, FieldVector::Active)?;
        Ok(DiffusionReaction {
            d,
            k,
            shift,
            shared: Rc::clone(shared),
            qf: RefCell::new(qf),
            op,
            context: Cell::new(context),
            work: RefCell::new(Work {
                x: disc.local_vector()?,
                y: disc.local_vector()?,
            }),
        })
    }

    /// Passes the current coefficients to the QFunction, if they changed.
    fn update_context(&self) -> Result<()> {
        let context = [self.d, self.shift + self.k];
        if self.context.get() != context {
            self.qf.borrow_mut().set_context(&context)?;
            self.context.set(context);
        }
        Ok(())
    }

    /// `y = A x_local` into the global vector `y`, for the local vector `x_local` with its
    /// essential values (for example set by [`Discretization::set_essential_values`]): the
    /// operator applied to the function `x_local`, tested against the functions of the global
    /// dofs. Collective.
    pub fn apply_local(&self, x_local: &Vector, y: &mut impl AsVecMut) -> Result<()> {
        let mut work = self.work.borrow_mut();
        self.apply_into(x_local, &mut work.y, y)
    }

    /// `y = A x_local` through the local vector `y_local`.
    fn apply_into(
        &self,
        x_local: &Vector,
        y_local: &mut Vector,
        y: &mut impl AsVecMut,
    ) -> Result<()> {
        self.update_context()?;
        // libCEED views the arrays of the local vectors in place.
        let applied =
            x_local.with_array(|x| y_local.with_array_mut(|y| self.op.apply_slices(x, y)));
        applied???;
        y.zero()?;
        self.shared
            .dm
            .local_to_global(y_local, InsertMode::Add, y)?;
        Ok(())
    }

    /// `y = A x` for the global vectors `x` and `y`, with zero essential values. Collective.
    pub(crate) fn mult_global(&self, x: &impl AsVec, y: &mut impl AsVecMut) -> Result<()> {
        let mut work = self.work.borrow_mut();
        let Work {
            x: x_local,
            y: y_local,
        } = &mut *work;
        // Essential dofs are not in `x`: they stay zero.
        x_local.zero()?;
        self.shared.dm.global_to_local(x, x_local)?;
        self.apply_into(x_local, y_local, y)
    }

    fn diagonal_global(&self, d: &mut impl AsVecMut) -> Result<()> {
        self.update_context()?;
        let mut diagonal = CVector::new(&self.shared.ceed, self.shared.local_size)?;
        self.op.assemble_diagonal(&mut diagonal)?;
        let mut work = self.work.borrow_mut();
        diagonal.with_array(|values| work.y.with_array_mut(|y| y.copy_from_slice(values)))??;
        d.zero()?;
        self.shared
            .dm
            .local_to_global(&work.y, InsertMode::Add, d)?;
        Ok(())
    }

    /// The operator as an AIJ matrix on global vectors, assembled by libCEED in coordinate
    /// format (for coarse levels and direct solvers). Rows and columns of essential dofs are left
    /// out, like the dofs themselves; entries of rows owned by another process are sent to it.
    /// Collective.
    pub fn assemble(&self) -> Result<Matrix> {
        let (rows, cols) = self.op.assemble_coo_symbolic()?;
        let shared = &self.shared;
        let global = |local: Vec<i64>| -> Result<Vec<i64>> {
            local
                .into_iter()
                .map(|i| {
                    usize::try_from(i)
                        .ok()
                        .and_then(|i| shared.local_to_global.get(i).copied())
                        .ok_or_else(|| {
                            PdeError::Unsupported(format!(
                                "assembled entry at local index {i} of a local vector of length \
                                 {}",
                                shared.local_size
                            ))
                        })
                })
                .collect()
        };
        let (rows, cols) = (global(rows)?, global(cols)?);
        let (local_rows, global_rows) = shared.global_sizes;
        let comm = shared.dm.comm();
        let mut matrix = Matrix::aij_coo(&comm, local_rows, global_rows, &rows, &cols)?;
        self.reassemble(&mut matrix)?;
        Ok(matrix)
    }

    /// Assembles the operator with its current coefficients again into `matrix`, from
    /// [`DiffusionReaction::assemble`] of an operator on the same discretization: the entries
    /// keep their places (the coordinate pattern depends on the discretization only), their
    /// values are computed again. Collective.
    pub fn reassemble(&self, matrix: &mut Matrix) -> Result<()> {
        self.update_context()?;
        let values = self.op.assemble_coo_values()?;
        values.with_array(|values| matrix.set_values_coo(values))??;
        Ok(())
    }

    /// The discretization of the operator.
    pub fn discretization(&self) -> Discretization {
        Discretization::from_shared(Rc::clone(&self.shared))
    }
}

impl ShellOperator for DiffusionReaction {
    fn mult(&self, x: &VecRef<'_>, y: &mut VecMut<'_>) -> petsc::Result<()> {
        self.mult_global(x, y).map_err(PdeError::into_petsc)
    }

    fn diagonal(&self, d: &mut VecMut<'_>) -> petsc::Result<()> {
        self.diagonal_global(d).map_err(PdeError::into_petsc)
    }
}

/// A [`DiffusionReaction`] shared between a shell matrix (as its [`ShellOperator`]) and the code
/// which changes its coefficients while PETSc owns the matrix: an implicit time integrator
/// changes the shift each step.
#[derive(Clone, Debug)]
pub struct SharedOperator(pub Rc<RefCell<DiffusionReaction>>);

impl ShellOperator for SharedOperator {
    fn mult(&self, x: &VecRef<'_>, y: &mut VecMut<'_>) -> petsc::Result<()> {
        self.0.borrow().mult(x, y)
    }

    fn diagonal(&self, d: &mut VecMut<'_>) -> petsc::Result<()> {
        self.0.borrow().diagonal(d)
    }
}

/// The values of a field of a [`Discretization`] at its quadrature points, in their layout.
#[derive(Debug)]
pub(crate) struct PointValues {
    op: Operator,
}

impl PointValues {
    pub(crate) fn new(disc: &Discretization) -> Result<PointValues> {
        let shared = disc.shared();
        let ceed = &shared.ceed;
        // SAFETY: the fields match what Values reads and writes; no context.
        let mut qf =
            unsafe { QFunction::interior(ceed, 1, Some(qf::slidesim_pde_Values), qf::VALUES) }?;
        qf.add_input("u", 1, EvalMode::Interp)?;
        qf.add_output("u_q", 1, EvalMode::None)?;
        let mut op = Operator::new(ceed, &qf)?;
        op.set_field(
            "u",
            Some(&shared.restriction),
            Some(&shared.basis),
            FieldVector::Active,
        )?;
        op.set_field(
            "u_q",
            Some(&shared.point_restriction),
            None,
            FieldVector::Active,
        )?;
        Ok(PointValues { op })
    }

    /// The values of the function `x_local` (a local vector, with its essential values) at the
    /// quadrature points of the local cells, into `values`.
    pub(crate) fn apply(&self, x_local: &Vector, values: &mut [f64]) -> Result<()> {
        x_local.with_array(|x| self.op.apply_slices(x, values))??;
        Ok(())
    }
}

/// A field given at the quadrature points of a [`Discretization`] (a source in mM/s), and its
/// integral against the test functions, `int f v`.
///
/// The values are stored as the discretization lays out values at points: point `q` of local
/// cell `e` at `e * Q + q`, in the order of [`Discretization::quadrature_points`].
#[derive(Debug)]
pub struct SourceField {
    values: Vec<f64>,
    shared: Rc<Shared>,
    op: Operator,
    /// A local vector for results.
    work: RefCell<Vector>,
}

impl SourceField {
    /// The field on `disc`, zero at every point.
    pub fn new(disc: &Discretization) -> Result<SourceField> {
        let shared = disc.shared();
        let ceed = &shared.ceed;
        // SAFETY: the fields match what Mass reads and writes; no context.
        let mut qf =
            unsafe { QFunction::interior(ceed, 1, Some(qf::slidesim_pde_Mass), qf::MASS) }?;
        qf.add_input("q_data", GEOMETRY_2D_SIZE, EvalMode::None)?;
        qf.add_input("f", 1, EvalMode::None)?;
        qf.add_output("v", 1, EvalMode::Interp)?;
        let mut op = Operator::new(ceed, &qf)?;
        op.set_field(
            "q_data",
            Some(&shared.q_data_restriction),
            None,
            FieldVector::Passive(&shared.q_data),
        )?;
        op.set_field(
            "f",
            Some(&shared.point_restriction),
            None,
            FieldVector::Active,
        )?;
        op.set_field(
            "v",
            Some(&shared.restriction),
            Some(&shared.basis),
            FieldVector::Active,
        )?;
        Ok(SourceField {
            values: vec![0.0; disc.num_quadrature_points()],
            shared: Rc::clone(shared),
            op,
            work: RefCell::new(disc.local_vector()?),
        })
    }

    /// The values at the local quadrature points.
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// The values at the local quadrature points, for writing.
    pub fn values_mut(&mut self) -> &mut [f64] {
        &mut self.values
    }

    /// `b = int f v` for the test functions `v` of the global dofs, into the global vector `b`.
    /// Collective.
    pub fn apply(&self, b: &mut impl AsVecMut) -> Result<()> {
        let mut local = self.work.borrow_mut();
        local.with_array_mut(|y| self.op.apply_slices(&self.values, y))??;
        b.zero()?;
        self.shared
            .dm
            .local_to_global(&*local, InsertMode::Add, b)?;
        Ok(())
    }
}
