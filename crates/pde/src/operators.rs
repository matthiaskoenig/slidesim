//! Operators on a [`Discretization`]: transport (mass, reaction, advection and diffusion with
//! coefficients at the quadrature points, stabilized by SUPG) and sources at quadrature points.

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use ceed::{CVector, EvalMode, FieldVector, Operator, QFunction};
use petsc::{AsVec, AsVecMut, InsertMode, Matrix, ShellOperator, VecMut, VecRef, Vector};

use crate::discretization::Shared;
use crate::qfunctions::{self as qf, GEOMETRY_2D_SIZE, TRANSPORT_DATA_SIZE};
use crate::{Discretization, PdeError, PointData, Result};

/// The terms of the transport equation `m u_t + r u + vel . grad u - div(D grad u) = f` a
/// [`Transport`] operator applies (`m`, `r`, `D`, the velocity `vel` and the SUPG parameter
/// `tau` from its [`PointData`]), each tested with the SUPG test function
/// `v + tau vel . grad v` except diffusion, which is tested with `v`: the residual SUPG weighs
/// leaves out `-div(D grad u)`, which for P1 vanishes in each cell where `D` is constant (the
/// usual practice for linear elements; it makes the method inconsistent by `tau D lap u`, an
/// error of order `h^2` where diffusion dominates). Without velocity (or with `tau` zero) this
/// is the Galerkin form.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Terms {
    /// The steady terms, `R + A + K_D`: `R` the mass matrix weighted with `r`, `A` advection
    /// (`int (vel . grad u) v`), `K_D` the stiffness matrix of `D`, with the SUPG terms
    /// `int tau (vel . grad v) (r u + vel . grad u)`.
    Steady,
    /// The mass term, `M_m`: the mass matrix weighted with `m`, with the SUPG term
    /// `int tau (vel . grad v) m u`.
    Mass,
    /// The Jacobian of an implicit time step, `shift M_m + R + A + K_D` (SUPG terms included).
    Jacobian,
}

impl Terms {
    /// The value of `TransportContext.terms` in `transport.h`.
    fn context(self) -> f64 {
        match self {
            Terms::Steady => 0.0,
            Terms::Mass => 1.0,
            Terms::Jacobian => 2.0,
        }
    }
}

/// The transport operator on global vectors of a [`Discretization`], with the coefficients of a
/// [`PointData`] at its quadrature points: of the weak form of
/// `shift m u + r u + vel . grad u - div(D grad u)` (stabilized by SUPG, see [`Terms`]), the
/// terms [`Terms`] selects. With a velocity the operator is not symmetric. `shift` is the shift
/// of an implicit time step (in 1/s), which [`Terms::Jacobian`] applies.
///
/// The coefficients are fixed; the shift may be changed between applications. As a
/// [`ShellOperator`] it is a Jacobian: essential dofs are zero in its input (and absent from its
/// output); the residual of nonzero essential values comes from [`Transport::apply_local`].
/// Each operator has its own libCEED operator and work vectors, and shares the restrictions,
/// basis and geometry of the discretization, and its point data with other operators of it.
#[derive(Debug)]
pub struct Transport {
    pub shift: f64,
    terms: Terms,
    data: Rc<PointData>,
    shared: Rc<Shared>,
    qf: RefCell<QFunction>,
    op: Operator,
    /// The context the QFunction holds: `[shift, terms]`.
    context: Cell<[f64; 2]>,
    /// Of [`Terms::Jacobian`], once assembled: the values of the coordinate entries of libCEED
    /// of the steady terms and of the mass term, of which the values at a shift `s` are
    /// `steady + s mass` (the Jacobian is affine in the shift).
    affine: RefCell<Option<Affine>>,
    /// Local vectors: the input of [`ShellOperator::mult`], and results.
    work: RefCell<Work>,
}

/// The values of the coordinate entries of a Jacobian, by term.
#[derive(Debug)]
struct Affine {
    steady: Vec<f64>,
    mass: Vec<f64>,
}

#[derive(Debug)]
struct Work {
    x: Vector,
    y: Vector,
}

impl Transport {
    /// The operator of the terms `terms` with the coefficients `data` on `disc`, with shift 0.
    ///
    /// Errors: [`PdeError::Unsupported`] unless `data` has a value per quadrature point of
    /// `disc`.
    pub fn new(disc: &Discretization, data: Rc<PointData>, terms: Terms) -> Result<Transport> {
        let n = disc.num_quadrature_points();
        if data.len() != n {
            return Err(PdeError::Unsupported(format!(
                "point data of {} points on a discretization of {n} quadrature points",
                data.len()
            )));
        }
        let shared = disc.shared();
        let ceed = &shared.ceed;
        let dim = shared.basis.dim();
        // SAFETY: the fields and the context (two values, `TransportContext`) match what
        // Transport2D reads and writes.
        let mut qf = unsafe {
            QFunction::interior(
                ceed,
                1,
                Some(qf::slidesim_pde_Transport2D),
                qf::TRANSPORT_2D,
            )
        }?;
        qf.add_input("u", 1, EvalMode::Interp)?;
        qf.add_input("du", dim, EvalMode::Grad)?;
        qf.add_input("q_data", GEOMETRY_2D_SIZE, EvalMode::None)?;
        qf.add_input("p_data", TRANSPORT_DATA_SIZE, EvalMode::None)?;
        qf.add_output("v", 1, EvalMode::Interp)?;
        qf.add_output("dv", dim, EvalMode::Grad)?;
        let shift = 0.0;
        let context = [shift, terms.context()];
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
        data.with_packed(ceed, disc.points_per_cell(), |p_data| {
            // libCEED keeps a reference to the vector.
            op.set_field(
                "p_data",
                Some(&shared.p_data_restriction),
                None,
                FieldVector::Passive(p_data),
            )?;
            Ok(())
        })?;
        op.set_field("v", r, b, FieldVector::Active)?;
        op.set_field("dv", r, b, FieldVector::Active)?;
        Ok(Transport {
            shift,
            terms,
            data,
            shared: Rc::clone(shared),
            qf: RefCell::new(qf),
            op,
            context: Cell::new(context),
            affine: RefCell::new(None),
            work: RefCell::new(Work {
                x: disc.local_vector()?,
                y: disc.local_vector()?,
            }),
        })
    }

    /// The terms the operator applies.
    pub fn terms(&self) -> Terms {
        self.terms
    }

    /// The coefficients at the quadrature points.
    pub fn data(&self) -> &Rc<PointData> {
        &self.data
    }

    /// The entries of the backflow term of the data ([`PointData::with_backflow`]), which the
    /// steady terms and the Jacobian apply (the mass term does not).
    fn backflow(&self) -> &[(usize, usize, f64)] {
        match self.terms {
            Terms::Mass => &[],
            Terms::Steady | Terms::Jacobian => self.data.backflow(),
        }
    }

    /// Passes the current shift to the QFunction, if it changed.
    fn update_context(&self) -> Result<()> {
        let context = [self.shift, self.terms.context()];
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
        let applied = x_local.with_array(|x| {
            y_local.with_array_mut(|y| {
                self.op.apply_slices(x, y)?;
                for &(i, j, value) in self.backflow() {
                    y[i] += value * x[j];
                }
                Ok::<_, ceed::Error>(())
            })
        });
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
        diagonal.with_array(|values| {
            work.y.with_array_mut(|y| {
                y.copy_from_slice(values);
                for &(i, j, value) in self.backflow() {
                    if i == j {
                        y[i] += value;
                    }
                }
            })
        })??;
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
        let (mut rows, mut cols) = self.op.assemble_coo_symbolic()?;
        // The backflow term after the entries of libCEED, in the order of `reassemble`.
        for &(i, j, _) in self.backflow() {
            rows.push(i as i64);
            cols.push(j as i64);
        }
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

    /// Assembles the operator with its current shift again into `matrix`, from
    /// [`Transport::assemble`] of this operator or of one on the same discretization with the
    /// same terms and the same backflow entries ([`PointData::with_backflow`]): the coordinate
    /// pattern is that of libCEED for the discretization followed by the backflow entries, so
    /// the entries keep their places and only their values are computed again, with the shift
    /// of the time of the call. The Jacobian ([`Terms::Jacobian`]) assembles its steady and
    /// mass terms once, at its first assembly, and combines them for each shift (equal to its
    /// assembly at the shift up to rounding): the assembly of libCEED costs several applications
    /// of the operator, the combination one pass over the entries. Collective.
    pub fn reassemble(&self, matrix: &mut Matrix) -> Result<()> {
        let mut values = match self.terms {
            Terms::Jacobian => {
                let mut affine = self.affine.borrow_mut();
                if affine.is_none() {
                    // At shift 0 the Jacobian is the steady terms.
                    *affine = Some(Affine {
                        steady: self.coo_values([0.0, Terms::Jacobian.context()])?,
                        mass: self.coo_values([1.0, Terms::Mass.context()])?,
                    });
                }
                let affine = affine.as_ref().expect("assembled above");
                affine
                    .steady
                    .iter()
                    .zip(&affine.mass)
                    .map(|(s, m)| s + self.shift * m)
                    .collect()
            }
            Terms::Steady | Terms::Mass => {
                self.update_context()?;
                self.op.assemble_coo_values()?.with_array(<[f64]>::to_vec)?
            }
        };
        values.extend(self.backflow().iter().map(|&(_, _, value)| value));
        matrix.set_values_coo(&values)?;
        Ok(())
    }

    /// The values of the coordinate entries of libCEED of the operator with the context
    /// `context` (`[shift, terms]`) instead of its own, which it sets again at its next use.
    fn coo_values(&self, context: [f64; 2]) -> Result<Vec<f64>> {
        self.qf.borrow_mut().set_context(&context)?;
        self.context.set(context);
        Ok(self.op.assemble_coo_values()?.with_array(<[f64]>::to_vec)?)
    }

    /// The discretization of the operator.
    pub fn discretization(&self) -> Discretization {
        Discretization::from_shared(Rc::clone(&self.shared))
    }
}

impl ShellOperator for Transport {
    fn mult(&self, x: &VecRef<'_>, y: &mut VecMut<'_>) -> petsc::Result<()> {
        self.mult_global(x, y).map_err(PdeError::into_petsc)
    }

    fn diagonal(&self, d: &mut VecMut<'_>) -> petsc::Result<()> {
        self.diagonal_global(d).map_err(PdeError::into_petsc)
    }
}

/// A [`Transport`] shared between a shell matrix (as its [`ShellOperator`]) and the code which
/// changes its shift while PETSc owns the matrix: an implicit time integrator changes the shift
/// each step.
#[derive(Clone, Debug)]
pub struct SharedOperator(pub Rc<RefCell<Transport>>);

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
/// integral against the test functions, `int f v`; or, stabilized
/// ([`SourceField::stabilized`]), against the SUPG test functions of a [`Transport`] operator,
/// `int f (v + tau vel . grad v)`.
///
/// The values are stored as the discretization lays out values at points: point `q` of local
/// cell `e` at `e * Q + q`, in the order of [`Discretization::quadrature_points`].
#[derive(Debug)]
pub struct SourceField {
    values: Vec<f64>,
    shared: Rc<Shared>,
    op: Operator,
    /// The point data of the SUPG test functions, whose packed values the operator reads.
    _data: Option<Rc<PointData>>,
    /// A local vector for results.
    work: RefCell<Vector>,
}

impl SourceField {
    /// The field on `disc`, zero at every point, integrated against the test functions `v`.
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
        SourceField::with_operator(disc, op, None)
    }

    /// The field on `disc`, zero at every point, integrated against the SUPG test functions
    /// `v + tau vel . grad v` of the velocity and `tau` of `data` (see [`Terms`]): the source
    /// of the equation of a [`Transport`] operator with `data`.
    ///
    /// Errors: [`PdeError::Unsupported`] unless `data` has a value per quadrature point of
    /// `disc`.
    pub fn stabilized(disc: &Discretization, data: Rc<PointData>) -> Result<SourceField> {
        let n = disc.num_quadrature_points();
        if data.len() != n {
            return Err(PdeError::Unsupported(format!(
                "point data of {} points on a discretization of {n} quadrature points",
                data.len()
            )));
        }
        let shared = disc.shared();
        let ceed = &shared.ceed;
        let dim = shared.basis.dim();
        // SAFETY: the fields match what TransportSource2D reads and writes; no context.
        let mut qf = unsafe {
            QFunction::interior(
                ceed,
                1,
                Some(qf::slidesim_pde_TransportSource2D),
                qf::TRANSPORT_SOURCE_2D,
            )
        }?;
        qf.add_input("q_data", GEOMETRY_2D_SIZE, EvalMode::None)?;
        qf.add_input("p_data", TRANSPORT_DATA_SIZE, EvalMode::None)?;
        qf.add_input("f", 1, EvalMode::None)?;
        qf.add_output("v", 1, EvalMode::Interp)?;
        qf.add_output("dv", dim, EvalMode::Grad)?;
        let mut op = Operator::new(ceed, &qf)?;
        op.set_field(
            "q_data",
            Some(&shared.q_data_restriction),
            None,
            FieldVector::Passive(&shared.q_data),
        )?;
        data.with_packed(ceed, disc.points_per_cell(), |p_data| {
            // libCEED keeps a reference to the vector.
            op.set_field(
                "p_data",
                Some(&shared.p_data_restriction),
                None,
                FieldVector::Passive(p_data),
            )?;
            Ok(())
        })?;
        op.set_field(
            "f",
            Some(&shared.point_restriction),
            None,
            FieldVector::Active,
        )?;
        let (r, b) = (Some(&shared.restriction), Some(&shared.basis));
        op.set_field("v", r, b, FieldVector::Active)?;
        op.set_field("dv", r, b, FieldVector::Active)?;
        SourceField::with_operator(disc, op, Some(data))
    }

    fn with_operator(
        disc: &Discretization,
        op: Operator,
        data: Option<Rc<PointData>>,
    ) -> Result<SourceField> {
        Ok(SourceField {
            values: vec![0.0; disc.num_quadrature_points()],
            shared: Rc::clone(disc.shared()),
            op,
            _data: data,
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

    /// `b = int f v` for the test functions `v` of the global dofs (the SUPG test functions if
    /// stabilized), into the global vector `b`. Collective.
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
