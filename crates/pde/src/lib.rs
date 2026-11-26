//! P1 finite elements on DMPlex meshes, with matrix-free libCEED operators.
//!
//! A [`Discretization`] puts a P1 `PetscFE` on one level of a [`mesh::Mesh`] and builds the
//! libCEED objects of it, as Ratel and HONEE do: element restrictions from
//! `DMPlexGetLocalOffsets`, bases from the `PetscFE` tabulation, and the quadrature geometry
//! (`w det J`, `J^-1`) stored per quadrature point. Essential (Dirichlet) boundaries are
//! constraints of its section: global vectors leave their dofs out. Its operators work on
//! PETSc vectors, with libCEED viewing the arrays of local vectors in place:
//! - [`Transport`]: mass, reaction, advection and tensor diffusion with coefficients given at the
//!   quadrature points ([`PointData`]), stabilized by SUPG, as a [`petsc::ShellOperator`], with
//!   its diagonal and its assembled matrix;
//! - [`SourceField`]: values at the quadrature points, integrated against the test functions
//!   (or the SUPG test functions of a [`Transport`]).
//!
//! Coefficients of a study, numbers or tables over maps, become values at the quadrature points
//! of a discretization through [`coefficient_values`].
//!
//! [`configure_multigrid`] makes a linear solver geometric multigrid over the levels of the mesh
//! ([`Discretization::hierarchy`]), with the operator discretized again on every level.
//!
//! A [`Problem`] steps one field of a study in time (BDF2, PETSc `TS`) on the finest level,
//! preconditioned by that multigrid, with the coefficients of the field (and its porosity) at
//! the quadrature points of every level: diffusion and reaction, and advection by the velocity
//! of a [`Flow`], stabilized by SUPG. Its [`MassBalance`] comes from its discrete equations.
//!
//! A [`Flow`] solves the steady Darcy perfusion of a study once: the pressure with an
//! anisotropic permeability along the gradient of a map, the equilibrated fluxes through the
//! edges and their conservative (Raviart-Thomas) velocity at the quadrature points of every
//! level, and the flows through the pressure boundaries.
//!
//! The QFunctions are C headers in `qfunctions/` ([`QFUNCTION_DIR`]), compiled by `build.rs`
//! for the CPU backends; backends which compile them at run time read them from there.
//!
//! Supported: 2D triangle meshes, degree 1. Every constructor is collective on the
//! communicator of the mesh.

mod balance;
mod coefficients;
mod discretization;
mod flow;
mod fluxes;
mod multigrid;
mod operators;
mod problem;

pub use balance::{BoundaryInflow, MassBalance, UNTAGGED};
pub use coefficients::{PointData, coefficient_values};
pub use discretization::Discretization;
pub use flow::Flow;
pub use multigrid::{Multigrid, configure_multigrid};
pub use operators::{SharedOperator, SourceField, Terms, Transport};
pub use problem::{Inputs, Problem};

/// The absolute path of the directory of the QFunction sources.
pub const QFUNCTION_DIR: &str = env!("SLIDESIM_PDE_QFUNCTIONS");

#[derive(Debug, thiserror::Error)]
pub enum PdeError {
    /// A discretization this crate does not provide (yet).
    #[error("unsupported: {0}")]
    Unsupported(String),
    /// A cell whose Jacobian determinant is not positive at a quadrature point: inverted
    /// (clockwise) or degenerate. `cell` is its global number on the level, `det` the
    /// determinant against PETSc's reference cell. The same error on every process.
    #[error("inverted element: cell {cell} has Jacobian determinant {det} (not positive)")]
    InvertedElement { cell: i64, det: f64 },
    #[error("mesh has no level {level} (it has {levels})")]
    NoLevel { level: usize, levels: usize },
    #[error("tag '{tag}' is not an essential boundary of the discretization; essential: {}",
        .essential.join(", "))]
    NotEssential { tag: String, essential: Vec<String> },
    /// Operators which are not one per level of the mesh, on the discretizations of
    /// [`Discretization::hierarchy`], for [`configure_multigrid`].
    #[error("not a multigrid hierarchy: {0}")]
    NotHierarchy(String),
    /// A coefficient which cannot be evaluated from its map at a quadrature point: outside the
    /// raster, or on a no-value pixel. The message names the map, the point and the area tag
    /// of its cell, and counts the points which fail. The same error on every process.
    #[error("{message}")]
    Map { message: String },
    /// The time integration failed: a solver diverged or a step failed. `reason` is the PETSc
    /// converged reason of the innermost solver which failed (for example `KSP DIVERGED_ITS`),
    /// `t` the time reached (in s), at the last step which succeeded.
    #[error("time integration failed at t = {t} s: {reason}")]
    Diverged { reason: String, t: f64 },
    /// A steady solve which did not converge: `solve` names it (for example `perfusion`),
    /// `reason` is the PETSc converged reason of its solver (for example `KSP DIVERGED_ITS`),
    /// with where it failed. The same error on every process.
    #[error("{solve} solve failed: {reason}")]
    SolveFailed { solve: String, reason: String },
    /// A time to step to which is before the current time, or not finite.
    #[error("cannot step to t = {t} s from t = {time} s")]
    InvalidTime { t: f64, time: f64 },
    /// A tag through which the flow enters the domain (net inflow above 1e-10 of the largest
    /// net flow through a tag, [`Flow::tag_flows`]), without a dirichlet condition of a field
    /// with advection. The same error on every process.
    #[error(
        "tag '{tag}' has inflow; field '{field}' with advection needs a dirichlet condition there"
    )]
    Inflow { tag: String, field: String },
    /// A quantity of a [`Problem`] which needs a time step (the rate of change, for
    /// [`Problem::mass_balance`]), asked for at the time `t` (in s) before the first step, or
    /// before the first step after a new initial condition.
    #[error("no time step since t = {t} s: the rate of change is known at the end of a step")]
    NotStepped { t: f64 },
    /// A [`Problem`] whose time integration failed earlier (`failure`, at the time `t` in s): its
    /// solvers may be left in any state, so it does not step again.
    #[error("the problem failed at t = {t} s and cannot step again: {failure}")]
    Poisoned { failure: String, t: f64 },
    #[error(transparent)]
    Mesh(#[from] mesh::MeshError),
    #[error(transparent)]
    Petsc(#[from] petsc::Error),
    #[error(transparent)]
    Ceed(#[from] ceed::Error),
}

pub type Result<T, E = PdeError> = std::result::Result<T, E>;

impl PdeError {
    /// The error as a PETSc error, for callbacks called by PETSc.
    fn into_petsc(self) -> petsc::Error {
        match self {
            PdeError::Petsc(e) => e,
            e => petsc::Error {
                code: petsc::sys::PETSC_ERR_LIB,
                message: e.to_string(),
            },
        }
    }
}

/// The QFunctions of `qfunctions/qfunctions.c` and their sources.
mod qfunctions {
    use std::ffi::{c_int, c_void};

    use ceed::sys::{CeedInt, CeedScalar};

    /// Declares the exported QFunctions (`int name(void *ctx, CeedInt Q, const CeedScalar *const
    /// *in, CeedScalar *const *out)`).
    macro_rules! qfunctions {
        ($($name:ident),*) => {
            unsafe extern "C" {
                $(
                    pub fn $name(
                        ctx: *mut c_void,
                        q: CeedInt,
                        input: *const *const CeedScalar,
                        output: *const *mut CeedScalar,
                    ) -> c_int;
                )*
            }
        };
    }

    qfunctions!(
        slidesim_pde_SetupGeometry2D,
        slidesim_pde_QuadraturePoints2D,
        slidesim_pde_Mass,
        slidesim_pde_Values,
        slidesim_pde_Transport2D,
        slidesim_pde_TransportSource2D,
        slidesim_pde_Velocity2D
    );

    /// The source of a QFunction, `/abs/dir/file.h:name`.
    macro_rules! source {
        ($file_name:literal) => {
            concat!(env!("SLIDESIM_PDE_QFUNCTIONS"), "/", $file_name)
        };
    }

    pub const SETUP_GEOMETRY_2D: &str = source!("geometry.h:SetupGeometry2D");
    pub const QUADRATURE_POINTS_2D: &str = source!("geometry.h:QuadraturePoints2D");
    pub const MASS: &str = source!("mass.h:Mass");
    pub const VALUES: &str = source!("mass.h:Values");
    pub const TRANSPORT_2D: &str = source!("transport.h:Transport2D");
    pub const TRANSPORT_SOURCE_2D: &str = source!("transport.h:TransportSource2D");
    pub const VELOCITY_2D: &str = source!("velocity.h:Velocity2D");

    /// The number of components of the quadrature data (`GEOMETRY_2D_SIZE` of `geometry.h`).
    pub const GEOMETRY_2D_SIZE: usize = 5;

    /// The number of components of the point data of `Transport2D` (`TRANSPORT_DATA_SIZE` of
    /// `transport.h`).
    pub const TRANSPORT_DATA_SIZE: usize = 8;
}
