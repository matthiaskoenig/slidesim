//! Cell models on the coarse elements of a mesh: one instance of an SBML model per element
//! ("cell"), coupled to the fields of a study by operator splitting.
//!
//! - [`CellSetup`] chooses the cells (the coarse elements in the `areas` of a [`CellsSpec`]) and
//!   gives each its parameters: every [`contracts::CellParameter`] as the mean of its coefficient
//!   over the element's tissue (the quadrature points of its finest children in the areas), the
//!   replacement rule applied, and the volumes of the blood and of the cells from the porosity.
//! - A [`CellSolver`] advances the cells over an interval, exchanging the linked species with
//!   the blood (`pde::Exchange` gives the means of the fields and takes the changes back):
//!   [`EngineCells`] runs one sbmlsim-core engine instance ([`sbmlsim_engine::Stepper`]) per
//!   cell, on the model compiled to native code by sbmlsim-core's C target, without
//!   WebAssembly; [`BatchedCells`] runs the cells of a process as one libCEED operator of the
//!   QFunction of the model (sbmlsim-core's QFunction target, a Rosenbrock integrator per
//!   point), the fast path, ready for GPUs. [`Cells`] is the path the spec chooses
//!   ([`contracts::CellsSpec::solver`]).
//! - Both need native code: the C target compiles the model with a C compiler at the start of a
//!   run. They are the feature `native` (on by default). Without it, in the browser build of
//!   slidesim (wasm32-wasip1, `crates/slidesim-wasm`), the cells run on the batched kernel of
//!   sbmlsim-core (`kernel/1`, [`KernelLayout`]: the model and RODAS4 in a WebAssembly module
//!   which the host compiles and links), `KernelCells`, for `"solver": "batched"`; `"engine"`
//!   is refused ([`CellsError::EngineNeedsNative`]).
//!
//! Units: time in s (the model's time unit is `CellsSpec::time_unit_s` s), concentrations in
//! the model's units (mM for the fields of slidesim), lengths in um. The model keeps its own
//! volume units: only the ratio of the blood and cell volumes matters.
//!
//! Processes: each process holds the cells of its coarse elements. Errors one process finds
//! (a map without value in a cell's tissue, a cell model which fails) are errors on every
//! process, the same on each ([`CellsError::Element`] names the element by its element id,
//! the same on any number of processes, and its centroid), so that no process waits for
//! another which left a collective call. A cell's step control is its own: it depends on no
//! other cell and not on the number of processes. With [`CellsSpec::balance`] the processes
//! share the work of each advance by the step attempts of their cells in the advance before:
//! a process advances some cells of others and sends some of its own, whose results come back,
//! the same as where they were (`balance`). The first cell which fails is the first in the
//! order of the processes and their cells, wherever it was advanced.

#[cfg(feature = "native")]
mod balance;
#[cfg(feature = "native")]
mod batched;
#[cfg(feature = "native")]
mod engine;
mod kernel;
mod model;
mod parameters;

use ceed::Ceed;
use contracts::{CellSolverKind, CellsSpec};
use petsc::Comm;
use petsc::sys::PetscInt;

#[cfg(feature = "native")]
pub use batched::BatchedCells;
#[cfg(feature = "native")]
pub use engine::EngineCells;
#[cfg(not(feature = "native"))]
pub use kernel::KernelCells;
pub use kernel::KernelLayout;
pub use model::CellModel;
pub use parameters::CellSetup;

/// A solver of the cells of a process: advances every cell over an interval, exchanging the
/// species linked to fields ([`contracts::CellsSpec::species`]) with the blood.
pub trait CellSolver {
    /// The number of cells of this process.
    fn n_cells(&self) -> usize;

    /// The model ids of the species exchanged with the blood, in the order of
    /// [`contracts::CellsSpec::species`].
    fn species(&self) -> &[String];

    /// The ids of the states of the model, in the order of [`CellSolver::states`].
    fn state_ids(&self) -> &[String];

    /// Advances every cell from `t0` to `t1` (s). `blood` holds per cell the concentrations of
    /// the linked species in the blood (`n_cells * species().len()` values, cell by cell, in the
    /// order of [`CellSolver::species`]), which are the cell's values at `t0`; it is replaced by
    /// their values at `t1`. Collective.
    ///
    /// Errors (on every process, the same): [`CellsError::Element`] naming the first element
    /// whose model fails, on the lowest process with one. After an error the cells are not
    /// advanced again: every later call returns the same error and leaves the cells as they are.
    fn advance(&mut self, t0: f64, t1: f64, blood: &mut [f64]) -> Result<CellStats>;

    /// The states of every cell (`n_cells * state_ids().len()` values, cell by cell), as the
    /// model stores them (a species of a concentration as its amount, in the model's units).
    fn states(&self) -> &[f64];

    /// The values [`CellSolver::outputs`] writes per cell: every state of the model, in its
    /// order, then the outputs of the spec ([`contracts::CellsSpec::outputs`]) which are not
    /// states (targets of assignment rules), in their order.
    fn output_values(&self) -> &[OutputValue];

    /// The values of [`CellSolver::output_values`] of every cell at the time the cells reached
    /// (`n_cells * output_values().len()` values, cell by cell), into `out`: a species whose
    /// unit is a concentration as its concentration (its amount over the volume of its
    /// compartment, in the model's units), any other value as the model has it; the targets of
    /// assignment rules evaluated from the state of the cell.
    fn outputs(&self, out: &mut Vec<f64>);
}

/// The cells of a process on the path of their spec ([`CellsSpec::solver`]), boxed.
#[derive(Debug)]
pub enum Cells {
    /// [`CellSolverKind::Engine`].
    #[cfg(feature = "native")]
    Engine(Box<EngineCells>),
    /// [`CellSolverKind::Batched`].
    #[cfg(feature = "native")]
    Batched(Box<BatchedCells>),
    /// [`CellSolverKind::Batched`] without native code: on the batched kernel of sbmlsim-core.
    #[cfg(not(feature = "native"))]
    Kernel(Box<KernelCells>),
}

impl Cells {
    /// The cells of `setup` with the model `model`, compiled and checked for `spec`
    /// ([`CellModel::new`]), on the path of `spec`: [`EngineCells::with_model`] or
    /// [`BatchedCells::with_model`] (on `ceed`). Collective.
    #[cfg(feature = "native")]
    pub fn new(
        setup: &CellSetup,
        spec: &CellsSpec,
        model: &CellModel,
        ceed: &Ceed,
    ) -> Result<Cells> {
        Ok(match spec.solver {
            CellSolverKind::Engine => {
                Cells::Engine(Box::new(EngineCells::with_model(setup, spec, model)?))
            }
            CellSolverKind::Batched => Cells::Batched(Box::new(BatchedCells::with_model(
                setup, spec, model, ceed,
            )?)),
        })
    }

    /// The cells of `setup` with the model `model`, the kernel checked for `spec`
    /// ([`CellModel::from_kernel`]), on the batched kernel (`KernelCells::with_model`); `ceed`
    /// is not used. Collective.
    ///
    /// Errors: [`CellsError::EngineNeedsNative`] for the engine path, and those of
    /// `KernelCells::with_model`.
    #[cfg(not(feature = "native"))]
    pub fn new(
        setup: &CellSetup,
        spec: &CellsSpec,
        model: &CellModel,
        _ceed: &Ceed,
    ) -> Result<Cells> {
        match spec.solver {
            CellSolverKind::Engine => Err(CellsError::EngineNeedsNative),
            CellSolverKind::Batched => Ok(Cells::Kernel(Box::new(KernelCells::with_model(
                setup, spec, model,
            )?))),
        }
    }

    fn solver(&self) -> &dyn CellSolver {
        match self {
            #[cfg(feature = "native")]
            Cells::Engine(cells) => cells.as_ref(),
            #[cfg(feature = "native")]
            Cells::Batched(cells) => cells.as_ref(),
            #[cfg(not(feature = "native"))]
            Cells::Kernel(cells) => cells.as_ref(),
        }
    }
}

impl CellSolver for Cells {
    fn n_cells(&self) -> usize {
        self.solver().n_cells()
    }

    fn species(&self) -> &[String] {
        self.solver().species()
    }

    fn state_ids(&self) -> &[String] {
        self.solver().state_ids()
    }

    fn advance(&mut self, t0: f64, t1: f64, blood: &mut [f64]) -> Result<CellStats> {
        match self {
            #[cfg(feature = "native")]
            Cells::Engine(cells) => cells.advance(t0, t1, blood),
            #[cfg(feature = "native")]
            Cells::Batched(cells) => cells.advance(t0, t1, blood),
            #[cfg(not(feature = "native"))]
            Cells::Kernel(cells) => cells.advance(t0, t1, blood),
        }
    }

    fn states(&self) -> &[f64] {
        self.solver().states()
    }

    fn output_values(&self) -> &[OutputValue] {
        self.solver().output_values()
    }

    fn outputs(&self, out: &mut Vec<f64>) {
        self.solver().outputs(out);
    }
}

/// The smallest step of a cell on the batched paths, in the time unit of its model: a cell which
/// needs a smaller one fails.
pub(crate) const H_MIN: f64 = 1e-10;

/// The most step attempts of a cell in one advance on the batched paths: a cell which has not
/// reached the end of the advance after as many fails.
pub(crate) const MAX_ATTEMPTS: u32 = 1_000_000;

/// The message of a cell of a batched path which did not reach `t1` (s) from `t0` in
/// [`MAX_ATTEMPTS`] step attempts, at `reached` (s).
pub(crate) fn unfinished_message(t0: f64, t1: f64, reached: f64) -> String {
    format!(
        "the cell model did not reach t = {t1} s from t = {t0} s in {MAX_ATTEMPTS} step attempts \
         (it is at t = {reached} s)"
    )
}

/// The message of a cell of a batched path which failed from `t0` to `t1` (s) at `reached`.
pub(crate) fn failed_message(t0: f64, t1: f64, reached: f64) -> String {
    format!(
        "the cell model failed from t = {t0} s to t = {t1} s at t = {reached} s: its rejected \
         steps (an error above the tolerances, values which are not finite or a singular \
         Jacobian) made its step smaller than {H_MIN} (in the time unit of the model), or too \
         small to advance the time"
    )
}

/// A value of the cells which [`CellSolver::outputs`] writes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OutputValue {
    /// Its id in the model.
    pub id: String,
    /// For a species written as a concentration, its compartment; `None` for other values.
    pub compartment: Option<String>,
}

/// The work of one [`CellSolver::advance`] on this process: of the cells it advanced (with
/// [`CellsSpec::balance`], some of other processes and not all of its own).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct CellStats {
    /// The steps of the integrators of the cells, summed.
    pub steps: u64,
    /// The rejected step attempts (failed error tests and nonlinear solves), summed.
    pub rejected: u64,
    /// The wall time of the integration of the cells (s), without sending cells and results to
    /// and from other processes.
    pub seconds: f64,
}

#[derive(Debug, thiserror::Error)]
pub enum CellsError {
    /// An area of `cells.areas` which is not an area tag of the mesh.
    #[error("cells.areas: the mesh has no area tag '{area}'; area tags: {}", .available.join(", "))]
    UnknownArea {
        area: String,
        available: Vec<String>,
    },
    /// An id of the study which the model does not have as it is used: `what` is the entry of
    /// the study (for example `cells.species`), `expected` what the id must be, `available`
    /// the ids of the model which are.
    #[error("{what} '{id}': the model has no {expected} '{id}'; it has: {}",
        .available.join(", "))]
    UnknownId {
        what: String,
        id: String,
        expected: String,
        available: Vec<String>,
    },
    /// A tolerance of the cell solver which is not positive.
    #[error("cells.{what} must be > 0, got {value}")]
    Tolerance { what: &'static str, value: f64 },
    /// The model cannot be translated, compiled or loaded. The same on every process.
    #[error("cell model: {message}")]
    Model { message: String },
    /// The engine path in a build without native code (the browser build, which runs the
    /// batched path on the kernel of sbmlsim-core).
    #[error(r#"cells.solver "engine" needs native code; the browser build runs "batched""#)]
    EngineNeedsNative,
    /// A model the batched path does not integrate: `feature` names what it has. The same on
    /// every process.
    #[error(
        "cell model: cells.solver \"batched\" does not integrate {feature}; the engine path \
         does (\"solver\": \"engine\")"
    )]
    Unsupported { feature: String },
    /// A failure at one element: `element` is its element id (`mesh::Mesh::element_ids`, the
    /// same on any number of processes), `centroid` its centroid (um). The same on every
    /// process.
    #[error("element {element} at ({}, {}) um: {message}", .centroid[0], .centroid[1])]
    Element {
        element: u64,
        centroid: [f64; 2],
        message: String,
    },
    /// Inputs which do not fit together: a setup whose lists differ in length, a map the
    /// study names but the maps lack, blood values not one per cell and species.
    #[error("cells: {0}")]
    Invalid(String),
    #[error(transparent)]
    Pde(#[from] pde::PdeError),
    #[error(transparent)]
    Petsc(#[from] petsc::Error),
    #[error(transparent)]
    Ceed(#[from] ceed::Error),
}

pub type Result<T, E = CellsError> = std::result::Result<T, E>;

/// The first error of the processes of `comm`, on every process: `local` where no process has
/// one, else the error of the lowest process with one, the same on every process (an
/// [`CellsError::Element`] and a [`CellsError::Unsupported`] as they are, any other error as
/// its message in a [`CellsError::Model`] or [`CellsError::Invalid`]). Collective.
pub(crate) fn agree<T>(comm: &Comm, local: Result<T>) -> Result<T> {
    agree_first(comm, local, position(comm.rank(), 0))
}

/// As [`agree`], the error of the process whose error has the smallest `key` (the
/// [`position`] of the cell which failed, on the process whose cell it is), on every process.
/// Collective.
pub(crate) fn agree_first<T>(comm: &Comm, local: Result<T>, key: u64) -> Result<T> {
    debug_assert!(key < 1 << f64::MANTISSA_DIGITS, "{key} is exact as a float");
    // The smallest key with an error, as the maximum of its negation (exact as a float).
    let mine = if local.is_err() {
        -(key as f64)
    } else {
        f64::NEG_INFINITY
    };
    let best = comm.max_real(mine)?;
    if best == f64::NEG_INFINITY {
        return local;
    }
    let rank = PetscInt::try_from(comm.rank()).expect("a rank is a PETSc integer");
    let root = comm.max_int(if mine == best { rank } else { -1 })?;
    let root = usize::try_from(root).expect("a rank");
    let mut bytes = match &local {
        Err(e) if comm.rank() == root => encode(e),
        _ => Vec::new(),
    };
    comm.broadcast_bytes(root, &mut bytes)?;
    Err(decode(&bytes))
}

/// The key of [`agree_first`] of the cell `index` of the process `rank`: the processes in the
/// order of their ranks, the cells of each in their order.
pub(crate) fn position(rank: usize, index: usize) -> u64 {
    ((rank as u64) << 32) | index as u64
}

/// The error of a failed advance of the cells, kept: after it, the cells are not advanced again,
/// so that a retry cannot advance the cells which had advanced before the failure a second time.
/// Every process keeps the same error ([`agree`]), so every process refuses alike.
#[derive(Debug, Default)]
pub(crate) struct Poison(Option<Vec<u8>>);

impl Poison {
    /// The kept error, if there is one.
    pub(crate) fn check(&self) -> Result<()> {
        match &self.0 {
            Some(bytes) => Err(decode(bytes)),
            None => Ok(()),
        }
    }

    /// `result`, its error kept.
    pub(crate) fn keep<T>(&mut self, result: Result<T>) -> Result<T> {
        if let Err(e) = &result {
            self.0 = Some(encode(e));
        }
        result
    }
}

/// The kinds of errors [`agree`] sends.
const ELEMENT: u8 = 1;
const MODEL: u8 = 2;
const OTHER: u8 = 3;
const UNSUPPORTED: u8 = 4;

/// `e` as bytes for [`decode`]: its kind, an element's number and centroid, the message.
fn encode(e: &CellsError) -> Vec<u8> {
    let mut bytes = Vec::new();
    match e {
        CellsError::Element {
            element,
            centroid,
            message,
        } => {
            bytes.push(ELEMENT);
            bytes.extend(element.to_le_bytes());
            bytes.extend(centroid[0].to_le_bytes());
            bytes.extend(centroid[1].to_le_bytes());
            bytes.extend(message.as_bytes());
        }
        CellsError::Model { message } => {
            bytes.push(MODEL);
            bytes.extend(message.as_bytes());
        }
        CellsError::Unsupported { feature } => {
            bytes.push(UNSUPPORTED);
            bytes.extend(feature.as_bytes());
        }
        CellsError::Invalid(message) => {
            bytes.push(OTHER);
            bytes.extend(message.as_bytes());
        }
        e => {
            bytes.push(OTHER);
            bytes.extend(e.to_string().as_bytes());
        }
    }
    bytes
}

/// The error of the bytes of [`encode`].
fn decode(bytes: &[u8]) -> CellsError {
    let text = |b: &[u8]| String::from_utf8_lossy(b).into_owned();
    let f64_at = |i: usize| f64::from_le_bytes(bytes[i..i + 8].try_into().expect("8 bytes"));
    match bytes.first() {
        Some(&ELEMENT) => CellsError::Element {
            element: u64::from_le_bytes(bytes[1..9].try_into().expect("8 bytes")),
            centroid: [f64_at(9), f64_at(17)],
            message: text(&bytes[25..]),
        },
        Some(&MODEL) => CellsError::Model {
            message: text(&bytes[1..]),
        },
        Some(&UNSUPPORTED) => CellsError::Unsupported {
            feature: text(&bytes[1..]),
        },
        _ => CellsError::Invalid(text(bytes.get(1..).unwrap_or_default())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every kind of error comes back as it was sent.
    #[test]
    fn errors_round_trip() {
        let element = CellsError::Element {
            element: (7 << 40) | 3,
            centroid: [1.5, -2.25],
            message: "it failed: é".into(),
        };
        match decode(&encode(&element)) {
            CellsError::Element {
                element,
                centroid,
                message,
            } => {
                assert_eq!(element, (7 << 40) | 3);
                assert_eq!(centroid, [1.5, -2.25]);
                assert_eq!(message, "it failed: é");
            }
            e => panic!("{e:?}"),
        }
        let model = CellsError::Model {
            message: "no compiler".into(),
        };
        let CellsError::Model { message } = decode(&encode(&model)) else {
            panic!("not a model error");
        };
        assert_eq!(message, "no compiler");
        let unsupported = CellsError::Unsupported {
            feature: "a model with events".into(),
        };
        let CellsError::Unsupported { feature } = decode(&encode(&unsupported)) else {
            panic!("not an unsupported feature");
        };
        assert_eq!(feature, "a model with events");
        let other = CellsError::Invalid("lengths".into());
        assert_eq!(decode(&encode(&other)).to_string(), "cells: lengths");
    }
}
