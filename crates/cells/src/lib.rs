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
//!   WebAssembly.
//!
//! Units: time in s (the model's time unit is `CellsSpec::time_unit_s` s), concentrations in
//! the model's units (mM for the fields of slidesim), lengths in um. The model keeps its own
//! volume units: only the ratio of the blood and cell volumes matters.
//!
//! Processes: each process holds the cells of its coarse elements. Errors one process finds
//! (a map without value in a cell's tissue, a cell model which fails) are errors on every
//! process, the same on each ([`CellsError::Element`] names the element by its global number
//! on the coarse level and its centroid), so that no process waits for another which left a
//! collective call. A cell's step control is its own: it depends on no other cell and not on
//! the number of processes.

mod engine;
mod parameters;

use petsc::Comm;
use petsc::sys::PetscInt;

pub use engine::EngineCells;
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
    /// advanced again.
    fn advance(&mut self, t0: f64, t1: f64, blood: &mut [f64]) -> Result<CellStats>;

    /// The states of every cell (`n_cells * state_ids().len()` values, cell by cell), as the
    /// model stores them (a species of a concentration as its amount, in the model's units).
    fn states(&self) -> &[f64];
}

/// The work of one [`CellSolver::advance`] on this process.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct CellStats {
    /// The steps of the integrators of the cells, summed.
    pub steps: u64,
    /// The rejected step attempts (failed error tests and nonlinear solves), summed.
    pub rejected: u64,
    /// The wall time of advancing the cells (s).
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
    /// A failure at one element: `cell` is its global number on the coarse level of the mesh,
    /// `centroid` its centroid (um). The same on every process.
    #[error("element {cell} at ({}, {}) um: {message}", .centroid[0], .centroid[1])]
    Element {
        cell: i64,
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
}

pub type Result<T, E = CellsError> = std::result::Result<T, E>;

/// The first error of the processes of `comm`, on every process: `local` where no process has
/// one, else the error of the lowest process with one, the same on every process (an
/// [`CellsError::Element`] as it is, any other error as its message in a
/// [`CellsError::Model`] or [`CellsError::Invalid`]). Collective.
pub(crate) fn agree<T>(comm: &Comm, local: Result<T>) -> Result<T> {
    let rank = PetscInt::try_from(comm.rank()).expect("a rank is a PETSc integer");
    // The lowest rank with an error, as the maximum of its negation.
    let key = comm.max_int(if local.is_err() { -rank } else { PetscInt::MIN })?;
    if key == PetscInt::MIN {
        return local;
    }
    let root = usize::try_from(-key).expect("a rank");
    let mut bytes = match &local {
        Err(e) if comm.rank() == root => encode(e),
        _ => Vec::new(),
    };
    comm.broadcast_bytes(root, &mut bytes)?;
    Err(decode(&bytes))
}

/// The kinds of errors [`agree`] sends.
const ELEMENT: u8 = 1;
const MODEL: u8 = 2;
const OTHER: u8 = 3;

/// `e` as bytes for [`decode`]: its kind, an element's number and centroid, the message.
fn encode(e: &CellsError) -> Vec<u8> {
    let mut bytes = Vec::new();
    match e {
        CellsError::Element {
            cell,
            centroid,
            message,
        } => {
            bytes.push(ELEMENT);
            bytes.extend(cell.to_le_bytes());
            bytes.extend(centroid[0].to_le_bytes());
            bytes.extend(centroid[1].to_le_bytes());
            bytes.extend(message.as_bytes());
        }
        CellsError::Model { message } => {
            bytes.push(MODEL);
            bytes.extend(message.as_bytes());
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
            cell: i64::from_le_bytes(bytes[1..9].try_into().expect("8 bytes")),
            centroid: [f64_at(9), f64_at(17)],
            message: text(&bytes[25..]),
        },
        Some(&MODEL) => CellsError::Model {
            message: text(&bytes[1..]),
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
            cell: -3,
            centroid: [1.5, -2.25],
            message: "it failed: é".into(),
        };
        match decode(&encode(&element)) {
            CellsError::Element {
                cell,
                centroid,
                message,
            } => {
                assert_eq!(cell, -3);
                assert_eq!(centroid, [1.5, -2.25]);
                assert_eq!(message, "it failed: é");
            }
            e => panic!("{e:?}"),
        }
        let model = CellsError::Model {
            message: "no compiler".into(),
        };
        assert!(
            matches!(decode(&encode(&model)), CellsError::Model { message } if message == "no compiler")
        );
        let other = CellsError::Invalid("lengths".into());
        assert_eq!(decode(&encode(&other)).to_string(), "cells: lengths");
    }
}
