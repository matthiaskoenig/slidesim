//! Versioned JSON input contracts of slidesim: `scene/1` and `study/1`.
//!
//! Units are fixed: length um, time s, concentration mM; diffusion um^2/s,
//! decay 1/s, source mM/s, flux mM*um/s.

mod error;
mod scene;
mod study;

pub use error::ContractError;
pub use scene::{LengthUnit, MeshFormat, MeshRef, SCENE_VERSION, Scene, Units, schema_scene};
pub use study::{
    BoundarySpec, Condition, FieldSpec, MeshOptions, STUDY_VERSION, SolverOptions, Study, TimeSpec,
    schema_study,
};

use std::path::{Path, PathBuf};

/// Directory against which relative paths of the contract at `origin` resolve.
fn base_dir(origin: &Path) -> PathBuf {
    let parent = origin.parent().unwrap_or(Path::new(""));
    std::path::absolute(parent).unwrap_or_else(|_| parent.to_path_buf())
}

/// Resolve `p` against `base` (absolute paths are kept).
fn resolve(base: &Path, p: &Path) -> PathBuf {
    base.join(p)
}

fn read_json(path: &Path) -> Result<serde_json::Value, ContractError> {
    let text = std::fs::read_to_string(path).map_err(|e| ContractError::Io {
        path: path.to_path_buf(),
        source: e,
    })?;
    serde_json::from_str(&text).map_err(|e| ContractError::Parse {
        path: path.to_path_buf(),
        message: e.to_string(),
    })
}

fn check_version(value: &serde_json::Value, expected: &'static str) -> Result<(), ContractError> {
    match value.get("version").and_then(|v| v.as_str()) {
        Some(v) if v == expected => Ok(()),
        Some(v) => Err(ContractError::UnsupportedVersion {
            found: v.to_string(),
            expected,
        }),
        None => Err(ContractError::Invalid(format!(
            "missing string field 'version', expected '{expected}'"
        ))),
    }
}
