//! Versioned JSON input contracts of slidesim: `scene/1`, `polygons/1` and `study/1`, and the
//! output contract `result/1`.
//!
//! Units are fixed: length um, time s, concentration mM; diffusion um^2/s,
//! decay 1/s, source mM/s, flux mM*um/s, pressure Pa, permeability um^2, viscosity Pa*s,
//! velocity um/s, porosity dimensionless, `g0` 1/um.

mod error;
mod polygons;
mod result;
mod scene;
mod study;

pub use error::ContractError;
pub use polygons::{AreaPolygon, EdgeChain, POLYGONS_VERSION, Polygons, schema_polygons};
pub use result::{RESULT_VERSION, RasterMeta, ResultField, ResultMeta, ResultTag, schema_result};
pub use scene::{
    HoleSpec, LengthUnit, LobuleSize, LobuleSpec, MapSpec, MeshSource, SCENE_VERSION, Scene,
    SizeField, Units, ZonationSpec, schema_scene,
};
pub use study::{
    BoundarySpec, CellParameter, CellSolverKind, CellsSpec, Coefficient, Condition, Direction,
    FieldSpec, MeshOptions, Perfusion, PressureBoundary, Replace, STUDY_VERSION, SolverOptions,
    SpeciesLink, Study, Table, TimeSpec, VolumeLink, check_cells, check_maps, schema_study,
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
