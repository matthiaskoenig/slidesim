use std::path::{Path, PathBuf};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{ContractError, base_dir, check_version, read_json, resolve};

pub const SCENE_VERSION: &str = "scene/1";

/// `scene/1`: the geometry description written by slideviz.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Scene {
    /// Contract version, exactly `scene/1`.
    #[schemars(extend("const" = "scene/1"))]
    pub version: String,
    /// Spatial dimension of the mesh (2 or 3).
    #[schemars(range(min = 2, max = 3))]
    pub dimension: u8,
    pub units: Units,
    pub mesh: MeshRef,
    /// Raster maps over the tissue (e.g. a zonation coordinate); studies refer to them by id.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub maps: Vec<MapSpec>,
}

/// A raster map: one float value per pixel, placed in the scene's um coordinates.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct MapSpec {
    /// Unique map id: an ASCII letter, then ASCII letters, digits and `_`.
    #[schemars(extend("pattern" = "^[A-Za-z][A-Za-z0-9_]*$"))]
    pub id: String,
    /// Path of the raster file; relative paths resolve against the directory of the scene
    /// file (`Scene::load` stores the absolute path).
    pub path: PathBuf,
    /// Units of the map values, free text (e.g. `1` for a dimensionless coordinate).
    pub units: String,
    /// Pixel size in um, must be > 0.
    #[schemars(extend("exclusiveMinimum" = 0))]
    pub um_per_px: f64,
    /// Position in um of the corner of the raster the pixel grid starts at (pixel 0, 0).
    pub origin_um: [f64; 2],
    /// Pixel value that marks "no data", if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub no_value: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Units {
    /// Length unit of the mesh coordinates; only `um` (micrometre) is supported.
    pub length: LengthUnit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub enum LengthUnit {
    #[serde(rename = "um")]
    Um,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct MeshRef {
    /// Mesh file format.
    pub format: MeshFormat,
    /// Path of the mesh file; relative paths resolve against the directory of
    /// the scene file (`Scene::load` stores the absolute path).
    pub path: PathBuf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub enum MeshFormat {
    #[serde(rename = "gmsh")]
    Gmsh,
}

impl Scene {
    /// Load and validate a scene file; `mesh.path` is resolved to an absolute path.
    pub fn load(path: impl AsRef<Path>) -> Result<Scene, ContractError> {
        let path = path.as_ref();
        Self::from_value(read_json(path)?, path)
    }

    /// Validate a parsed scene; `origin` is the file it came from.
    pub fn from_value(value: serde_json::Value, origin: &Path) -> Result<Scene, ContractError> {
        check_version(&value, SCENE_VERSION)?;
        if let Some(u) = value.pointer("/units/length").and_then(|v| v.as_str())
            && u != "um"
        {
            return Err(ContractError::UnsupportedLengthUnit {
                found: u.to_string(),
            });
        }
        let mut scene: Scene = serde_json::from_value(value).map_err(|e| ContractError::Parse {
            path: origin.to_path_buf(),
            message: e.to_string(),
        })?;
        if !(2..=3).contains(&scene.dimension) {
            return Err(ContractError::Invalid(format!(
                "dimension must be 2 or 3, got {}",
                scene.dimension
            )));
        }
        let mut ids = std::collections::HashSet::new();
        for m in &mut scene.maps {
            let id = &m.id;
            let valid_id = id.starts_with(|c: char| c.is_ascii_alphabetic())
                && id.chars().all(|c| c.is_ascii_alphanumeric() || c == '_');
            if !valid_id {
                return Err(ContractError::Invalid(format!(
                    "map id '{id}' must match ^[A-Za-z][A-Za-z0-9_]*$"
                )));
            }
            if !ids.insert(id.clone()) {
                return Err(ContractError::Invalid(format!("duplicate map id '{id}'")));
            }
            if !(m.um_per_px.is_finite() && m.um_per_px > 0.0) {
                return Err(ContractError::Invalid(format!(
                    "map '{id}': um_per_px must be > 0, got {}",
                    m.um_per_px
                )));
            }
            if !m.origin_um.iter().all(|x| x.is_finite())
                || m.no_value.is_some_and(|v| !v.is_finite())
            {
                return Err(ContractError::Invalid(format!(
                    "map '{id}': origin_um and no_value must be finite"
                )));
            }
            m.path = resolve(&base_dir(origin), &m.path);
        }
        scene.mesh.path = resolve(&base_dir(origin), &scene.mesh.path);
        Ok(scene)
    }
}

pub fn schema_scene() -> serde_json::Value {
    serde_json::to_value(schemars::schema_for!(Scene)).expect("schema is JSON")
}
