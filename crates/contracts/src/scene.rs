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
        scene.mesh.path = resolve(&base_dir(origin), &scene.mesh.path);
        Ok(scene)
    }
}

pub fn schema_scene() -> serde_json::Value {
    serde_json::to_value(schemars::schema_for!(Scene)).expect("schema is JSON")
}
