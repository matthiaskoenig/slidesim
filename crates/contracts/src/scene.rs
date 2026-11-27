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
    pub mesh: MeshSource,
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
    /// Position `[x, y]` in um of the centre of pixel (row 0, col 0); rows run along y, columns
    /// along x.
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

/// Where the mesh of the scene comes from.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "format", deny_unknown_fields)]
pub enum MeshSource {
    /// A gmsh `.msh` file; its physical groups are the tags.
    #[serde(rename = "gmsh")]
    Gmsh {
        /// Path of the mesh file; relative paths resolve against the directory of the scene
        /// file (`Scene::load` stores the absolute path).
        path: PathBuf,
    },
    /// Tagged polygons (`polygons/1`), meshed by slidesim with the given size field.
    #[serde(rename = "polygons")]
    Polygons {
        /// Path of the `polygons/1` file; relative paths resolve against the directory of the
        /// scene file (`Scene::load` stores the absolute path).
        path: PathBuf,
        /// Target element size over the domain.
        size: SizeField,
    },
    /// Parametric hexagonal lobules, generated and meshed by slidesim.
    #[serde(rename = "lobules")]
    Lobules(LobuleSpec),
}

impl MeshSource {
    /// The mesh or polygons file, if the source reads one.
    pub fn path(&self) -> Option<&Path> {
        match self {
            MeshSource::Gmsh { path } | MeshSource::Polygons { path, .. } => Some(path),
            MeshSource::Lobules(_) => None,
        }
    }
}

/// Element size as a function of the distance to the nearest edge of the `walls` tags: `min_um`
/// up to `dist_min_um`, linear up to `max_um` at `dist_max_um`, `max_um` beyond. Without walls
/// the size is `max_um` everywhere.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SizeField {
    /// Element size near the walls in um, must be > 0 and <= `max_um`.
    #[schemars(extend("exclusiveMinimum" = 0))]
    pub min_um: f64,
    /// Element size far from the walls in um.
    #[schemars(extend("exclusiveMinimum" = 0))]
    pub max_um: f64,
    /// Distance in um up to which the size is `min_um`, must be >= 0 and < `dist_max_um`.
    #[schemars(range(min = 0))]
    pub dist_min_um: f64,
    /// Distance in um from which the size is `max_um`.
    #[schemars(extend("exclusiveMinimum" = 0))]
    pub dist_max_um: f64,
    /// Edge tags the distance is measured from.
    #[serde(default)]
    pub walls: Vec<String>,
}

/// Parametric liver lobules: a hexagonal lobule with a central vein and a portal vein at every
/// corner, optionally surrounded by rings of neighbours, with the veins displaced by `jitter`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct LobuleSpec {
    /// Circumradius of the hexagonal lobule in um, must be > 0.
    #[serde(default = "default_lobule_radius")]
    #[schemars(extend("exclusiveMinimum" = 0))]
    pub lobule_radius_um: f64,
    /// Rings of neighbouring lobules around the central one (0: a single lobule).
    #[serde(default)]
    pub rings: u32,
    /// Largest displacement of a vein as a fraction of `lobule_radius_um`, `0 <= jitter <= 0.25`.
    #[serde(default)]
    #[schemars(range(min = 0, max = 0.25))]
    pub jitter: f64,
    /// Seed of the displacements.
    #[serde(default)]
    pub seed: u64,
    /// The central vein; no default for the tag.
    #[serde(deserialize_with = "de_centre")]
    #[schemars(schema_with = "centre_schema")]
    pub centre: HoleSpec,
    /// The portal veins at the corners; no default for the tag.
    #[serde(deserialize_with = "de_corner")]
    #[schemars(schema_with = "corner_schema")]
    pub corner: HoleSpec,
    /// Tag of the outer boundary of the domain; no default.
    pub outer_tag: String,
    /// Tag of the tissue area; no default.
    pub area_tag: String,
    #[serde(default)]
    pub size: LobuleSize,
    /// The zonation raster generated over the domain; no default for the map id.
    pub zonation: ZonationSpec,
}

fn default_lobule_radius() -> f64 {
    300.0
}

/// `HoleSpec` as read: the radius may be missing, the caller supplies its default.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct HoleIn {
    radius_um: Option<f64>,
    tag: String,
}

fn hole_with_default<'de, D: serde::Deserializer<'de>>(
    d: D,
    radius_um: f64,
) -> Result<HoleSpec, D::Error> {
    let h = HoleIn::deserialize(d)?;
    Ok(HoleSpec {
        radius_um: h.radius_um.unwrap_or(radius_um),
        tag: h.tag,
    })
}

fn de_centre<'de, D: serde::Deserializer<'de>>(d: D) -> Result<HoleSpec, D::Error> {
    hole_with_default(d, 30.0)
}

fn de_corner<'de, D: serde::Deserializer<'de>>(d: D) -> Result<HoleSpec, D::Error> {
    hole_with_default(d, 25.0)
}

fn hole_schema(radius_um: f64, what: &str) -> schemars::Schema {
    schemars::json_schema!({
        "type": "object",
        "description": format!("{what}; a circular vein cut out of the tissue."),
        "properties": {
            "radius_um": {
                "description": "Radius in um, must be > 0.",
                "type": "number",
                "exclusiveMinimum": 0,
                "default": radius_um
            },
            "tag": {"description": "Edge tag of the vein wall.", "type": "string"}
        },
        "required": ["tag"],
        "additionalProperties": false
    })
}

fn centre_schema(_: &mut schemars::SchemaGenerator) -> schemars::Schema {
    hole_schema(30.0, "The central vein")
}

fn corner_schema(_: &mut schemars::SchemaGenerator) -> schemars::Schema {
    hole_schema(25.0, "A portal vein at a corner")
}

/// A circular vein cut out of the tissue.
///
/// Read leniently by `LobuleSpec` (the radius defaults to 30 for the centre and 25 for the
/// corners); written with both keys. Its schema is `hole_schema`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HoleSpec {
    /// Radius in um, must be > 0.
    pub radius_um: f64,
    /// Edge tag of the vein wall.
    pub tag: String,
}

/// Element sizes of the lobule mesh.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct LobuleSize {
    /// Element size at the vein walls in um (default 5), must be > 0 and <= `interior_um`.
    #[serde(default = "default_wall")]
    #[schemars(extend("exclusiveMinimum" = 0))]
    pub wall_um: f64,
    /// Element size in the interior in um (default 20).
    #[serde(default = "default_interior")]
    #[schemars(extend("exclusiveMinimum" = 0))]
    pub interior_um: f64,
}

fn default_wall() -> f64 {
    5.0
}

fn default_interior() -> f64 {
    20.0
}

impl Default for LobuleSize {
    fn default() -> Self {
        LobuleSize {
            wall_um: default_wall(),
            interior_um: default_interior(),
        }
    }
}

/// The zonation map generated for the lobules.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ZonationSpec {
    /// Map id a study refers to: an ASCII letter, then ASCII letters, digits and `_`; must not
    /// be the id of an entry of `maps`.
    #[schemars(extend("pattern" = "^[A-Za-z][A-Za-z0-9_]*$"))]
    pub map: String,
    /// Pixel size in um (default 2), must be > 0.
    #[serde(default = "default_um_per_px")]
    #[schemars(extend("exclusiveMinimum" = 0))]
    pub um_per_px: f64,
}

fn default_um_per_px() -> f64 {
    2.0
}

impl LobuleSpec {
    /// The size field of the lobule mesh: `wall_um` up to `2 wall_um` from the walls (centre and
    /// corner veins), linear to `interior_um` at `10 interior_um`.
    pub fn size_field(&self) -> SizeField {
        SizeField {
            min_um: self.size.wall_um,
            max_um: self.size.interior_um,
            dist_min_um: 2.0 * self.size.wall_um,
            dist_max_um: 10.0 * self.size.interior_um,
            walls: vec![self.centre.tag.clone(), self.corner.tag.clone()],
        }
    }
}

fn invalid(message: String) -> ContractError {
    ContractError::Invalid(message)
}

fn valid_map_id(id: &str) -> bool {
    id.starts_with(|c: char| c.is_ascii_alphabetic())
        && id.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
}

impl SizeField {
    fn validate(&self, prefix: &str) -> Result<(), ContractError> {
        let pos = |key: &str, v: f64| {
            if v.is_finite() && v > 0.0 {
                Ok(())
            } else {
                Err(invalid(format!("{prefix}.{key} must be > 0, got {v}")))
            }
        };
        pos("min_um", self.min_um)?;
        pos("max_um", self.max_um)?;
        pos("dist_max_um", self.dist_max_um)?;
        if !(self.dist_min_um.is_finite() && self.dist_min_um >= 0.0) {
            return Err(invalid(format!(
                "{prefix}.dist_min_um must be >= 0, got {}",
                self.dist_min_um
            )));
        }
        if self.min_um > self.max_um {
            return Err(invalid(format!(
                "{prefix}.min_um ({}) must be <= {prefix}.max_um ({})",
                self.min_um, self.max_um
            )));
        }
        if self.dist_min_um >= self.dist_max_um {
            return Err(invalid(format!(
                "{prefix}.dist_min_um ({}) must be < {prefix}.dist_max_um ({})",
                self.dist_min_um, self.dist_max_um
            )));
        }
        Ok(())
    }
}

impl LobuleSpec {
    fn validate(&self, map_ids: &[&str]) -> Result<(), ContractError> {
        let r = self.lobule_radius_um;
        if !(r.is_finite() && r > 0.0) {
            return Err(invalid(format!("lobule_radius_um must be > 0, got {r}")));
        }
        if !(self.jitter.is_finite() && (0.0..=0.25).contains(&self.jitter)) {
            return Err(invalid(format!(
                "jitter must be within 0 and 0.25, got {}",
                self.jitter
            )));
        }
        for (key, h) in [("centre", &self.centre), ("corner", &self.corner)] {
            if !(h.radius_um.is_finite() && h.radius_um > 0.0) {
                return Err(invalid(format!(
                    "{key}.radius_um must be > 0, got {}",
                    h.radius_um
                )));
            }
        }
        // The distance between a vein and its neighbours stays above (1 - 2 jitter) r / 2.
        let gap = (1.0 - 2.0 * self.jitter) * r / 2.0;
        if self.centre.radius_um + self.corner.radius_um >= gap {
            return Err(invalid(format!(
                "centre.radius_um + corner.radius_um ({}) must be < (1 - 2 jitter) lobule_radius_um / 2 ({gap}): the veins would touch",
                self.centre.radius_um + self.corner.radius_um
            )));
        }
        if 2.0 * self.corner.radius_um >= gap {
            return Err(invalid(format!(
                "2 corner.radius_um ({}) must be < (1 - 2 jitter) lobule_radius_um / 2 ({gap}): the veins would touch",
                2.0 * self.corner.radius_um
            )));
        }
        let size = &self.size;
        for (key, v) in [("wall_um", size.wall_um), ("interior_um", size.interior_um)] {
            if !(v.is_finite() && v > 0.0) {
                return Err(invalid(format!("size.{key} must be > 0, got {v}")));
            }
        }
        if size.wall_um > size.interior_um {
            return Err(invalid(format!(
                "size.wall_um ({}) must be <= size.interior_um ({})",
                size.wall_um, size.interior_um
            )));
        }
        let z = &self.zonation;
        if !valid_map_id(&z.map) {
            return Err(invalid(format!(
                "zonation.map '{}' must match ^[A-Za-z][A-Za-z0-9_]*$",
                z.map
            )));
        }
        if !(z.um_per_px.is_finite() && z.um_per_px > 0.0) {
            return Err(invalid(format!(
                "zonation.um_per_px must be > 0, got {}",
                z.um_per_px
            )));
        }
        if map_ids.contains(&z.map.as_str()) {
            return Err(invalid(format!(
                "zonation.map '{}' is also the id of an entry of maps",
                z.map
            )));
        }
        let tags = [
            ("outer_tag", &self.outer_tag),
            ("area_tag", &self.area_tag),
            ("centre.tag", &self.centre.tag),
            ("corner.tag", &self.corner.tag),
        ];
        for (i, (key, tag)) in tags.iter().enumerate() {
            if tag.is_empty() {
                return Err(invalid(format!("{key} must not be empty")));
            }
            if let Some((other, _)) = tags[..i].iter().find(|(_, t)| t == tag) {
                return Err(invalid(format!(
                    "{key} '{tag}' equals {other}: the tags must differ"
                )));
            }
        }
        Ok(())
    }
}

impl Scene {
    /// Load and validate a scene file; the mesh path is resolved to an absolute path.
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
        match &mut scene.mesh {
            MeshSource::Gmsh { path } => *path = resolve(&base_dir(origin), path),
            MeshSource::Polygons { path, size } => {
                *path = resolve(&base_dir(origin), path);
                size.validate("size")?;
            }
            MeshSource::Lobules(spec) => {
                let ids: Vec<&str> = scene.maps.iter().map(|m| m.id.as_str()).collect();
                spec.validate(&ids)?;
            }
        }
        Ok(scene)
    }

    /// The ids of the maps a study may refer to: those of `maps` and the zonation map of a
    /// lobule source.
    pub fn map_ids(&self) -> Vec<&str> {
        let mut ids: Vec<&str> = self.maps.iter().map(|m| m.id.as_str()).collect();
        if let MeshSource::Lobules(spec) = &self.mesh {
            ids.push(&spec.zonation.map);
        }
        ids
    }
}

pub fn schema_scene() -> serde_json::Value {
    serde_json::to_value(schemars::schema_for!(Scene)).expect("schema is JSON")
}
