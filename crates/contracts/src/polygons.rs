use std::collections::HashMap;
use std::path::Path;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{ContractError, check_version, read_json};

pub const POLYGONS_VERSION: &str = "polygons/1";

/// `polygons/1`: tagged areas and edges as a GeoJSON FeatureCollection, in the um coordinates of
/// the scene.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct FeatureCollection {
    /// Always `FeatureCollection`.
    #[serde(rename = "type")]
    #[schemars(extend("const" = "FeatureCollection"))]
    kind: String,
    /// Contract version, exactly `polygons/1`.
    #[schemars(extend("const" = "polygons/1"))]
    version: String,
    /// Free metadata, ignored by slidesim.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    meta: Option<serde_json::Value>,
    features: Vec<Feature>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct Feature {
    /// Always `Feature`.
    #[serde(rename = "type")]
    #[schemars(extend("const" = "Feature"))]
    kind: String,
    properties: Properties,
    geometry: Geometry,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct Properties {
    /// `area` (Polygon or MultiPolygon) or `edge` (LineString or MultiLineString).
    kind: FeatureKind,
    /// Tag name; a tag is either an area tag or an edge tag.
    tag: String,
    /// Integer tag value, must be > 0 (default 1).
    #[serde(default = "one")]
    #[schemars(range(min = 1))]
    value: i64,
}

fn one() -> i64 {
    1
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
enum FeatureKind {
    Area,
    Edge,
}

type Point = [f64; 2];

/// GeoJSON geometry; coordinates are `[x, y]` in um. The first ring of a polygon is its outer
/// boundary, the others are holes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", deny_unknown_fields)]
enum Geometry {
    Polygon { coordinates: Vec<Vec<Point>> },
    MultiPolygon { coordinates: Vec<Vec<Vec<Point>>> },
    LineString { coordinates: Vec<Point> },
    MultiLineString { coordinates: Vec<Vec<Point>> },
}

impl Geometry {
    fn name(&self) -> &'static str {
        match self {
            Geometry::Polygon { .. } => "Polygon",
            Geometry::MultiPolygon { .. } => "MultiPolygon",
            Geometry::LineString { .. } => "LineString",
            Geometry::MultiLineString { .. } => "MultiLineString",
        }
    }
}

/// One polygon of an area tag: the outer ring first, then the holes.
#[derive(Debug, Clone, PartialEq)]
pub struct AreaPolygon {
    pub tag: String,
    pub value: u32,
    pub rings: Vec<Vec<[f64; 2]>>,
}

/// One chain of an edge tag.
#[derive(Debug, Clone, PartialEq)]
pub struct EdgeChain {
    pub tag: String,
    pub value: u32,
    pub points: Vec<[f64; 2]>,
}

/// The loaded `polygons/1` file. Geometry (closed rings, no self-intersections, ...) is checked
/// by the mesher.
#[derive(Debug, Clone, PartialEq)]
pub struct Polygons {
    pub areas: Vec<AreaPolygon>,
    pub edges: Vec<EdgeChain>,
}

impl Polygons {
    /// Load and validate a `polygons/1` file.
    pub fn load(path: impl AsRef<Path>) -> Result<Polygons, ContractError> {
        let path = path.as_ref();
        Self::from_value(read_json(path)?, path)
    }

    /// Validate a parsed `polygons/1` document; `origin` is the file it came from.
    pub fn from_value(value: serde_json::Value, origin: &Path) -> Result<Polygons, ContractError> {
        check_version(&value, POLYGONS_VERSION)?;
        let fc: FeatureCollection =
            serde_json::from_value(value).map_err(|e| ContractError::Parse {
                path: origin.to_path_buf(),
                message: e.to_string(),
            })?;
        if fc.kind != "FeatureCollection" {
            return Err(ContractError::Invalid(format!(
                "type must be 'FeatureCollection', got '{}'",
                fc.kind
            )));
        }
        let mut out = Polygons {
            areas: Vec::new(),
            edges: Vec::new(),
        };
        let mut kinds: HashMap<String, FeatureKind> = HashMap::new();
        for (i, f) in fc.features.into_iter().enumerate() {
            let p = f.properties;
            let bad = |what: String| ContractError::Invalid(format!("feature {i}: {what}"));
            if f.kind != "Feature" {
                return Err(bad(format!("type must be 'Feature', got '{}'", f.kind)));
            }
            if p.tag.is_empty() {
                return Err(bad("tag must not be empty".to_string()));
            }
            let value = u32::try_from(p.value)
                .ok()
                .filter(|v| *v > 0)
                .ok_or_else(|| bad(format!("value must be an integer > 0, got {}", p.value)))?;
            if let Some(&other) = kinds.get(&p.tag)
                && other != p.kind
            {
                return Err(bad(format!(
                    "tag '{}' is used as an area and as an edge",
                    p.tag
                )));
            }
            kinds.insert(p.tag.clone(), p.kind);
            let tag = p.tag;
            match (p.kind, f.geometry) {
                (FeatureKind::Area, Geometry::Polygon { coordinates }) => {
                    out.areas.push(AreaPolygon {
                        tag,
                        value,
                        rings: coordinates,
                    });
                }
                (FeatureKind::Area, Geometry::MultiPolygon { coordinates }) => {
                    out.areas
                        .extend(coordinates.into_iter().map(|rings| AreaPolygon {
                            tag: tag.clone(),
                            value,
                            rings,
                        }));
                }
                (FeatureKind::Edge, Geometry::LineString { coordinates }) => {
                    out.edges.push(EdgeChain {
                        tag,
                        value,
                        points: coordinates,
                    });
                }
                (FeatureKind::Edge, Geometry::MultiLineString { coordinates }) => {
                    out.edges
                        .extend(coordinates.into_iter().map(|points| EdgeChain {
                            tag: tag.clone(),
                            value,
                            points,
                        }));
                }
                (kind, g) => {
                    let (name, want) = match kind {
                        FeatureKind::Area => ("area", "Polygon or MultiPolygon"),
                        FeatureKind::Edge => ("edge", "LineString or MultiLineString"),
                    };
                    return Err(bad(format!(
                        "tag '{tag}': kind '{name}' needs a {want}, got a {}",
                        g.name()
                    )));
                }
            }
        }
        Ok(out)
    }
}

pub fn schema_polygons() -> serde_json::Value {
    serde_json::to_value(schemars::schema_for!(FeatureCollection)).expect("schema is JSON")
}
