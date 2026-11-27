//! `result/1`: what a viewer reads from a finished run (`result.zarr`).
//!
//! The store is a Zarr v3 group with zstd compression; its root attributes are [`ResultMeta`].
//! All coordinates are um, concentrations mM. A time index `t < written` is complete and
//! readable alone.
//!
//! | Array | dtype | shape | chunks | Meaning |
//! | --- | --- | --- | --- | --- |
//! | `time` | f64 | `[T]` | all | output times in s |
//! | `mesh/vertices` | f64 | `[V, 2]` | all | vertex coordinates in um |
//! | `mesh/triangles` | u32 | `[N, 3]` | all | vertex indices, counterclockwise |
//! | `mesh/element` | u64 | `[N]` | all | coarse element id of each finest triangle |
//! | `mesh/tags/<name>` | u32 | `[k]` or `[k, 2]` | all | area tag: triangle indices; edge tag: vertex index pairs |
//! | `fields/<id>` | f64 | `[T, V]` | `[1, V]` | P1 nodal values on the finest mesh |
//! | `cells/element` | u64 | `[C]` | all | coarse element ids holding cells |
//! | `cells/<value>` | f64 | `[T, C]` | `[1, C]` | per-element cell value |
//! | `raster/<field>/<level>` | f32 | `[T, rows, cols]` | `[1, 256, 256]` | optional; NaN outside the mesh; level k+1 is the mean of the valid pixels of 2 x 2 of level k |
//!
//! Element ids are partition independent: `Mesh::element_ids` numbers the coarse cells of the
//! undistributed mesh; the id of a finest triangle is `(coarse << (2 * level)) | child bits`. The
//! triangles of `mesh/triangles` are sorted by that fine id, the vertices are numbered by first
//! appearance in that order, and `mesh/element` holds the coarse id of each triangle.
//!
//! `written` is the number of output times reached so far; in a finished run `complete` is true
//! and `written == times.len()`. The live API (`wasm/js/slidesim.d.ts`) says the same: a created
//! simulation has `written == 1` (the initial state at `times[0]`), each step reaches one more.

use std::collections::HashSet;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::ContractError;

pub const RESULT_VERSION: &str = "result/1";

/// The root attributes of `result.zarr`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ResultMeta {
    /// Contract version, exactly `result/1`.
    #[schemars(extend("const" = "result/1"))]
    pub version: String,
    /// The study file the run was made from.
    pub study: String,
    /// The scene file of the study.
    pub scene: String,
    /// Uniform refinements of the mesh beyond the coarse one.
    pub refinements: u32,
    /// Output times in s, strictly increasing; the length of the `time` array.
    pub times: Vec<f64>,
    /// Number of time indices written so far; indices below it are complete.
    pub written: usize,
    /// Whether the run finished, then `written == times.len()`.
    pub complete: bool,
    /// The fields stored under `fields/<id>`.
    pub fields: Vec<ResultField>,
    /// The per-element cell values stored under `cells/<value>`.
    pub cell_values: Vec<String>,
    /// The tags stored under `mesh/tags/<name>`.
    pub tags: Vec<ResultTag>,
    /// The raster pyramid under `raster/`, when written.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raster: Option<RasterMeta>,
}

/// A stored field.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ResultField {
    /// Field id, unique.
    pub id: String,
    /// Units of the values, `mM`.
    pub units: String,
}

/// A stored tag.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ResultTag {
    pub name: String,
    /// 2 for an area (triangle indices), 1 for an edge (vertex index pairs).
    #[schemars(range(min = 1, max = 2))]
    pub dim: u32,
}

/// The raster pyramid of the fields.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RasterMeta {
    /// Pixel size of level 0 in um, must be > 0.
    #[schemars(extend("exclusiveMinimum" = 0))]
    pub um_per_px: f64,
    /// Coordinates in um of the corner of pixel (0, 0) of level 0.
    pub origin_um: [f64; 2],
    /// Number of levels, at least 1.
    #[schemars(range(min = 1))]
    pub levels: u32,
    /// `[rows, cols]` per level, level 0 first.
    pub shapes: Vec<[u64; 2]>,
}

impl ResultMeta {
    /// Check the rules of the contract; the message names the field.
    pub fn check(&self) -> Result<(), ContractError> {
        let bad = |m: String| Err(ContractError::Invalid(m));
        if self.version != RESULT_VERSION {
            return Err(ContractError::UnsupportedVersion {
                found: self.version.clone(),
                expected: RESULT_VERSION,
            });
        }
        if self.written > self.times.len() {
            return bad(format!(
                "written ({}) must not exceed the number of times ({})",
                self.written,
                self.times.len()
            ));
        }
        if self.complete && self.written != self.times.len() {
            return bad(format!(
                "complete requires written ({}) to equal the number of times ({})",
                self.written,
                self.times.len()
            ));
        }
        if self
            .times
            .windows(2)
            .any(|w| w[0].partial_cmp(&w[1]) != Some(std::cmp::Ordering::Less))
        {
            return bad("times must be strictly increasing".to_string());
        }
        let mut seen = HashSet::new();
        for f in &self.fields {
            if !seen.insert(f.id.as_str()) {
                return bad(format!("duplicate field id '{}' in fields", f.id));
            }
        }
        let mut names = HashSet::new();
        for t in &self.tags {
            if !(1..=2).contains(&t.dim) {
                return bad(format!(
                    "tag '{}': dim must be 1 or 2, got {}",
                    t.name, t.dim
                ));
            }
            if !names.insert(t.name.as_str()) {
                return bad(format!("duplicate tag name '{}' in tags", t.name));
            }
        }
        let mut values = HashSet::new();
        for v in &self.cell_values {
            if !values.insert(v.as_str()) {
                return bad(format!("duplicate name '{v}' in cell_values"));
            }
        }
        if let Some(r) = &self.raster {
            if r.levels < 1 {
                return bad("raster levels must be >= 1".to_string());
            }
            if r.shapes.len() != r.levels as usize {
                return bad(format!(
                    "raster shapes has {} entries, levels is {}",
                    r.shapes.len(),
                    r.levels
                ));
            }
            if let Some(l) = r.shapes.iter().position(|s| s[0] == 0 || s[1] == 0) {
                return bad(format!("raster shapes of level {l} must be non-zero"));
            }
            if r.um_per_px.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
                return bad(format!("raster um_per_px must be > 0, got {}", r.um_per_px));
            }
        }
        Ok(())
    }
}

pub fn schema_result() -> serde_json::Value {
    serde_json::to_value(schemars::schema_for!(ResultMeta)).expect("schema is JSON")
}
