use std::collections::HashSet;
use std::path::{Path, PathBuf};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{ContractError, base_dir, check_version, read_json, resolve};

pub const STUDY_VERSION: &str = "study/1";

/// `study/1`: the simulation description.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Study {
    /// Contract version, exactly `study/1`.
    #[schemars(extend("const" = "study/1"))]
    pub version: String,
    /// Path of the `scene/1` file; relative paths resolve against the
    /// directory of the study file (`Study::load` stores the absolute path).
    pub scene: PathBuf,
    /// Diffusing fields, at least one, with unique ids.
    #[schemars(length(min = 1))]
    pub fields: Vec<FieldSpec>,
    pub time: TimeSpec,
    #[serde(default)]
    pub mesh: MeshOptions,
    #[serde(default)]
    pub solver: SolverOptions,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FieldSpec {
    /// Unique field id.
    pub id: String,
    /// Diffusion coefficient in um^2/s, must be > 0.
    #[schemars(extend("exclusiveMinimum" = 0))]
    pub diffusion: f64,
    /// First-order decay rate in 1/s, must be >= 0.
    #[schemars(range(min = 0))]
    #[serde(default)]
    pub decay: f64,
    /// Uniform source term in mM/s.
    #[serde(default)]
    pub source: f64,
    /// Uniform initial concentration in mM.
    #[serde(default)]
    pub initial: f64,
    /// Boundary conditions by mesh tag; tags not listed are zero flux.
    #[serde(default)]
    pub boundaries: Vec<BoundarySpec>,
}

/// Boundary condition on one mesh tag, serialized as
/// `{"tag": "...", "dirichlet": 0.1}` or `{"tag": "...", "flux": 1.0}`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(try_from = "BoundaryRepr", into = "BoundaryRepr")]
#[schemars(with = "BoundaryRepr")]
pub struct BoundarySpec {
    /// Name of the boundary tag in the mesh.
    pub tag: String,
    pub condition: Condition,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Condition {
    /// Fixed concentration in mM.
    Dirichlet { value: f64 },
    /// Inward flux in mM*um/s.
    Flux { value: f64 },
}

#[derive(Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[schemars(extend("oneOf" = [{"required": ["dirichlet"]}, {"required": ["flux"]}]))]
struct BoundaryRepr {
    /// Name of the boundary tag in the mesh.
    tag: String,
    /// Fixed concentration in mM (exclusive with `flux`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    dirichlet: Option<f64>,
    /// Inward flux in mM*um/s (exclusive with `dirichlet`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    flux: Option<f64>,
}

impl TryFrom<BoundaryRepr> for BoundarySpec {
    type Error = String;
    fn try_from(r: BoundaryRepr) -> Result<Self, String> {
        let condition = match (r.dirichlet, r.flux) {
            (Some(value), None) => Condition::Dirichlet { value },
            (None, Some(value)) => Condition::Flux { value },
            _ => {
                return Err(format!(
                    "boundary '{}': exactly one of 'dirichlet' and 'flux' is required",
                    r.tag
                ));
            }
        };
        Ok(BoundarySpec {
            tag: r.tag,
            condition,
        })
    }
}

impl From<BoundarySpec> for BoundaryRepr {
    fn from(b: BoundarySpec) -> Self {
        let (dirichlet, flux) = match b.condition {
            Condition::Dirichlet { value } => (Some(value), None),
            Condition::Flux { value } => (None, Some(value)),
        };
        BoundaryRepr {
            tag: b.tag,
            dirichlet,
            flux,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TimeSpec {
    /// End time in s, must be > 0.
    #[schemars(extend("exclusiveMinimum" = 0))]
    pub end: f64,
    /// Time step in s (initial step if adaptive), must be > 0.
    #[schemars(extend("exclusiveMinimum" = 0))]
    pub dt: f64,
    /// Use adaptive time stepping.
    #[serde(default)]
    pub adaptive: bool,
    /// Output times in s, sorted and within [0, end].
    #[serde(default)]
    pub outputs: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct MeshOptions {
    /// Number of uniform mesh refinements.
    #[serde(default)]
    pub refinements: u32,
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SolverOptions {
    /// PETSc options string, e.g. `-ksp_type cg -pc_type jacobi`.
    #[serde(default)]
    pub petsc_options: String,
}

fn invalid(msg: String) -> ContractError {
    ContractError::Invalid(msg)
}

fn finite(what: &str, x: f64) -> Result<(), ContractError> {
    if x.is_finite() {
        Ok(())
    } else {
        Err(invalid(format!("{what} must be finite, got {x}")))
    }
}

impl Study {
    /// Load and validate a study file; `scene` is resolved to an absolute path.
    pub fn load(path: impl AsRef<Path>) -> Result<Study, ContractError> {
        let path = path.as_ref();
        Self::from_value(read_json(path)?, path)
    }

    /// Validate a parsed study; `origin` is the file it came from.
    pub fn from_value(value: serde_json::Value, origin: &Path) -> Result<Study, ContractError> {
        check_version(&value, STUDY_VERSION)?;
        let mut study: Study = serde_json::from_value(value).map_err(|e| ContractError::Parse {
            path: origin.to_path_buf(),
            message: e.to_string(),
        })?;
        study.validate()?;
        study.scene = resolve(&base_dir(origin), &study.scene);
        Ok(study)
    }

    fn validate(&self) -> Result<(), ContractError> {
        if self.fields.is_empty() {
            return Err(invalid("fields must not be empty".into()));
        }
        let mut ids = HashSet::new();
        for f in &self.fields {
            if !ids.insert(f.id.as_str()) {
                return Err(invalid(format!("duplicate field id '{}'", f.id)));
            }
            f.validate()?;
        }
        self.time.validate()
    }
}

impl FieldSpec {
    fn validate(&self) -> Result<(), ContractError> {
        let id = &self.id;
        for (name, x) in [
            ("diffusion", self.diffusion),
            ("decay", self.decay),
            ("source", self.source),
            ("initial", self.initial),
        ] {
            finite(&format!("field '{id}': {name}"), x)?;
        }
        if self.diffusion <= 0.0 {
            return Err(invalid(format!(
                "field '{id}': diffusion must be > 0, got {}",
                self.diffusion
            )));
        }
        if self.decay < 0.0 {
            return Err(invalid(format!(
                "field '{id}': decay must be >= 0, got {}",
                self.decay
            )));
        }
        let mut tags = HashSet::new();
        for b in &self.boundaries {
            let (Condition::Dirichlet { value } | Condition::Flux { value }) = b.condition;
            finite(&format!("field '{id}': boundary '{}' value", b.tag), value)?;
            if !tags.insert(b.tag.as_str()) {
                return Err(invalid(format!(
                    "field '{id}': duplicate boundary tag '{}'",
                    b.tag
                )));
            }
        }
        Ok(())
    }
}

impl TimeSpec {
    fn validate(&self) -> Result<(), ContractError> {
        finite("time.end", self.end)?;
        finite("time.dt", self.dt)?;
        if self.end <= 0.0 {
            return Err(invalid(format!("time.end must be > 0, got {}", self.end)));
        }
        if self.dt <= 0.0 {
            return Err(invalid(format!("time.dt must be > 0, got {}", self.dt)));
        }
        let ok = self.outputs.iter().all(|t| (0.0..=self.end).contains(t))
            && self.outputs.windows(2).all(|w| w[0] <= w[1]);
        if !ok {
            return Err(invalid(
                "time.outputs must be sorted and within [0, end]".into(),
            ));
        }
        Ok(())
    }
}

pub fn schema_study() -> serde_json::Value {
    serde_json::to_value(schemars::schema_for!(Study)).expect("schema is JSON")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn study() -> Study {
        let text = include_str!("../../../testdata/contracts/study-minimal.json");
        serde_json::from_str(text).unwrap()
    }

    #[test]
    fn non_finite_numbers_are_rejected() {
        let mut s = study();
        s.fields[0].diffusion = f64::NAN;
        assert_eq!(
            s.validate().unwrap_err().to_string(),
            "field 'oxygen': diffusion must be finite, got NaN"
        );
        let mut s = study();
        s.fields[0].boundaries[0].condition = Condition::Flux {
            value: f64::INFINITY,
        };
        assert_eq!(
            s.validate().unwrap_err().to_string(),
            "field 'oxygen': boundary 'inlet' value must be finite, got inf"
        );
        let mut s = study();
        s.time.end = f64::INFINITY;
        assert_eq!(
            s.validate().unwrap_err().to_string(),
            "time.end must be finite, got inf"
        );
    }
}
