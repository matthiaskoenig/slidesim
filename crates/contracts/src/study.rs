use std::collections::HashSet;
use std::path::{Path, PathBuf};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{ContractError, Scene, base_dir, check_version, read_json, resolve};

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
    /// Darcy perfusion of the tissue; required for fields with `advection`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub perfusion: Option<Perfusion>,
    /// Cell models coupled to the fields; their `outputs` are checked against the model's
    /// states and assignment rule targets at study start, not here.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cells: Option<CellsSpec>,
    pub time: TimeSpec,
    #[serde(default)]
    pub mesh: MeshOptions,
    #[serde(default)]
    pub solver: SolverOptions,
    /// What the result store holds besides the mesh values.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result: Option<ResultOptions>,
}

/// Options of the result store `result/1`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ResultOptions {
    /// Also write the fields as a raster pyramid (`raster/<field>/<level>`), the frame of a slide
    /// image a viewer overlays the results on.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raster: Option<RasterOptions>,
}

/// The raster pyramid of the fields (`result.raster`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RasterOptions {
    /// Pixel size of level 0 in um, must be > 0.
    #[schemars(extend("exclusiveMinimum" = 0))]
    pub um_per_px: f64,
    /// Coordinates in um of the corner of pixel (0, 0) of level 0; default: the lower corner of
    /// the mesh's bounding box.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub origin_um: Option<[f64; 2]>,
    /// Number of levels, at least 1; level k+1 halves level k.
    #[schemars(range(min = 1))]
    pub levels: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FieldSpec {
    /// Unique field id: an ASCII letter, then ASCII letters, digits and `_` (it names the
    /// PETSc options prefix of the field, `<id>_`).
    #[schemars(extend("pattern" = "^[A-Za-z][A-Za-z0-9_]*$"))]
    pub id: String,
    /// Diffusion coefficient in um^2/s, must be > 0 (a number or a table over a map).
    #[schemars(schema_with = "positive_coefficient_schema")]
    pub diffusion: Coefficient,
    /// First-order decay rate in 1/s, must be >= 0 (a number or a table over a map).
    #[schemars(schema_with = "nonnegative_coefficient_schema")]
    #[serde(default = "zero")]
    pub decay: Coefficient,
    /// Source term in mM/s (a number or a table over a map).
    #[schemars(schema_with = "any_coefficient_schema")]
    #[serde(default = "zero")]
    pub source: Coefficient,
    /// Fraction of the volume the field lives in (dimensionless), must be > 0 (a number or a
    /// table over a map).
    #[schemars(schema_with = "positive_coefficient_schema")]
    #[serde(default = "one", skip_serializing_if = "is_one")]
    pub porosity: Coefficient,
    /// Advect the field with the Darcy velocity of `Study::perfusion`.
    #[serde(default, skip_serializing_if = "is_false")]
    pub advection: bool,
    /// Uniform initial concentration in mM.
    #[serde(default)]
    pub initial: f64,
    /// Boundary conditions by mesh tag; tags not listed are zero flux.
    #[serde(default)]
    pub boundaries: Vec<BoundarySpec>,
}

fn zero() -> Coefficient {
    Coefficient::Constant(0.0)
}

fn one() -> Coefficient {
    Coefficient::Constant(1.0)
}

fn is_one(c: &Coefficient) -> bool {
    *c == Coefficient::Constant(1.0)
}

fn is_false(b: &bool) -> bool {
    !*b
}

fn is_true(b: &bool) -> bool {
    *b
}

fn yes() -> bool {
    true
}

/// A coefficient: a constant (a JSON number) or a piecewise linear function of the values of a
/// scene map (a JSON object, see [`Table`]).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Coefficient {
    Constant(f64),
    Table(Table),
}

/// `{"map": "zonation", "points": [[0, 500], [1, 900]]}`: the coefficient is the linear
/// interpolation of the `[value, output]` points at the map value, held constant beyond them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Table {
    /// Id of a map of the scene.
    pub map: String,
    /// `[value, output]` points, at least 2, the values strictly increasing.
    #[schemars(length(min = 2))]
    pub points: Vec<[f64; 2]>,
}

impl Coefficient {
    /// The map a table refers to; `None` for a constant.
    pub fn maps(&self) -> Option<&str> {
        match self {
            Coefficient::Constant(_) => None,
            Coefficient::Table(t) => Some(&t.map),
        }
    }

    /// The coefficient at the map value `value`: linear interpolation between the points, the
    /// end outputs beyond them; the constant itself for a constant.
    pub fn eval(&self, value: f64) -> f64 {
        let t = match self {
            Coefficient::Constant(c) => return *c,
            Coefficient::Table(t) => t,
        };
        let (first, last) = (t.points[0], t.points[t.points.len() - 1]);
        if value <= first[0] {
            return first[1];
        }
        if value >= last[0] {
            return last[1];
        }
        let i = t.points.partition_point(|p| p[0] <= value);
        let (a, b) = (t.points[i - 1], t.points[i]);
        a[1] + (b[1] - a[1]) * (value - a[0]) / (b[0] - a[0])
    }

    /// Check finiteness, the table shape and `check` on the constant or on every table output.
    fn validate(&self, what: &str, rule: Rule) -> Result<(), ContractError> {
        let outputs: Vec<f64> = match self {
            Coefficient::Constant(c) => vec![*c],
            Coefficient::Table(t) => {
                if t.points.len() < 2 {
                    return Err(invalid(format!("{what}: table needs at least 2 points")));
                }
                for p in &t.points {
                    finite(&format!("{what}: table point"), p[0])?;
                }
                if !t.points.windows(2).all(|w| w[0][0] < w[1][0]) {
                    return Err(invalid(format!(
                        "{what}: table values must be strictly increasing"
                    )));
                }
                t.points.iter().map(|p| p[1]).collect()
            }
        };
        for x in outputs {
            finite(what, x)?;
            match rule {
                Rule::Positive if x <= 0.0 => {
                    return Err(invalid(format!("{what} must be > 0, got {x}")));
                }
                Rule::NonNegative if x < 0.0 => {
                    return Err(invalid(format!("{what} must be >= 0, got {x}")));
                }
                _ => {}
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
enum Rule {
    Positive,
    NonNegative,
    Any,
}

/// Schema of a [`Coefficient`] whose constant and table outputs satisfy `output`.
fn coefficient_schema(output: serde_json::Value) -> schemars::Schema {
    let mut number = serde_json::json!({"type": "number"});
    number
        .as_object_mut()
        .expect("object")
        .extend(output.as_object().expect("object").clone());
    schemars::json_schema!({
        "anyOf": [
            number,
            {
                "type": "object",
                "required": ["map", "points"],
                "additionalProperties": false,
                "properties": {
                    "map": {"type": "string", "description": "Id of a map of the scene."},
                    "points": {
                        "type": "array",
                        "minItems": 2,
                        "description": "`[value, output]` points, the values strictly increasing.",
                        "items": {
                            "type": "array",
                            "minItems": 2,
                            "maxItems": 2,
                            "prefixItems": [{"type": "number"}, number]
                        }
                    }
                }
            }
        ]
    })
}

fn positive_coefficient_schema(_: &mut schemars::SchemaGenerator) -> schemars::Schema {
    coefficient_schema(serde_json::json!({"exclusiveMinimum": 0}))
}

fn nonnegative_coefficient_schema(_: &mut schemars::SchemaGenerator) -> schemars::Schema {
    coefficient_schema(serde_json::json!({"minimum": 0}))
}

fn any_coefficient_schema(_: &mut schemars::SchemaGenerator) -> schemars::Schema {
    coefficient_schema(serde_json::json!({}))
}

/// Darcy perfusion: the pressure field `p` solves `div(K grad p) = 0` with the pressures of
/// `boundaries`; the flow is `u = -(K / viscosity) grad p`, with `K` anisotropic: `k_along`
/// along the direction of the `direction` map, `k_across` across it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Perfusion {
    /// Dynamic viscosity in Pa*s, must be > 0.
    #[schemars(extend("exclusiveMinimum" = 0))]
    pub viscosity: f64,
    /// Permeability along the flow direction in um^2, must be > 0.
    #[schemars(schema_with = "positive_coefficient_schema")]
    pub k_along: Coefficient,
    /// Permeability across the flow direction in um^2, must be > 0.
    #[schemars(schema_with = "positive_coefficient_schema")]
    pub k_across: Coefficient,
    pub direction: Direction,
    /// Pressure boundary conditions by mesh tag, at least one, with unique tags.
    #[schemars(length(min = 1))]
    pub boundaries: Vec<PressureBoundary>,
}

/// The flow direction: that of the gradient of a scene map (e.g. a zonation coordinate).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Direction {
    /// Id of a map of the scene.
    pub map: String,
    /// Gradient magnitude in 1/um below which the direction fades to isotropic, must be > 0.
    #[schemars(extend("exclusiveMinimum" = 0))]
    pub g0: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PressureBoundary {
    /// Name of the boundary tag in the mesh.
    pub tag: String,
    /// Fixed pressure in Pa.
    pub pressure: f64,
}

/// Check that every map a coefficient table or the perfusion direction names is in
/// `scene.maps`.
pub fn check_maps(study: &Study, scene: &Scene) -> Result<(), ContractError> {
    let mut named: Vec<&str> = Vec::new();
    for f in &study.fields {
        for c in [&f.diffusion, &f.decay, &f.source, &f.porosity] {
            named.extend(c.maps());
        }
    }
    if let Some(p) = &study.perfusion {
        named.extend(p.k_along.maps());
        named.extend(p.k_across.maps());
        named.push(&p.direction.map);
    }
    let available: Vec<String> = scene.map_ids().into_iter().map(String::from).collect();
    match named.iter().find(|m| !available.iter().any(|a| a == *m)) {
        Some(m) => Err(ContractError::UnknownMap {
            map: m.to_string(),
            available,
        }),
        None => Ok(()),
    }
}

/// How the cell models are solved.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum CellSolverKind {
    Batched,
    Engine,
}

fn batched() -> CellSolverKind {
    CellSolverKind::Batched
}

fn default_rtol() -> f64 {
    1e-6
}

fn default_atol() -> f64 {
    1e-9
}

/// Cell models at the elements of the mesh: one model instance per element, coupled to the
/// fields by operator splitting.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CellsSpec {
    /// SBML model, relative to the study file (`Study::load` leaves it as written).
    pub model: PathBuf,
    /// Solver of the cell models, default `batched`.
    #[serde(default = "batched")]
    pub solver: CellSolverKind,
    /// Seconds per model time unit, must be > 0.
    #[schemars(extend("exclusiveMinimum" = 0))]
    pub time_unit_s: f64,
    /// Area tags of the mesh that hold cells; default all area tags.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub areas: Option<Vec<String>>,
    /// Model species exchanged with fields.
    pub species: Vec<SpeciesLink>,
    /// Compartments of the model whose volumes follow the porosity.
    pub volumes: VolumeLink,
    /// Model parameters set per element.
    pub parameters: Vec<CellParameter>,
    /// Splitting step in s, must be > 0 and >= `time.dt`.
    #[schemars(extend("exclusiveMinimum" = 0))]
    pub split_step: f64,
    /// Relative tolerance of the cell solver, default 1e-6.
    #[serde(default = "default_rtol")]
    pub rtol: f64,
    /// Absolute tolerance of the cell solver, default 1e-9.
    #[serde(default = "default_atol")]
    pub atol: f64,
    /// Extra assignment rule targets written per cell (states and rules are checked at study
    /// start against the compiled model); default none.
    #[serde(default)]
    pub outputs: Vec<String>,
    /// Spread the work of the cells evenly over the processes, default true: before each step
    /// of the cells, cells move from processes of more work to processes of less, by their step
    /// attempts in the step before (the mesh partition stays). A cell takes the same steps
    /// wherever it is advanced, so the results are the same either way, bit for bit. The
    /// engine path does not move the cells of a model with events, discontinuities or
    /// algebraic rules (its integrator keeps state from one step to the next).
    #[serde(default = "yes", skip_serializing_if = "is_true")]
    pub balance: bool,
}

/// A model species exchanged with a field.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SpeciesLink {
    /// Species id in the model.
    pub species: String,
    /// Id of a field of the study.
    pub field: String,
}

/// Compartment ids of the model, set per element to the porosity (blood) and 1 - porosity
/// (cells).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct VolumeLink {
    pub blood: String,
    pub cells: String,
}

/// A model parameter set per element.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CellParameter {
    /// Parameter id in the model.
    pub id: String,
    /// Value (a number or a table over a map).
    #[schemars(schema_with = "any_coefficient_schema")]
    pub value: Coefficient,
    /// Replace the value where a map is high.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub replace: Option<Replace>,
}

/// Where the element mean of `map` is above `above`, the parameter takes `value` instead.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Replace {
    /// Id of a map of the scene.
    pub map: String,
    pub above: f64,
    #[schemars(schema_with = "any_coefficient_schema")]
    pub value: Coefficient,
}

/// Check the `cells` section against the study and the scene: linked fields and named maps
/// exist, `split_step > 0` and `time.dt <= split_step`. The `outputs` are checked against the
/// compiled model at study start.
pub fn check_cells(study: &Study, scene: &Scene) -> Result<(), ContractError> {
    let Some(cells) = &study.cells else {
        return Ok(());
    };
    let fields: Vec<String> = study.fields.iter().map(|f| f.id.clone()).collect();
    for link in &cells.species {
        if !fields.contains(&link.field) {
            return Err(invalid(format!(
                "cells.species '{}': unknown field '{}', the study defines: [{}]",
                link.species,
                link.field,
                fields.join(", ")
            )));
        }
    }
    let available: Vec<String> = scene.map_ids().into_iter().map(String::from).collect();
    for p in &cells.parameters {
        let mut named: Vec<&str> = p.value.maps().into_iter().collect();
        if let Some(r) = &p.replace {
            named.push(&r.map);
            named.extend(r.value.maps());
        }
        if let Some(m) = named.iter().find(|m| !available.contains(&m.to_string())) {
            return Err(ContractError::UnknownMap {
                map: m.to_string(),
                available,
            });
        }
    }
    if !(cells.split_step.is_finite() && cells.split_step > 0.0) {
        return Err(invalid(format!(
            "cells.split_step must be > 0, got {}",
            cells.split_step
        )));
    }
    if study.time.dt > cells.split_step {
        return Err(invalid(format!(
            "time.dt ({}) must be <= cells.split_step ({})",
            study.time.dt, cells.split_step
        )));
    }
    Ok(())
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

/// The default of [`SolverOptions::linear_rtol`]: the setting at which runs on different numbers
/// of processes agree within about 1e-10.
pub const DEFAULT_LINEAR_RTOL: f64 = 1e-10;

fn default_linear_rtol() -> f64 {
    DEFAULT_LINEAR_RTOL
}

fn is_default_linear_rtol(rtol: &f64) -> bool {
    *rtol == DEFAULT_LINEAR_RTOL
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SolverOptions {
    /// Relative tolerance of the linear solves of the transport (the residual of each solve
    /// reduced by it), in (0, 1), default 1e-10. With fixed steps the time integration has no
    /// error estimate to derive it from: the study sets it. `-ksp_rtol` in `petsc_options`
    /// overrides it.
    #[serde(
        default = "default_linear_rtol",
        skip_serializing_if = "is_default_linear_rtol"
    )]
    #[schemars(extend("exclusiveMinimum" = 0, "exclusiveMaximum" = 1))]
    pub linear_rtol: f64,
    /// PETSc options string, e.g. `-ksp_type cg -pc_type jacobi`.
    #[serde(default)]
    pub petsc_options: String,
}

impl Default for SolverOptions {
    fn default() -> SolverOptions {
        SolverOptions {
            linear_rtol: DEFAULT_LINEAR_RTOL,
            petsc_options: String::new(),
        }
    }
}

impl SolverOptions {
    /// The default options with the PETSc options `petsc_options`.
    pub fn with_petsc_options(petsc_options: impl Into<String>) -> SolverOptions {
        SolverOptions {
            petsc_options: petsc_options.into(),
            ..SolverOptions::default()
        }
    }

    fn validate(&self) -> Result<(), ContractError> {
        let rtol = self.linear_rtol;
        if !(rtol > 0.0 && rtol < 1.0) {
            return Err(invalid(format!(
                "solver.linear_rtol must be in (0, 1), got {rtol}"
            )));
        }
        Ok(())
    }
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
        if let Some(p) = &self.perfusion {
            p.validate()?;
        }
        if self.perfusion.is_none()
            && let Some(f) = self.fields.iter().find(|f| f.advection)
        {
            return Err(invalid(format!(
                "field '{}': advection needs a 'perfusion' section",
                f.id
            )));
        }
        if let Some(c) = &self.cells {
            c.validate()?;
        }
        self.solver.validate()?;
        if let Some(r) = self.result.as_ref().and_then(|r| r.raster.as_ref()) {
            r.validate()?;
        }
        self.time.validate()
    }
}

impl RasterOptions {
    fn validate(&self) -> Result<(), ContractError> {
        finite("result.raster.um_per_px", self.um_per_px)?;
        if self.um_per_px <= 0.0 {
            return Err(invalid(format!(
                "result.raster.um_per_px must be > 0, got {}",
                self.um_per_px
            )));
        }
        if self.levels < 1 {
            return Err(invalid("result.raster.levels must be >= 1".into()));
        }
        if let Some([x, y]) = self.origin_um {
            finite("result.raster.origin_um", x)?;
            finite("result.raster.origin_um", y)?;
        }
        Ok(())
    }
}

impl CellsSpec {
    fn validate(&self) -> Result<(), ContractError> {
        for (what, x) in [
            ("time_unit_s", self.time_unit_s),
            ("rtol", self.rtol),
            ("atol", self.atol),
        ] {
            finite(&format!("cells.{what}"), x)?;
        }
        if self.time_unit_s <= 0.0 {
            return Err(invalid(format!(
                "cells.time_unit_s must be > 0, got {}",
                self.time_unit_s
            )));
        }
        for p in &self.parameters {
            let id = &p.id;
            p.value
                .validate(&format!("cells.parameters '{id}': value"), Rule::Any)?;
            if let Some(r) = &p.replace {
                finite(&format!("cells.parameters '{id}': replace.above"), r.above)?;
                r.value.validate(
                    &format!("cells.parameters '{id}': replace.value"),
                    Rule::Any,
                )?;
            }
        }
        Ok(())
    }
}

impl FieldSpec {
    fn validate(&self) -> Result<(), ContractError> {
        let id = &self.id;
        let valid_id = id.starts_with(|c: char| c.is_ascii_alphabetic())
            && id.chars().all(|c| c.is_ascii_alphanumeric() || c == '_');
        if !valid_id {
            return Err(invalid(format!(
                "field id '{id}' must match ^[A-Za-z][A-Za-z0-9_]*$"
            )));
        }
        finite(&format!("field '{id}': initial"), self.initial)?;
        for (name, c, rule) in [
            ("diffusion", &self.diffusion, Rule::Positive),
            ("decay", &self.decay, Rule::NonNegative),
            ("source", &self.source, Rule::Any),
            ("porosity", &self.porosity, Rule::Positive),
        ] {
            c.validate(&format!("field '{id}': {name}"), rule)?;
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

impl Perfusion {
    fn validate(&self) -> Result<(), ContractError> {
        finite("perfusion.viscosity", self.viscosity)?;
        if self.viscosity <= 0.0 {
            return Err(invalid(format!(
                "perfusion.viscosity must be > 0, got {}",
                self.viscosity
            )));
        }
        self.k_along.validate("perfusion.k_along", Rule::Positive)?;
        self.k_across
            .validate("perfusion.k_across", Rule::Positive)?;
        finite("perfusion.direction.g0", self.direction.g0)?;
        if self.direction.g0 <= 0.0 {
            return Err(invalid(format!(
                "perfusion.direction.g0 must be > 0, got {}",
                self.direction.g0
            )));
        }
        if self.boundaries.is_empty() {
            return Err(invalid(
                "perfusion.boundaries must have at least one boundary".into(),
            ));
        }
        let mut tags = HashSet::new();
        for b in &self.boundaries {
            finite(
                &format!("perfusion.boundaries: tag '{}' pressure", b.tag),
                b.pressure,
            )?;
            if !tags.insert(b.tag.as_str()) {
                return Err(invalid(format!(
                    "perfusion.boundaries: duplicate tag '{}'",
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
    fn raster_options_are_checked() {
        let with = |raster: serde_json::Value| {
            let mut s = study();
            s.result =
                Some(serde_json::from_value(serde_json::json!({ "raster": raster })).unwrap());
            s.validate()
        };
        with(serde_json::json!({"um_per_px": 2.0, "levels": 3})).unwrap();
        with(serde_json::json!({"um_per_px": 2.0, "levels": 1, "origin_um": [-5.0, 1.0]})).unwrap();
        for (raster, message) in [
            (
                serde_json::json!({"um_per_px": 0.0, "levels": 1}),
                "result.raster.um_per_px must be > 0",
            ),
            (
                serde_json::json!({"um_per_px": -1.0, "levels": 1}),
                "result.raster.um_per_px must be > 0",
            ),
            (
                serde_json::json!({"um_per_px": 1.0, "levels": 0}),
                "result.raster.levels must be >= 1",
            ),
        ] {
            let e = with(raster).unwrap_err().to_string();
            assert!(e.contains(message), "{e}");
        }
        let mut s = study();
        s.result = Some(ResultOptions {
            raster: Some(RasterOptions {
                um_per_px: f64::NAN,
                origin_um: None,
                levels: 1,
            }),
        });
        assert!(
            s.validate()
                .unwrap_err()
                .to_string()
                .contains("result.raster.um_per_px")
        );
        s.result = Some(ResultOptions {
            raster: Some(RasterOptions {
                um_per_px: 1.0,
                origin_um: Some([0.0, f64::INFINITY]),
                levels: 1,
            }),
        });
        assert!(
            s.validate()
                .unwrap_err()
                .to_string()
                .contains("result.raster.origin_um")
        );
    }

    #[test]
    fn non_finite_numbers_are_rejected() {
        let mut s = study();
        s.fields[0].diffusion = Coefficient::Constant(f64::NAN);
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
        // A table point (JSON has no NaN: only a study built in code can carry one).
        let mut s = study();
        s.fields[0].decay = Coefficient::Table(Table {
            map: "m".into(),
            points: vec![[0.0, 0.1], [f64::NAN, 0.2]],
        });
        assert_eq!(
            s.validate().unwrap_err().to_string(),
            "field 'oxygen': decay: table point must be finite, got NaN"
        );
        let mut s = study();
        s.fields[0].source = Coefficient::Table(Table {
            map: "m".into(),
            points: vec![[0.0, 0.1], [1.0, f64::NEG_INFINITY]],
        });
        assert_eq!(
            s.validate().unwrap_err().to_string(),
            "field 'oxygen': source must be finite, got -inf"
        );
        let mut s = study();
        s.time.end = f64::INFINITY;
        assert_eq!(
            s.validate().unwrap_err().to_string(),
            "time.end must be finite, got inf"
        );
    }
}
