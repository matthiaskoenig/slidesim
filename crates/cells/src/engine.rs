//! The engine path: one sbmlsim-core engine instance per cell ([`EngineCells`]).

use std::sync::Arc;
use std::time::Instant;

use contracts::CellsSpec;
use petsc::Comm;
use sbmlsim_abi::{EntityKind, IntegratorSettings, ModelDescription, Tolerance};
use sbmlsim_c::{CModel, Library};
use sbmlsim_engine::{ModelFns, Stepper};

use crate::{CellSetup, CellSolver, CellStats, CellsError, OutputValue, Result, agree};

/// The cells of a process on sbmlsim-core's engine: the model translated to C and compiled
/// once per process (`sbmlsim_c`, cached across processes and runs, see
/// [`sbmlsim_c::Library::build`]), and one [`Stepper`] per cell, initialized with the cell's
/// parameters ([`CellSetup::parameters`]) as a simulation initializes its model.
///
/// An advance starts each stepper from the cell's state with the linked species set to their
/// concentrations in the blood (as amounts, times the volume of their compartment, unless the
/// model stores the concentration), and integrates it with CVODE (or IDA) and the tolerances of
/// the spec, events included. Each stepper chooses its own steps, starting every advance
/// afresh, so a cell does not depend on any other cell or on the number of processes.
pub struct EngineCells {
    steppers: Vec<Stepper>,
    species: Vec<String>,
    state_ids: Vec<String>,
    /// Of every cell, `n_cells * state_ids.len()`.
    states: Vec<f64>,
    /// Of each linked species.
    links: Vec<Link>,
    time_unit_s: f64,
    global: Vec<i64>,
    centroids: Vec<[f64; 2]>,
    comm: Comm,
    /// The state of a cell during an advance.
    work: Vec<f64>,
    /// The model, for the assignment rules of the outputs.
    model: CModel,
    /// The number of constants behind the parameters.
    n_constants: usize,
    /// The values of the outputs and how each is computed.
    outputs: Vec<OutputValue>,
    sources: Vec<Source>,
    /// Whether an output is the target of an assignment rule.
    rules: bool,
    /// The time the cells reached (s).
    t: f64,
}

/// How an output is computed from a cell.
#[derive(Debug, Clone, Copy)]
enum Source {
    /// A state, divided by a volume.
    State(usize, Volume),
    /// A parameter after the assignment rules.
    Parameter(usize),
}

impl std::fmt::Debug for EngineCells {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EngineCells")
            .field("cells", &self.steppers.len())
            .field("species", &self.species)
            .field("state_ids", &self.state_ids)
            .field("time_unit_s", &self.time_unit_s)
            .finish_non_exhaustive()
    }
}

/// A species exchanged with the blood: its state, and how its value is a concentration.
#[derive(Debug, Clone, Copy)]
struct Link {
    state: usize,
    volume: Volume,
}

/// The volume which divides the value of a species into its concentration.
#[derive(Debug, Clone, Copy)]
enum Volume {
    /// The model stores the concentration.
    None,
    /// The compartment is a parameter of the model.
    Parameter(usize),
    /// The compartment is a state of the model.
    State(usize),
}

impl Volume {
    /// The volume, of the parameters `p` and the state `y`.
    fn of(self, p: &[f64], y: &[f64]) -> f64 {
        match self {
            Volume::None => 1.0,
            Volume::Parameter(i) => p[i],
            Volume::State(i) => y[i],
        }
    }
}

/// A cell model compiled to native code for [`EngineCells`], and checked against a
/// [`CellsSpec`]: at the start of a study, before anything is solved.
pub struct CellModel {
    model: CModel,
    description: Arc<ModelDescription>,
    links: Vec<Link>,
}

impl std::fmt::Debug for CellModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CellModel")
            .field("id", &self.description.id)
            .finish_non_exhaustive()
    }
}

impl CellModel {
    /// The model `model_sbml` (an SBML document), translated to C and compiled once per process
    /// (`sbmlsim_c`, cached across processes and runs, see [`sbmlsim_c::Library::build`]), and
    /// checked as `spec` uses it: every linked species is a species state of the model in the
    /// blood compartment, both volumes are settable compartments, every parameter a settable
    /// entity, every output a state or the target of an assignment rule, and both tolerances
    /// positive. Collective.
    ///
    /// Errors: [`CellsError::Tolerance`], [`CellsError::UnknownId`] (listing what the model
    /// has) and [`CellsError::Invalid`] for the checks; on every process, [`CellsError::Model`]
    /// if the model cannot be translated, compiled or loaded.
    pub fn new(comm: &Comm, spec: &CellsSpec, model_sbml: &str) -> Result<CellModel> {
        for (what, value) in [("rtol", spec.rtol), ("atol", spec.atol)] {
            if !(value.is_finite() && value > 0.0) {
                return Err(CellsError::Tolerance { what, value });
            }
        }
        let (model, description) = agree(comm, compile(model_sbml))?;
        let links = links(&description, spec)?;
        check_volumes(&description, spec)?;
        check_parameters(&description, spec)?;
        check_outputs(&description, spec)?;
        Ok(CellModel {
            model,
            description,
            links,
        })
    }
}

impl EngineCells {
    /// The cells of `setup` with the model `model_sbml` (an SBML document) as `spec` describes
    /// it: [`CellModel::new`], then [`EngineCells::with_model`]. Collective.
    pub fn new(setup: &CellSetup, spec: &CellsSpec, model_sbml: &str) -> Result<EngineCells> {
        let model = CellModel::new(&setup.comm(), spec, model_sbml)?;
        EngineCells::with_model(setup, spec, &model)
    }

    /// The cells of `setup` with the model `model`, compiled and checked for `spec`.
    /// Collective.
    ///
    /// Errors: [`CellsError::Invalid`] for a setup whose lists are not one entry per cell; on
    /// every process, [`CellsError::Element`] for a cell whose model does not initialize.
    pub fn with_model(
        setup: &CellSetup,
        spec: &CellsSpec,
        model: &CellModel,
    ) -> Result<EngineCells> {
        let comm = setup.comm();
        agree(&comm, setup.check())?;
        let CellModel {
            model,
            description,
            links,
        } = model;
        let links = links.clone();
        let settings = IntegratorSettings {
            relative_tolerance: spec.rtol,
            absolute_tolerance: Tolerance::Scalar(spec.atol),
            ..IntegratorSettings::default()
        };
        let created = setup
            .parameters
            .iter()
            .enumerate()
            .map(|(i, changes)| {
                Stepper::new(
                    Box::new(model.clone()),
                    Arc::clone(description),
                    settings.clone(),
                    changes,
                )
                .map_err(|e| CellsError::Element {
                    cell: setup.global[i],
                    centroid: setup.centroids[i],
                    message: format!("the cell model does not initialize: {e}"),
                })
            })
            .collect::<Result<Vec<Stepper>>>();
        let steppers = agree(&comm, created)?;
        let states = steppers
            .iter()
            .flat_map(|s| s.state().iter().copied())
            .collect();
        let (outputs, sources) = outputs(description, spec);
        let rules = sources.iter().any(|s| matches!(s, Source::Parameter(_)));
        Ok(EngineCells {
            steppers,
            species: spec.species.iter().map(|l| l.species.clone()).collect(),
            state_ids: description.states.iter().map(|s| s.id.clone()).collect(),
            states,
            links,
            time_unit_s: spec.time_unit_s,
            global: setup.global.clone(),
            centroids: setup.centroids.clone(),
            comm,
            work: Vec::with_capacity(description.states.len()),
            model: model.clone(),
            n_constants: description.n_constants as usize,
            outputs,
            sources,
            rules,
            t: 0.0,
        })
    }

    /// An estimate of the memory of the cells of this process in bytes: the steppers
    /// ([`Stepper::memory_bytes`], complete after the first advance) and the states. The
    /// description and the code of the model, which the cells share, are not counted.
    pub fn memory_bytes(&self) -> usize {
        self.steppers
            .iter()
            .map(Stepper::memory_bytes)
            .sum::<usize>()
            + self.states.capacity() * size_of::<f64>()
    }

    /// The advance of every cell of this process; the first failure an error of its element.
    fn advance_local(&mut self, t0: f64, t1: f64, blood: &mut [f64]) -> Result<CellStats> {
        let (n, ns) = (self.steppers.len(), self.state_ids.len());
        let nb = self.links.len();
        if blood.len() != n * nb {
            return Err(CellsError::Invalid(format!(
                "{} blood values for {n} cells of {nb} species",
                blood.len()
            )));
        }
        if !(t0.is_finite() && t1.is_finite() && t0 <= t1) {
            return Err(CellsError::Invalid(format!(
                "an advance of the cells from t = {t0} s to t = {t1} s"
            )));
        }
        let start = Instant::now();
        let (m0, m1) = (t0 / self.time_unit_s, t1 / self.time_unit_s);
        let mut stats = CellStats::default();
        for (i, stepper) in self.steppers.iter_mut().enumerate() {
            let state = &mut self.states[i * ns..(i + 1) * ns];
            let blood = &mut blood[i * nb..(i + 1) * nb];
            self.work.clear();
            self.work.extend_from_slice(state);
            for (link, c) in self.links.iter().zip(blood.iter()) {
                let volume = link.volume.of(stepper.parameters(), &self.work);
                self.work[link.state] = c * volume;
            }
            let step = stepper
                .advance(&self.work, m0, m1)
                .map_err(|e| CellsError::Element {
                    cell: self.global[i],
                    centroid: self.centroids[i],
                    message: format!("the cell model failed from t = {t0} s to t = {t1} s: {e}"),
                })?;
            stats.steps += step.steps;
            stats.rejected += step.rejected;
            state.copy_from_slice(stepper.state());
            for (link, c) in self.links.iter().zip(blood.iter_mut()) {
                *c = state[link.state] / link.volume.of(stepper.parameters(), state);
            }
        }
        stats.seconds = start.elapsed().as_secs_f64();
        self.t = t1;
        Ok(stats)
    }
}

impl CellSolver for EngineCells {
    fn n_cells(&self) -> usize {
        self.steppers.len()
    }

    fn species(&self) -> &[String] {
        &self.species
    }

    fn state_ids(&self) -> &[String] {
        &self.state_ids
    }

    fn advance(&mut self, t0: f64, t1: f64, blood: &mut [f64]) -> Result<CellStats> {
        let local = self.advance_local(t0, t1, blood);
        agree(&self.comm, local)
    }

    fn states(&self) -> &[f64] {
        &self.states
    }

    fn output_values(&self) -> &[OutputValue] {
        &self.outputs
    }

    fn outputs(&self, out: &mut Vec<f64>) {
        out.clear();
        let ns = self.state_ids.len();
        let t = self.t / self.time_unit_s;
        let mut y = Vec::with_capacity(ns);
        let mut p = Vec::new();
        for (stepper, state) in self
            .steppers
            .iter()
            .zip(self.states.chunks_exact(ns.max(1)))
        {
            y.clear();
            y.extend_from_slice(state);
            p.clear();
            p.extend_from_slice(stepper.parameters());
            if self.rules {
                p.resize(p.len() + self.n_constants, 0.0);
                // The first call writes the constants behind the parameters, which the rules
                // may read; the second evaluates the rules with them.
                for _ in 0..2 {
                    self.model.begin_parameters(&p);
                    self.model.assign(t, &mut y, &mut p);
                    self.model.end_parameters();
                }
                // The rules may write states too; the outputs of the states stay those of the
                // integration.
                y.copy_from_slice(state);
            }
            for source in &self.sources {
                out.push(match *source {
                    Source::State(i, volume) => y[i] / volume.of(&p, &y),
                    Source::Parameter(i) => p[i],
                });
            }
        }
    }
}

/// The outputs of the cells (see [`CellSolver::output_values`]) and how each is computed: the
/// states, a species of a concentration stored as its amount divided by its compartment, then
/// the outputs of `spec` which are parameters (targets of assignment rules; the check of the
/// outputs makes sure every output is a state or one of these).
fn outputs(description: &ModelDescription, spec: &CellsSpec) -> (Vec<OutputValue>, Vec<Source>) {
    let mut values = Vec::new();
    let mut sources = Vec::new();
    for (i, state) in description.states.iter().enumerate() {
        let (compartment, volume) = match &state.kind {
            EntityKind::Species {
                compartment,
                has_only_substance_units: false,
                stores_concentration,
            } => {
                let volume = if *stores_concentration {
                    Some(Volume::None)
                } else if let Some(k) = description
                    .parameters
                    .iter()
                    .position(|p| &p.id == compartment)
                {
                    Some(Volume::Parameter(k))
                } else {
                    description
                        .states
                        .iter()
                        .position(|s| &s.id == compartment)
                        .map(Volume::State)
                };
                match volume {
                    Some(volume) => (Some(compartment.clone()), volume),
                    None => (None, Volume::None),
                }
            }
            _ => (None, Volume::None),
        };
        values.push(OutputValue {
            id: state.id.clone(),
            compartment,
        });
        sources.push(Source::State(i, volume));
    }
    for id in &spec.outputs {
        if description.states.iter().any(|s| &s.id == id) {
            continue;
        }
        if let Some(k) = description.parameters.iter().position(|p| &p.id == id) {
            let compartment = match &description.parameters[k].kind {
                EntityKind::Species {
                    compartment,
                    has_only_substance_units: false,
                    ..
                } => Some(compartment.clone()),
                _ => None,
            };
            values.push(OutputValue {
                id: id.clone(),
                compartment,
            });
            sources.push(Source::Parameter(k));
        }
    }
    (values, sources)
}

/// The model of `sbml` compiled to native code, and its description.
fn compile(sbml: &str) -> Result<(CModel, Arc<ModelDescription>)> {
    let model_error = |e: sbmlsim_c::CError| CellsError::Model {
        message: e.to_string(),
    };
    let source = sbmlsim_c::generate(sbml).map_err(model_error)?;
    let library = Library::build(&source.code).map_err(model_error)?;
    let model = CModel::new(library, &source.description).map_err(model_error)?;
    Ok((model, Arc::new(source.description)))
}

/// The links of the species of `spec` to the states of the model: each a species state in the
/// blood compartment.
fn links(description: &ModelDescription, spec: &CellsSpec) -> Result<Vec<Link>> {
    let blood = &spec.volumes.blood;
    spec.species
        .iter()
        .map(|link| {
            let id = &link.species;
            let found = description
                .states
                .iter()
                .enumerate()
                .find_map(|(i, s)| match &s.kind {
                    EntityKind::Species {
                        compartment,
                        stores_concentration,
                        ..
                    } if &s.id == id => Some((i, compartment, *stores_concentration)),
                    _ => None,
                });
            let Some((state, compartment, stores_concentration)) = found else {
                return Err(unknown(
                    "cells.species",
                    id,
                    "species which is a state",
                    description.states.iter().filter(|s| is_species(&s.kind)),
                ));
            };
            if compartment != blood {
                return Err(CellsError::Invalid(format!(
                    "cells.species '{id}' is in the compartment '{compartment}', not in the \
                     blood compartment '{blood}' (cells.volumes.blood)"
                )));
            }
            let volume = if stores_concentration {
                Volume::None
            } else if let Some(i) = description.parameters.iter().position(|p| &p.id == blood) {
                Volume::Parameter(i)
            } else if let Some(i) = description.states.iter().position(|s| &s.id == blood) {
                Volume::State(i)
            } else {
                return Err(CellsError::Invalid(format!(
                    "the compartment '{blood}' of the species '{id}' is neither a state nor a \
                     parameter of the model"
                )));
            };
            Ok(Link { state, volume })
        })
        .collect()
}

/// Both volumes of `spec` are settable compartments of the model.
fn check_volumes(description: &ModelDescription, spec: &CellsSpec) -> Result<()> {
    let compartment = |e: &&sbmlsim_abi::Entity| e.settable && e.kind == EntityKind::Compartment;
    for (what, id) in [
        ("cells.volumes.blood", &spec.volumes.blood),
        ("cells.volumes.cells", &spec.volumes.cells),
    ] {
        if !description
            .parameters
            .iter()
            .filter(compartment)
            .any(|p| &p.id == id)
        {
            return Err(unknown(
                what,
                id,
                "constant compartment",
                description.parameters.iter().filter(compartment),
            ));
        }
    }
    Ok(())
}

/// Every parameter of `spec` is a settable entity of the model (a parameter, a compartment or
/// a species, whose initial value it sets).
fn check_parameters(description: &ModelDescription, spec: &CellsSpec) -> Result<()> {
    let entities = || {
        description
            .parameters
            .iter()
            .chain(&description.states)
            .filter(|e| e.settable)
    };
    for p in &spec.parameters {
        if !entities().any(|e| e.id == p.id) {
            return Err(unknown(
                "cells.parameters",
                &p.id,
                "settable entity",
                description.parameters.iter().filter(|e| e.settable),
            ));
        }
    }
    Ok(())
}

/// Every output of `spec` is a state of the model or the target of an assignment rule (a
/// parameter which no change may set).
fn check_outputs(description: &ModelDescription, spec: &CellsSpec) -> Result<()> {
    let outputs = || {
        description
            .states
            .iter()
            .chain(description.parameters.iter().filter(|p| !p.settable))
    };
    for id in &spec.outputs {
        if !outputs().any(|e| &e.id == id) {
            return Err(unknown(
                "cells.outputs",
                id,
                "state or assignment rule target",
                outputs(),
            ));
        }
    }
    Ok(())
}

fn is_species(kind: &EntityKind) -> bool {
    matches!(kind, EntityKind::Species { .. })
}

/// [`CellsError::UnknownId`] of `id` of `what`, with the ids of `available`.
fn unknown<'a>(
    what: &str,
    id: &str,
    expected: &str,
    available: impl Iterator<Item = &'a sbmlsim_abi::Entity>,
) -> CellsError {
    CellsError::UnknownId {
        what: what.to_string(),
        id: id.to_string(),
        expected: expected.to_string(),
        available: available.map(|e| e.id.clone()).collect(),
    }
}
