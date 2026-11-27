//! The cell model of a study: compiled once, loaded by every process, and checked against the
//! spec ([`CellModel`]), the links of its species to the blood, and the outputs of a cell
//! ([`Outputs`]), which every path shares.

#[cfg(feature = "native")]
use std::fs;
#[cfg(feature = "native")]
use std::io::Write as _;
#[cfg(feature = "native")]
use std::path::PathBuf;
use std::sync::Arc;

use contracts::{CellSolverKind, CellsSpec};
#[cfg(feature = "native")]
use petsc::Comm;
use sbmlsim_abi::{EntityKind, ModelDescription};
#[cfg(feature = "native")]
use sbmlsim_c::{CError, CModel, Library, QSource};
#[cfg(feature = "native")]
use sbmlsim_engine::ModelFns;

#[cfg(feature = "native")]
use crate::agree;
#[cfg(not(feature = "native"))]
use crate::kernel::KernelLayout;
use crate::{CellsError, OutputValue, Result};

/// How an output is computed from a cell.
#[derive(Debug, Clone, Copy)]
enum Source {
    /// A state, divided by a volume.
    State(usize, Volume),
    /// A parameter after the assignment rules.
    Parameter(usize),
}

/// A species exchanged with the blood: its state, and how its value is a concentration.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Link {
    pub(crate) state: usize,
    pub(crate) volume: Volume,
}

/// The volume which divides the value of a species into its concentration.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Volume {
    /// The model stores the concentration.
    None,
    /// The compartment is a parameter of the model.
    Parameter(usize),
    /// The compartment is a state of the model.
    State(usize),
}

impl Volume {
    /// The volume, of the parameters `p` and the state `y`.
    pub(crate) fn of(self, p: &[f64], y: &[f64]) -> f64 {
        match self {
            Volume::None => 1.0,
            Volume::Parameter(i) => p[i],
            Volume::State(i) => y[i],
        }
    }
}

/// A cell model compiled to native code and checked against a [`CellsSpec`]: at the start of a
/// study, before anything is solved. Both paths read the model of sbmlsim-core's C target (the
/// engine path integrates it, the batched path initializes its cells and evaluates their rules
/// with it); for the batched path ([`CellSolverKind::Batched`]) it also holds the QFunction of
/// the model. Without the feature `native`, the batched kernel of the model instead
/// ([`CellModel::from_kernel`]), which the host compiled and links.
pub struct CellModel {
    #[cfg(feature = "native")]
    pub(crate) model: CModel,
    pub(crate) description: Arc<ModelDescription>,
    pub(crate) links: Vec<Link>,
    /// For the batched path.
    #[cfg(feature = "native")]
    pub(crate) qfunction: Option<Arc<QModel>>,
    /// The layout of the points of the kernel.
    #[cfg(not(feature = "native"))]
    pub(crate) kernel: KernelLayout,
}

impl std::fmt::Debug for CellModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut debug = f.debug_struct("CellModel");
        debug.field("id", &self.description.id);
        #[cfg(feature = "native")]
        debug.field("batched", &self.qfunction.is_some());
        debug.finish_non_exhaustive()
    }
}

/// Both tolerances of `spec` are positive.
fn check_tolerances(spec: &CellsSpec) -> Result<()> {
    for (what, value) in [("rtol", spec.rtol), ("atol", spec.atol)] {
        if !(value.is_finite() && value > 0.0) {
            return Err(CellsError::Tolerance { what, value });
        }
    }
    Ok(())
}

/// The checks of the model `description` against `spec`, on every path ([`CellModel::new`]):
/// its links.
fn check(description: &ModelDescription, spec: &CellsSpec) -> Result<Vec<Link>> {
    let links = links(description, spec)?;
    check_volumes(description, spec)?;
    check_parameters(description, spec)?;
    check_outputs(description, spec)?;
    Ok(links)
}

impl CellModel {
    /// The cell model of the batched kernel `kernel` (`kernel.json` of sbmlsim-core, which the
    /// host compiled from the SBML of the study and links), checked as `spec` uses it, as
    /// [`CellModel::new`] checks a model: on [`KernelLayout::model`]. Without the feature
    /// `native`.
    ///
    /// Errors: [`CellsError::EngineNeedsNative`] for the engine path; [`CellsError::Model`] for
    /// a kernel whose states or parameters are not those of its model; [`CellsError::Unsupported`]
    /// for a model the kernel does not integrate (events, delays, algebraic rules), and for an
    /// output which is the target of a rule of a kernel which cannot evaluate rules (no
    /// `assigned_ids`), and the errors of the checks of [`CellModel::new`].
    #[cfg(not(feature = "native"))]
    pub fn from_kernel(spec: &CellsSpec, kernel: KernelLayout) -> Result<CellModel> {
        if spec.solver == CellSolverKind::Engine {
            return Err(CellsError::EngineNeedsNative);
        }
        check_tolerances(spec)?;
        kernel.check()?;
        let description = Arc::new(kernel.model.clone());
        let links = check(&description, spec)?;
        if kernel.assigned_ids.is_none()
            && let Some(id) = spec
                .outputs
                .iter()
                .find(|id| description.parameters.iter().any(|p| &p.id == *id))
        {
            return Err(CellsError::Model {
                message: format!(
                    "cells.outputs '{id}' is the target of an assignment rule, and the kernel \
                     cannot evaluate rules (its kernel.json has no assigned_ids: a kernel of \
                     sbmlsim-core without sbmlsim_kernel_assign)"
                ),
            });
        }
        Ok(CellModel {
            description,
            links,
            kernel,
        })
    }

    /// The model `model_sbml` (an SBML document), translated to C and compiled once (`sbmlsim_c`,
    /// cached across processes and runs, see [`sbmlsim_c::Library::build`]: process 0 first,
    /// then the others, which load it from the cache, see `first_on_root`), and
    /// checked as `spec` uses it: every linked species is a species state of the model in the
    /// blood compartment, both volumes are settable compartments, every parameter a settable
    /// entity, every output a state or the target of an assignment rule, and both tolerances
    /// positive. For the batched path, also the QFunction of the model, compiled the same way
    /// (see `QModel`). Collective.
    ///
    /// Errors: [`CellsError::Tolerance`], [`CellsError::UnknownId`] (listing what the model
    /// has) and [`CellsError::Invalid`] for the checks; on every process,
    /// [`CellsError::Model`] if the model cannot be translated, compiled or loaded, and
    /// [`CellsError::Unsupported`] for a model the batched path does not integrate (events,
    /// delays, algebraic rules). With the feature `native`.
    #[cfg(feature = "native")]
    pub fn new(comm: &Comm, spec: &CellsSpec, model_sbml: &str) -> Result<CellModel> {
        check_tolerances(spec)?;
        let (model, description) = first_on_root(comm, || compile(model_sbml))?;
        let links = check(&description, spec)?;
        let qfunction = match spec.solver {
            CellSolverKind::Batched => Some(Arc::new(first_on_root(comm, || {
                QModel::new(model_sbml, &description)
            })?)),
            CellSolverKind::Engine => None,
        };
        Ok(CellModel {
            model,
            description,
            links,
            qfunction,
        })
    }
}

/// `build()` on process 0, then on the others: a library which is not in the cache of the
/// compiled models is compiled by process 0 alone, and the others load it from the cache,
/// instead of every process compiling it at once. The error of process 0 on every process (the
/// others do not build then), else the first error of the others. Collective.
#[cfg(feature = "native")]
fn first_on_root<T>(comm: &Comm, build: impl Fn() -> Result<T>) -> Result<T> {
    let first = (comm.rank() == 0).then(&build);
    let local = match agree(comm, first.transpose())? {
        Some(built) => Ok(built),
        None => build(),
    };
    agree(comm, local)
}

/// The QFunction of a model for the batched path ([`sbmlsim_c::generate_qfunction`]): the model
/// with the Rosenbrock integrator RODAS4, step attempts per point until it is done or failed, at
/// most a number per application (`RosenbrockContext::attempts`).
#[cfg(feature = "native")]
pub(crate) struct QModel {
    pub(crate) q: QSource,
    /// The library of [`QSource::host_code`], whose [`QSource::host_symbol`] the CPU backends of
    /// libCEED call.
    pub(crate) library: Arc<Library>,
    /// The file of [`QSource::code`], the source libCEED's JIT backends compile, beside the
    /// library in the cache.
    pub(crate) source: PathBuf,
}

#[cfg(feature = "native")]
impl QModel {
    /// The QFunction of `sbml`, whose model of the C target is the one of `description`.
    ///
    /// Errors: [`CellsError::Unsupported`] for events, delays and algebraic rules, and for what
    /// else the QFunction target refuses; [`CellsError::Model`] if it cannot be generated,
    /// compiled or loaded.
    fn new(sbml: &str, description: &ModelDescription) -> Result<QModel> {
        let refused = [
            (!description.events.is_empty(), "events"),
            (description.n_delays > 0, "delays"),
            (description.n_algebraic > 0, "algebraic rules"),
        ];
        if let Some((_, feature)) = refused.iter().find(|(has, _)| *has) {
            return Err(CellsError::Unsupported {
                feature: format!("a model with {feature}"),
            });
        }
        let model_error = |e: CError| CellsError::Model {
            message: e.to_string(),
        };
        let q = sbmlsim_c::generate_qfunction(sbml).map_err(|e| match e {
            CError::Unsupported { feature } => CellsError::Unsupported { feature },
            e => model_error(e),
        })?;
        let ids = |entities: &[sbmlsim_abi::Entity]| -> Vec<String> {
            entities.iter().map(|e| e.id.clone()).collect()
        };
        if q.state_ids != ids(&description.states)
            || q.parameter_ids != ids(&description.parameters)
        {
            return Err(CellsError::Model {
                message: format!(
                    "the QFunction of the model {} has other states or parameters than its \
                     model of the C target",
                    description.id
                ),
            });
        }
        let library = Library::build(&q.host_code()).map_err(model_error)?;
        let source = library.path().with_extension("qfunction.c");
        write_once(&source, &q.code).map_err(|e| CellsError::Model {
            message: format!("cannot write {}: {e}", source.display()),
        })?;
        Ok(QModel { q, library, source })
    }
}

/// Writes `text` into the file `path` unless it holds it already: into a file of this process
/// beside it, renamed into place, so that another process never reads a part of it.
#[cfg(feature = "native")]
fn write_once(path: &std::path::Path, text: &str) -> std::io::Result<()> {
    if fs::read(path).is_ok_and(|bytes| bytes == text.as_bytes()) {
        return Ok(());
    }
    let temporary = path.with_extension(format!("{}.tmp", std::process::id()));
    let mut file = fs::File::create(&temporary)?;
    file.write_all(text.as_bytes())?;
    drop(file);
    fs::rename(&temporary, path).inspect_err(|_| {
        let _ = fs::remove_file(&temporary);
    })
}

/// The outputs of the cells ([`crate::CellSolver::output_values`]) and how each is computed from
/// the state and the parameters of a cell, the same on both paths.
pub(crate) struct Outputs {
    #[cfg(feature = "native")]
    model: CModel,
    values: Vec<OutputValue>,
    sources: Vec<Source>,
    /// Whether an output is the target of an assignment rule.
    rules: bool,
    /// The number of constants of the model, behind its parameters.
    #[cfg(feature = "native")]
    n_constants: usize,
}

/// The vectors of the model for [`Outputs::push`], kept from one cell to the next (unused
/// without native code).
#[derive(Debug, Default)]
#[cfg_attr(not(feature = "native"), allow(dead_code))]
pub(crate) struct Work {
    y: Vec<f64>,
    p: Vec<f64>,
}

impl Outputs {
    /// The outputs of `spec` with the model `model`.
    pub(crate) fn new(model: &CellModel, spec: &CellsSpec) -> Outputs {
        let (values, sources) = outputs(&model.description, spec);
        Outputs {
            #[cfg(feature = "native")]
            model: model.model.clone(),
            rules: sources.iter().any(|s| matches!(s, Source::Parameter(_))),
            values,
            sources,
            #[cfg(feature = "native")]
            n_constants: model.description.n_constants as usize,
        }
    }

    /// Whether an output is the target of an assignment rule.
    #[cfg_attr(feature = "native", allow(dead_code))]
    pub(crate) fn has_rules(&self) -> bool {
        self.rules
    }

    /// The values of the outputs.
    pub(crate) fn values(&self) -> &[OutputValue] {
        &self.values
    }

    /// Appends the outputs of a cell of the state `state` and the parameters `parameters` (in
    /// the order of the description, without the constants) at the time `t` of the model to
    /// `out`: a state as it is or as its concentration, the target of an assignment rule
    /// evaluated at the state.
    ///
    /// The rules follow the model contract of sbmlsim-core (`model/2`, see [`sbmlsim_abi`]):
    /// the other functions of a model read the constants (subexpressions of the parameters
    /// alone) from `p` behind the parameters, and `model_assign` computes them there after the
    /// targets of the rules, which is why the engine calls it after every change of the
    /// parameters. So a first call computes the constants of the cell's parameters, and a
    /// second one the targets of the rules with them.
    ///
    /// Without the feature `native` the caller evaluates the rules with the kernel
    /// (`sbmlsim_kernel_assign` of all cells at once, see `KernelCells`), and `parameters` holds
    /// their targets at `t` and the state already.
    pub(crate) fn push(
        &self,
        t: f64,
        state: &[f64],
        parameters: &[f64],
        work: &mut Work,
        out: &mut Vec<f64>,
    ) {
        let p = self.rules(t, state, parameters, work);
        // The rules may write states too; the outputs of the states stay those of the
        // integration.
        for source in &self.sources {
            out.push(match *source {
                Source::State(i, volume) => state[i] / volume.of(p, state),
                Source::Parameter(k) => p[k],
            });
        }
    }

    /// The parameters of a cell with the targets of the rules evaluated at its state (see
    /// [`Outputs::push`]), in `work`.
    #[cfg(feature = "native")]
    fn rules<'w>(
        &self,
        t: f64,
        state: &[f64],
        parameters: &[f64],
        work: &'w mut Work,
    ) -> &'w [f64] {
        let Work { y, p } = work;
        p.clear();
        p.extend_from_slice(parameters);
        if self.rules {
            p.resize(parameters.len() + self.n_constants, 0.0);
            y.clear();
            y.extend_from_slice(state);
            self.model.begin_parameters(p);
            // The constants of the parameters, then the rules with them.
            self.model.assign(t, y, p);
            y.copy_from_slice(state);
            self.model.assign(t, y, p);
            self.model.end_parameters();
        }
        p
    }

    /// The parameters of a cell as they are: without native code the caller evaluated the rules
    /// with the kernel (see [`Outputs::push`]).
    #[cfg(not(feature = "native"))]
    fn rules<'a>(
        &self,
        _t: f64,
        _state: &[f64],
        parameters: &'a [f64],
        _work: &mut Work,
    ) -> &'a [f64] {
        parameters
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
#[cfg(feature = "native")]
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
                entities(),
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
