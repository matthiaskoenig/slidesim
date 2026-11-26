//! The engine path: one sbmlsim-core engine instance per cell ([`EngineCells`]).

use std::sync::Arc;
use std::time::Instant;

use contracts::CellsSpec;
use petsc::Comm;
use sbmlsim_abi::{IntegratorSettings, Tolerance};
use sbmlsim_engine::Stepper;

use crate::model::{Link, Outputs, Work};
use crate::{CellModel, CellSetup, CellSolver, CellStats, CellsError, OutputValue, Result, agree};

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
    outputs: Outputs,
    /// The time the cells reached (s).
    t: f64,
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
        let outputs = Outputs::new(model, spec);
        let CellModel {
            model,
            description,
            links,
            ..
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
            outputs,
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
        self.outputs.values()
    }

    fn outputs(&self, out: &mut Vec<f64>) {
        out.clear();
        let ns = self.state_ids.len();
        let t = self.t / self.time_unit_s;
        let mut work = Work::default();
        for (stepper, state) in self
            .steppers
            .iter()
            .zip(self.states.chunks_exact(ns.max(1)))
        {
            self.outputs
                .push(t, state, stepper.parameters(), &mut work, out);
        }
    }
}
