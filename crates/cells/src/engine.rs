//! The engine path: one sbmlsim-core engine instance per cell ([`EngineCells`]).

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use contracts::CellsSpec;
use petsc::{Comm, OwnedComm};
use sbmlsim_abi::{IntegratorSettings, ModelDescription, Tolerance};
use sbmlsim_c::CModel;
use sbmlsim_engine::Stepper;

use crate::balance::Balance;
use crate::model::{Link, Outputs, Work};
use crate::{
    CellModel, CellSetup, CellSolver, CellStats, CellsError, OutputValue, Poison, Result, agree,
    agree_first, position,
};

/// The cells of a process on sbmlsim-core's engine: the model translated to C, compiled once
/// and loaded by every process ([`CellModel::new`]; `sbmlsim_c`, cached across processes and
/// runs, see [`sbmlsim_c::Library::build`]), and one [`Stepper`] per cell, initialized with the
/// cell's parameters ([`CellSetup::parameters`]) as a simulation initializes its model.
///
/// An advance starts each stepper from the cell's state with the linked species set to their
/// concentrations in the blood (as amounts, times the volume of their compartment, unless the
/// model stores the concentration), and integrates it with CVODE (or IDA) and the tolerances of
/// the spec, events included. Each stepper chooses its own steps, starting every advance
/// afresh, so a cell does not depend on any other cell or on the number of processes.
///
/// With the balance of the work ([`CellsSpec::balance`], `Balance`), a process advances some
/// cells of others: each on a stepper of its own for the cell, made from the cell's parameters
/// and kept while the cell comes, from the cell's state with the blood. A stepper keeps nothing
/// else from one advance to the next for a model without events, discontinuities and algebraic
/// rules, so a cell takes the same steps on either; the cells of any other model stay.
pub struct EngineCells {
    steppers: Vec<Stepper>,
    species: Vec<String>,
    state_ids: Vec<String>,
    /// Of every cell, `n_cells * state_ids.len()`.
    states: Vec<f64>,
    /// Of each linked species.
    links: Vec<Link>,
    time_unit_s: f64,
    elements: Vec<u64>,
    centroids: Vec<[f64; 2]>,
    comm: OwnedComm,
    /// The error of a failed advance, after which the cells are not advanced again.
    poison: Poison,
    /// The state of every cell with the blood, at the start of an advance.
    inputs: Vec<f64>,
    /// The step attempts of every cell in an advance.
    attempts: Vec<u64>,
    /// The work of the cells spread over the processes.
    balance: Balance,
    /// The parameters of the model a cell sets ([`CellSetup::parameters`], the same for every
    /// cell), with the balance on.
    change_ids: Vec<String>,
    /// Their values, of every cell, with the balance on.
    changes: Vec<f64>,
    /// The steppers of the cells of other processes this process advanced in the last advance,
    /// by their element ids.
    guests: HashMap<u64, Stepper>,
    model: CModel,
    description: Arc<ModelDescription>,
    settings: IntegratorSettings,
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
        let comm = setup.owned_comm();
        agree(&comm.comm(), setup.check())?;
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
                    element: setup.elements[i],
                    centroid: setup.centroids[i],
                    message: format!("the cell model does not initialize: {e}"),
                })
            })
            .collect::<Result<Vec<Stepper>>>();
        let steppers = agree(&comm.comm(), created)?;
        let states: Vec<f64> = steppers
            .iter()
            .flat_map(|s| s.state().iter().copied())
            .collect();
        let movable = description.events.is_empty()
            && description.n_roots == 0
            && description.n_algebraic == 0;
        let change_ids = if spec.balance && movable && comm.comm().size() > 1 {
            change_ids(&comm.comm(), setup)?
        } else {
            None
        };
        let n = steppers.len();
        Ok(EngineCells {
            inputs: vec![0.0; states.len()],
            attempts: vec![0; n],
            balance: Balance::new(comm.clone(), n, change_ids.is_some()),
            changes: match &change_ids {
                Some(_) => setup.parameters.iter().flatten().map(|(_, v)| *v).collect(),
                None => Vec::new(),
            },
            change_ids: change_ids.unwrap_or_default(),
            guests: HashMap::new(),
            model: model.clone(),
            description: Arc::clone(description),
            settings,
            steppers,
            species: spec.species.iter().map(|l| l.species.clone()).collect(),
            state_ids: description.states.iter().map(|s| s.id.clone()).collect(),
            states,
            links,
            time_unit_s: spec.time_unit_s,
            elements: setup.elements.clone(),
            centroids: setup.centroids.clone(),
            comm,
            poison: Poison::default(),
            outputs,
            t: 0.0,
        })
    }

    /// An estimate of the memory of the cells of this process in bytes: the steppers
    /// ([`Stepper::memory_bytes`], complete after the first advance), those of the cells of
    /// other processes of the last advance, and the states. The description and the code of the
    /// model, which the cells share, are not counted.
    pub fn memory_bytes(&self) -> usize {
        self.steppers
            .iter()
            .chain(self.guests.values())
            .map(Stepper::memory_bytes)
            .sum::<usize>()
            + (self.states.capacity() + self.inputs.capacity() + self.changes.capacity())
                * size_of::<f64>()
            + self.attempts.capacity() * size_of::<u64>()
    }

    /// The advance of every cell of this process; a failure the error of the first cell which
    /// failed, with its [`position`]. With the balance on, the cells this process keeps and
    /// those of other processes it is sent are advanced here, and the others of this process
    /// elsewhere. Collective, if the balance is on.
    fn advance_local(
        &mut self,
        t0: f64,
        t1: f64,
        blood: &mut [f64],
    ) -> Result<CellStats, (u64, CellsError)> {
        let (n, ns) = (self.steppers.len(), self.state_ids.len());
        let nb = self.links.len();
        let rank = self.comm.comm().rank();
        let here = |e: CellsError| (position(rank, 0), e);
        let valid = if blood.len() != n * nb {
            Err(CellsError::Invalid(format!(
                "{} blood values for {n} cells of {nb} species",
                blood.len()
            )))
        } else if !(t0.is_finite() && t1.is_finite() && t0 <= t1) {
            Err(CellsError::Invalid(format!(
                "an advance of the cells from t = {t0} s to t = {t1} s"
            )))
        } else {
            Ok(())
        };
        // Where a process cannot advance, none does: the error of the one which cannot.
        if !self.balance.plan(valid.is_ok()).map_err(here)? {
            return valid.map(|()| CellStats::default()).map_err(here);
        }
        let (m0, m1) = (t0 / self.time_unit_s, t1 / self.time_unit_s);
        // The state of each cell with the concentrations of the blood.
        self.inputs.copy_from_slice(&self.states);
        for (i, stepper) in self.steppers.iter().enumerate() {
            let y = &mut self.inputs[i * ns..(i + 1) * ns];
            for (link, c) in self.links.iter().zip(&blood[i * nb..(i + 1) * nb]) {
                let volume = link.volume.of(stepper.parameters(), y);
                y[link.state] = c * volume;
            }
        }
        // A cell sent: its element id (its bits), its index, its centroid, its parameters and
        // its state; back: its state, its steps and its rejected steps.
        let nc = self.change_ids.len();
        let (sent, back) = (4 + nc + ns, ns + 2);
        let (inputs, elements, centroids) = (&self.inputs, &self.elements, &self.centroids);
        let changes = &self.changes;
        let guests = self
            .balance
            .scatter(sent, |i, out| {
                out.push(f64::from_bits(elements[i]));
                out.push(i as f64);
                out.extend_from_slice(&centroids[i]);
                out.extend_from_slice(&changes[i * nc..(i + 1) * nc]);
                out.extend_from_slice(&inputs[i * ns..(i + 1) * ns]);
            })
            .map_err(here)?;
        let start = Instant::now();
        let mut stats = CellStats::default();
        // The first failure, by its position.
        let mut failure: Option<(u64, CellsError)> = None;
        let mut fail = |key: u64, e: CellsError| {
            if failure.as_ref().is_none_or(|(k, _)| key < *k) {
                failure = Some((key, e));
            }
        };
        let kept = self.balance.kept();
        for i in 0..kept {
            let stepper = &mut self.steppers[i];
            let y = &self.inputs[i * ns..(i + 1) * ns];
            match stepper.advance(y, m0, m1) {
                Ok(step) => {
                    stats.steps += step.steps;
                    stats.rejected += step.rejected;
                    self.attempts[i] = step.steps + step.rejected;
                    self.states[i * ns..(i + 1) * ns].copy_from_slice(stepper.state());
                }
                Err(e) => fail(
                    position(rank, i),
                    CellsError::Element {
                        element: self.elements[i],
                        centroid: self.centroids[i],
                        message: format!(
                            "the cell model failed from t = {t0} s to t = {t1} s: {e}"
                        ),
                    },
                ),
            }
        }
        // The cells of other processes, on steppers of their own: those of the last advance
        // kept, the others made.
        let mut last = std::mem::take(&mut self.guests);
        let mut results = Vec::with_capacity(self.balance.guests() * back);
        let mut from = 0;
        let mut received = self.balance.received().iter().copied().enumerate();
        let mut left = 0;
        for record in guests.chunks_exact(sent) {
            while left == 0 {
                let (q, count) = received.next().expect("a process of the cell");
                (from, left) = (q, count);
            }
            left -= 1;
            let element = record[0].to_bits();
            let index = record[1] as usize;
            let centroid = [record[2], record[3]];
            let y = &record[4 + nc..];
            let error = |message: String| CellsError::Element {
                element,
                centroid,
                message,
            };
            let stepper = match last.remove(&element) {
                Some(stepper) => Ok(stepper),
                None => {
                    let changes: Vec<(String, f64)> = self
                        .change_ids
                        .iter()
                        .cloned()
                        .zip(record[4..4 + nc].iter().copied())
                        .collect();
                    Stepper::new(
                        Box::new(self.model.clone()),
                        Arc::clone(&self.description),
                        self.settings.clone(),
                        &changes,
                    )
                    .map_err(|e| error(format!("the cell model does not initialize: {e}")))
                }
            };
            let advanced = stepper.and_then(|mut stepper| {
                let step = stepper.advance(y, m0, m1).map_err(|e| {
                    error(format!(
                        "the cell model failed from t = {t0} s to t = {t1} s: {e}"
                    ))
                })?;
                results.extend_from_slice(stepper.state());
                self.guests.insert(element, stepper);
                Ok(step)
            });
            match advanced {
                Ok(step) => {
                    stats.steps += step.steps;
                    stats.rejected += step.rejected;
                    results.extend([step.steps as f64, step.rejected as f64]);
                }
                Err(e) => {
                    fail(position(from, index), e);
                    results.extend(y.iter().copied().chain([0.0, 0.0]));
                }
            }
        }
        drop(last);
        stats.seconds = start.elapsed().as_secs_f64();
        let (states, attempts) = (&mut self.states, &mut self.attempts);
        self.balance
            .gather(back, &results, |i, record| {
                states[i * ns..(i + 1) * ns].copy_from_slice(&record[..ns]);
                attempts[i] = (record[ns] + record[ns + 1]) as u64;
            })
            .map_err(here)?;
        if let Some(failure) = failure {
            return Err(failure);
        }
        self.balance.weights_mut().copy_from_slice(&self.attempts);
        for (i, stepper) in self.steppers.iter().enumerate() {
            let state = &self.states[i * ns..(i + 1) * ns];
            for (link, c) in self.links.iter().zip(&mut blood[i * nb..(i + 1) * nb]) {
                *c = state[link.state] / link.volume.of(stepper.parameters(), state);
            }
        }
        self.t = t1;
        Ok(stats)
    }
}

/// The ids of the parameters every cell of every process sets ([`CellSetup::parameters`]), if
/// they are the same for every cell, else `None`. Collective.
fn change_ids(comm: &Comm, setup: &CellSetup) -> Result<Option<Vec<String>>> {
    let ids = |cell: &[(String, f64)]| cell.iter().map(|(id, _)| id.clone()).collect::<Vec<_>>();
    // 0: no cells, 1: the ids of every cell, 2: cells of different ids.
    let mine = match setup.parameters.first() {
        None => vec![0],
        Some(first) if setup.parameters.iter().all(|c| ids(c) == ids(first)) => {
            let mut bytes = vec![1];
            bytes.extend(ids(first).join("\0").bytes());
            bytes
        }
        Some(_) => vec![2],
    };
    let all = comm.exchange_bytes(&vec![mine; comm.size()])?;
    if all.iter().any(|b| b[0] == 2) {
        return Ok(None);
    }
    let mut lists = all.iter().filter(|b| b[0] == 1).map(|b| &b[1..]);
    let Some(first) = lists.next() else {
        return Ok(None);
    };
    if lists.any(|l| l != first) {
        return Ok(None);
    }
    let text = String::from_utf8_lossy(first);
    Ok(Some(if text.is_empty() {
        Vec::new()
    } else {
        text.split('\0').map(String::from).collect()
    }))
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
        self.poison.check()?;
        let (local, key) = match self.advance_local(t0, t1, blood) {
            Ok(stats) => (Ok(stats), 0),
            Err((key, e)) => (Err(e), key),
        };
        let agreed = agree_first(&self.comm.comm(), local, key);
        self.poison.keep(agreed)
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
