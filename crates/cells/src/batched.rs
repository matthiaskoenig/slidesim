//! The batched path: the cells of a process as one libCEED operator of the QFunction of their
//! model ([`BatchedCells`]).

use std::ffi::c_void;
use std::os::raw::c_int;
use std::sync::Arc;
use std::time::Instant;

use ceed::sys::{CeedInt, CeedScalar};
use ceed::{CVector, Ceed, ElemRestriction, EvalMode, FieldVector, Operator, QFunction};
use contracts::CellsSpec;
use petsc::Comm;
use sbmlsim_abi::{IntegratorSettings, Tolerance};
use sbmlsim_c::{
    CONTROL, CONTROL_DONE, CONTROL_FAILED, CONTROL_T, COUNTS, INPUTS, Library, OUTPUTS,
    RosenbrockContext,
};
use sbmlsim_engine::Stepper;

use crate::model::{Link, Outputs, QModel, Work};
use crate::{CellModel, CellSetup, CellSolver, CellStats, CellsError, OutputValue, Result, agree};

/// The smallest step of a cell, in the time unit of its model: a cell which needs a smaller one
/// fails.
const H_MIN: f64 = 1e-10;

/// The most step attempts of a cell in one advance: a cell which has not reached the end of
/// the advance after as many fails.
const MAX_ATTEMPTS: u64 = 1_000_000;

/// The QFunction of a library, of the signature of `CeedQFunctionUser`.
type Function = unsafe extern "C" fn(
    *mut c_void,
    CeedInt,
    *const *const CeedScalar,
    *const *mut CeedScalar,
) -> c_int;

// The context of the QFunction is its four values, in the order of `RosenbrockContext`.
const _: () = assert!(size_of::<RosenbrockContext>() == 4 * size_of::<f64>());

/// The cells of a process as one libCEED operator: one point per cell (one element of one node
/// per cell, strided restrictions, no basis), the QFunction of the model
/// ([`sbmlsim_c::generate_qfunction`]: the model with the Rosenbrock integrator RODAS4, one step
/// attempt per point and application) compiled once per process (`sbmlsim_c`, cached across
/// processes and runs).
///
/// The cells are initialized as the engine path initializes its steppers (a
/// [`sbmlsim_engine::Stepper`] of the model of the C target per cell, the defaults, the cell's
/// parameters, the initial assignments), so both paths start from the same states.
///
/// An advance sets the linked species to their concentrations in the blood (as amounts, times
/// the volume of their compartment, unless the model stores the concentration), then applies
/// the operator until every cell has reached the end of the advance or failed, the outputs of
/// one application the inputs of the next (two operators, which read the fields of one set of
/// vectors and write the other). Each cell takes its own steps with the tolerances of the spec
/// and keeps its step from one advance to the next, so a cell depends on no other cell, the
/// number of processes, the backend or its blocks. A cell fails where its next step would be
/// smaller than [`H_MIN`] (in the time unit of the model), or after [`MAX_ATTEMPTS`] step
/// attempts in one advance.
pub struct BatchedCells {
    species: Vec<String>,
    state_ids: Vec<String>,
    /// Of every cell, `n_cells * state_ids.len()`, after the last advance.
    states: Vec<f64>,
    /// The states during an advance, as long as `states`.
    next: Vec<f64>,
    /// Of every cell, in the order of the parameters of the description (the field
    /// `parameters` of the QFunction).
    parameters: Vec<f64>,
    /// The control of every cell (`t`, `h`, done, failed), after the last application.
    control: Vec<f64>,
    /// The accepted and rejected steps of every cell in the last advance.
    counts: Vec<f64>,
    /// Of each linked species.
    links: Vec<Link>,
    outputs: Outputs,
    time_unit_s: f64,
    rtol: f64,
    atol: f64,
    global: Vec<i64>,
    centroids: Vec<[f64; 2]>,
    comm: Comm,
    /// None where this process has no cells.
    batch: Option<Batch>,
    /// The time the cells reached (s).
    t: f64,
}

impl std::fmt::Debug for BatchedCells {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BatchedCells")
            .field("cells", &self.global.len())
            .field("species", &self.species)
            .field("state_ids", &self.state_ids)
            .field("time_unit_s", &self.time_unit_s)
            .finish_non_exhaustive()
    }
}

/// The operators of the cells of a process and their vectors.
struct Batch {
    qfunction: QFunction,
    /// `operators[k]` reads the states, controls and counts of the set `k` of vectors (and the
    /// parameters) and writes those of the other set.
    operators: [Operator; 2],
    state: [CVector; 2],
    control: [CVector; 2],
    counts: [CVector; 2],
    /// The set of vectors which holds the cells.
    current: usize,
    /// The library of the function of the QFunction, dropped after the QFunction and the
    /// operators (the fields are dropped in their order).
    _library: Arc<Library>,
}

impl Batch {
    /// The operators of the QFunction of `qmodel` on `ceed` for the `n` cells of the
    /// `parameters`, with `context`.
    fn new(
        ceed: &Ceed,
        qmodel: &QModel,
        parameters: &[f64],
        n: usize,
        context: &RosenbrockContext,
    ) -> Result<Batch> {
        let q = &qmodel.q;
        let (ns, np) = (q.state_ids.len(), q.parameter_ids.len());
        let model_error = |e: sbmlsim_c::CError| CellsError::Model {
            message: e.to_string(),
        };
        // SAFETY: the type is the signature `generate_qfunction` writes for the host symbol; the
        // function is only called through the operators, which `Batch` drops before the
        // library.
        let function: Function =
            unsafe { qmodel.library.symbol(&q.host_symbol()) }.map_err(model_error)?;
        let source = format!("{}:{}", qmodel.source.display(), q.function);
        // SAFETY: the fields added below are the inputs and outputs of the QFunction in their
        // order and sizes ([`sbmlsim_c::INPUTS`], [`sbmlsim_c::OUTPUTS`]), and its context is
        // the four values of `RosenbrockContext`, set before any application.
        let mut qfunction = unsafe { QFunction::interior(ceed, 1, Some(function), &source) }?;
        let inputs = [ns, np, CONTROL, COUNTS];
        let outputs = [ns, CONTROL, COUNTS];
        for (name, size) in INPUTS.iter().zip(inputs) {
            qfunction.add_input(name, size, EvalMode::None)?;
        }
        for (name, size) in OUTPUTS.iter().zip(outputs) {
            qfunction.add_output(name, size, EvalMode::None)?;
        }
        qfunction.set_context(&words(context))?;
        // The values of a field point by point, as the caller has them.
        let restriction =
            |size: usize| ElemRestriction::strided_with(ceed, n, 1, size, n * size, [1, 1, size]);
        let restrictions = [
            restriction(ns)?,
            restriction(np)?,
            restriction(CONTROL)?,
            restriction(COUNTS)?,
        ];
        let pair = |size: usize| -> Result<[CVector; 2]> {
            Ok([CVector::new(ceed, n * size)?, CVector::new(ceed, n * size)?])
        };
        let (state, control, counts) = (pair(ns)?, pair(CONTROL)?, pair(COUNTS)?);
        let parameters = CVector::from_slice(ceed, parameters)?;
        let operator = |k: usize| -> Result<Operator> {
            let mut op = Operator::new(ceed, &qfunction)?;
            let read = [&state[k], &parameters, &control[k], &counts[k]];
            for ((name, r), v) in INPUTS.iter().zip(&restrictions).zip(read) {
                op.set_field(name, Some(r), None, FieldVector::Passive(v))?;
            }
            let written = [&state[1 - k], &control[1 - k], &counts[1 - k]];
            let r = [&restrictions[0], &restrictions[2], &restrictions[3]];
            for ((name, r), v) in OUTPUTS.iter().zip(r).zip(written) {
                op.set_field(name, Some(r), None, FieldVector::Passive(v))?;
            }
            Ok(op)
        };
        let operators = [operator(0)?, operator(1)?];
        Ok(Batch {
            qfunction,
            operators,
            state,
            control,
            counts,
            current: 0,
            _library: Arc::clone(&qmodel.library),
        })
    }

    /// Applies the operators until every cell is done or failed, or has made `MAX_ATTEMPTS`
    /// step attempts: from the states `fill` writes and the controls `control`, with `context`.
    /// Afterwards `control` holds the controls, `counts` the steps of this run and `states` the
    /// states. Whether every cell is done or failed.
    fn run(
        &mut self,
        context: &RosenbrockContext,
        fill: impl FnOnce(&mut [f64]),
        control: &mut [f64],
        counts: &mut [f64],
        states: &mut [f64],
    ) -> Result<bool> {
        let k = self.current;
        self.state[k].with_array_mut(fill)?;
        self.control[k].with_array_mut(|a| a.copy_from_slice(control))?;
        self.counts[k].set_value(0.0)?;
        self.qfunction.set_context(&words(context))?;
        let mut attempts = 0;
        while !finished(control) {
            if attempts == MAX_ATTEMPTS {
                break;
            }
            self.operators[self.current].apply_passive()?;
            self.current = 1 - self.current;
            self.control[self.current].with_array(|a| control.copy_from_slice(a))?;
            attempts += 1;
        }
        let k = self.current;
        self.counts[k].with_array(|a| counts.copy_from_slice(a))?;
        self.state[k].with_array(|a| states.copy_from_slice(a))?;
        Ok(finished(control))
    }
}

/// The context as the values of the QFunction's context.
fn words(context: &RosenbrockContext) -> [f64; 4] {
    [context.t1, context.rtol, context.atol, context.h_min]
}

/// Whether every cell of the controls `control` is done or failed.
fn finished(control: &[f64]) -> bool {
    control
        .chunks(CONTROL)
        .all(|c| c[CONTROL_DONE] != 0.0 || c[CONTROL_FAILED] != 0.0)
}

impl BatchedCells {
    /// The cells of `setup` with the model `model_sbml` (an SBML document) as `spec` describes
    /// it, on `ceed`: [`CellModel::new`] (for the batched path, whatever `spec.solver` says),
    /// then [`BatchedCells::with_model`]. Collective.
    pub fn new(
        setup: &CellSetup,
        spec: &CellsSpec,
        model_sbml: &str,
        ceed: &Ceed,
    ) -> Result<BatchedCells> {
        let spec = CellsSpec {
            solver: contracts::CellSolverKind::Batched,
            ..spec.clone()
        };
        let model = CellModel::new(&setup.comm(), &spec, model_sbml)?;
        BatchedCells::with_model(setup, &spec, &model, ceed)
    }

    /// The cells of `setup` with the model `model`, compiled and checked for `spec` on the
    /// batched path, on `ceed`. Collective.
    ///
    /// Errors: [`CellsError::Invalid`] for a setup whose lists are not one entry per cell and
    /// for a model compiled for the engine path; on every process, [`CellsError::Element`] for
    /// a cell whose model does not initialize, and the errors of libCEED.
    pub fn with_model(
        setup: &CellSetup,
        spec: &CellsSpec,
        model: &CellModel,
        ceed: &Ceed,
    ) -> Result<BatchedCells> {
        let comm = setup.comm();
        agree(&comm, setup.check())?;
        let Some(qmodel) = &model.qfunction else {
            return Err(CellsError::Invalid(
                "the cell model is compiled for the engine path, not for the batched path"
                    .to_string(),
            ));
        };
        let description = &model.description;
        let settings = IntegratorSettings {
            relative_tolerance: spec.rtol,
            absolute_tolerance: Tolerance::Scalar(spec.atol),
            ..IntegratorSettings::default()
        };
        let initialized = setup
            .parameters
            .iter()
            .enumerate()
            .map(|(i, changes)| {
                let stepper = Stepper::new(
                    Box::new(model.model.clone()),
                    Arc::clone(description),
                    settings.clone(),
                    changes,
                )
                .map_err(|e| CellsError::Element {
                    cell: setup.global[i],
                    centroid: setup.centroids[i],
                    message: format!("the cell model does not initialize: {e}"),
                })?;
                Ok((stepper.state().to_vec(), stepper.parameters().to_vec()))
            })
            .collect::<Result<Vec<_>>>();
        let initialized = agree(&comm, initialized)?;
        let n = initialized.len();
        let states: Vec<f64> = initialized.iter().flat_map(|c| c.0.clone()).collect();
        let parameters: Vec<f64> = initialized.iter().flat_map(|c| c.1.clone()).collect();
        let context = RosenbrockContext {
            t1: 0.0,
            rtol: spec.rtol,
            atol: spec.atol,
            h_min: H_MIN,
        };
        let batch = (n > 0).then(|| Batch::new(ceed, qmodel, &parameters, n, &context));
        let batch = agree(&comm, batch.transpose())?;
        Ok(BatchedCells {
            species: spec.species.iter().map(|l| l.species.clone()).collect(),
            state_ids: qmodel.q.state_ids.clone(),
            next: vec![0.0; states.len()],
            states,
            parameters,
            // At t = 0, a step to estimate, neither done nor failed.
            control: vec![0.0; n * CONTROL],
            counts: vec![0.0; n * COUNTS],
            links: model.links.clone(),
            outputs: Outputs::new(model, spec),
            time_unit_s: spec.time_unit_s,
            rtol: spec.rtol,
            atol: spec.atol,
            global: setup.global.clone(),
            centroids: setup.centroids.clone(),
            comm,
            batch,
            t: 0.0,
        })
    }

    /// An estimate of the memory of the cells of this process in bytes: the vectors of the
    /// operators (two sets of the states, controls and counts, and the parameters) and their
    /// copies on the host. The QFunction and its library, which the cells share, are not
    /// counted.
    pub fn memory_bytes(&self) -> usize {
        let n = self.global.len();
        let (ns, np) = (self.state_ids.len(), self.parameters.len() / n.max(1));
        let host = self.states.capacity()
            + self.next.capacity()
            + self.parameters.capacity()
            + self.control.capacity()
            + self.counts.capacity();
        let operators = n * (2 * (ns + CONTROL + COUNTS) + np);
        (host + operators) * size_of::<f64>()
    }

    /// The advance of every cell of this process; a failure an error of the first element
    /// which failed.
    fn advance_local(&mut self, t0: f64, t1: f64, blood: &mut [f64]) -> Result<CellStats> {
        let n = self.global.len();
        let (ns, nb) = (self.state_ids.len(), self.links.len());
        let np = self.parameters.len() / n.max(1);
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
        let mut stats = CellStats::default();
        let Some(batch) = &mut self.batch else {
            self.t = t1;
            return Ok(stats);
        };
        let (m0, m1) = (t0 / self.time_unit_s, t1 / self.time_unit_s);
        // Every cell which has not failed starts at t0, with the step it ended the last
        // advance with.
        for control in self.control.chunks_mut(CONTROL) {
            if control[CONTROL_FAILED] == 0.0 {
                control[CONTROL_T] = m0;
                control[CONTROL_DONE] = 0.0;
            }
        }
        let context = RosenbrockContext {
            t1: m1,
            rtol: self.rtol,
            atol: self.atol,
            h_min: H_MIN,
        };
        let (states, parameters, links) = (&self.states, &self.parameters, &self.links);
        let blood_in: &[f64] = blood;
        let fill = |state: &mut [f64]| {
            state.copy_from_slice(states);
            for i in 0..n {
                let y = &mut state[i * ns..(i + 1) * ns];
                let p = &parameters[i * np..(i + 1) * np];
                for (link, c) in links.iter().zip(&blood_in[i * nb..(i + 1) * nb]) {
                    let volume = link.volume.of(p, y);
                    y[link.state] = c * volume;
                }
            }
        };
        let finished = batch.run(
            &context,
            fill,
            &mut self.control,
            &mut self.counts,
            &mut self.next,
        )?;
        let element = |i: usize, message: String| CellsError::Element {
            cell: self.global[i],
            centroid: self.centroids[i],
            message,
        };
        let reached = |i: usize| self.control[i * CONTROL + CONTROL_T] * self.time_unit_s;
        if !finished {
            let i = (0..n)
                .find(|&i| {
                    let c = &self.control[i * CONTROL..(i + 1) * CONTROL];
                    c[CONTROL_DONE] == 0.0 && c[CONTROL_FAILED] == 0.0
                })
                .expect("a cell which is not finished");
            return Err(element(
                i,
                format!(
                    "the cell model did not reach t = {t1} s from t = {t0} s in {MAX_ATTEMPTS} \
                     step attempts (it is at t = {} s)",
                    reached(i)
                ),
            ));
        }
        if let Some(i) = (0..n).find(|&i| self.control[i * CONTROL + CONTROL_FAILED] != 0.0) {
            return Err(element(
                i,
                format!(
                    "the cell model failed from t = {t0} s to t = {t1} s: its step fell below \
                     {H_MIN} (in the time unit of the model) at t = {} s",
                    reached(i)
                ),
            ));
        }
        for c in self.counts.chunks(COUNTS) {
            stats.steps += c[0] as u64;
            stats.rejected += c[1] as u64;
        }
        std::mem::swap(&mut self.states, &mut self.next);
        for i in 0..n {
            let y = &self.states[i * ns..(i + 1) * ns];
            let p = &self.parameters[i * np..(i + 1) * np];
            for (link, c) in self.links.iter().zip(&mut blood[i * nb..(i + 1) * nb]) {
                *c = y[link.state] / link.volume.of(p, y);
            }
        }
        stats.seconds = start.elapsed().as_secs_f64();
        self.t = t1;
        Ok(stats)
    }
}

impl CellSolver for BatchedCells {
    fn n_cells(&self) -> usize {
        self.global.len()
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
        let n = self.global.len();
        let ns = self.state_ids.len();
        let np = self.parameters.len() / n.max(1);
        let t = self.t / self.time_unit_s;
        let mut work = Work::default();
        for i in 0..n {
            self.outputs.push(
                t,
                &self.states[i * ns..(i + 1) * ns],
                &self.parameters[i * np..(i + 1) * np],
                &mut work,
                out,
            );
        }
    }
}
