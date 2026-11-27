//! The batched path: the cells of a process as one libCEED operator of the QFunction of their
//! model ([`BatchedCells`]).

use std::ffi::c_void;
use std::os::raw::c_int;
use std::sync::Arc;
use std::time::Instant;

use ceed::sys::{CeedInt, CeedScalar};
use ceed::{CVector, Ceed, ElemRestriction, EvalMode, FieldVector, Operator, QFunction};
use contracts::CellsSpec;
use petsc::OwnedComm;
use sbmlsim_abi::{IntegratorSettings, Tolerance};
use sbmlsim_c::{
    CONTROL, CONTROL_DONE, CONTROL_FAILED, CONTROL_T, COUNTS, INPUTS, Library, OUTPUTS,
    RosenbrockContext,
};
use sbmlsim_engine::Stepper;

use crate::balance::Balance;
use crate::model::{Link, Outputs, QModel, Work};
use crate::{
    CellModel, CellSetup, CellSolver, CellStats, CellsError, OutputValue, Poison, Result, agree,
};

/// The smallest step of a cell, in the time unit of its model: a cell which needs a smaller one
/// fails.
const H_MIN: f64 = 1e-10;

/// The most step attempts of a cell in one advance: a cell which has not reached the end of
/// the advance after as many fails.
const MAX_ATTEMPTS: u32 = 1_000_000;

/// The QFunction of a library, of the signature of `CeedQFunctionUser`.
type Function = unsafe extern "C" fn(
    *mut c_void,
    CeedInt,
    *const *const CeedScalar,
    *const *mut CeedScalar,
) -> c_int;

// The context of the QFunction is its five values, in the order of `RosenbrockContext`.
const _: () = assert!(size_of::<RosenbrockContext>() == 5 * size_of::<f64>());

/// The cells of a process as one libCEED operator: one point per cell (one element of one node
/// per cell, strided restrictions, no basis), the QFunction of the model
/// ([`sbmlsim_c::generate_qfunction`]: the model with the Rosenbrock integrator RODAS4, step
/// attempts per point until it reaches the end of the advance, at most `MAX_ATTEMPTS`, 10^6)
/// compiled once and loaded by every process ([`CellModel::new`]; `sbmlsim_c`, cached across
/// processes and runs).
///
/// The cells are initialized as the engine path initializes its steppers (a
/// [`sbmlsim_engine::Stepper`] of the model of the C target per cell, the defaults, the cell's
/// parameters, the initial assignments), so both paths start from the same states.
///
/// An advance sets the linked species to their concentrations in the blood (as amounts, times
/// the volume of their compartment, unless the model stores the concentration), then applies
/// the operator once (it reads the fields of one set of vectors and writes the other): each cell
/// steps until it reaches the end of the advance or fails, so a process reads and writes its
/// cells once per advance and a cell which is done waits for no other. (One attempt per
/// application, which the QFunction allows for calls of bounded work, passes over every cell as
/// often as the cell of the most attempts needs, the cells which are done included.) Each cell
/// takes its own steps with the tolerances of the spec and keeps its step from one advance to
/// the next, so a cell depends on no other cell, the number of processes, the backend or its
/// blocks. A cell fails where its next step would be smaller than `H_MIN` = 1e-10 (in the time
/// unit of the model) or too small to advance the time, which rejected steps cause (an error
/// estimate above the tolerances, values which are not finite, a singular Jacobian), or after
/// `MAX_ATTEMPTS` step attempts in one advance.
///
/// With the balance of the work ([`CellsSpec::balance`], `Balance`), the points of the operator
/// are the cells this process keeps and those other processes send it (their states with the
/// blood, parameters and controls), whose states, controls and counts go back after the
/// application. Where the points do not fit, the operator is made again with room for an eighth
/// more; the points beyond those of an advance are done, so the QFunction copies them.
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
    /// The number of parameters of a cell.
    n_parameters: usize,
    /// The control of every cell (`t`, `h`, done, failed), after the last advance.
    control: Vec<f64>,
    /// The accepted and rejected steps of every cell in the last advance.
    counts: Vec<f64>,
    /// The points of the operator in an advance: the cells this process keeps, then those of
    /// other processes it advances ([`Balance`]); their states (with the blood), parameters and
    /// controls, then their states, controls and counts after it.
    points: Points,
    /// Of each linked species.
    links: Vec<Link>,
    outputs: Outputs,
    time_unit_s: f64,
    rtol: f64,
    atol: f64,
    elements: Vec<u64>,
    centroids: Vec<[f64; 2]>,
    comm: OwnedComm,
    /// The error of a failed advance, after which the cells are not advanced again.
    poison: Poison,
    /// The operator of the points, of room for at least as many; none before an advance with
    /// points on a process without cells.
    batch: Option<Batch>,
    /// The work of the cells spread over the processes.
    balance: Balance,
    /// The QFunction of the model, for a larger operator.
    qmodel: Arc<QModel>,
    ceed: Ceed,
    /// The time the cells reached (s).
    t: f64,
}

impl std::fmt::Debug for BatchedCells {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BatchedCells")
            .field("cells", &self.elements.len())
            .field("species", &self.species)
            .field("state_ids", &self.state_ids)
            .field("time_unit_s", &self.time_unit_s)
            .finish_non_exhaustive()
    }
}

/// The points of an advance on a process ([`BatchedCells::points`]).
#[derive(Default)]
struct Points {
    states: Vec<f64>,
    parameters: Vec<f64>,
    control: Vec<f64>,
    next: Vec<f64>,
    counts: Vec<f64>,
}

/// The operator of the points of a process and its vectors, of room for `capacity` points: the
/// points beyond those of an advance are done, so the QFunction copies them.
struct Batch {
    capacity: usize,
    qfunction: QFunction,
    /// Reads the states, controls and counts of the first of each pair of vectors (and the
    /// parameters) and writes those of the second.
    operator: Operator,
    state: [CVector; 2],
    parameters: CVector,
    control: [CVector; 2],
    counts: [CVector; 2],
    /// The library of the function of the QFunction, dropped after the QFunction and the
    /// operators (the fields are dropped in their order).
    _library: Arc<Library>,
}

impl Batch {
    /// The operator of the QFunction of `qmodel` on `ceed` for `n` points, with `context`.
    fn new(ceed: &Ceed, qmodel: &QModel, n: usize, context: &RosenbrockContext) -> Result<Batch> {
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
        // the five values of `RosenbrockContext`, set before any application.
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
        let parameters = CVector::new(ceed, n * np)?;
        let mut operator = Operator::new(ceed, &qfunction)?;
        let read = [&state[0], &parameters, &control[0], &counts[0]];
        for ((name, r), v) in INPUTS.iter().zip(&restrictions).zip(read) {
            operator.set_field(name, Some(r), None, FieldVector::Passive(v))?;
        }
        let written = [&state[1], &control[1], &counts[1]];
        let r = [&restrictions[0], &restrictions[2], &restrictions[3]];
        for ((name, r), v) in OUTPUTS.iter().zip(r).zip(written) {
            operator.set_field(name, Some(r), None, FieldVector::Passive(v))?;
        }
        Ok(Batch {
            capacity: n,
            qfunction,
            operator,
            state,
            parameters,
            control,
            counts,
            _library: Arc::clone(&qmodel.library),
        })
    }

    /// Applies the operator once, with `context` (whose attempts are [`MAX_ATTEMPTS`]), to the
    /// points of `points` (at most `capacity`): every point steps until it is done or failed, or
    /// has made `MAX_ATTEMPTS` step attempts, from its state, parameters and control. Afterwards
    /// `points.control` holds the controls, `points.counts` the steps of this run and
    /// `points.next` the states.
    fn run(&mut self, context: &RosenbrockContext, points: &mut Points) -> Result<()> {
        let n = points.control.len() / CONTROL;
        debug_assert!(n <= self.capacity);
        // The points beyond `n`: done at the end of the advance, copied.
        let unused = [context.t1, 0.0, 1.0, 0.0];
        /// Writes `values`, then `rest` over and over.
        fn fill<'a>(values: &'a [f64], rest: &'a [f64]) -> impl FnOnce(&mut [f64]) + 'a {
            move |a: &mut [f64]| {
                a[..values.len()].copy_from_slice(values);
                for (k, v) in a[values.len()..].iter_mut().enumerate() {
                    *v = rest[k % rest.len()];
                }
            }
        }
        self.state[0].with_array_mut(fill(&points.states, &[0.0]))?;
        self.parameters
            .with_array_mut(fill(&points.parameters, &[0.0]))?;
        self.control[0].with_array_mut(fill(&points.control, &unused))?;
        self.counts[0].set_value(0.0)?;
        self.qfunction.set_context(&words(context))?;
        self.operator.apply_passive()?;
        /// Takes the first `count` values into `out`.
        fn take(out: &mut Vec<f64>, count: usize) -> impl FnOnce(&[f64]) + '_ {
            move |a: &[f64]| {
                out.clear();
                out.extend_from_slice(&a[..count]);
            }
        }
        let ns = points.states.len() / n.max(1);
        self.control[1].with_array(take(&mut points.control, n * CONTROL))?;
        self.counts[1].with_array(take(&mut points.counts, n * COUNTS))?;
        self.state[1].with_array(take(&mut points.next, n * ns))?;
        Ok(())
    }
}

/// The context as the values of the QFunction's context.
fn words(context: &RosenbrockContext) -> [f64; 5] {
    [
        context.t1,
        context.rtol,
        context.atol,
        context.h_min,
        context.attempts,
    ]
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
        let comm = setup.owned_comm();
        agree(&comm.comm(), setup.check())?;
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
                    element: setup.elements[i],
                    centroid: setup.centroids[i],
                    message: format!("the cell model does not initialize: {e}"),
                })?;
                Ok((stepper.state().to_vec(), stepper.parameters().to_vec()))
            })
            .collect::<Result<Vec<_>>>();
        let initialized = agree(&comm.comm(), initialized)?;
        let n = initialized.len();
        let states: Vec<f64> = initialized.iter().flat_map(|c| c.0.clone()).collect();
        let parameters: Vec<f64> = initialized.iter().flat_map(|c| c.1.clone()).collect();
        let context = RosenbrockContext {
            t1: 0.0,
            rtol: spec.rtol,
            atol: spec.atol,
            h_min: H_MIN,
            attempts: f64::from(MAX_ATTEMPTS),
        };
        let batch = (n > 0).then(|| Batch::new(ceed, qmodel, n, &context));
        let batch = agree(&comm.comm(), batch.transpose())?;
        Ok(BatchedCells {
            points: Points::default(),
            balance: Balance::new(comm.clone(), n, spec.balance),
            qmodel: Arc::clone(qmodel),
            ceed: ceed.clone(),
            n_parameters: qmodel.q.parameter_ids.len(),
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
            elements: setup.elements.clone(),
            centroids: setup.centroids.clone(),
            comm,
            poison: Poison::default(),
            batch,
            t: 0.0,
        })
    }

    /// An estimate of the memory of the cells of this process in bytes: the vectors of the
    /// operators (two sets of the states, controls and counts, and the parameters) and their
    /// copies on the host. The QFunction and its library, which the cells share, are not
    /// counted.
    pub fn memory_bytes(&self) -> usize {
        let (ns, np) = (self.state_ids.len(), self.n_parameters);
        let points = &self.points;
        let host = self.states.capacity()
            + self.next.capacity()
            + self.parameters.capacity()
            + self.control.capacity()
            + self.counts.capacity()
            + points.states.capacity()
            + points.parameters.capacity()
            + points.control.capacity()
            + points.next.capacity()
            + points.counts.capacity();
        let capacity = self.batch.as_ref().map_or(0, |b| b.capacity);
        let operators = capacity * (2 * (ns + CONTROL + COUNTS) + np);
        (host + operators) * size_of::<f64>()
    }

    /// The advance of every cell of this process; a failure an error of the first element
    /// which failed. With the balance on, the cells this process keeps and those of other
    /// processes it is sent are advanced here, and the others of this process elsewhere
    /// ([`Balance`]). Collective, if the balance is on.
    fn advance_local(&mut self, t0: f64, t1: f64, blood: &mut [f64]) -> Result<CellStats> {
        let n = self.elements.len();
        let (ns, nb, np) = (self.state_ids.len(), self.links.len(), self.n_parameters);
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
        if !self.balance.plan(valid.is_ok())? {
            return valid.map(|()| CellStats::default());
        }
        let mut stats = CellStats::default();
        let (m0, m1) = (t0 / self.time_unit_s, t1 / self.time_unit_s);
        // Every cell which has not failed starts at t0, with the step it ended the last
        // advance with.
        for control in self.control.chunks_mut(CONTROL) {
            if control[CONTROL_FAILED] == 0.0 {
                control[CONTROL_T] = m0;
                control[CONTROL_DONE] = 0.0;
            }
        }
        // The state of each cell with the concentrations of the blood, into `next`.
        self.next.copy_from_slice(&self.states);
        for i in 0..n {
            let y = &mut self.next[i * ns..(i + 1) * ns];
            let p = &self.parameters[i * np..(i + 1) * np];
            for (link, c) in self.links.iter().zip(&blood[i * nb..(i + 1) * nb]) {
                let volume = link.volume.of(p, y);
                y[link.state] = c * volume;
            }
        }
        // The points: the cells kept, then those of other processes.
        let (next, parameters, control) = (&self.next, &self.parameters, &self.control);
        let sent = ns + np + CONTROL;
        let guests = self.balance.scatter(sent, |i, out| {
            out.extend_from_slice(&next[i * ns..(i + 1) * ns]);
            out.extend_from_slice(&parameters[i * np..(i + 1) * np]);
            out.extend_from_slice(&control[i * CONTROL..(i + 1) * CONTROL]);
        })?;
        let kept = self.balance.kept();
        let points = &mut self.points;
        points.states.clear();
        points.states.extend_from_slice(&next[..kept * ns]);
        points.parameters.clear();
        points
            .parameters
            .extend_from_slice(&parameters[..kept * np]);
        points.control.clear();
        points.control.extend_from_slice(&control[..kept * CONTROL]);
        for record in guests.chunks_exact(sent) {
            points.states.extend_from_slice(&record[..ns]);
            points.parameters.extend_from_slice(&record[ns..ns + np]);
            points.control.extend_from_slice(&record[ns + np..]);
        }
        let m = kept + self.balance.guests();
        let context = RosenbrockContext {
            t1: m1,
            rtol: self.rtol,
            atol: self.atol,
            h_min: H_MIN,
            attempts: f64::from(MAX_ATTEMPTS),
        };
        let start = Instant::now();
        // An error here is reported after the results went back, so that no process waits.
        let ran = if m == 0 {
            Ok(())
        } else {
            self.run_points(&context, m)
        };
        stats.seconds = start.elapsed().as_secs_f64();
        let points = &mut self.points;
        if ran.is_err() {
            points.next.resize(m * ns, f64::NAN);
            points.control.resize(m * CONTROL, 0.0);
            points.counts.resize(m * COUNTS, 0.0);
        }
        for c in points.counts.chunks(COUNTS) {
            stats.steps += c[0] as u64;
            stats.rejected += c[1] as u64;
        }
        // The results: of the cells kept, and of those of other processes, sent back.
        self.next[..kept * ns].copy_from_slice(&points.next[..kept * ns]);
        self.control[..kept * CONTROL].copy_from_slice(&points.control[..kept * CONTROL]);
        self.counts[..kept * COUNTS].copy_from_slice(&points.counts[..kept * COUNTS]);
        let back = ns + CONTROL + COUNTS;
        let results: Vec<f64> = (kept..m)
            .flat_map(|j| {
                points.next[j * ns..(j + 1) * ns]
                    .iter()
                    .chain(&points.control[j * CONTROL..(j + 1) * CONTROL])
                    .chain(&points.counts[j * COUNTS..(j + 1) * COUNTS])
                    .copied()
            })
            .collect();
        let (next, control, counts) = (&mut self.next, &mut self.control, &mut self.counts);
        self.balance.gather(back, &results, |i, record| {
            next[i * ns..(i + 1) * ns].copy_from_slice(&record[..ns]);
            control[i * CONTROL..(i + 1) * CONTROL].copy_from_slice(&record[ns..ns + CONTROL]);
            counts[i * COUNTS..(i + 1) * COUNTS].copy_from_slice(&record[ns + CONTROL..]);
        })?;
        ran?;
        for (w, c) in self
            .balance
            .weights_mut()
            .iter_mut()
            .zip(self.counts.chunks(COUNTS))
        {
            *w = (c[0] + c[1]) as u64;
        }
        let element = |i: usize, message: String| CellsError::Element {
            element: self.elements[i],
            centroid: self.centroids[i],
            message,
        };
        let reached = |i: usize| self.control[i * CONTROL + CONTROL_T] * self.time_unit_s;
        if !finished(&self.control) {
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
                    "the cell model failed from t = {t0} s to t = {t1} s at t = {} s: its \
                     rejected steps (an error above the tolerances, values which are not finite \
                     or a singular Jacobian) made its step smaller than {H_MIN} (in the time \
                     unit of the model), or too small to advance the time",
                    reached(i)
                ),
            ));
        }
        std::mem::swap(&mut self.states, &mut self.next);
        for i in 0..n {
            let y = &self.states[i * ns..(i + 1) * ns];
            let p = &self.parameters[i * np..(i + 1) * np];
            for (link, c) in self.links.iter().zip(&mut blood[i * nb..(i + 1) * nb]) {
                *c = y[link.state] / link.volume.of(p, y);
            }
        }
        self.t = t1;
        Ok(stats)
    }

    /// Applies the operator to the `m` points of [`BatchedCells::points`], for an operator of
    /// room for them first if there is none (an eighth more than `m`, so that a few more points
    /// in later advances fit).
    fn run_points(&mut self, context: &RosenbrockContext, m: usize) -> Result<()> {
        if self.batch.as_ref().is_none_or(|b| b.capacity < m) {
            self.batch = None;
            self.batch = Some(Batch::new(&self.ceed, &self.qmodel, m + m / 8, context)?);
        }
        self.batch
            .as_mut()
            .expect("an operator")
            .run(context, &mut self.points)
    }
}

impl CellSolver for BatchedCells {
    fn n_cells(&self) -> usize {
        self.elements.len()
    }

    fn species(&self) -> &[String] {
        &self.species
    }

    fn state_ids(&self) -> &[String] {
        &self.state_ids
    }

    fn advance(&mut self, t0: f64, t1: f64, blood: &mut [f64]) -> Result<CellStats> {
        self.poison.check()?;
        let local = self.advance_local(t0, t1, blood);
        let agreed = agree(&self.comm.comm(), local);
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
        let n = self.elements.len();
        let (ns, np) = (self.state_ids.len(), self.n_parameters);
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
