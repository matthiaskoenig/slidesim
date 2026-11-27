//! A `study/1` as a simulation in memory ([`Simulation`]): the fields, the Darcy flow and the
//! cells of a study, stepped from one output time to the next, with the values of each output
//! time (the fields at the vertices, the values of the cells, the summary) for the caller to
//! write or show. `slidesim run` is a loop over [`Simulation::next_output`] which writes the
//! VTU files and `summary.json`.
//!
//! [`Simulation::new`] checks that the maps the study names are in the scene and the `cells`
//! section against the study and the scene, translates the cell model to C, compiles it and
//! checks it against the study (before anything is solved), makes the mesh (distributed over the
//! processes, refined `mesh.refinements` times; tagged polygons and lobules meshed by process 0
//! first, the quality of such a mesh reported in one line to the caller's log, or the mesh of the
//! inputs),
//! reads the maps over the cells of each process (the zonation map of lobules computed there,
//! `zonation_map`), solves the Darcy [`Flow`] if the study has a `perfusion` section, and
//! builds one [`Problem`] per field and, with `cells`, the cells and their exchange with the
//! fields (`Coupled`).
//!
//! [`Simulation::next_output`] steps the fields to each output time of `time.outputs` in turn
//! (the last is `time.end`, as `study/1` requires): each field alone, or with `cells` all fields
//! and the cells together by Strang splitting (`Coupled`). At each output time it records the
//! values of the summary ([`Summary`], `summary.json`). The caller reads or writes the state of
//! each output time before the next call: [`Simulation::field_values`],
//! [`Simulation::cell_values`], [`Simulation::write_vtu`]. [`Simulation::write_vtu`] writes every
//! field as
//! `<out>/<field id>_<index>.vtu`, `<index>` the position of the time in `time.outputs`: point
//! data `<field id>.mM` on the finest mesh level, Dirichlet values included, one file of the
//! pieces of every process; with `cells`, the cells as `<out>/cells_<index>.vtu`: cell data of
//! the coarse elements, one array per value named after it (NaN in elements without a cell):
//! every state of the model (a species of a concentration as its concentration, in the model's
//! units), the outputs of `cells.outputs` (targets of assignment rules) and the parameters of
//! the element (`cells.parameters` and the two volumes). [`Simulation::write_flow`] writes the
//! flow as `<out>/pressure.vtu` (point data `pressure.Pa`) and `<out>/velocity.vtu` (the point
//! vectors `velocity`, the conservative velocity the fields are carried by, and
//! `primal_velocity`, `-K grad p / mu`, um/s).
//!
//! [`ResultWriter`] writes the state of each output time into the store `result/1`,
//! `<out>/result.zarr`, for viewers: the canonical finest mesh, the fields at its vertices and
//! the cell values per element (see its module); [`read_result`] reads such a store. Neither is
//! built for WASI (Zarr's filesystem store needs positioned file I/O): the browser build serves
//! the arrays from memory, from a [`ResultLayout`] and the values of each output time.
//!
//! The library prints nothing: what it reports goes to the log the caller gives
//! [`Simulation::new`] (`slidesim run` prints it, the browser build keeps it as its array `log`),
//! and the notes of a run are the caller's to show ([`Simulation::notes`]).
//!
//! With the feature `native` (the default) the cell model is the SBML of the study, compiled to
//! native code at the start ([`Inputs::model_sbml`]); without it (the browser build) the batched
//! kernel of the model which the host compiled and links (`Inputs::kernel`, `cells::KernelCells`).
//!
//! Every process calls the functions marked collective, in the same order. An error which every
//! process meets at the same point carries the context [`OnEveryProcess`]; after any other
//! error of a collective function, the other processes may wait for the failing one forever.

mod coupled;
mod inputs;
mod phases;
pub mod raster;
mod result;
mod summary;
pub mod vtu;

use std::fmt;
use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result, ensure};
use ceed::Ceed;
use cells::CellModel;
use contracts::{LobuleSpec, MeshSource, Scene, Study};
use maps::{Map, Maps};
use mesh::Mesh;
use mesher::{Lobules, Quality};
use pde::{Flow, PdeError, Problem};
use petsc::{Comm, Petsc};

use crate::coupled::Coupled;
pub use crate::inputs::Inputs;
use crate::phases::Phases;
pub use crate::result::ResultLayout;
#[cfg(not(target_os = "wasi"))]
pub use crate::result::{RESULT_STORE, ResultData, ResultWriter, read_result};
pub use crate::summary::{
    Attempts, BalanceSummary, CellsSummary, FieldSummary, InflowSummary, IslandsSummary,
    PerfusionSummary, Seconds, SolverSummary, Summary,
};

/// The context of an error which every process meets at the same point of a simulation, with
/// the same message: a collective error, after which no process waits for another (`slidesim
/// run` ends each process alone, without `MPI_Abort`).
#[derive(Debug)]
pub struct OnEveryProcess(pub &'static str);

impl fmt::Display for OnEveryProcess {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.0)
    }
}

/// The error and its causes, `error: cause: cause`, without a cause whose text the ones before
/// already show (errors which print their source in their own message): how `slidesim run` and
/// the browser build report an error.
pub fn causes(e: &anyhow::Error) -> String {
    let mut text = String::new();
    for cause in e.chain() {
        let cause = cause.to_string();
        if text.contains(&cause) {
            continue;
        }
        if !text.is_empty() {
            text.push_str(": ");
        }
        text.push_str(&cause);
    }
    text
}

/// The quality of a mesh in one line.
pub fn quality_line(q: &Quality) -> String {
    let mut line = format!(
        "mesh: {} triangles, {} vertices, min angle {:.2} deg, SICN min {:.3} mean {:.3}",
        q.triangles, q.vertices, q.min_angle_deg, q.min_sicn, q.mean_sicn
    );
    if let Some([lo, hi]) = q.edge_ratio {
        line.push_str(&format!(", edge length / size {lo:.2} to {hi:.2}"));
    }
    line
}

/// `x` rounded to `digits` significant digits, for a log line (all digits before the point).
fn significant(x: f64, digits: i32) -> String {
    let magnitude = if x == 0.0 || !x.is_finite() {
        0
    } else {
        x.abs().log10().floor() as i32
    };
    let decimals = usize::try_from(digits - 1 - magnitude).unwrap_or(0);
    format!("{x:.decimals$}")
}

/// A study on the processes of a [`Petsc`]: its fields, flow and cells, stepped from output time
/// to output time (see the crate documentation).
///
/// After [`Simulation::new`] the fields and the cells are at the start, t = 0, and no output
/// time is reached. Each [`Simulation::next_output`] steps to the next output time and returns
/// its index; the first returns `Some(0)` without a step if the first output time is 0 (the
/// initial state, as `slidesim run` writes output 0). The live API of the browser build and
/// `result/1` (simulation at output index 0 when it is created, each step to the next index) is
/// a `Simulation::new` followed by one `next_output`.
pub struct Simulation<'p> {
    // Declared in the order of their drop: the cells and the problems before what they were
    // made of.
    coupled: Option<Coupled>,
    problems: Vec<Problem>,
    flow: Option<Flow>,
    mesh: Mesh,
    _model: Option<CellModel>,
    study: Study,
    comm: Comm<'p>,
    phases: Phases<'p>,
    summary: Summary,
    /// The index in `time.outputs` of the next output time.
    next: usize,
    /// The index of the output time reached last, while it is the current state.
    current: Option<usize>,
    /// Whether [`Simulation::next_output`] returned `None`: the summary of the whole run is
    /// complete.
    finished: bool,
}

impl<'p> Simulation<'p> {
    /// The simulation of `inputs` on the processes of `petsc`, for a process started at
    /// `started` (the start of the phase `setup` of the summary): every check, the setup and the
    /// flow (see the crate documentation). Gives `log` the quality of a mesh it generates in one
    /// line ([`quality_line`], on process 0), as soon as it is made. Collective.
    ///
    /// Errors: those of [`Inputs::load`]'s checks; a study without fields, or with `cells` and
    /// no cell model in the inputs (these with [`OnEveryProcess`]); the cell model (one which
    /// does not compile or does not fit the study), the mesh (with the context
    /// [`OnEveryProcess`] if every process has the error, see
    /// [`mesh::MeshError::is_collective`]), the maps (with [`OnEveryProcess`]), the flow, the
    /// problems and the cells.
    pub fn new(
        petsc: &'p Petsc,
        inputs: Inputs,
        started: Instant,
        log: &mut dyn FnMut(&str),
    ) -> Result<Simulation<'p>> {
        inputs::check_inputs(&inputs)?;
        let comm = petsc.comm();
        let model = cell_model(&comm, &inputs)?;
        let Inputs {
            study, scene, mesh, ..
        } = inputs;
        let mesh = match &mesh {
            Some(tagged) => {
                ensure!(
                    scene.dimension == 2,
                    OnEveryProcess("a mesh of the inputs is 2D, the scene is 3D")
                );
                Mesh::from_tagged(petsc, Some(tagged), study.mesh.refinements)
            }
            None => Mesh::load(petsc, &scene, study.mesh.refinements),
        }
        .map_err(|e| {
            let everywhere = e.is_collective();
            let e = anyhow::Error::new(e);
            if everywhere {
                e.context(OnEveryProcess("cannot load the mesh"))
            } else {
                e.context("cannot load the mesh")
            }
        })?;
        if let Some((_, quality)) = mesh.generated() {
            log(&quality_line(quality));
        }
        let ceed = Ceed::new(ceed::DEFAULT_RESOURCE).context("cannot initialize libCEED")?;
        let mut phases = Phases::new(comm, started);
        let maps = read_maps(&comm, &study, &scene, &mesh)?;
        phases.end("setup")?;
        let flow = study
            .perfusion
            .as_ref()
            .map(|perfusion| {
                Flow::solve(&mesh, perfusion, &maps, &ceed, &study.solver.petsc_options)
                    .context("perfusion")
            })
            .transpose()?;
        phases.end("pressure")?;
        let problem_inputs = pde::Inputs {
            maps: &maps,
            flow: flow.as_ref(),
        };
        let problems = study
            .fields
            .iter()
            .map(|field| {
                Problem::new(
                    &mesh,
                    field,
                    &study.time,
                    &ceed,
                    &study.solver,
                    &problem_inputs,
                )
                .with_context(|| format!("field '{}'", field.id))
            })
            .collect::<Result<Vec<_>>>()?;
        let coupled = match (&study.cells, &model) {
            (Some(spec), Some(model)) => Some(Coupled::new(
                &study, spec, model, &mesh, &maps, &problems, &ceed,
            )?),
            _ => None,
        };
        phases.end("setup")?;
        // The loosest tolerance of the fields: PETSc options may set one per field.
        let mut linear_rtol = 0.0_f64;
        for problem in &problems {
            linear_rtol = linear_rtol.max(problem.linear_rtol()?);
        }
        let summary = Summary {
            fields: study
                .fields
                .iter()
                .map(|field| (field.id.clone(), FieldSummary::default()))
                .collect(),
            solver: SolverSummary { linear_rtol },
            perfusion: flow.as_ref().map(|flow| PerfusionSummary {
                tag_flows: flow.tag_flows().iter().cloned().collect(),
                pressure_islands: IslandsSummary {
                    count: flow.islands().count,
                    area_um2: flow.islands().area,
                },
            }),
            cells: coupled.as_ref().map(|coupled| CellsSummary {
                count: coupled.count(),
                ..CellsSummary::default()
            }),
            phases: phases.seconds().clone(),
        };
        Ok(Simulation {
            coupled,
            problems,
            flow,
            mesh,
            _model: model,
            study,
            comm,
            phases,
            summary,
            next: 0,
            current: None,
            finished: false,
        })
    }

    /// Steps to the next output time and records its values in the summary: `Some(index)`, its
    /// index in `time.outputs` (from 0). The first call at an output time 0 does not step (the
    /// initial state). After the last output time, which is `time.end`, records the statistics
    /// of the cells and the phases of the whole run and returns `None` without a step: the
    /// fields and the cells stay at the last output time. Later calls return `None` and do
    /// nothing. The time since the end of the last call (or of [`Simulation::new`]) counts as
    /// the phase `output`: the caller's outputs. Collective.
    ///
    /// Errors: those of the steps of the fields and the cells and of the values of the summary;
    /// the simulation is not to be stepped further after one.
    pub fn next_output(&mut self) -> Result<Option<usize>> {
        if self.finished {
            return Ok(None);
        }
        self.current = None;
        self.phases.end("output")?;
        let Some(&t) = self.study.time.outputs.get(self.next) else {
            // `study/1` ends `time.outputs` at `time.end`: the run is there already (up to the
            // rounding of the time of the steps).
            let end = self.study.time.end;
            debug_assert!(
                (self.time() - end).abs() <= 1e-12 * end.abs(),
                "time.outputs ends at time.end: the time is {}, time.end {end}",
                self.time()
            );
            if let (Some(coupled), Some(cells)) = (&self.coupled, &mut self.summary.cells) {
                let stats = coupled.stats(&self.comm)?;
                (cells.steps, cells.rejected, cells.seconds, cells.attempts) =
                    (stats.steps, stats.rejected, stats.seconds, stats.attempts);
            }
            self.phases.end("output")?;
            self.summary.phases = self.phases.seconds().clone();
            self.finished = true;
            return Ok(None);
        };
        self.advance(t)?;
        for (field, problem) in self.study.fields.iter().zip(&mut self.problems) {
            let id = &field.id;
            let values = self
                .summary
                .fields
                .get_mut(id)
                .expect("every field has a summary");
            values.times.push(problem.time());
            values.l2_norm.push(problem.l2_norm()?);
            values.total_mass.push(problem.total_mass()?);
            let balance = match problem.interval_balance() {
                Ok(balance) => Some(balance.into()),
                // An interval of no length (the output of time 0).
                Err(PdeError::NotStepped { .. }) => None,
                // Another time integration, as the notes say.
                Err(_) if problem.without_interval_balance().is_some() => None,
                Err(e) => return Err(e).with_context(|| format!("field '{id}'")),
            };
            values.mass_balance.push(balance);
            problem.start_interval()?;
        }
        if let (Some(coupled), Some(cells)) = (&self.coupled, &mut self.summary.cells) {
            for (id, amount) in coupled.amounts(&self.comm)? {
                cells.amounts.entry(id).or_default().push(amount);
            }
            for (field, amount) in coupled.exchanged() {
                let id = self.study.fields[field].id.clone();
                cells.exchanged.entry(id).or_default().push(amount);
            }
        }
        self.summary.phases = self.phases.seconds().clone();
        let index = self.next;
        self.next += 1;
        self.current = Some(index);
        Ok(Some(index))
    }

    /// Advances the fields to the time `t`: each alone, or with the cells all together.
    /// Collective.
    fn advance(&mut self, t: f64) -> Result<()> {
        match &mut self.coupled {
            Some(coupled) => coupled.advance(&self.study, &mut self.problems, t, &mut self.phases),
            None => {
                for (field, problem) in self.study.fields.iter().zip(self.problems.iter_mut()) {
                    problem
                        .step_to(t)
                        .with_context(|| format!("field '{}'", field.id))?;
                }
                self.phases.end("transport")
            }
        }
    }

    /// The time the fields reached (s): 0 at the start, an output time after a
    /// [`Simulation::next_output`] which returned its index, still the last output time
    /// (`time.end`) after the one which returned `None`. (A study has a field:
    /// [`Simulation::new`] checks it.)
    pub fn time(&self) -> f64 {
        self.problems[0].time()
    }

    /// The output times of the study, `time.outputs` (s).
    pub fn output_times(&self) -> &[f64] {
        &self.study.time.outputs
    }

    /// The study.
    pub fn study(&self) -> &Study {
        &self.study
    }

    /// The mesh: distributed over the processes, its finest level that of the fields.
    pub fn mesh(&self) -> &Mesh {
        &self.mesh
    }

    /// The ids of the fields, in the order of the study.
    pub fn field_ids(&self) -> Vec<&str> {
        self.study.fields.iter().map(|f| f.id.as_str()).collect()
    }

    /// The field `field` (its index in [`Simulation::field_ids`]) at each local vertex of the
    /// finest level now ([`Problem::vertex_values`]: ghosts and Dirichlet values included), the
    /// input of [`Mesh::gather_vertex_values`]. Collective.
    ///
    /// Panics if there is no field `field`.
    pub fn field_values(&self, field: usize) -> Result<Vec<f64>> {
        Ok(self.problems[field].vertex_values()?)
    }

    /// The names of the values of a cell, in the order of a row of [`Simulation::cell_values`]:
    /// the states of the model (a species of a concentration as its concentration),
    /// `cells.outputs`, then the parameters of the element (`cells.parameters` and the two
    /// volumes). Empty without cells.
    pub fn cell_value_names(&self) -> Vec<String> {
        self.coupled
            .as_ref()
            .map_or_else(Vec::new, |c| c.names().to_vec())
    }

    /// The local cells now, `None` without cells: the element id of each ([`Mesh::element_ids`]),
    /// and its values, one row per cell of one value per name of
    /// [`Simulation::cell_value_names`].
    pub fn cell_values(&self) -> Option<(&[u64], Vec<f64>)> {
        self.coupled.as_ref().map(|c| (c.elements(), c.table()))
    }

    /// The summary: the values of the output times so far and, after the end, the statistics of
    /// the cells and the phases of the whole run; `summary.json` of `slidesim run`.
    pub fn summary(&self) -> &Summary {
        &self.summary
    }

    /// Remarks on the run for its log, one line each: the fields without a mass balance over
    /// the output intervals (another time integration than BDF2, chosen by PETSc options), and
    /// the pressure islands of the flow, if any.
    pub fn notes(&self) -> Vec<String> {
        let mut notes = Vec::new();
        for (field, problem) in self.study.fields.iter().zip(&self.problems) {
            if let Some(why) = problem.without_interval_balance() {
                notes.push(format!(
                    "field '{}': no mass balance over the output intervals (null in \
                     summary.json), it follows the steps of BDF of order 2: {why}",
                    field.id
                ));
            }
        }
        if let Some(flow) = &self.flow {
            let islands = flow.islands();
            if islands.count > 0 {
                let plural = if islands.count == 1 { "" } else { "s" };
                notes.push(format!(
                    "perfusion: {} pressure island{plural} of {} um^2 in all, which no pressure \
                     boundary reaches: pressure 0 and no flow there",
                    islands.count,
                    significant(islands.area, 4)
                ));
            }
        }
        notes
    }

    /// Writes the state of the output time `index` into the directory `out` (see the crate
    /// documentation): `<out>/<field id>_<index>.vtu` per field and, with cells,
    /// `<out>/cells_<index>.vtu`. Collective.
    ///
    /// Errors: `index` is not the output time [`Simulation::next_output`] returned last, or a
    /// file cannot be written.
    pub fn write_vtu(&self, out: &Path, index: usize) -> Result<()> {
        ensure!(
            self.current == Some(index),
            "the simulation is not at the output time {index}"
        );
        for (field, problem) in self.study.fields.iter().zip(&self.problems) {
            let path = out.join(format!("{}_{index}.vtu", field.id));
            vtu::write_problem(problem, &path)
                .with_context(|| format!("cannot write {}", path.display()))?;
        }
        if let Some(coupled) = &self.coupled {
            coupled.write(&out.join(format!("cells_{index}.vtu")))?;
        }
        Ok(())
    }

    /// Checks the study's `result.raster`, if any, against the mesh: a raster of more than
    /// [`raster::MAX_PIXELS`] pixels in level 0 is refused (the message names
    /// `result.raster.um_per_px`). Cheap, so a run makes it before it writes anything
    /// ([`ResultWriter::create`] makes it again). Collective; the error is the same on every
    /// process.
    pub fn check_result_options(&self) -> Result<()> {
        let Some(options) = self.study().result.as_ref().and_then(|r| r.raster.as_ref()) else {
            return Ok(());
        };
        let comm = self.mesh().finest().comm();
        let local = self.mesh().local_bounds();
        // The maximum of the negated minima and of the maxima; -inf from a process without cells.
        let reduce = |x: f64| comm.max_real(x);
        let mut bounds = [[0.0; 2]; 2];
        for d in 0..2 {
            let lo = local.map_or(f64::NEG_INFINITY, |b| -b[0][d]);
            let hi = local.map_or(f64::NEG_INFINITY, |b| b[1][d]);
            bounds[0][d] = -reduce(lo)?;
            bounds[1][d] = reduce(hi)?;
        }
        let origin = options.origin_um.unwrap_or(bounds[0]);
        raster::Raster::level0_shape(bounds, options.um_per_px, origin)
            .map(|_| ())
            .map_err(|e| e.context(result::CANNOT_WRITE))
    }

    /// Writes the flow as `<out>/pressure.vtu` and `<out>/velocity.vtu`; nothing without
    /// perfusion. Collective.
    pub fn write_flow(&self, out: &Path) -> Result<()> {
        match &self.flow {
            Some(flow) => vtu::write_flow(flow, out),
            None => Ok(()),
        }
    }
}

/// The cell model of the study of `inputs`, checked against it: its SBML compiled to native code
/// ([`CellModel::new`]); none without `cells`. Collective.
#[cfg(feature = "native")]
fn cell_model(comm: &Comm, inputs: &Inputs) -> Result<Option<CellModel>> {
    match (&inputs.study.cells, &inputs.model_sbml) {
        (Some(spec), Some(sbml)) => Ok(Some(CellModel::new(comm, spec, sbml).context("cells")?)),
        _ => Ok(None),
    }
}

/// The cell model of the study of `inputs`, checked against it: the batched kernel of the inputs
/// ([`CellModel::from_kernel`]); none without `cells`.
#[cfg(not(feature = "native"))]
fn cell_model(_comm: &Comm, inputs: &Inputs) -> Result<Option<CellModel>> {
    match (&inputs.study.cells, &inputs.kernel) {
        (Some(spec), Some(kernel)) => Ok(Some(
            CellModel::from_kernel(spec, kernel.clone()).context("cells")?,
        )),
        _ => Ok(None),
    }
}

/// The maps the study names, read for the cells of this process (nothing for a process without
/// cells). A process which cannot read a map fails every process, with the context
/// [`OnEveryProcess`]. Collective.
fn read_maps(comm: &Comm, study: &Study, scene: &Scene, mesh: &Mesh) -> Result<Maps> {
    let mut ids: Vec<&str> = Vec::new();
    for field in &study.fields {
        for c in [
            &field.diffusion,
            &field.decay,
            &field.source,
            &field.porosity,
        ] {
            ids.extend(c.maps());
        }
    }
    if let Some(p) = &study.perfusion {
        ids.extend(p.k_along.maps());
        ids.extend(p.k_across.maps());
        ids.push(&p.direction.map);
    }
    if let Some(cells) = &study.cells {
        for p in &cells.parameters {
            ids.extend(p.value.maps());
            if let Some(r) = &p.replace {
                ids.push(&r.map);
                ids.extend(r.value.maps());
            }
        }
    }
    ids.sort_unstable();
    ids.dedup();
    // The zonation map of a lobule source is computed, not read.
    let zonation = match &scene.mesh {
        MeshSource::Lobules(spec) if ids.contains(&spec.zonation.map.as_str()) => Some(spec),
        _ => None,
    };
    ids.retain(|id| zonation.is_none_or(|spec| *id != spec.zonation.map));
    let read = match mesh.local_bounds() {
        Some(bounds) => Maps::read(&scene.maps, &ids, bounds)
            .map_err(anyhow::Error::from)
            .and_then(|mut maps| {
                if let Some(spec) = zonation {
                    maps.insert(zonation_map(spec, bounds)?);
                }
                Ok(maps)
            }),
        None => Ok(Maps::default()),
    };
    // Agreed: an error of one process is an error of every process.
    let failed = comm.max_int(read.is_err().into())?;
    match read {
        Ok(maps) if failed == 0 => Ok(maps),
        Ok(_) => Err(anyhow::anyhow!("another process cannot read a map")
            .context(OnEveryProcess("cannot read the maps"))),
        Err(e) => Err(e.context(OnEveryProcess("cannot read the maps"))),
    }
}

/// The zonation map of the lobules of `spec` over the window `bounds` (the local bounds of
/// the cells of a process): the relative position between the walls ([`Lobules::position`], 0
/// on the corner walls, 1 on the centre walls) at the pixel centres of a raster of
/// `zonation.um_per_px` whose first pixel is centred at the lower corner of the bounding box of
/// the lobules and which covers it.
fn zonation_map(spec: &LobuleSpec, bounds: [[f64; 2]; 2]) -> Result<Map> {
    let lobules = Lobules::new(spec).context("lobules")?;
    let [lo, hi] = lobules.bounds();
    let h = spec.zonation.um_per_px;
    let pixels = |axis: usize| (((hi[axis] - lo[axis]) / h).ceil() as usize + 1).max(2);
    let shape = [pixels(1), pixels(0)];
    Ok(Map::from_fn(
        &spec.zonation.map,
        h,
        lo,
        shape,
        bounds,
        |x| lobules.position(x),
    ))
}

#[cfg(test)]
mod tests {
    use super::{OnEveryProcess, causes, significant};

    #[test]
    fn log_values_have_significant_digits() {
        assert_eq!(significant(1.000_000_000_000_2, 4), "1.000");
        assert_eq!(significant(0.012_345_6, 4), "0.01235");
        assert_eq!(significant(12_345.67, 4), "12346");
        assert_eq!(significant(0.0, 4), "0.000");
    }

    /// The chain of an error in one line, without a cause whose text an earlier one shows.
    #[test]
    fn causes_skip_repeated_text() {
        let e = anyhow::anyhow!("no such file")
            .context("cannot read map.zarr: no such file")
            .context(OnEveryProcess("cannot read the maps"));
        assert_eq!(
            causes(&e),
            "cannot read the maps: cannot read map.zarr: no such file"
        );
    }
}
