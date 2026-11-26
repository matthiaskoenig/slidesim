//! `slidesim run`: a study end to end.
//!
//! Loads the study and its scene, checks that the maps the study names are in the scene and the
//! `cells` section against the study and the scene, translates the cell model to C, compiles it
//! and checks it against the study (before anything is solved), loads the mesh (distributed over
//! the processes, refined `mesh.refinements` times), reads the maps over the cells of each
//! process, solves the Darcy [`Flow`] if the study has a `perfusion` section, builds one
//! [`Problem`] per field and, with `cells`, the cells and their exchange with the fields
//! ([`Coupled`]). It removes the outputs `<out>/<field id>_<index>.vtu` and
//! `<out>/cells_<index>.vtu` of an earlier run (`pressure.vtu` and `velocity.vtu` included;
//! other files in `<out>` stay), writes the flow as `<out>/pressure.vtu` (point data
//! `pressure.Pa`) and `<out>/velocity.vtu` (the point vectors `velocity`, the conservative
//! velocity the fields are carried by, and `primal_velocity`, `-K grad p / mu`, um/s), and steps
//! the fields to each output time of `time.outputs` in turn, then to `time.end`: each field
//! alone, or with `cells` all fields and the cells together by Strang splitting ([`Coupled`]).
//! At each output time (time 0 before any step) it writes every field as
//! `<out>/<field id>_<index>.vtu`, `<index>` the position of the time in `time.outputs`: point
//! data `<field id>.mM` on the finest mesh level, Dirichlet values included, one file of the
//! pieces of every process; with `cells`, the cells as `<out>/cells_<index>.vtu`: cell data of
//! the coarse elements, one array per value named after it (NaN in elements without a cell):
//! every state of the model (a species of a concentration as its concentration, in the model's
//! units), the outputs of `cells.outputs` (targets of assignment rules) and the parameters of
//! the element (`cells.parameters` and the two volumes). When every field is done, process 0
//! writes `<out>/summary.json`:
//!
//! ```text
//! {"fields": {"<field id>": {
//!     "times": [...], "l2_norm": [...], "total_mass": [...],
//!     "mass_balance": [null | {
//!         "storage_rate": ..., "source": ..., "advection_defect": ..., "exchange": ...,
//!         "boundary": {"<tag>": {"advective": ..., "diffusive": ..., "resupply": ...}}}]}},
//!  "perfusion": {"tag_flows": {"<tag>": <um^2/s>},
//!                "pressure_islands": {"count": <n>, "area_um2": <um^2>}},
//!  "cells": {"count": <n>, "steps": <n>, "rejected": <n>, "seconds": {"mean": ..., "max": ...},
//!            "exchanged": {"<field id>": [...]}, "amounts": {"<species id>": [...]}},
//!  "phases": {"setup": ..., "pressure": ..., "transport": ..., "exchange": ...,
//!             "cells": ..., "output": ...}}
//! ```
//!
//! with, per output time, the time reached (s), the L2 norm of the field (mM*um), its amount of
//! substance in the domain, the integral of the field weighed by the porosity (mM*um^2,
//! [`Problem::total_mass`]), and its mass balance over the interval since the output time
//! before (since t = 0 for the first), each term its mean rate there
//! ([`Problem::interval_balance`], every term in mM*um^2/s; `null` for an interval of no
//! length, as at time 0):
//! `storage_rate = source + sum over the tags of (advective + diffusive + resupply) + exchange
//! - advection_defect`, the inflow through each tag of the boundary positive into the domain
//! ([`pde::BoundaryInflow`]), `exchange` what the cells gave to the field, and `resupply` what
//! the Dirichlet values of a tag resupplied when the cells changed the field next to them.
//! `storage_rate` is the change of `total_mass` over the interval divided by its length, and
//! the terms add up to it exactly, up to the residuals of the linear solves.
//! `cells` is there for a study with cells: their number, the steps and rejected steps of
//! their integrators summed over the cells and the run, the time of a process in its cells (s,
//! the mean over the processes and the largest; the phase `cells` is the largest and the wait
//! of the others for it), and per output time the amount each
//! linked field received from the cells since t = 0 (mM*um^2) and the amount of each species
//! of the cell compartment in all cells, in the units of the fields (mM*um^2: its
//! concentration times the cell volume of the model, times the blood volume of the element in
//! um^2 over the blood volume of the model). `perfusion` is there for a study with perfusion:
//! the flow through each pressure tag ([`Flow::tag_flows`], positive into the domain), and the
//! pressure islands, the connected components of the mesh which no pressure tag reaches, whose
//! pressure is 0 and which have no flow ([`Flow::islands`]): their number and area (0 and 0.0
//! without). With islands, the run says so in one line. `phases` is the wall time
//! of process 0 in each phase of the run (s), the phases separated by barriers: `setup` (from the
//! start of the process: PETSc, the study, the cell model, the mesh, the maps, the problems of
//! the fields and the cells), `pressure` (the flow, zero without perfusion), `transport` (the
//! steps of the fields), `exchange` (the means of the fields over the cells and the changes
//! given back), `cells` (the steps of the cells) and `output` (the files and the values of the
//! summary); together the time of the run but the writing of `summary.json` itself.

use std::collections::BTreeMap;
use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result};
use ceed::Ceed;
use cells::CellModel;
use contracts::{Scene, Study, check_cells, check_maps};
use maps::Maps;
use mesh::Mesh;
use pde::{Flow, Inputs, MassBalance, PdeError, Problem};
use petsc::{Comm, Petsc, Vector, Viewer};
use serde::Serialize;

use crate::coupled::{Coupled, Seconds, linked_fields};

/// `summary.json`.
#[derive(Debug, Serialize)]
struct Summary {
    fields: BTreeMap<String, FieldSummary>,
    #[serde(skip_serializing_if = "Option::is_none")]
    perfusion: Option<PerfusionSummary>,
    #[serde(skip_serializing_if = "Option::is_none")]
    cells: Option<CellsSummary>,
    /// The wall time of process 0 in each phase: s.
    phases: BTreeMap<&'static str, f64>,
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

/// The wall time of process 0 in each phase of a run, the phases separated by barriers, so that
/// a phase ends when every process ended it.
#[derive(Debug)]
pub(crate) struct Phases<'a> {
    comm: Comm<'a>,
    /// The end of the last phase.
    last: Instant,
    seconds: BTreeMap<&'static str, f64>,
}

impl<'a> Phases<'a> {
    /// The phases of a run started at `started`, each at zero seconds.
    fn new(comm: Comm<'a>, started: Instant) -> Phases<'a> {
        let seconds = [
            "setup",
            "pressure",
            "transport",
            "exchange",
            "cells",
            "output",
        ]
        .into_iter()
        .map(|name| (name, 0.0))
        .collect();
        Phases {
            comm,
            last: started,
            seconds,
        }
    }

    /// Ends a time spent in the phase `name`, from the end of the last phase, when every process
    /// ended it. Collective.
    pub(crate) fn end(&mut self, name: &'static str) -> Result<()> {
        self.comm.barrier()?;
        let now = Instant::now();
        *self.seconds.entry(name).or_default() += (now - self.last).as_secs_f64();
        self.last = now;
        Ok(())
    }
}

/// The cells of the study.
#[derive(Debug, Default, Serialize)]
struct CellsSummary {
    /// The number of cells.
    count: usize,
    /// The steps of the integrators of the cells and the rejected step attempts, summed.
    steps: u64,
    rejected: u64,
    /// The time of each process in its cells (without the wait for the others) over the
    /// processes: the mean and the largest; their ratio is the imbalance of the work.
    seconds: Seconds,
    /// Per linked field and output time, the amount the cells gave to the field since t = 0:
    /// mM*um^2.
    exchanged: BTreeMap<String, Vec<f64>>,
    /// Per species of the cell compartment and output time, its amount in all cells: mM*um^2.
    amounts: BTreeMap<String, Vec<f64>>,
}

/// The flow of the study.
#[derive(Debug, Serialize)]
struct PerfusionSummary {
    /// The flow through each pressure tag, positive into the domain: um^2/s.
    tag_flows: BTreeMap<String, f64>,
    pressure_islands: IslandsSummary,
}

/// [`pde::Floating`] of [`Flow::islands`] in `summary.json`.
#[derive(Debug, Serialize)]
struct IslandsSummary {
    count: usize,
    area_um2: f64,
}

/// The values of one field at the output times.
#[derive(Debug, Default, Serialize)]
struct FieldSummary {
    /// s.
    times: Vec<f64>,
    /// mM*um.
    l2_norm: Vec<f64>,
    /// The amount of substance, `int phi c` (porosity `phi`): mM*um^2.
    total_mass: Vec<f64>,
    /// The mass balance over the interval since the output time before, `null` for an interval
    /// of no length.
    mass_balance: Vec<Option<BalanceSummary>>,
}

/// [`pde::MassBalance`] in `summary.json`: mM*um^2/s.
#[derive(Debug, Serialize)]
struct BalanceSummary {
    storage_rate: f64,
    source: f64,
    advection_defect: f64,
    exchange: f64,
    /// The inflow through each tag of the boundary, positive into the domain.
    boundary: BTreeMap<String, InflowSummary>,
}

/// [`pde::BoundaryInflow`] in `summary.json`: mM*um^2/s.
#[derive(Debug, Serialize)]
struct InflowSummary {
    advective: f64,
    diffusive: f64,
    resupply: f64,
}

impl From<MassBalance> for BalanceSummary {
    fn from(balance: MassBalance) -> BalanceSummary {
        BalanceSummary {
            storage_rate: balance.storage_rate,
            source: balance.source,
            advection_defect: balance.advection_defect,
            exchange: balance.exchange,
            boundary: balance
                .boundary
                .into_iter()
                .map(|b| {
                    let inflow = InflowSummary {
                        advective: b.advective,
                        diffusive: b.diffusive,
                        resupply: b.resupply,
                    };
                    (b.tag, inflow)
                })
                .collect(),
        }
    }
}

/// Runs the study of the file `study_path` and writes its results into the directory `out` (see
/// the module documentation), for a process started at `started`. Collective.
pub fn run(petsc: &Petsc, study_path: &Path, out: &Path, started: Instant) -> Result<()> {
    let study = Study::load(study_path).context("cannot load the study")?;
    let scene = Scene::load(&study.scene).context("cannot load the scene")?;
    check_maps(&study, &scene)?;
    check_cells(&study, &scene)?;
    let comm = petsc.comm();
    let model = study
        .cells
        .as_ref()
        .map(|spec| {
            linked_fields(&study, spec)?;
            let path = study_path
                .parent()
                .unwrap_or(Path::new("."))
                .join(&spec.model);
            let sbml = std::fs::read_to_string(&path)
                .with_context(|| format!("cannot read the cell model {}", path.display()))?;
            CellModel::new(&comm, spec, &sbml).context("cells")
        })
        .transpose()?;
    let mesh = Mesh::load(petsc, &scene, study.mesh.refinements).context("cannot load the mesh")?;
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
    let inputs = Inputs {
        maps: &maps,
        flow: flow.as_ref(),
    };
    let mut problems = study
        .fields
        .iter()
        .map(|field| {
            Problem::new(
                &mesh,
                field,
                &study.time,
                &ceed,
                &study.solver.petsc_options,
                &inputs,
            )
            .with_context(|| format!("field '{}'", field.id))
        })
        .collect::<Result<Vec<_>>>()?;
    let mut coupled = match (&study.cells, &model) {
        (Some(spec), Some(model)) => Some(Coupled::new(
            &study, spec, model, &mesh, &maps, &problems, &ceed,
        )?),
        _ => None,
    };
    phases.end("setup")?;
    // Process 0 writes every file, after it removed the outputs of an earlier run.
    if comm.rank() == 0 {
        std::fs::create_dir_all(out)
            .with_context(|| format!("cannot create the output directory {}", out.display()))?;
        remove_earlier_outputs(&study, out)?;
    }
    comm.barrier()?;
    if comm.rank() == 0 {
        for (field, problem) in study.fields.iter().zip(&problems) {
            if let Some(why) = problem.without_interval_balance() {
                println!(
                    "field '{}': no mass balance over the output intervals (null in \
                     summary.json), it follows the steps of BDF of order 2: {why}",
                    field.id
                );
            }
        }
    }
    if let Some(flow) = &flow {
        let islands = flow.islands();
        if islands.count > 0 && comm.rank() == 0 {
            let plural = if islands.count == 1 { "" } else { "s" };
            println!(
                "perfusion: {} pressure island{plural} of {} um^2 in all, which no pressure \
                 boundary reaches: pressure 0 and no flow there",
                islands.count,
                significant(islands.area, 4)
            );
        }
        write_flow(flow, out)?;
    }
    phases.end("output")?;
    let mut summary = Summary {
        fields: study
            .fields
            .iter()
            .map(|field| (field.id.clone(), FieldSummary::default()))
            .collect(),
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
        phases: BTreeMap::new(),
    };
    for (index, &t) in study.time.outputs.iter().enumerate() {
        advance(&study, &mut problems, coupled.as_mut(), t, &mut phases)?;
        for (field, problem) in study.fields.iter().zip(&mut problems) {
            let id = &field.id;
            let values = summary
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
                // Another time integration, as the log says.
                Err(_) if problem.without_interval_balance().is_some() => None,
                Err(e) => return Err(e).with_context(|| format!("field '{id}'")),
            };
            values.mass_balance.push(balance);
            problem.start_interval()?;
            let path = out.join(format!("{id}_{index}.vtu"));
            write_vtu(problem, &path)
                .with_context(|| format!("cannot write {}", path.display()))?;
        }
        if let (Some(coupled), Some(cells)) = (&coupled, &mut summary.cells) {
            let amounts = coupled.write(&comm, &out.join(format!("cells_{index}.vtu")))?;
            for (id, amount) in amounts {
                cells.amounts.entry(id).or_default().push(amount);
            }
            for (field, amount) in coupled.exchanged() {
                let id = study.fields[field].id.clone();
                cells.exchanged.entry(id).or_default().push(amount);
            }
        }
        phases.end("output")?;
    }
    advance(
        &study,
        &mut problems,
        coupled.as_mut(),
        study.time.end,
        &mut phases,
    )?;
    if let (Some(coupled), Some(cells)) = (&coupled, &mut summary.cells) {
        let stats = coupled.stats(&comm)?;
        (cells.steps, cells.rejected, cells.seconds) = (stats.steps, stats.rejected, stats.seconds);
    }
    phases.end("output")?;

    summary.phases = phases.seconds;
    if comm.rank() == 0 {
        let path = out.join("summary.json");
        let text = serde_json::to_string_pretty(&summary)? + "\n";
        std::fs::write(&path, text).with_context(|| format!("cannot write {}", path.display()))?;
    }
    Ok(())
}

/// Advances the fields of `problems` to the time `t`: each alone, or with the cells of
/// `coupled` all together. Collective.
fn advance(
    study: &Study,
    problems: &mut [Problem],
    coupled: Option<&mut Coupled>,
    t: f64,
    phases: &mut Phases<'_>,
) -> Result<()> {
    match coupled {
        Some(coupled) => coupled.advance(study, problems, t, phases),
        None => {
            for (field, problem) in study.fields.iter().zip(problems.iter_mut()) {
                problem
                    .step_to(t)
                    .with_context(|| format!("field '{}'", field.id))?;
            }
            phases.end("transport")
        }
    }
}

/// The maps the study names, read for the cells of this process (nothing for a process without
/// cells). A process which cannot read a map fails every process (collective).
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
    let read = match mesh.local_bounds() {
        Some(bounds) => Maps::read(&scene.maps, &ids, bounds).map_err(anyhow::Error::from),
        None => Ok(Maps::default()),
    };
    let failed = comm.max_int(read.is_err().into())?;
    match read {
        Ok(maps) if failed == 0 => Ok(maps),
        Ok(_) => Err(anyhow::anyhow!("another process cannot read a map")),
        Err(e) => Err(e.context("cannot read the maps")),
    }
}

/// Writes the pressure and the velocity of `flow` as `<out>/pressure.vtu` and
/// `<out>/velocity.vtu`. Collective.
fn write_flow(flow: &Flow, out: &Path) -> Result<()> {
    let comm = flow.discretization().dm().comm();
    for (name, vector) in [
        ("pressure.vtu", flow.local_pressure()?),
        ("velocity.vtu", flow.nodal_velocity()?),
    ] {
        let path = out.join(name);
        write_vector(&comm, &vector, &path)
            .with_context(|| format!("cannot write {}", path.display()))?;
    }
    Ok(())
}

/// Writes the local vector `vector` as the VTU file `path`. Collective.
fn write_vector(comm: &Comm, vector: &Vector, path: &Path) -> Result<()> {
    let mut viewer = Viewer::vtk(comm, path)?;
    vector.view(&mut viewer)?;
    viewer.flush()?;
    Ok(())
}

/// Removes the files `<field id>_<index>.vtu` of the fields of `study` and `cells_<index>.vtu`
/// from the directory `out`:
/// outputs of an earlier run, which a viewer would take for a part of the series of this run (for
/// example the files of more output times). Other files stay.
fn remove_earlier_outputs(study: &Study, out: &Path) -> Result<()> {
    let entries =
        std::fs::read_dir(out).with_context(|| format!("cannot read {}", out.display()))?;
    for entry in entries {
        let entry = entry.with_context(|| format!("cannot read {}", out.display()))?;
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            continue;
        };
        if name == "pressure.vtu"
            || name == "velocity.vtu"
            || is_output(name, "cells")
            || study.fields.iter().any(|field| is_output(name, &field.id))
        {
            let path = entry.path();
            std::fs::remove_file(&path)
                .with_context(|| format!("cannot remove {}", path.display()))?;
        }
    }
    Ok(())
}

/// Whether the file name `name` is that of an output of the field `id`: `<id>_<index>.vtu`.
fn is_output(name: &str, id: &str) -> bool {
    name.strip_prefix(id)
        .and_then(|rest| rest.strip_prefix('_'))
        .and_then(|rest| rest.strip_suffix(".vtu"))
        .is_some_and(|index| !index.is_empty() && index.bytes().all(|b| b.is_ascii_digit()))
}

/// Writes the solution of `problem` as the VTU file `path`. Collective.
fn write_vtu(problem: &Problem, path: &Path) -> Result<()> {
    let solution = problem.local_solution()?;
    write_vector(&problem.discretization().dm().comm(), &solution, path)
}

#[cfg(test)]
mod tests {
    use super::{is_output, significant};

    #[test]
    fn log_values_have_significant_digits() {
        assert_eq!(significant(1.000_000_000_000_2, 4), "1.000");
        assert_eq!(significant(0.012_345_6, 4), "0.01235");
        assert_eq!(significant(12_345.67, 4), "12346");
        assert_eq!(significant(0.0, 4), "0.000");
    }

    #[test]
    fn outputs_are_named_after_the_field_and_the_index() {
        assert!(is_output("oxygen_0.vtu", "oxygen"));
        assert!(is_output("oxygen_12.vtu", "oxygen"));
        assert!(is_output("o2_x_3.vtu", "o2_x"));
        for name in [
            "oxygen_.vtu",
            "oxygen_x.vtu",
            "oxygen_1.vtu.bak",
            "oxygen_1.pvd",
            "oxygen0.vtu",
            "oxygen_x_1.vtu",
            "summary.json",
        ] {
            assert!(!is_output(name, "oxygen"), "{name}");
        }
        assert!(!is_output("oxygen_1.vtu", "o2"));
    }
}
