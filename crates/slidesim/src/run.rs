//! `slidesim run`: a study end to end.
//!
//! Loads the study and its scene, checks that the maps the study names are in the scene (before
//! anything is solved), loads the mesh (distributed over the processes, refined
//! `mesh.refinements` times), reads the maps over the cells of each process, solves the Darcy
//! [`Flow`] if the study has a `perfusion` section and writes it as `<out>/pressure.vtu` (point
//! data `pressure.Pa`) and `<out>/velocity.vtu` (the point vector `velocity`, um/s). It removes
//! the outputs `<out>/<field id>_<index>.vtu` of an earlier run (`pressure.vtu` and
//! `velocity.vtu` included; other files in `<out>` stay), builds one [`Problem`] per field, and
//! steps the fields to each output time of `time.outputs` in turn, then to `time.end`. At each
//! output time (time 0 before any step) it writes every field as `<out>/<field id>_<index>.vtu`,
//! `<index>` the position of the time in `time.outputs`: point data `<field id>.mM` on the finest
//! mesh level, Dirichlet values included, one file of the pieces of every process. When every
//! field is done, process 0 writes `<out>/summary.json`:
//!
//! ```text
//! {"fields": {"<field id>": {
//!     "times": [...], "l2_norm": [...], "total_mass": [...],
//!     "mass_balance": [null | {
//!         "storage_rate": ..., "source": ..., "advection_defect": ...,
//!         "boundary": {"<tag>": {"advective": ..., "diffusive": ...}}}]}},
//!  "perfusion": {"tag_flows": {"<tag>": <um^2/s>}}}
//! ```
//!
//! with, per output time, the time reached (s), the L2 norm of the field (mM*um), its amount of
//! substance in the domain, the integral of the field weighed by the porosity (mM*um^2,
//! [`Problem::total_mass`]), and its mass balance at the end of the last step
//! ([`pde::MassBalance`], every term in mM*um^2/s; `null` at a time before the first step):
//! `storage_rate = source + sum over the tags of (advective + diffusive) - advection_defect`,
//! the inflow through each tag of the boundary positive into the domain
//! ([`pde::BoundaryInflow`]). `perfusion` is there for a study with perfusion: the flow through
//! each pressure tag ([`Flow::tag_flows`], positive into the domain).

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result};
use ceed::Ceed;
use contracts::{Scene, Study, check_maps};
use maps::Maps;
use mesh::Mesh;
use pde::{Flow, Inputs, MassBalance, PdeError, Problem};
use petsc::{Comm, Petsc, Vector, Viewer};
use serde::Serialize;

/// `summary.json`.
#[derive(Debug, Serialize)]
struct Summary {
    fields: BTreeMap<String, FieldSummary>,
    #[serde(skip_serializing_if = "Option::is_none")]
    perfusion: Option<PerfusionSummary>,
}

/// The flow of the study.
#[derive(Debug, Serialize)]
struct PerfusionSummary {
    /// The flow through each pressure tag, positive into the domain: um^2/s.
    tag_flows: BTreeMap<String, f64>,
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
    /// The mass balance at the end of the last step, `null` before the first step.
    mass_balance: Vec<Option<BalanceSummary>>,
}

/// [`pde::MassBalance`] in `summary.json`: mM*um^2/s.
#[derive(Debug, Serialize)]
struct BalanceSummary {
    storage_rate: f64,
    source: f64,
    advection_defect: f64,
    /// The inflow through each tag of the boundary, positive into the domain.
    boundary: BTreeMap<String, InflowSummary>,
}

/// [`pde::BoundaryInflow`] in `summary.json`: mM*um^2/s.
#[derive(Debug, Serialize)]
struct InflowSummary {
    advective: f64,
    diffusive: f64,
}

impl From<MassBalance> for BalanceSummary {
    fn from(balance: MassBalance) -> BalanceSummary {
        BalanceSummary {
            storage_rate: balance.storage_rate,
            source: balance.source,
            advection_defect: balance.advection_defect,
            boundary: balance
                .boundary
                .into_iter()
                .map(|b| {
                    let inflow = InflowSummary {
                        advective: b.advective,
                        diffusive: b.diffusive,
                    };
                    (b.tag, inflow)
                })
                .collect(),
        }
    }
}

/// Runs the study of the file `study_path` and writes its results into the directory `out` (see
/// the module documentation). Collective.
pub fn run(petsc: &Petsc, study_path: &Path, out: &Path) -> Result<()> {
    let study = Study::load(study_path).context("cannot load the study")?;
    let scene = Scene::load(&study.scene).context("cannot load the scene")?;
    check_maps(&study, &scene)?;
    let mesh = Mesh::load(petsc, &scene, study.mesh.refinements).context("cannot load the mesh")?;
    let ceed = Ceed::new(ceed::DEFAULT_RESOURCE).context("cannot initialize libCEED")?;
    let comm = petsc.comm();
    // Process 0 writes every file, after it removed the outputs of an earlier run.
    if comm.rank() == 0 {
        std::fs::create_dir_all(out)
            .with_context(|| format!("cannot create the output directory {}", out.display()))?;
        remove_earlier_outputs(&study, out)?;
    }
    comm.barrier()?;

    let maps = read_maps(&comm, &study, &scene, &mesh)?;
    let flow = study
        .perfusion
        .as_ref()
        .map(|perfusion| {
            Flow::solve(&mesh, perfusion, &maps, &ceed, &study.solver.petsc_options)
                .context("perfusion")
        })
        .transpose()?;
    if let Some(flow) = &flow {
        write_flow(flow, out)?;
    }
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
    let mut summary = Summary {
        fields: study
            .fields
            .iter()
            .map(|field| (field.id.clone(), FieldSummary::default()))
            .collect(),
        perfusion: flow.as_ref().map(|flow| PerfusionSummary {
            tag_flows: flow.tag_flows().iter().cloned().collect(),
        }),
    };
    for (index, &t) in study.time.outputs.iter().enumerate() {
        for (field, problem) in study.fields.iter().zip(&mut problems) {
            let id = &field.id;
            problem
                .step_to(t)
                .with_context(|| format!("field '{id}'"))?;
            let values = summary
                .fields
                .get_mut(id)
                .expect("every field has a summary");
            values.times.push(problem.time());
            values.l2_norm.push(problem.l2_norm()?);
            values.total_mass.push(problem.total_mass()?);
            let balance = match problem.mass_balance() {
                Ok(balance) => Some(balance.into()),
                // Before the first step there is no balance (the output of time 0).
                Err(PdeError::NotStepped { .. }) => None,
                Err(e) => return Err(e).with_context(|| format!("field '{id}'")),
            };
            values.mass_balance.push(balance);
            let path = out.join(format!("{id}_{index}.vtu"));
            write_vtu(problem, &path)
                .with_context(|| format!("cannot write {}", path.display()))?;
        }
    }
    for (field, problem) in study.fields.iter().zip(&mut problems) {
        problem
            .step_to(study.time.end)
            .with_context(|| format!("field '{}'", field.id))?;
    }

    if comm.rank() == 0 {
        let path = out.join("summary.json");
        let text = serde_json::to_string_pretty(&summary)? + "\n";
        std::fs::write(&path, text).with_context(|| format!("cannot write {}", path.display()))?;
    }
    Ok(())
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

/// Removes the files `<field id>_<index>.vtu` of the fields of `study` from the directory `out`:
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
    use super::is_output;

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
