//! `slidesim run`: a study end to end.
//!
//! Loads the study, its scene and the mesh (distributed over the processes, refined
//! `mesh.refinements` times), removes the outputs `<out>/<field id>_<index>.vtu` of an earlier
//! run (other files in `<out>` stay), builds one [`Problem`] per field, and steps the fields to
//! each output time of `time.outputs` in turn, then to `time.end`. At each output time (time 0
//! before any step) it writes every field as `<out>/<field id>_<index>.vtu`, `<index>` the
//! position of the time in `time.outputs`: point data `<field id>.mM` on the finest mesh level,
//! Dirichlet values included, one file of the pieces of every process. When every field is done,
//! process 0 writes `<out>/summary.json`:
//!
//! `{"fields": {"<field id>": {"times": [...], "l2_norm": [...], "total_mass": [...]}}}`
//!
//! with, per output time, the time reached (s), the L2 norm of the field (mM*um) and its
//! integral over the domain (mM*um^2).

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result};
use ceed::Ceed;
use contracts::{Scene, Study};
use mesh::Mesh;
use pde::Problem;
use petsc::{Petsc, Viewer};
use serde::Serialize;

/// `summary.json`.
#[derive(Debug, Serialize)]
struct Summary {
    fields: BTreeMap<String, FieldSummary>,
}

/// The values of one field at the output times.
#[derive(Debug, Default, Serialize)]
struct FieldSummary {
    /// s.
    times: Vec<f64>,
    /// mM*um.
    l2_norm: Vec<f64>,
    /// mM*um^2.
    total_mass: Vec<f64>,
}

/// Runs the study of the file `study_path` and writes its results into the directory `out` (see
/// the module documentation). Collective.
pub fn run(petsc: &Petsc, study_path: &Path, out: &Path) -> Result<()> {
    let study = Study::load(study_path).context("cannot load the study")?;
    let scene = Scene::load(&study.scene).context("cannot load the scene")?;
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
        if study.fields.iter().any(|field| is_output(name, &field.id)) {
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
    let mut viewer = Viewer::vtk(&problem.discretization().dm().comm(), path)?;
    solution.view(&mut viewer)?;
    viewer.flush()?;
    Ok(())
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
