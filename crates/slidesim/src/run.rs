//! `slidesim run`: a study end to end, a loop over the library `simulation`.
//!
//! Loads the inputs of the study ([`Inputs::load`]: the study, its scene and its cell model) and
//! makes the [`Simulation`] (every check, the cell model compiled, the mesh, the maps, the flow,
//! the problems of the fields and the cells; its log, the quality of a generated mesh in one line,
//! printed). Then process 0 removes the outputs `<out>/<field id>_<index>.vtu` and
//! `<out>/cells_<index>.vtu` of an earlier run (`pressure.vtu` and `velocity.vtu` included;
//! other files in `<out>` stay) and prints the notes of the simulation
//! ([`Simulation::notes`]: fields without a mass balance over the output intervals, pressure
//! islands of the flow), and the run writes the flow as `<out>/pressure.vtu` and
//! `<out>/velocity.vtu` ([`Simulation::write_flow`]). It steps to each output time in turn, the
//! last at `time.end` ([`Simulation::next_output`], output 0 at time 0 before any step), and
//! writes the fields and the cells there ([`Simulation::write_vtu`]:
//! `<out>/<field id>_<index>.vtu`, `<out>/cells_<index>.vtu`). Beside the VTU files it writes the
//! store `result/1`, `<out>/result.zarr` ([`ResultWriter`]: made before the first output time,
//! an earlier store removed whole, each output time written as it is reached; `study` names the
//! study by the file name given on the command line). At the end, process 0 writes the summary
//! ([`Simulation::summary`], see [`simulation::Summary`]) as `<out>/summary.json`.

use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result};
use contracts::Study;
use petsc::Petsc;
use simulation::{Inputs, ResultWriter, Simulation};

/// Runs the study of the file `study_path` and writes its results into the directory `out` (see
/// the module documentation), for a process started at `started`. Collective.
pub fn run(petsc: &Petsc, study_path: &Path, out: &Path, started: Instant) -> Result<()> {
    let inputs = Inputs::load(study_path)?;
    // The log of the simulation (the quality of a generated mesh, on process 0) to stdout.
    let mut simulation = Simulation::new(petsc, inputs, started, &mut |line| println!("{line}"))?;
    let comm = petsc.comm();
    // Refuse what the result store cannot hold before anything is written or removed.
    simulation.check_result_options()?;
    // Process 0 writes every file, after it removed the outputs of an earlier run.
    if comm.rank() == 0 {
        std::fs::create_dir_all(out)
            .with_context(|| format!("cannot create the output directory {}", out.display()))?;
        remove_earlier_outputs(simulation.study(), out)?;
    }
    comm.barrier()?;
    if comm.rank() == 0 {
        for note in simulation.notes() {
            println!("{note}");
        }
    }
    simulation.write_flow(out)?;
    let mut result = ResultWriter::create(out, &simulation, study_path)?;
    while let Some(index) = simulation.next_output()? {
        simulation.write_vtu(out, index)?;
        result.write(&simulation, index)?;
    }
    if comm.rank() == 0 {
        let path = out.join("summary.json");
        let text = serde_json::to_string_pretty(simulation.summary())? + "\n";
        std::fs::write(&path, text).with_context(|| format!("cannot write {}", path.display()))?;
    }
    Ok(())
}

/// Removes the files `<field id>_<index>.vtu` of the fields of `study` and `cells_<index>.vtu`
/// and the store `result.zarr` from the directory `out`:
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
            || name == simulation::RESULT_STORE
            || name == "velocity.vtu"
            || is_output(name, "cells")
            || study.fields.iter().any(|field| is_output(name, &field.id))
        {
            let path = entry.path();
            if path.is_dir() {
                std::fs::remove_dir_all(&path)
            } else {
                std::fs::remove_file(&path)
            }
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
