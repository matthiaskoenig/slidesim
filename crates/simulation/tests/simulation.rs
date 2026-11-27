//! Tests of the library: a simulation steps from output time to output time, and a mesh given in
//! its inputs replaces the meshing of the scene. nextest runs each test in its own process, so
//! every test initializes PETSc once. The agreement of the library with `slidesim run` is a test
//! of the command line (`crates/slidesim/tests/simulation.rs`), whose binary it runs.

use std::path::{Path, PathBuf};
use std::time::Instant;

use petsc::Petsc;
use simulation::{Inputs, OnEveryProcess, Simulation};

fn studies(path: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../testdata/studies")
        .join(path)
}

/// Sets the cache of the compiled cell models to the target directory, not the user's cache.
/// First in each test, before PETSc starts threads or anything reads the environment.
fn c_cache() {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("sbmlsim-c-cache");
    // SAFETY: nextest runs each test in its own process; this is the first thing it does.
    unsafe { std::env::set_var("SBMLSIM_C_CACHE", dir) };
}

/// `text`, a `summary.json`, without its wall times: `phases`, the last key, and the `seconds` of
/// the cells.
fn without_timings(text: &str) -> String {
    let at = text.find(",\n  \"phases\": {").expect("phases last");
    let mut kept = Vec::new();
    let mut lines = text[..at].lines();
    while let Some(line) = lines.next() {
        if line.trim_start().starts_with("\"seconds\": {") {
            // Up to the end of the object.
            lines.find(|l| l.trim_start().starts_with('}'));
            continue;
        }
        kept.push(line);
    }
    kept.join("\n")
}

/// `summary.json` of `simulation` as `slidesim run` writes it, without its wall times.
fn summary_without_timings(simulation: &Simulation) -> String {
    without_timings(&serde_json::to_string_pretty(simulation.summary()).unwrap())
}

/// The simulation of `cells_toy/spt` shortened to 4 output times and an end after the last:
/// `next_output` returns the indices in order, the first at time 0 without a step, each at its
/// output time, then `None` at the end of the study, and `None` again after it.
#[test]
fn next_output_visits_every_output() {
    c_cache();
    let mut inputs = Inputs::load(&studies("cells_toy/spt.study.json")).unwrap();
    let times = [0.0, 60.0, 120.0, 240.0];
    inputs.study.time.outputs = times.to_vec();
    inputs.study.time.end = 300.0;
    let petsc = Petsc::init(&[]).unwrap();
    let mut simulation = Simulation::new(&petsc, inputs, Instant::now(), &mut |_| {}).unwrap();
    assert_eq!(simulation.output_times(), times);
    assert_eq!(simulation.time(), 0.0);
    assert_eq!(simulation.field_ids(), ["S", "P"]);
    let names = simulation.cell_value_names();
    assert!(names.iter().any(|n| n == "protein"), "{names:?}");
    let vertices = simulation.mesh().finest().depth_stratum(0).unwrap().len();
    for (index, &t) in times.iter().enumerate() {
        assert_eq!(simulation.next_output().unwrap(), Some(index));
        assert_eq!(simulation.time(), t);
        let summary = serde_json::to_value(simulation.summary()).unwrap();
        let recorded = summary["fields"]["S"]["times"].as_array().unwrap();
        assert_eq!(recorded.len(), index + 1);
        assert_eq!(recorded[index].as_f64(), Some(t));
        let amounts = summary["cells"]["amounts"]["T"].as_array().unwrap();
        assert_eq!(amounts.len(), index + 1);
        for field in 0..2 {
            assert_eq!(simulation.field_values(field).unwrap().len(), vertices);
        }
        let (elements, values) = simulation.cell_values().expect("a study with cells");
        assert!(!elements.is_empty());
        assert_eq!(values.len(), elements.len() * names.len());
        if index == 0 {
            // The initial state: the cells start from the model's values, the fields from 0.
            assert!(simulation.field_values(0).unwrap().contains(&0.0));
            assert_eq!(summary["cells"]["steps"].as_u64(), Some(0));
        }
    }
    assert_eq!(simulation.next_output().unwrap(), None);
    assert_eq!(simulation.time(), 300.0);
    let summary = serde_json::to_value(simulation.summary()).unwrap();
    assert!(summary["cells"]["steps"].as_u64().unwrap() > 0);
    let phases = summary["phases"].as_object().unwrap();
    assert_eq!(phases.len(), 6);
    assert!(phases["cells"].as_f64().unwrap() > 0.0);
    assert_eq!(simulation.next_output().unwrap(), None);
    assert_eq!(simulation.time(), 300.0);
}

/// A study whose first output time is after the start: the first `next_output` steps to it.
#[test]
fn first_output_after_the_start_steps() {
    let mut inputs = Inputs::load(&studies("tissue_oxygen/study.json")).unwrap();
    inputs.study.time.outputs = vec![1.0, 2.0];
    inputs.study.time.end = 2.0;
    let petsc = Petsc::init(&[]).unwrap();
    let mut simulation = Simulation::new(&petsc, inputs, Instant::now(), &mut |_| {}).unwrap();
    assert_eq!(simulation.time(), 0.0);
    assert!(simulation.cell_values().is_none());
    assert_eq!(simulation.next_output().unwrap(), Some(0));
    assert_eq!(simulation.time(), 1.0);
    assert_eq!(simulation.next_output().unwrap(), Some(1));
    assert_eq!(simulation.time(), 2.0);
    assert_eq!(simulation.next_output().unwrap(), None);
    assert_eq!(simulation.time(), 2.0);
}

/// The linear lobule study (shortened) on the mesh `slidesim mesh` writes for its scene, read
/// back from its JSON, has the summary of the run which meshes the scene itself, bitwise.
#[test]
fn mesh_override_matches_generated() {
    c_cache();
    let load = || {
        let mut inputs = Inputs::load(&studies("lobules/linear.study.json")).unwrap();
        inputs.study.time.outputs = vec![0.0, 150.0, 300.0];
        inputs.study.time.end = 300.0;
        inputs
    };
    let petsc = Petsc::init(&[]).unwrap();
    // The summary and the log.
    let run = |inputs: Inputs| {
        let mut log = Vec::new();
        let mut simulation = Simulation::new(&petsc, inputs, Instant::now(), &mut |line| {
            log.push(line.to_string())
        })
        .unwrap();
        while simulation.next_output().unwrap().is_some() {}
        (summary_without_timings(&simulation), log)
    };
    let (generated, log) = run(load());
    // The quality of the generated mesh, to the log only.
    assert_eq!(log.len(), 1, "{log:?}");
    assert!(
        log[0].starts_with("mesh: ") && log[0].contains("min angle"),
        "{log:?}"
    );
    let mut inputs = load();
    let (tagged, _) = mesh::mesh_serial(&petsc, &inputs.scene).unwrap();
    // As `slidesim mesh` writes `mesh.json` and a reader reads it.
    let json = serde_json::to_string(&tagged).unwrap();
    inputs.mesh = Some(serde_json::from_str(&json).unwrap());
    let (given, log) = run(inputs);
    assert_eq!(given, generated);
    assert!(log.is_empty(), "no generated mesh, nothing to log: {log:?}");
}

/// Inputs built in memory (as the browser build builds them from JSON) are checked before
/// anything is made, with errors which every process has: a study with cells but no cell model,
/// and a study without fields (whose time would be no field's).
#[test]
fn invalid_inputs_fail_on_every_process() {
    let petsc = Petsc::init(&[]).unwrap();
    let mut inputs = Inputs::load(&studies("cells_toy/spt.study.json")).unwrap();
    inputs.model_sbml = None;
    let e = Simulation::new(&petsc, inputs, Instant::now(), &mut |_| {})
        .err()
        .expect("no cell model");
    assert!(e.downcast_ref::<OnEveryProcess>().is_some(), "{e:#}");
    assert!(
        format!("{e:#}").contains("the inputs have no cell model"),
        "{e:#}"
    );

    let mut inputs = Inputs::load(&studies("tissue_oxygen/study.json")).unwrap();
    inputs.study.fields.clear();
    let e = Simulation::new(&petsc, inputs, Instant::now(), &mut |_| {})
        .err()
        .expect("no fields");
    assert!(e.downcast_ref::<OnEveryProcess>().is_some(), "{e:#}");
    assert!(
        format!("{e:#}").contains("the study has no fields"),
        "{e:#}"
    );
}
