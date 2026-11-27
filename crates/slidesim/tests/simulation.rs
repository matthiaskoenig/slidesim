//! The library `simulation` against the command line: the summary a [`Simulation`] holds in memory
//! is the `summary.json` `slidesim run` writes, but for its wall times (`phases`,
//! `cells.seconds`). A test of this package, which builds the binary: the tests of the library
//! cannot run it. nextest runs each test in its own process, so the test initializes PETSc once.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

use petsc::Petsc;
use simulation::{Inputs, Simulation};
use test_support::run_with_timeout;

/// The C compiler cache of the tests: in the target directory, not the user's cache.
fn c_cache() -> PathBuf {
    Path::new(env!("CARGO_TARGET_TMPDIR")).join("sbmlsim-c-cache")
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

#[test]
fn simulation_matches_cli() {
    let cache = c_cache();
    // SAFETY: nextest runs each test in its own process; this is the first thing it does.
    unsafe { std::env::set_var("SBMLSIM_C_CACHE", &cache) };
    let study = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../testdata/studies/cells_toy/spt.study.json");
    let out = Path::new(env!("CARGO_TARGET_TMPDIR")).join("simulation_matches_cli");
    if out.exists() {
        std::fs::remove_dir_all(&out).unwrap();
    }
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_slidesim"));
    cmd.arg("run")
        .arg(&study)
        .arg("--out")
        .arg(&out)
        .env_remove("LD_LIBRARY_PATH")
        .env("SBMLSIM_C_CACHE", &cache);
    let run = run_with_timeout(cmd, Duration::from_secs(300));
    assert!(
        run.status.is_some_and(|s| s.success()),
        "{:?}\n{}\n{}",
        run.status,
        run.stdout,
        run.stderr
    );
    let cli = std::fs::read_to_string(out.join("summary.json")).unwrap();

    let petsc = Petsc::init(&[]).unwrap();
    let inputs = Inputs::load(&study).unwrap();
    let mut simulation = Simulation::new(&petsc, inputs, Instant::now(), &mut |_| {}).unwrap();
    while simulation.next_output().unwrap().is_some() {}
    let library = serde_json::to_string_pretty(simulation.summary()).unwrap() + "\n";
    assert_eq!(without_timings(&library), without_timings(&cli));
}
