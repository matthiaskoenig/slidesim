//! End to end tests of `slidesim run`: the binary as a user runs it, on one process and under
//! `mpiexec`, on `testdata/studies/tissue_oxygen` (oxygen from the vessel walls of
//! `tissue_small.msh`). The binary runs without `LD_LIBRARY_PATH`: it finds PETSc and libCEED by
//! its rpath.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

use serde_json::Value;
use test_support::{Mpiexec, Run, run_with_timeout};

/// How long one run may take: a hang (processes waiting for one which failed) fails the test.
const TIMEOUT: Duration = Duration::from_secs(300);

/// The output times of the study.
const OUTPUTS: [f64; 4] = [0.0, 10.0, 30.0, 60.0];

fn study() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/studies/tissue_oxygen/study.json")
}

/// An empty directory for the files of the test `name`.
fn scratch(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("cli")
        .join(name);
    if dir.exists() {
        std::fs::remove_dir_all(&dir).unwrap();
    }
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// `slidesim run <study> --out <out> -- <petsc_options>`, without `LD_LIBRARY_PATH`.
fn args(study: &Path, out: &Path, petsc_options: &[&str]) -> Vec<String> {
    let mut args = vec![
        "run".to_string(),
        study.display().to_string(),
        "--out".to_string(),
        out.display().to_string(),
    ];
    if !petsc_options.is_empty() {
        args.push("--".to_string());
        args.extend(petsc_options.iter().map(|o| o.to_string()));
    }
    args
}

fn slidesim(args: &[String]) -> Run {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_slidesim"));
    cmd.args(args).env_remove("LD_LIBRARY_PATH");
    finished(run_with_timeout(cmd, TIMEOUT))
}

/// `slidesim` on `ranks` processes, or `None` without `mpiexec`.
fn slidesim_parallel(ranks: usize, args: &[String]) -> Option<Run> {
    let Some(mpiexec) = Mpiexec::find() else {
        eprintln!("skipped: no mpiexec");
        return None;
    };
    let mut cmd = mpiexec.command(ranks, env!("CARGO_BIN_EXE_slidesim"));
    cmd.args(args).env_remove("LD_LIBRARY_PATH");
    Some(finished(run_with_timeout(cmd, TIMEOUT)))
}

/// `run`, which did not time out.
fn finished(run: Run) -> Run {
    assert!(
        run.status.is_some(),
        "timed out after {TIMEOUT:?}:\n{}\n{}",
        run.stdout,
        run.stderr
    );
    run
}

fn assert_success(run: &Run) {
    assert!(
        run.status.unwrap().success(),
        "{:?}:\n{}\n{}",
        run.status,
        run.stdout,
        run.stderr
    );
}

/// The field `field` of `<out>/summary.json`: its arrays by name.
fn summary(out: &Path, field: &str) -> [Vec<f64>; 3] {
    let text = std::fs::read_to_string(out.join("summary.json")).unwrap();
    let summary: Value = serde_json::from_str(&text).unwrap();
    let field = &summary["fields"][field];
    ["times", "l2_norm", "total_mass"].map(|key| {
        field[key]
            .as_array()
            .unwrap_or_else(|| panic!("no {key} in {text}"))
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect()
    })
}

/// The names of the point data arrays of the `.vtu` file `path`.
fn point_data(path: &Path) -> Vec<String> {
    let bytes = std::fs::read(path).unwrap();
    let text = String::from_utf8_lossy(&bytes);
    let start = text.find("<PointData>").expect("point data");
    let end = start
        + text[start..]
            .find("</PointData>")
            .expect("end of point data");
    text[start..end]
        .split("Name=\"")
        .skip(1)
        .map(|rest| rest[..rest.find('"').unwrap()].to_string())
        .collect()
}

/// The largest relative difference of `a` and `b`, entry by entry.
fn max_relative_difference(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs() / x.abs().max(y.abs()))
        .fold(0.0, f64::max)
}

#[test]
fn runs_tissue_oxygen() {
    // Existing files in the output directory are overwritten.
    let out = scratch("runs_tissue_oxygen");
    std::fs::write(out.join("oxygen_0.vtu"), "stale").unwrap();
    std::fs::write(out.join("summary.json"), "stale").unwrap();
    // PETSc options after `--`, per field with the prefix `<field id>_`.
    let run = slidesim(&args(&study(), &out, &["-oxygen_ts_monitor"]));
    assert_success(&run);
    assert!(
        run.stdout.contains("120 TS (oxygen_) dt 0.5 time 60."),
        "{}",
        run.stdout
    );

    for index in 0..OUTPUTS.len() {
        let path = out.join(format!("oxygen_{index}.vtu"));
        assert_eq!(point_data(&path), ["oxygen.mM"], "{}", path.display());
    }
    assert!(!out.join(format!("oxygen_{}.vtu", OUTPUTS.len())).exists());

    let [times, l2_norm, total_mass] = summary(&out, "oxygen");
    assert_eq!(times, OUTPUTS);
    assert_eq!(l2_norm.len(), OUTPUTS.len());
    assert!(l2_norm.iter().all(|&n| n > 0.0), "{l2_norm:?}");
    // From zero inside, oxygen enters from the vessel walls towards a steady state.
    assert!(total_mass[0] > 0.0, "{total_mass:?}");
    assert!(total_mass.windows(2).all(|w| w[1] > w[0]), "{total_mass:?}");
    let change = (total_mass[3] - total_mass[2]).abs() / total_mass[3];
    assert!(change < 0.01, "{change}: {total_mass:?}");
}

#[test]
fn outputs_of_an_earlier_run_are_removed() {
    // A short run with two output times, in a directory with the outputs of a run with more.
    let dir = scratch("outputs_of_an_earlier_run_are_removed");
    let mut doc: Value = serde_json::from_str(&std::fs::read_to_string(study()).unwrap()).unwrap();
    doc["scene"] = Value::from(study().with_file_name("scene.json").display().to_string());
    doc["mesh"]["refinements"] = Value::from(0);
    doc["time"] = serde_json::json!({"end": 1.0, "dt": 0.5, "outputs": [0.5, 1.0]});
    let short = dir.join("study.json");
    std::fs::write(&short, doc.to_string()).unwrap();
    let out = dir.join("out");
    let stale = ["oxygen_2.vtu", "oxygen_10.vtu"];
    // Not outputs of slidesim for the field 'oxygen': kept.
    let other = ["oxygen_final.vtu", "oxygen_2.vtu.bak", "notes.txt"];

    for ranks in [1, 2] {
        std::fs::create_dir_all(&out).unwrap();
        for file in stale.iter().chain(&other) {
            std::fs::write(out.join(file), "earlier").unwrap();
        }
        let command = args(&short, &out, &[]);
        let run = if ranks == 1 {
            slidesim(&command)
        } else {
            let Some(run) = slidesim_parallel(ranks, &command) else {
                return;
            };
            run
        };
        assert_success(&run);
        for index in 0..2 {
            let path = out.join(format!("oxygen_{index}.vtu"));
            assert_eq!(point_data(&path), ["oxygen.mM"], "{}", path.display());
        }
        for file in stale {
            assert!(!out.join(file).exists(), "{file} is left");
        }
        for file in other {
            assert!(out.join(file).exists(), "{file} is removed");
        }
    }
}

#[test]
fn unknown_tag_fails_cleanly() {
    let dir = scratch("unknown_tag_fails_cleanly");
    let text = std::fs::read_to_string(study()).unwrap();
    let mut doc: Value = serde_json::from_str(&text).unwrap();
    let scene = study().with_file_name("scene.json");
    doc["scene"] = Value::from(scene.display().to_string());
    doc["fields"][0]["boundaries"][0]["tag"] = Value::from("portal_wall");
    let bad = dir.join("study.json");
    std::fs::write(&bad, serde_json::to_string(&doc).unwrap()).unwrap();
    let out = dir.join("out");
    let expected = "mesh has no tag 'portal_wall'; available:";

    let run = slidesim(&args(&bad, &out, &[]));
    assert_eq!(run.status.unwrap().code(), Some(1), "{}", run.stderr);
    assert!(run.stderr.contains(expected), "{}", run.stderr);
    assert!(!run.stderr.contains("panicked"), "{}", run.stderr);

    // On two processes every process fails: the run ends, no process waits for another.
    if let Some(run) = slidesim_parallel(2, &args(&bad, &out, &[])) {
        assert!(!run.status.unwrap().success(), "{}", run.stderr);
        assert!(run.stderr.contains(expected), "{}", run.stderr);
        assert!(!run.stderr.contains("panicked"), "{}", run.stderr);
    }
}

#[test]
fn abort_keeps_the_monitor_output() {
    // Process 0 alone fails, at the end (it writes summary.json, here a directory): it ends every
    // process (`MPI_Abort`), and the output of the monitors up to then is printed.
    let dir = scratch("abort_keeps_the_monitor_output");
    let mut doc: Value = serde_json::from_str(&std::fs::read_to_string(study()).unwrap()).unwrap();
    doc["scene"] = Value::from(study().with_file_name("scene.json").display().to_string());
    doc["mesh"]["refinements"] = Value::from(0);
    doc["time"] = serde_json::json!({"end": 2.0, "dt": 0.5, "outputs": [2.0]});
    let short = dir.join("study.json");
    std::fs::write(&short, doc.to_string()).unwrap();
    let out = dir.join("out");
    std::fs::create_dir_all(out.join("summary.json")).unwrap();

    let Some(run) = slidesim_parallel(2, &args(&short, &out, &["-oxygen_ts_monitor"])) else {
        return;
    };
    assert!(!run.status.unwrap().success(), "{}", run.stderr);
    assert!(
        run.stderr.contains("[rank 0] error: cannot write"),
        "{}",
        run.stderr
    );
    assert!(!run.stderr.contains("panicked"), "{}", run.stderr);
    assert!(
        run.stdout.contains("4 TS (oxygen_) dt 0.5 time 2."),
        "{}",
        run.stdout
    );
}

#[test]
fn bad_petsc_option_fails_cleanly() {
    let out = scratch("bad_petsc_option_fails_cleanly").join("out");
    let run = slidesim(&args(&study(), &out, &["-oxygen_ksp_type", "no_such_ksp"]));
    assert_eq!(run.status.unwrap().code(), Some(1), "{}", run.stderr);
    assert!(run.stderr.contains("no_such_ksp"), "{}", run.stderr);
    assert!(!run.stderr.contains("panicked"), "{}", run.stderr);
}

/// The relative tolerances of the outer linear solver of the field `oxygen` in the output of
/// `-oxygen_ksp_view`, one per solve.
fn outer_rtols(stdout: &str) -> Vec<String> {
    let mut lines = stdout.lines();
    let mut rtols = Vec::new();
    while lines.any(|line| line.starts_with("KSP Object: (oxygen_)")) {
        let tolerances = lines
            .find_map(|line| line.trim().strip_prefix("tolerances: relative="))
            .expect("tolerances of the KSP");
        rtols.push(tolerances.split(',').next().unwrap().to_string());
    }
    rtols
}

#[test]
fn petsc_options_precedence() {
    // A short run of the study: no refinement, one step.
    let dir = scratch("petsc_options_precedence");
    let mut doc: Value = serde_json::from_str(&std::fs::read_to_string(study()).unwrap()).unwrap();
    doc["scene"] = Value::from(study().with_file_name("scene.json").display().to_string());
    doc["mesh"]["refinements"] = Value::from(0);
    doc["time"] = serde_json::json!({"end": 0.5, "dt": 0.5, "outputs": [0.5]});
    let defaults = dir.join("defaults.json");
    std::fs::write(&defaults, doc.to_string()).unwrap();
    doc["solver"] = serde_json::json!({"petsc_options": "-ksp_rtol 1e-6"});
    let with_options = dir.join("with_options.json");
    std::fs::write(&with_options, doc.to_string()).unwrap();
    let out = dir.join("out");
    let rtols = |study: &Path, command_line: &[&str]| {
        let mut options = vec!["-oxygen_ksp_view"];
        options.extend(command_line);
        let run = slidesim(&args(study, &out, &options));
        assert_success(&run);
        let rtols = outer_rtols(&run.stdout);
        assert!(!rtols.is_empty(), "{}", run.stdout);
        rtols
    };

    // Lowest first: the defaults of the problem, the study, the command line.
    assert!(rtols(&defaults, &[]).iter().all(|r| r == "1e-10"));
    assert!(rtols(&with_options, &[]).iter().all(|r| r == "1e-06"));
    let rtols = rtols(&with_options, &["-oxygen_ksp_rtol", "1e-2"]);
    assert!(rtols.iter().all(|r| r == "0.01"), "{rtols:?}");
}

#[test]
fn usage_error_exits_1() {
    let run = slidesim(&["frobnicate".to_string()]);
    assert_eq!(run.status.unwrap().code(), Some(1), "{}", run.stderr);
    assert!(run.stderr.contains("Usage"), "{}", run.stderr);
    let run = slidesim(&["run".to_string(), "--help".to_string()]);
    assert_success(&run);
    assert!(run.stdout.contains("oxygen_ksp_monitor"), "{}", run.stdout);
}

#[test]
fn two_ranks_match_one() {
    let dir = scratch("two_ranks_match_one");
    // A missing output directory is created, with its parents.
    let serial = dir.join("serial/out");
    assert_success(&slidesim(&args(&study(), &serial, &[])));
    let parallel = dir.join("parallel/out");
    let Some(run) = slidesim_parallel(2, &args(&study(), &parallel, &[])) else {
        return;
    };
    assert_success(&run);
    for index in 0..OUTPUTS.len() {
        let path = parallel.join(format!("oxygen_{index}.vtu"));
        assert_eq!(point_data(&path), ["oxygen.mM"], "{}", path.display());
    }

    let [times_1, l2_norm_1, total_mass_1] = summary(&serial, "oxygen");
    let [times_2, l2_norm_2, total_mass_2] = summary(&parallel, "oxygen");
    assert_eq!(times_1, times_2);
    let l2_norm = max_relative_difference(&l2_norm_1, &l2_norm_2);
    let total_mass = max_relative_difference(&total_mass_1, &total_mass_2);
    eprintln!(
        "relative differences, 2 processes against 1: l2_norm {l2_norm:e}, total_mass {total_mass:e}"
    );
    assert!(l2_norm < 1e-10, "{l2_norm_1:?}\n{l2_norm_2:?}");
    assert!(total_mass < 1e-10, "{total_mass_1:?}\n{total_mass_2:?}");
}
