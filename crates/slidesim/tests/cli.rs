//! End to end tests of `slidesim run`: the binary as a user runs it, on one process and under
//! `mpiexec`, on `testdata/studies/tissue_oxygen` (oxygen from the vessel walls of
//! `tissue_small.msh`) and `testdata/studies/lobule_toy` (a tracer carried by the Darcy flow
//! between two holes, with a map), and with cell models on `testdata/studies/cells_toy`. The
//! binary runs without `LD_LIBRARY_PATH`: it finds PETSc and libCEED by its rpath. The cell
//! models are compiled into a cache in the target directory (`SBMLSIM_C_CACHE`).

mod common;

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, SystemTime};

use common::{cells_by_centroid, points_by_position, relative_difference, vtu_array};
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
    slidesim_with_env(args, &[])
}

/// The C compiler cache of the tests: in the target directory, not the user's cache.
fn c_cache() -> PathBuf {
    Path::new(env!("CARGO_TARGET_TMPDIR")).join("sbmlsim-c-cache")
}

/// [`slidesim`] with the environment variables `env` (name, value).
fn slidesim_with_env(args: &[String], env: &[(&str, &Path)]) -> Run {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_slidesim"));
    cmd.args(args)
        .env_remove("LD_LIBRARY_PATH")
        .env("SBMLSIM_C_CACHE", c_cache())
        .envs(env.iter().copied());
    finished(run_with_timeout(cmd, TIMEOUT))
}

/// `slidesim` on `ranks` processes, or `None` without `mpiexec`.
fn slidesim_parallel(ranks: usize, args: &[String]) -> Option<Run> {
    let Some(mpiexec) = Mpiexec::find() else {
        eprintln!("skipped: no mpiexec");
        return None;
    };
    let mut cmd = mpiexec.command(ranks, env!("CARGO_BIN_EXE_slidesim"));
    cmd.args(args)
        .env_remove("LD_LIBRARY_PATH")
        .env("SBMLSIM_C_CACHE", c_cache());
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

/// A copy of the tissue_oxygen study in `dir` with `refinements` mesh refinements and the other
/// settings of the original (the full study is too slow for the debug build of PETSc).
fn coarse_study(dir: &Path, refinements: u32) -> PathBuf {
    let mut doc: Value = serde_json::from_str(&std::fs::read_to_string(study()).unwrap()).unwrap();
    doc["scene"] = Value::from(study().with_file_name("scene.json").display().to_string());
    doc["mesh"]["refinements"] = Value::from(refinements);
    let path = dir.join("coarse_study.json");
    std::fs::write(&path, doc.to_string()).unwrap();
    path
}

fn lobule_study() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/studies/lobule_toy/study.json")
}

/// The centre of the central hole of `lobule_toy.msh`, um.
const CENTRAL: [f64; 2] = [310.0, 310.0];

/// `<out>/summary.json` as JSON.
fn summary_json(out: &Path) -> Value {
    serde_json::from_str(&std::fs::read_to_string(out.join("summary.json")).unwrap()).unwrap()
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

/// A time integration other than BDF2, chosen by PETSc options (another TS type, another BDF
/// order): the run goes on to the end, with a note in the log and `null` for the mass balances,
/// which follow the steps of BDF2.
#[test]
fn other_time_integration_runs_without_balance() {
    for (name, options) in [
        ("cn", ["-oxygen_ts_type", "cn"]),
        ("bdf_order_1", ["-oxygen_ts_bdf_order", "1"]),
    ] {
        let out = scratch(&format!(
            "other_time_integration_runs_without_balance_{name}"
        ));
        let run = slidesim(&args(&coarse_study(&out, 1), &out, &options));
        assert_success(&run);
        let notes: Vec<&str> = run
            .stdout
            .lines()
            .filter(|line| line.contains("no mass balance"))
            .collect();
        assert_eq!(notes.len(), 1, "{name}: {}", run.stdout);
        assert!(notes[0].starts_with("field 'oxygen': "), "{}", notes[0]);
        println!("{}", notes[0]);
        let [times, _, _] = summary(&out, "oxygen");
        assert_eq!(times, OUTPUTS, "{name}");
        assert!(
            balances(&out, "oxygen").iter().all(Option::is_none),
            "{name}: {:?}",
            balances(&out, "oxygen")
        );
    }
}

/// The mass balances of the field `field` of `<out>/summary.json`, one per output time: `None`
/// for an interval of no length, else the terms by name (`storage_rate`, `source`,
/// `advection_defect`, `exchange`, `<tag>.advective`, `<tag>.diffusive`, `<tag>.resupply`), in
/// the order of the file.
fn balances(out: &Path, field: &str) -> Vec<Option<Vec<(String, f64)>>> {
    let summary = summary_json(out);
    let balances = summary["fields"][field]["mass_balance"]
        .as_array()
        .unwrap_or_else(|| panic!("no mass_balance in {summary}"))
        .clone();
    balances
        .iter()
        .map(|balance| {
            if balance.is_null() {
                return None;
            }
            let number = |v: &Value| v.as_f64().unwrap_or_else(|| panic!("{balance}"));
            let mut terms: Vec<(String, f64)> =
                ["storage_rate", "source", "advection_defect", "exchange"]
                    .iter()
                    .map(|key| (key.to_string(), number(&balance[key])))
                    .collect();
            for (tag, inflow) in balance["boundary"].as_object().unwrap() {
                for part in ["advective", "diffusive", "resupply"] {
                    terms.push((format!("{tag}.{part}"), number(&inflow[part])));
                }
            }
            Some(terms)
        })
        .collect()
}

/// The term `name` of the balance `terms` (see [`balances`]).
fn term(terms: &[(String, f64)], name: &str) -> f64 {
    terms
        .iter()
        .find(|(n, _)| n == name)
        .unwrap_or_else(|| panic!("no {name} in {terms:?}"))
        .1
}

/// The largest absolute term of the balance `terms`.
fn scale(terms: &[(String, f64)]) -> f64 {
    terms.iter().fold(0.0, |m, (_, x)| m.max(x.abs()))
}

/// Asserts that the balance `terms` closes, `storage_rate = source + inflows + exchange -
/// advection_defect`, within 1e-8 of its largest term.
fn assert_closes(terms: &[(String, f64)]) {
    let inflows: f64 = terms
        .iter()
        .filter(|(name, _)| {
            [".advective", ".diffusive", ".resupply"]
                .iter()
                .any(|part| name.ends_with(part))
        })
        .map(|(_, x)| x)
        .sum();
    let residual =
        term(terms, "storage_rate") - term(terms, "source") - inflows - term(terms, "exchange")
            + term(terms, "advection_defect");
    assert!(
        residual.abs() <= 1e-8 * scale(terms),
        "residual {residual:e}: {terms:?}"
    );
}

/// The largest difference of the balances `a` and `b` (of the same output times), term by term,
/// relative to the largest term of the balance: terms which are rounding (zero flow, no source)
/// have no relative difference of their own.
fn max_balance_difference(
    a: &[Option<Vec<(String, f64)>>],
    b: &[Option<Vec<(String, f64)>>],
) -> f64 {
    assert_eq!(a.len(), b.len());
    let mut largest = 0.0_f64;
    for (a, b) in a.iter().zip(b) {
        let (Some(a), Some(b)) = (a, b) else {
            assert!(a.is_none() && b.is_none(), "{a:?}\n{b:?}");
            continue;
        };
        let names = |t: &[(String, f64)]| t.iter().map(|(n, _)| n.clone()).collect::<Vec<_>>();
        assert_eq!(names(a), names(b));
        let scale = scale(a).max(scale(b));
        for ((_, x), (_, y)) in a.iter().zip(b) {
            largest = largest.max((x - y).abs() / scale);
        }
    }
    largest
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
fn lobule_toy_flows_from_portal_to_central() {
    let out = scratch("lobule_toy_flows_from_portal_to_central");
    assert_success(&slidesim(&args(&lobule_study(), &out, &[])));
    let summary = summary_json(&out);
    let islands = &summary["perfusion"]["pressure_islands"];
    assert_eq!(islands["count"].as_u64(), Some(0), "{islands}");
    assert_eq!(islands["area_um2"].as_f64(), Some(0.0), "{islands}");
    let flows = &summary["perfusion"]["tag_flows"];
    let portal = flows["portal_wall"].as_f64().unwrap();
    let central = flows["central_wall"].as_f64().unwrap();
    assert!(portal > 0.0, "{flows}");
    assert!(central < 0.0, "{flows}");
    assert!((portal + central).abs() < 1e-8 * portal, "{flows}");
    assert_eq!(point_data(&out.join("pressure.vtu")), ["pressure.Pa"]);
    assert_eq!(
        point_data(&out.join("velocity.vtu")),
        ["velocity", "primal_velocity"]
    );
    // The mass balance of the tracer at each output time (none before the first step).
    let balances = balances(&out, "sucrose");
    assert_eq!(balances.len(), 3, "{balances:?}");
    assert!(balances[0].is_none(), "{balances:?}");
    for terms in balances[1..].iter().map(|b| b.as_ref().unwrap()) {
        assert_closes(terms);
        // The conservative velocity has no advection defect (the primal velocity of milestone
        // 2 had -411 and -229 against an inflow of 2105 um^2/s at 1 mM, about 20 % and 11 %).
        let defect = term(terms, "advection_defect");
        assert!(defect.abs() < 1e-10 * portal, "{defect} against {portal}");
        // The tracer enters at the portal wall (1 mM), carried and by diffusion, and leaves by
        // the flow through the central wall; none crosses the section edge, a wall without
        // flow (no diffusive flux there either: no condition).
        assert!(term(terms, "portal_wall.advective") > 0.0, "{terms:?}");
        assert!(term(terms, "portal_wall.diffusive") > 0.0, "{terms:?}");
        assert!(term(terms, "central_wall.advective") < 0.0, "{terms:?}");
        for tag in ["central_wall", "section_edge"] {
            assert_eq!(term(terms, &format!("{tag}.diffusive")), 0.0, "{terms:?}");
        }
        let section_edge = term(terms, "section_edge.advective");
        assert!(section_edge.abs() <= 1e-12 * portal, "{terms:?}");
        // The advective inflow at the portal wall is the flow times 1 mM: the normal component
        // of the conservative velocity there is the flux through each edge over its length.
        let advective = term(terms, "portal_wall.advective");
        assert!(
            (advective - portal).abs() < 1e-10 * portal,
            "{advective} against {portal}"
        );
    }
}

/// The mean of the tracer in the output `path` over the nodes within `radius` um of `centre`.
fn mean_near(path: &Path, centre: [f64; 2], radius: f64) -> f64 {
    let points = vtu_array(path, "Position");
    let sucrose = vtu_array(path, "sucrose.mM");
    assert_eq!(points.len(), 3 * sucrose.len());
    let near: Vec<f64> = points
        .as_chunks::<3>()
        .0
        .iter()
        .zip(&sucrose)
        .filter(|(x, _)| ((x[0] - centre[0]).powi(2) + (x[1] - centre[1]).powi(2)).sqrt() < radius)
        .map(|(_, &c)| c)
        .collect();
    assert!(near.len() > 10, "{}", near.len());
    near.iter().sum::<f64>() / near.len() as f64
}

/// The study of `lobule_study` on `island.msh` (whose rest has `portal_wall` and `central_wall`,
/// and an island of 1 um^2 bounded by `vessel_wall` alone, which no pressure reaches), with the
/// zonation map of `lobule_toy` scaled over it, written into `dir` with its scene.
fn island_study(dir: &Path) -> PathBuf {
    let lobule = lobule_study();
    let mut scene: Value = serde_json::from_str(
        &std::fs::read_to_string(lobule.with_file_name("scene.json")).unwrap(),
    )
    .unwrap();
    let testdata = lobule.parent().unwrap().join("../..");
    scene["mesh"]["path"] = Value::from(testdata.join("meshes/island.msh").display().to_string());
    let map = &mut scene["maps"][0];
    map["path"] = Value::from(lobule.with_file_name("zonation.zarr").display().to_string());
    // 106 pixels over [-0.1, 6.3] um.
    map["um_per_px"] = Value::from(0.0625);
    map["origin_um"] = serde_json::json!([-0.1, -0.1]);
    let scene_path = dir.join("scene.json");
    std::fs::write(&scene_path, scene.to_string()).unwrap();
    let mut study: Value =
        serde_json::from_str(&std::fs::read_to_string(&lobule).unwrap()).unwrap();
    study["scene"] = Value::from(scene_path.display().to_string());
    study["time"] =
        serde_json::json!({"end": 0.1, "dt": 0.05, "adaptive": false, "outputs": [0.0, 0.1]});
    let path = dir.join("island_study.json");
    std::fs::write(&path, study.to_string()).unwrap();
    path
}

/// A floating island of the mesh, which no pressure boundary reaches, has no flow: the run
/// succeeds, reports it in `summary.json` (`perfusion.pressure_islands`: one, its area) and says
/// so in one line.
#[test]
fn island_reports_pressure_islands() {
    let dir = scratch("island_reports_pressure_islands");
    let out = dir.join("out");
    let run = slidesim(&args(&island_study(&dir), &out, &[]));
    assert_success(&run);
    let islands = &summary_json(&out)["perfusion"]["pressure_islands"];
    assert_eq!(islands["count"].as_u64(), Some(1), "{islands}");
    let area = islands["area_um2"].as_f64().unwrap();
    assert!((area - 1.0).abs() < 1e-12, "{islands}");
    let lines: Vec<&str> = run
        .stdout
        .lines()
        .filter(|line| line.contains("pressure island"))
        .collect();
    assert_eq!(lines.len(), 1, "{}", run.stdout);
    assert!(lines[0].contains("1 pressure island "), "{}", lines[0]);
    // The mass balance of the tracer stays exact: the island exchanges nothing by advection.
    let portal = summary_json(&out)["perfusion"]["tag_flows"]["portal_wall"]
        .as_f64()
        .unwrap();
    assert!(portal > 0.0);
    for terms in balances(&out, "sucrose")[1..]
        .iter()
        .map(|b| b.as_ref().unwrap())
    {
        assert_closes(terms);
        let defect = term(terms, "advection_defect");
        assert!(defect.abs() < 1e-10 * portal, "{defect} against {portal}");
    }
}

/// The phases of a run in `summary.json`, the wall time of process 0 in each: together the time
/// of the run until it writes the summary. (Not until the process ends: it then still destroys
/// its objects and finalizes PETSc and MPI, about 0.1 s, which no phase can hold, and which would
/// be more than a tenth of this short run.)
#[test]
fn summary_has_phases() {
    phases_sum_to_the_run(&lobule_study(), "summary_has_phases");
}

/// As [`summary_has_phases`], with cells: the phases of the exchange and the cells.
#[test]
fn summary_has_phases_with_cells() {
    phases_sum_to_the_run(&cells_study(), "summary_has_phases_with_cells");
}

fn phases_sum_to_the_run(study: &Path, name: &str) {
    let out = scratch(name);
    // The cell model compiled before the clock starts.
    assert_success(&slidesim(&args(study, &out, &[])));
    let start = SystemTime::now();
    let run = slidesim(&args(study, &out, &[]));
    assert_success(&run);
    let written = std::fs::metadata(out.join("summary.json"))
        .and_then(|m| m.modified())
        .unwrap();
    let wall = written.duration_since(start).unwrap().as_secs_f64();
    let summary = summary_json(&out);
    let phases = summary["phases"]
        .as_object()
        .unwrap_or_else(|| panic!("no phases in {summary}"));
    let mut sum = 0.0;
    for name in [
        "setup",
        "pressure",
        "transport",
        "exchange",
        "cells",
        "output",
    ] {
        let seconds = phases
            .get(name)
            .and_then(Value::as_f64)
            .unwrap_or_else(|| panic!("no phase {name} in {phases:?}"));
        assert!(seconds >= 0.0, "{name}: {seconds}");
        sum += seconds;
    }
    println!("phases {phases:?} sum to {sum} s, the summary written after {wall} s");
    assert!(
        (sum - wall).abs() <= 0.1 * wall,
        "phases {phases:?} sum to {sum} s, the run wrote its summary after {wall} s"
    );
}

#[test]
fn lobule_toy_tracer_reaches_central() {
    let out = scratch("lobule_toy_tracer_reaches_central");
    assert_success(&slidesim(&args(&lobule_study(), &out, &[])));
    // Nodes within 10 um of the wall of a hole (radius 20 um). The portal hole is at (90, 90).
    let portal = [90.0, 90.0];
    let end = mean_near(&out.join("sucrose_2.vtu"), CENTRAL, 30.0);
    assert!(end > 0.1, "{end}");
    // The front is still on its way at t = 5: more tracer at the portal than at the central hole.
    let middle = out.join("sucrose_1.vtu");
    let (at_portal, at_central) = (
        mean_near(&middle, portal, 30.0),
        mean_near(&middle, CENTRAL, 30.0),
    );
    assert!(at_portal > 0.9, "{at_portal}");
    assert!(
        at_central < 0.5 * at_portal,
        "{at_central} against {at_portal}"
    );
}

#[test]
fn unknown_map_fails_before_solving() {
    let dir = scratch("unknown_map_fails_before_solving");
    let mut doc: Value =
        serde_json::from_str(&std::fs::read_to_string(lobule_study()).unwrap()).unwrap();
    doc["scene"] = Value::from(
        lobule_study()
            .with_file_name("scene.json")
            .display()
            .to_string(),
    );
    doc["perfusion"]["direction"]["map"] = Value::from("zonation2");
    let bad = dir.join("study.json");
    std::fs::write(&bad, doc.to_string()).unwrap();
    let out = dir.join("out");

    let run = slidesim(&args(&bad, &out, &[]));
    assert_eq!(run.status.unwrap().code(), Some(1), "{}", run.stderr);
    assert!(run.stderr.contains("zonation2"), "{}", run.stderr);
    assert!(
        run.stderr.contains("the scene defines: [zonation]"),
        "{}",
        run.stderr
    );
    assert!(!run.stderr.contains("panicked"), "{}", run.stderr);
    let vtus = std::fs::read_dir(&out)
        .map(|d| {
            d.flatten()
                .filter(|e| e.path().extension().is_some_and(|x| x == "vtu"))
                .count()
        })
        .unwrap_or(0);
    assert_eq!(vtus, 0);
}

#[test]
fn runs_tissue_oxygen() {
    // Existing files in the output directory are overwritten.
    let out = scratch("runs_tissue_oxygen");
    std::fs::write(out.join("oxygen_0.vtu"), "stale").unwrap();
    std::fs::write(out.join("summary.json"), "stale").unwrap();
    // PETSc options after `--`, per field with the prefix `<field id>_`.
    let run = slidesim(&args(&coarse_study(&out, 1), &out, &["-oxygen_ts_monitor"]));
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
    // The mass balance of each step without flow: no advective terms, the oxygen enters by
    // diffusion through the vessel walls (dirichlet) and nothing crosses the section edge
    // (no condition: no flux).
    let balances = balances(&out, "oxygen");
    assert_eq!(balances.len(), OUTPUTS.len());
    assert!(balances[0].is_none());
    for terms in balances[1..].iter().map(|b| b.as_ref().unwrap()) {
        assert_closes(terms);
        for (name, x) in terms {
            if name == "advection_defect" || name.ends_with(".advective") {
                assert_eq!(*x, 0.0, "{name}: {terms:?}");
            }
        }
        assert!(term(terms, "vessel_wall.diffusive") > 0.0, "{terms:?}");
        assert_eq!(term(terms, "section_edge.diffusive"), 0.0, "{terms:?}");
    }
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
    let study = coarse_study(&dir, 1);
    let serial = dir.join("serial/out");
    assert_success(&slidesim(&args(&study, &serial, &[])));
    let parallel = dir.join("parallel/out");
    let Some(run) = slidesim_parallel(2, &args(&study, &parallel, &[])) else {
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
    let balance =
        max_balance_difference(&balances(&serial, "oxygen"), &balances(&parallel, "oxygen"));
    eprintln!(
        "relative differences, 2 processes against 1: l2_norm {l2_norm:e}, \
         total_mass {total_mass:e}, mass_balance {balance:e}"
    );
    assert!(l2_norm < 1e-10, "{l2_norm_1:?}\n{l2_norm_2:?}");
    assert!(total_mass < 1e-10, "{total_mass_1:?}\n{total_mass_2:?}");
    assert!(balance < 1e-10, "{balance:e}");
}

#[test]
fn two_ranks_match_one_with_perfusion() {
    let dir = scratch("two_ranks_match_one_with_perfusion");
    let serial = dir.join("serial");
    assert_success(&slidesim(&args(&lobule_study(), &serial, &[])));
    let parallel = dir.join("parallel");
    let Some(run) = slidesim_parallel(2, &args(&lobule_study(), &parallel, &[])) else {
        return;
    };
    assert_success(&run);
    for name in ["pressure.vtu", "velocity.vtu", "sucrose_2.vtu"] {
        assert!(parallel.join(name).exists(), "{name}");
    }
    let [times_1, l2_norm_1, total_mass_1] = summary(&serial, "sucrose");
    let [times_2, l2_norm_2, total_mass_2] = summary(&parallel, "sucrose");
    assert_eq!(times_1, times_2);
    let l2_norm = max_relative_difference(&l2_norm_1[1..], &l2_norm_2[1..]);
    let total_mass = max_relative_difference(&total_mass_1[1..], &total_mass_2[1..]);
    let flows = |out: &Path| {
        let flows = summary_json(out)["perfusion"]["tag_flows"].clone();
        ["portal_wall", "central_wall"].map(|tag| flows[tag].as_f64().unwrap())
    };
    let flow = max_relative_difference(&flows(&serial), &flows(&parallel));
    // The mass balance, advection_defect included: each term against the largest of its
    // balance.
    let balance = max_balance_difference(
        &balances(&serial, "sucrose"),
        &balances(&parallel, "sucrose"),
    );
    eprintln!(
        "relative differences, 2 processes against 1: l2_norm {l2_norm:e}, \
         total_mass {total_mass:e}, flow {flow:e}, mass_balance {balance:e}"
    );
    assert!(l2_norm < 1e-10, "{l2_norm_1:?}\n{l2_norm_2:?}");
    assert!(total_mass < 1e-10, "{total_mass_1:?}\n{total_mass_2:?}");
    assert!(flow < 1e-10, "{flow}");
    assert!(balance < 1e-10, "{balance:e}");
}

fn cells_study() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/studies/cells_toy/cells.study.json")
}

/// The numbers of the JSON array `value`.
fn numbers(value: &Value) -> Vec<f64> {
    value
        .as_array()
        .unwrap_or_else(|| panic!("not an array: {value}"))
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect()
}

/// The study `study` with the cells on the path `solver` (`engine` or `batched`), written into
/// `dir` with the paths of its scene and its model made absolute.
fn with_solver(study: &Path, solver: &str, dir: &Path) -> PathBuf {
    let mut json: Value = serde_json::from_str(&std::fs::read_to_string(study).unwrap()).unwrap();
    let base = study.parent().unwrap();
    let absolute = |v: &Value| Value::from(base.join(v.as_str().unwrap()).display().to_string());
    json["scene"] = absolute(&json["scene"]);
    json["cells"]["model"] = absolute(&json["cells"]["model"]);
    json["cells"]["solver"] = Value::from(solver);
    let path = dir.join(format!("{solver}.study.json"));
    std::fs::write(&path, serde_json::to_string_pretty(&json).unwrap()).unwrap();
    path
}

/// The linear cell model on `cells_toy` on 2 and 3 processes, with fixed transport steps, on
/// both paths of the cells: the fields, the states of the cells and the integrals of
/// `summary.json` agree with one process within 1e-10, and the cells take the same steps (their
/// step control is their own).
#[test]
fn cells_parallel_matches_one() {
    let dir = scratch("cells_parallel_matches_one");
    for solver in ["engine", "batched"] {
        let study = with_solver(&cells_study(), solver, &dir);
        if !cells_parallel_matches_one_on(&study, &dir.join(solver)) {
            return;
        }
    }
}

/// [`cells_parallel_matches_one`] of `study`, with the outputs in `dir`; `false` without
/// `mpiexec`.
fn cells_parallel_matches_one_on(study: &Path, dir: &Path) -> bool {
    let serial = dir.join("serial");
    assert_success(&slidesim(&args(study, &serial, &[])));
    let last = 3;
    let cell_values = ["A_ext", "A", "uptake_rate", "k", "Vext"];
    let one = summary_json(&serial);
    for ranks in [2, 3] {
        let parallel = dir.join(format!("parallel_{ranks}"));
        let Some(run) = slidesim_parallel(ranks, &args(study, &parallel, &[])) else {
            return false;
        };
        assert_success(&run);
        let many = summary_json(&parallel);
        let field = |s: &Value, key: &str| numbers(&s["fields"]["substrate"][key]);
        let mut largest = 0.0_f64;
        for key in ["l2_norm", "total_mass"] {
            largest = largest.max(relative_difference(&field(&one, key), &field(&many, key)));
        }
        for (group, key) in [("amounts", "A"), ("exchanged", "substrate")] {
            let (a, b) = (&one["cells"][group][key], &many["cells"][group][key]);
            largest = largest.max(relative_difference(&numbers(a), &numbers(b)));
        }
        let balance = max_balance_difference(
            &balances(&serial, "substrate"),
            &balances(&parallel, "substrate"),
        );
        for key in ["count", "steps", "rejected"] {
            assert_eq!(one["cells"][key], many["cells"][key], "{key}");
        }
        // The fields, point by point.
        let path = |out: &Path| out.join(format!("substrate_{last}.vtu"));
        let (a, b) = (
            points_by_position(&path(&serial), "substrate.mM"),
            points_by_position(&path(&parallel), "substrate.mM"),
        );
        assert_eq!(a.len(), b.len());
        assert!(a.iter().zip(&b).all(|(x, y)| x.0 == y.0));
        let values = |p: &[([i64; 2], f64)]| p.iter().map(|x| x.1).collect::<Vec<_>>();
        let fields = relative_difference(&values(&a), &values(&b));
        // The cells, element by element.
        let path = |out: &Path| out.join(format!("cells_{last}.vtu"));
        let (a, b) = (
            cells_by_centroid(&path(&serial), &cell_values),
            cells_by_centroid(&path(&parallel), &cell_values),
        );
        assert_eq!(a.len(), b.len());
        let mut cells = 0.0_f64;
        for k in 0..cell_values.len() {
            assert!(a.iter().zip(&b).all(|(x, y)| x.0 == y.0));
            let column = |c: &[([i64; 2], Vec<f64>)]| c.iter().map(|x| x.1[k]).collect::<Vec<_>>();
            cells = cells.max(relative_difference(&column(&a), &column(&b)));
        }
        eprintln!(
            "{}: relative differences, {ranks} processes against 1: integrals {largest:e}, \
             mass_balance {balance:e}, field {fields:e}, cells {cells:e}",
            study.display()
        );
        assert!(largest < 1e-10, "{largest:e}");
        assert!(balance < 1e-10, "{balance:e}");
        assert!(fields < 1e-10, "{fields:e}");
        assert!(cells < 1e-10, "{cells:e}");
    }
    true
}

/// Without a C compiler for the cell model, the run fails at the start, before anything is
/// solved or written: exit code 1, an error naming the compiler.
#[test]
fn run_without_compiler_fails_before_solving() {
    let dir = scratch("run_without_compiler_fails_before_solving");
    let out = dir.join("out");
    // A cache of its own, so that the model is compiled.
    let cache = dir.join("cache");
    let run = slidesim_with_env(
        &args(&cells_study(), &out, &[]),
        &[
            ("CC", Path::new("/nonexistent/cc")),
            ("SBMLSIM_C_CACHE", &cache),
        ],
    );
    assert_eq!(
        run.status.unwrap().code(),
        Some(1),
        "{}\n{}",
        run.stdout,
        run.stderr
    );
    assert!(run.stderr.contains("/nonexistent/cc"), "{}", run.stderr);
    assert!(
        !out.exists(),
        "{:?}",
        std::fs::read_dir(&out).map(|d| d.count())
    );
}

/// The SPT cell model on `cells_toy` for 60 min, 5 mM of substrate at the portal wall: necrosis
/// starts in the cells of high protein (the central end of the zonation) and has not reached
/// those of low protein by the end.
#[test]
fn spt_toy_runs() {
    let study = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../testdata/studies/cells_toy/spt.study.json");
    let out = scratch("spt_toy_runs");
    assert_success(&slidesim(&args(&study, &out, &[])));
    let summary = summary_json(&out);
    assert_eq!(numbers(&summary["fields"]["S"]["times"]).len(), 5);
    let mut started = None;
    for index in 0..5 {
        let path = out.join(format!("cells_{index}.vtu"));
        let protein = vtu_array(&path, "protein");
        let necrosis = vtu_array(&path, "necrosis");
        let mut cells: Vec<(f64, f64)> = protein.into_iter().zip(necrosis).collect();
        cells.sort_by(|a, b| a.0.total_cmp(&b.0));
        let quarter = cells.len() / 4;
        let mean = |c: &[(f64, f64)]| c.iter().map(|x| x.1).sum::<f64>() / c.len() as f64;
        let (low, high) = (
            mean(&cells[..quarter]),
            mean(&cells[cells.len() - quarter..]),
        );
        eprintln!("output {index}: necrosis {low} in low protein, {high} in high protein");
        assert!(low <= high, "output {index}: {low} against {high}");
        if high > 0.1 && started.is_none() {
            started = Some(index);
            assert!(low < 0.01 * high, "output {index}: {low} against {high}");
        }
    }
    assert!(started.is_some(), "no necrosis");
    assert!(summary["cells"]["amounts"]["T"].as_array().is_some());
}
