//! End to end tests of coupled runs of cell models and fields by Strang splitting
//! (`slidesim run` with `cells`): the order of the splitting against the solution of a linear
//! system of ODEs (`testdata/studies/cells_manufactured`), and the conservation of the
//! substance in the blood and the cells (`testdata/studies/cells_toy/cells.study.json`).

mod common;

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

use common::vtu_array;
use serde_json::Value;
use test_support::{Run, run_with_timeout};

/// How long one run may take.
const TIMEOUT: Duration = Duration::from_secs(300);

fn testdata(path: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../testdata")
        .join(path)
}

/// An empty directory for the files of the test `name`.
fn scratch(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("coupled")
        .join(name);
    if dir.exists() {
        std::fs::remove_dir_all(&dir).unwrap();
    }
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// The C compiler cache of the tests: in the target directory, not the user's cache.
fn c_cache() -> PathBuf {
    Path::new(env!("CARGO_TARGET_TMPDIR")).join("sbmlsim-c-cache")
}

/// `slidesim run <study> --out <out>`, which succeeds.
fn run(study: &Path, out: &Path) {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_slidesim"));
    cmd.args([
        "run".as_ref(),
        study.as_os_str(),
        "--out".as_ref(),
        out.as_os_str(),
    ])
    .env_remove("LD_LIBRARY_PATH")
    .env("SBMLSIM_C_CACHE", c_cache());
    let Run {
        status,
        stdout,
        stderr,
    } = run_with_timeout(cmd, TIMEOUT);
    assert!(
        status.is_some_and(|s| s.success()),
        "{status:?}:\n{stdout}\n{stderr}"
    );
}

/// The study `path` with its scene and cell model as absolute paths, changed by `change`,
/// written into `dir`.
fn variant(path: &Path, dir: &Path, name: &str, change: impl FnOnce(&mut Value)) -> PathBuf {
    let mut doc: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let base = path.parent().unwrap();
    doc["scene"] = Value::from(
        base.join(doc["scene"].as_str().unwrap())
            .display()
            .to_string(),
    );
    doc["cells"]["model"] = Value::from(
        base.join(doc["cells"]["model"].as_str().unwrap())
            .display()
            .to_string(),
    );
    change(&mut doc);
    let study = dir.join(format!("{name}.study.json"));
    std::fs::write(&study, doc.to_string()).unwrap();
    study
}

fn summary(out: &Path) -> Value {
    serde_json::from_str(&std::fs::read_to_string(out.join("summary.json")).unwrap()).unwrap()
}

fn numbers(value: &Value) -> Vec<f64> {
    value
        .as_array()
        .unwrap_or_else(|| panic!("not an array: {value}"))
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect()
}

/// `exp(M t) x` for the 2x2 matrix `m` with real distinct eigenvalues.
fn exp_times(m: [[f64; 2]; 2], t: f64, x: [f64; 2]) -> [f64; 2] {
    let trace = m[0][0] + m[1][1];
    let det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
    let root = (trace * trace / 4.0 - det).sqrt();
    let (l1, l2) = (trace / 2.0 + root, trace / 2.0 - root);
    // exp(M t) = (e1 (M - l2 I) - e2 (M - l1 I)) / (l1 - l2) (Sylvester).
    let (e1, e2) = ((l1 * t).exp(), (l2 * t).exp());
    let mut out = [0.0; 2];
    for (i, o) in out.iter_mut().enumerate() {
        for j in 0..2 {
            let id = if i == j { 1.0 } else { 0.0 };
            let a = (e1 * (m[i][j] - l2 * id) - e2 * (m[i][j] - l1 * id)) / (l1 - l2);
            *o += a * x[j];
        }
    }
    out
}

/// The well-mixed manufactured problem: on the unit square with closed walls, a uniform field
/// `u` (porosity 0.5, decay `k_u` = 0.05/s, initial 1 mM) and identical cells of the linear model
/// (`a` = `A`, uptake `k` = 0.1/s, decay `d` = 0.01/s, `V_c/V_b` = 1) stay uniform, with
/// `u' = -k_u u - k (u - a) V_c/V_b` and `a' = k (u - a) - d a`: the solution is that of this
/// linear system. (Without the decay of the field the transport would leave the uniform field
/// as it is, and the splitting would be exact.) Split steps of 8, 4, 2 and 1 s with transport
/// steps of 1/16 of the split step, the cells integrated to 1e-12: the error at 32 s falls at
/// second order. The cell steps of adjacent split steps are merged into one but at the output at
/// 16 s, where they are apart: that output is of second order too.
#[test]
fn strang_order_two() {
    let dir = scratch("strang_order_two");
    let study = testdata("studies/cells_manufactured/study.json");
    let (k_u, k, d) = (0.05, 0.1, 0.01);
    let m = [[-k_u - k, k], [k, -k - d]];
    let mut errors = Vec::new();
    for split in [8.0, 4.0, 2.0, 1.0] {
        let name = format!("split_{split}");
        let study = variant(&study, &dir, &name, |doc| {
            doc["cells"]["split_step"] = Value::from(split);
            doc["time"]["dt"] = Value::from(split / 16.0);
        });
        let out = dir.join(&name);
        run(&study, &out);
        let mut error = [0.0; 2];
        for (index, t) in [(1, 16.0), (2, 32.0)] {
            let exact = exp_times(m, t, [1.0, 0.0]);
            let u = vtu_array(&out.join(format!("u_{index}.vtu")), "u.mM");
            let a = vtu_array(&out.join(format!("cells_{index}.vtu")), "A");
            // Uniform: the field and the cells.
            for values in [&u, &a] {
                let spread = values
                    .iter()
                    .fold(0.0_f64, |m, v| m.max((v - values[0]).abs()));
                assert!(spread <= 1e-12, "{spread:e}");
            }
            error[index - 1] = (u[0] - exact[0]).abs().max((a[0] - exact[1]).abs());
            eprintln!(
                "split step {split} s, t = {t} s: u {} a {} error {:e}",
                u[0],
                a[0],
                error[index - 1]
            );
        }
        errors.push(error);
    }
    for (k, t) in [16.0, 32.0].into_iter().enumerate() {
        let errors: Vec<f64> = errors.iter().map(|e| e[k]).collect();
        let orders: Vec<f64> = errors.windows(2).map(|e| (e[0] / e[1]).log2()).collect();
        eprintln!("t = {t} s: errors {errors:?}, orders {orders:?}");
        for order in &orders {
            assert!(
                *order >= 1.9,
                "t = {t} s: errors {errors:?}, orders {orders:?}"
            );
        }
    }
}

/// The linear model without decay, coupled to the field carried from the portal to the central
/// wall: at every output the amount in the blood and the cells changed by the flows through the
/// walls only (the resupply of the Dirichlet values included), within 1e-8 of the amount; the
/// mass balance of the field closes with the exchange, and the cells hold what they took.
#[test]
fn coupled_conservation() {
    conservation("coupled_conservation", false);
}

/// As [`coupled_conservation`], with adaptive steps of the fields: each steps to the end of a
/// split step, its last step shortened to end there.
#[test]
fn coupled_conservation_with_adaptive_steps() {
    conservation("coupled_conservation_with_adaptive_steps", true);
}

fn conservation(name: &str, adaptive: bool) {
    let dir = scratch(name);
    let study = variant(
        &testdata("studies/cells_toy/cells.study.json"),
        &dir,
        "cells",
        |doc| {
            assert_eq!(doc["cells"]["parameters"][1]["id"], "d");
            assert_eq!(doc["cells"]["parameters"][1]["value"], 0.0);
            doc["time"]["adaptive"] = Value::from(adaptive);
        },
    );
    let out = dir.join("out");
    run(&study, &out);
    let summary = summary(&out);
    let field = &summary["fields"]["substrate"];
    let times = numbers(&field["times"]);
    let mass = numbers(&field["total_mass"]);
    let cells = numbers(&summary["cells"]["amounts"]["A"]);
    let exchanged = numbers(&summary["cells"]["exchanged"]["substrate"]);
    let balances = field["mass_balance"].as_array().unwrap();
    assert!(balances[0].is_null());
    assert!(times.len() >= 3);
    let mut moved = 0.0;
    for k in 1..times.len() {
        let b = &balances[k];
        let term = |name: &str| b[name].as_f64().unwrap();
        let boundary: f64 = b["boundary"]
            .as_object()
            .unwrap()
            .values()
            .map(|i| {
                ["advective", "diffusive", "resupply"]
                    .iter()
                    .map(|p| i[*p].as_f64().unwrap())
                    .sum::<f64>()
            })
            .sum();
        let duration = times[k] - times[k - 1];
        // The balance of the field closes with the exchange.
        let residual = term("storage_rate")
            - (term("source") + boundary + term("exchange") - term("advection_defect"));
        let scale = term("storage_rate")
            .abs()
            .max(boundary.abs())
            .max(term("exchange").abs());
        assert!(
            residual.abs() <= 1e-8 * scale,
            "output {k}: {residual:e} of {scale:e}: {b}"
        );
        assert!(term("exchange") < 0.0, "the cells take up: {b}");
        let resupply = b["boundary"]["portal_wall"]["resupply"].as_f64().unwrap();
        assert!(resupply > 0.0, "{b}");
        // The exchange is what the cells took.
        let taken = exchanged[k] - exchanged[k - 1];
        assert!(
            (term("exchange") * duration - taken).abs() <= 1e-10 * taken.abs(),
            "{} against {taken}",
            term("exchange") * duration
        );
        // The blood and the cells: changed by the flows through the walls only.
        moved += boundary * duration;
        let total = mass[k] + cells[k];
        let expected = mass[0] + cells[0] + moved;
        let error = (total - expected).abs() / total.abs();
        eprintln!(
            "t = {} s: blood {} cells {} error {error:e}",
            times[k], mass[k], cells[k]
        );
        assert!(
            error <= 1e-8,
            "output {k}: {total} against {expected}: {error:e}"
        );
        assert!(cells[k] > cells[k - 1]);
        assert!((cells[k] + exchanged[k]).abs() <= 1e-10 * cells[k]);
    }
    // The counts of the cells, and every phase.
    assert!(summary["cells"]["count"].as_u64().unwrap() > 100);
    assert!(summary["cells"]["steps"].as_u64().unwrap() > 0);
    for phase in [
        "setup",
        "pressure",
        "transport",
        "exchange",
        "cells",
        "output",
    ] {
        assert!(summary["phases"][phase].as_f64().unwrap() > 0.0, "{phase}");
    }
}
