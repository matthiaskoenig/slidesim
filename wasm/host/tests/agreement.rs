//! The browser build against the native build: `slidesim.wasm` in wasmtime ([`slidesim_host::run`])
//! with the batched kernel of the SPT cell model, and the native `slidesim`, both from
//! `scripts/wasm/build.sh` (their paths in `target/wasm/artifacts.env`).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use serde_json::{Value, json};
use slidesim_host::{Outputs, run};
use zarrs::array::Array;
use zarrs::filesystem::FilesystemStore;
use zarrs::group::Group;

/// The repository.
fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// The artifacts of `scripts/wasm/build.sh`.
struct Artifacts {
    slidesim_wasm: PathBuf,
    slidesim_native: PathBuf,
    mathlib: PathBuf,
    kernels: PathBuf,
}

impl Artifacts {
    fn load() -> Artifacts {
        let path = root().join("target/wasm/artifacts.env");
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("{}: {e}; run scripts/wasm/build.sh", path.display()));
        let values: BTreeMap<&str, &str> = text
            .lines()
            .filter(|l| !l.starts_with('#'))
            .filter_map(|l| l.split_once('='))
            .collect();
        let get = |key: &str| PathBuf::from(values.get(key).unwrap_or_else(|| panic!("{key}")));
        Artifacts {
            slidesim_wasm: get("SLIDESIM_WASM"),
            slidesim_native: get("SLIDESIM_NATIVE"),
            mathlib: get("SBMLSIM_MATHLIB"),
            kernels: get("SBMLSIM_KERNELS"),
        }
    }

    /// The kernel of the SPT cell model.
    fn spt_kernel(&self) -> PathBuf {
        self.kernels.join("spt_liver_cell")
    }

    /// Runs `inputs` in `slidesim.wasm` with the SPT kernel.
    fn run_wasm(&self, inputs: &Value, max_pages: Option<u64>) -> anyhow::Result<Outputs> {
        run(
            &self.slidesim_wasm,
            &self.spt_kernel(),
            &self.mathlib,
            &inputs.to_string(),
            max_pages,
        )
    }

    /// Runs the native `slidesim` with `args`, on one process.
    fn native(&self, args: &[&Path]) {
        let output = Command::new(&self.slidesim_native)
            .args(args)
            // OpenMPI's hwloc probes X displays without this.
            .env("HWLOC_COMPONENTS", "-gl")
            .output()
            .expect("the native slidesim runs");
        assert!(
            output.status.success(),
            "slidesim {args:?}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

/// An empty directory of the test `name`.
fn fresh_dir(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(name);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn read_json(path: &Path) -> Value {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    serde_json::from_str(&text).unwrap()
}

fn lobules() -> PathBuf {
    root().join("testdata/studies/lobules")
}

/// A study of one field on the lobule, without flow and cells, of one step: cheap.
fn diffusion_study() -> Value {
    json!({
        "version": "study/1",
        "scene": "scene.json",
        "fields": [{
            "id": "S",
            "diffusion": 700.0,
            "decay": 0.0,
            "source": 0.0,
            "porosity": 0.15,
            "advection": false,
            "initial": 0.0,
            "boundaries": [{"tag": "portal_wall", "dirichlet": 5.0}]
        }],
        "time": {"end": 15.0, "dt": 15.0, "adaptive": false, "outputs": [0.0, 15.0]},
        "mesh": {"refinements": 0}
    })
}

/// The arrays of a native `result.zarr`.
struct Store {
    meta: Value,
    time: Vec<f64>,
    vertices: Vec<f64>,
    triangles: Vec<u32>,
    element: Vec<u64>,
    tags: BTreeMap<String, Vec<u32>>,
    /// Per field, `[T, V]`.
    fields: BTreeMap<String, Vec<f64>>,
    cell_elements: Vec<u64>,
    /// Per cell value, `[T, C]`.
    cells: BTreeMap<String, Vec<f64>>,
}

fn read_store(path: &Path) -> Store {
    let storage = Arc::new(FilesystemStore::new(path).unwrap());
    let root = Group::open(storage.clone(), "/").unwrap();
    let meta = Value::Object(root.attributes().clone());
    fn all<T: zarrs::array::ElementOwned>(storage: &Arc<FilesystemStore>, path: &str) -> Vec<T> {
        let array = Array::open(storage.clone(), path).unwrap();
        array.retrieve_array_subset(&array.subset_all()).unwrap()
    }
    let names = |key: &str, field: &str| -> Vec<String> {
        meta[key]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| {
                v.as_str()
                    .or_else(|| v[field].as_str())
                    .unwrap()
                    .to_string()
            })
            .collect()
    };
    Store {
        time: all(&storage, "/time"),
        vertices: all(&storage, "/mesh/vertices"),
        triangles: all(&storage, "/mesh/triangles"),
        element: all(&storage, "/mesh/element"),
        tags: names("tags", "name")
            .into_iter()
            .map(|n| {
                let values = all(&storage, &format!("/mesh/tags/{n}"));
                (n, values)
            })
            .collect(),
        fields: names("fields", "id")
            .into_iter()
            .map(|id| {
                let values = all(&storage, &format!("/fields/{id}"));
                (id, values)
            })
            .collect(),
        cell_elements: all(&storage, "/cells/element"),
        cells: names("cell_values", "")
            .into_iter()
            .map(|name| {
                let values = all(&storage, &format!("/cells/{name}"));
                (name, values)
            })
            .collect(),
        meta,
    }
}

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|v| v.to_bits()).collect()
}

/// Asserts that `a` and `b` (the arrays `what`) are equal bitwise, else reports how many values
/// differ and by how much at most.
fn assert_bitwise(what: &str, a: &[f64], b: &[f64]) {
    assert_eq!(a.len(), b.len(), "{what}: lengths");
    let differ: Vec<f64> = a
        .iter()
        .zip(b)
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .map(|(x, y)| (x - y).abs())
        .collect();
    assert!(
        differ.is_empty(),
        "{what}: {} of {} values differ, by at most {:e}",
        differ.len(),
        a.len(),
        differ.iter().fold(0.0_f64, |m, d| m.max(*d))
    );
}

/// The native run of the SPT study of one hour on the lobule (one process) and the wasm run on
/// the native mesh agree: the mesh arrays bitwise, the fields within 1e-5 of each field's
/// maximum at every output time, the cell values within `1e-4 |y| + 1e-8`.
#[test]
fn lobule_matches_native() {
    let a = Artifacts::load();
    let dir = fresh_dir("lobule_matches_native");
    let (scene, study) = (
        lobules().join("scene.json"),
        lobules().join("spt_1h.study.json"),
    );
    a.native(&[
        Path::new("mesh"),
        &scene,
        Path::new("--out"),
        &dir.join("mesh"),
    ]);
    a.native(&[
        Path::new("run"),
        &study,
        Path::new("--out"),
        &dir.join("native"),
    ]);
    let native = read_store(&dir.join("native/result.zarr"));
    let inputs = json!({
        "scene": read_json(&scene),
        "study": read_json(&study),
        "mesh": read_json(&dir.join("mesh/mesh.json")),
    });
    let wasm = a.run_wasm(&inputs, None).unwrap();

    assert_eq!(wasm.stdout, "", "slidesim.wasm prints nothing");
    assert_eq!(wasm.quality, Value::Null, "a mesh of the inputs");
    for key in ["times", "fields", "cell_values", "tags", "refinements"] {
        assert_eq!(wasm.meta[key], native.meta[key], "meta.{key}");
    }
    assert_eq!(wasm.meta["complete"], json!(true));
    assert_eq!(wasm.meta["written"], native.meta["written"]);
    assert_bitwise("time", &wasm.time, &native.time);
    assert_bitwise("mesh/vertices", &wasm.mesh.vertices, &native.vertices);
    assert_eq!(wasm.mesh.triangles, native.triangles);
    assert_eq!(wasm.mesh.element, native.element);
    assert_eq!(wasm.mesh.tags, native.tags);
    assert_eq!(wasm.cell_elements, native.cell_elements);
    assert!(!native.cell_elements.is_empty());
    let t = native.time.len();
    assert_eq!(wasm.outputs.len(), t);

    for (id, values) in &native.fields {
        let v = values.len() / t;
        let max = values.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
        let mut worst = 0.0_f64;
        for (k, output) in wasm.outputs.iter().enumerate() {
            assert_eq!(output.index, k);
            assert_eq!(output.time, native.time[k]);
            let row = &output.fields[id];
            assert_eq!(row.len(), v, "field {id}");
            for (w, n) in row.iter().zip(&values[k * v..(k + 1) * v]) {
                worst = worst.max((w - n).abs());
            }
        }
        eprintln!(
            "field {id}: max |wasm - native| {worst:.3e}, field maximum {max:.6e}, ratio {:.3e}",
            worst / max
        );
        assert!(
            worst <= 1e-5 * max,
            "field {id}: {worst:e} > 1e-5 * {max:e}"
        );
    }
    for (name, values) in &native.cells {
        let c = values.len() / t;
        assert_eq!(c, native.cell_elements.len());
        // The largest difference against the bound, and the difference there.
        let (mut ratio, mut at) = (0.0_f64, (0.0, 0.0));
        for (k, output) in wasm.outputs.iter().enumerate() {
            let row = &output.cells[name];
            for (w, n) in row.iter().zip(&values[k * c..(k + 1) * c]) {
                let r = (w - n).abs() / (1e-4 * n.abs() + 1e-8);
                if r > ratio {
                    (ratio, at) = (r, (*w, *n));
                }
            }
        }
        eprintln!(
            "cells {name}: largest |wasm - native| / (1e-4 |y| + 1e-8) {ratio:.3e} (wasm {:e}, native {:e})",
            at.0, at.1
        );
        assert!(ratio <= 1.0, "cells {name}: wasm {} native {}", at.0, at.1);
    }
}

/// The wasm build meshes the lobule itself within the mesher's bounds, the same mesh twice.
#[test]
fn wasm_generated_lobule_is_valid() {
    let a = Artifacts::load();
    let inputs = json!({
        "scene": read_json(&lobules().join("scene.json")),
        "study": diffusion_study(),
    });
    let first = a.run_wasm(&inputs, None).unwrap();
    let q = &first.quality;
    eprintln!(
        "wasm lobule (rings 0): {q}; create {:.2} s",
        first.create_seconds
    );
    assert!(q["min_angle_deg"].as_f64().unwrap() >= 20.0, "{q}");
    assert!(q["min_sicn"].as_f64().unwrap() >= 0.4, "{q}");
    let ratio = q["edge_ratio"].as_array().unwrap();
    assert!(
        ratio[0].as_f64().unwrap() >= 0.4 && ratio[1].as_f64().unwrap() <= 2.0,
        "{q}"
    );
    let tags: Vec<&str> = first.mesh.tags.keys().map(String::as_str).collect();
    assert_eq!(tags, ["central_wall", "field_cut", "portal_wall", "tissue"]);
    assert!(
        first.mesh.tags.values().all(|t| !t.is_empty()),
        "every tag has entities"
    );
    assert!(
        first.log.starts_with("mesh: "),
        "the quality line is logged: {}",
        first.log
    );
    assert_eq!(first.stdout, "", "slidesim.wasm prints nothing");

    let second = a.run_wasm(&inputs, None).unwrap();
    assert_eq!(first.quality, second.quality);
    assert_eq!(bits(&first.mesh.vertices), bits(&second.mesh.vertices));
    assert_eq!(first.mesh.triangles, second.mesh.triangles);
    assert_eq!(first.mesh.element, second.mesh.element);
    assert_eq!(first.mesh.tags, second.mesh.tags);
    for (x, y) in first.outputs.iter().zip(&second.outputs) {
        assert_eq!(bits(&x.fields["S"]), bits(&y.fields["S"]));
    }
}

/// The engine path needs native code: the browser build refuses it, naming the path it runs.
#[test]
fn engine_solver_is_refused() {
    let a = Artifacts::load();
    let mut study = read_json(&lobules().join("spt_1h.study.json"));
    study["cells"]["solver"] = json!("engine");
    let inputs = json!({"scene": read_json(&lobules().join("scene.json")), "study": study});
    let e = a.run_wasm(&inputs, None).unwrap_err();
    assert!(
        format!("{e:#}").contains(
            r#"cells.solver "engine" needs native code; the browser build runs "batched""#
        ),
        "{e:#}"
    );
}

/// The browser build reads no files: a gmsh scene is refused, naming its path.
#[test]
fn file_scene_is_refused() {
    let a = Artifacts::load();
    let scene = json!({
        "version": "scene/1",
        "dimension": 2,
        "units": {"length": "um"},
        "mesh": {"format": "gmsh", "path": "lobule_toy.msh"}
    });
    let inputs = json!({"scene": scene, "study": diffusion_study()});
    let e = a.run_wasm(&inputs, None).unwrap_err();
    let message = format!("{e:#}");
    eprintln!("{message}");
    assert!(message.contains("lobule_toy.msh"), "{message}");
}

/// A run which needs more memory than the cap fails with an error naming the cap, and the host
/// lives on.
#[test]
fn memory_cap_is_an_error() {
    let a = Artifacts::load();
    let inputs = json!({
        "scene": read_json(&lobules().join("scene.json")),
        "study": diffusion_study(),
    });
    // Enough to instantiate slidesim, not to mesh and run.
    let pages = 600;
    let e = a.run_wasm(&inputs, Some(pages)).unwrap_err();
    let message = format!("{e:#}");
    eprintln!("{message}");
    assert!(message.contains("maximum of 600 wasm pages"), "{message}");
    // The same host process runs the study without the cap.
    a.run_wasm(&inputs, None).unwrap();
}

/// The wall time of the mesher in wasm for 0 and 1 rings of lobules (measured for the report,
/// not checked): the time of `slidesim_create` of a cheap study.
#[test]
#[ignore]
fn mesher_wall_times() {
    let a = Artifacts::load();
    for scene in ["scene.json", "ring1.scene.json"] {
        let inputs =
            json!({"scene": read_json(&lobules().join(scene)), "study": diffusion_study()});
        let out = a.run_wasm(&inputs, None).unwrap();
        eprintln!(
            "{scene}: create {:.2} s, {}",
            out.create_seconds,
            out.log.trim_end()
        );
    }
}
