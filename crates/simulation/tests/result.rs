//! Tests of `result/1`, the store `result.zarr` of a run ([`ResultWriter`], [`read_result`]): the
//! mesh is the canonical mesh, every field and cell value at every output time is the value the
//! simulation had, bitwise, each output time is a chunk of its own, the root attributes are a
//! valid `ResultMeta`, a run on 2 or 3 processes writes the same store, and an unfinished run is
//! marked as such. nextest runs each test in its own process, so every test initializes PETSc
//! once; `result_is_partition_independent` runs itself again under `mpiexec`.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use contracts::{RESULT_VERSION, ResultMeta};
use mesh::CanonicalMesh;
use petsc::Petsc;
use simulation::{
    Inputs, OnEveryProcess, RESULT_STORE, ResultData, ResultWriter, Simulation, read_result,
};
use zarrs::array::Array;
use zarrs::filesystem::FilesystemStore;

/// The output times of the shortened `cells_toy/spt` study.
const TIMES: [f64; 4] = [0.0, 60.0, 120.0, 240.0];

/// Set in the runs of [`result_is_partition_independent`] under `mpiexec`: the directory of the
/// store of one process.
const REFERENCE: &str = "SLIDESIM_TEST_RESULT_REFERENCE";

fn studies(path: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../testdata/studies")
        .join(path)
}

/// The study file of the tests.
fn spt() -> PathBuf {
    studies("cells_toy/spt.study.json")
}

/// Sets the cache of the compiled cell models to the target directory, not the user's cache.
/// First in each test, before PETSc starts threads or anything reads the environment.
fn c_cache() {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("sbmlsim-c-cache");
    // SAFETY: nextest runs each test in its own process; this is the first thing it does.
    unsafe { std::env::set_var("SBMLSIM_C_CACHE", dir) };
}

/// An empty directory for the files of the test `name` (made by process 0, which writes the
/// store; the others wait for it at the first collective call).
fn scratch(name: &str, petsc: Option<&Petsc>) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("result")
        .join(name);
    if petsc.is_none_or(|p| p.comm().rank() == 0) {
        if dir.exists() {
            std::fs::remove_dir_all(&dir).unwrap();
        }
        std::fs::create_dir_all(&dir).unwrap();
    }
    dir
}

/// `cells_toy/spt` shortened to the output times [`TIMES`]; with `tight`, the linear solves to
/// 1e-12 and the cells to 1e-14 (rtol; atol 1e-16), the settings of an agreement across
/// processes to 1e-10. The cells decide it: at a cell rtol of 1e-12 the cells' substrate
/// `S_ext` of 2 and 3 processes differs from one process by 9e-11 (cells whose step control
/// decides otherwise on inputs which differ in the last digits), at 1e-14 by 1e-12; the linear
/// tolerance changes nothing there (1e-14 as 1e-12).
fn inputs(tight: bool) -> Inputs {
    let mut inputs = Inputs::load(&spt()).unwrap();
    inputs.study.time.outputs = TIMES.to_vec();
    inputs.study.time.end = 240.0;
    if tight {
        inputs.study.solver.linear_rtol = 1e-12;
        let cells = inputs.study.cells.as_mut().unwrap();
        cells.rtol = 1e-14;
        cells.atol = 1e-16;
    }
    inputs
}

/// The values the simulation has now, gathered on process 0 in the order of the store: each
/// field at the canonical vertices, the element ids of the cells and the column of each cell
/// value. `None` on the other processes.
type Gathered = (Vec<Vec<f64>>, Vec<u64>, Vec<Vec<f64>>);

fn gathered(simulation: &Simulation) -> Option<Gathered> {
    let mesh = simulation.mesh();
    let mut fields = Vec::new();
    for field in 0..simulation.field_ids().len() {
        let values = simulation.field_values(field).unwrap();
        fields.push(mesh.gather_vertex_values(&values).unwrap());
    }
    let width = simulation.cell_value_names().len();
    let (ids, table) = simulation.cell_values().expect("a study with cells");
    let rows = mesh.gather_element_values(ids, &table, width).unwrap();
    let fields: Option<Vec<Vec<f64>>> = fields.into_iter().collect();
    let (ids, table) = rows?;
    let columns = (0..width)
        .map(|k| table.iter().skip(k).step_by(width).copied().collect())
        .collect();
    Some((fields?, ids, columns))
}

/// Runs `inputs` and writes its store into `out`, stopping after `outputs` output times; the
/// canonical mesh and the values of each output time written, gathered on process 0.
fn run(
    petsc: &Petsc,
    inputs: Inputs,
    out: &Path,
    outputs: usize,
) -> (Option<CanonicalMesh>, Vec<Option<Gathered>>) {
    let mut simulation = Simulation::new(petsc, inputs, Instant::now()).unwrap();
    let mut writer = ResultWriter::create(out, &simulation, &spt()).unwrap();
    let canonical = simulation.mesh().canonical().unwrap();
    let mut values = Vec::new();
    while let Some(index) = simulation.next_output().unwrap() {
        if index == outputs {
            break;
        }
        writer.write(&simulation, index).unwrap();
        values.push(gathered(&simulation));
    }
    (canonical, values)
}

/// Whether `a` and `b` are the same numbers, bitwise.
fn bitwise(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

/// The row `t` of a `[T, n]` array, row by row.
fn row(values: &[f64], t: usize, n: usize) -> &[f64] {
    &values[t * n..(t + 1) * n]
}

/// The store of a finished run holds the canonical mesh, the output times and every field and
/// cell value at every output time, bitwise as the simulation had them; its root attributes
/// name the study and the scene by their file names.
#[test]
fn result_round_trip() {
    c_cache();
    let petsc = Petsc::init(&[]).unwrap();
    let out = scratch("result_round_trip", None);
    let (canonical, values) = run(&petsc, inputs(false), &out, TIMES.len());
    let canonical = canonical.unwrap();
    let data = read_result(&out.join(RESULT_STORE)).unwrap();
    let meta = &data.meta;
    assert_eq!(meta.version, RESULT_VERSION);
    assert_eq!(meta.study, "spt.study.json");
    assert_eq!(meta.scene, "scene.json");
    assert_eq!(meta.refinements, 1);
    assert_eq!(meta.times, TIMES);
    assert_eq!((meta.written, meta.complete), (TIMES.len(), true));
    let ids: Vec<&str> = meta.fields.iter().map(|f| f.id.as_str()).collect();
    assert_eq!(ids, ["S", "P"]);
    assert!(meta.fields.iter().all(|f| f.units == "mM"));
    assert!(meta.cell_values.iter().any(|v| v == "protein"));
    assert!(meta.raster.is_none());
    let tags: Vec<(&str, u32)> = meta.tags.iter().map(|t| (t.name.as_str(), t.dim)).collect();
    let expected: Vec<(&str, u32)> = canonical
        .tags
        .iter()
        .filter(|t| t.dim > 0)
        .map(|t| (t.name.as_str(), u32::from(t.dim)))
        .collect();
    assert_eq!(tags, expected);
    assert!(!tags.is_empty());
    assert_eq!(data.time, TIMES);
    let mut canonical = canonical;
    canonical.tags.retain(|t| t.dim > 0);
    assert!(data.mesh == canonical, "the mesh is not the canonical mesh");
    let n = data.mesh.vertices.len();
    let c = data.cell_elements.len();
    for (t, gathered) in values.iter().enumerate() {
        let (fields, ids, columns) = gathered.as_ref().unwrap();
        for ((id, stored), field) in data.fields.iter().zip(fields) {
            assert!(bitwise(row(stored, t, n), field), "field {id} at {t}");
        }
        assert_eq!(&data.cell_elements, ids);
        for ((name, stored), column) in data.cells.iter().zip(columns) {
            assert!(
                bitwise(row(stored, t, c), column),
                "cell value {name} at {t}"
            );
        }
    }
    let names: Vec<&str> = data.cells.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(names, meta.cell_values);
}

/// The root attributes of the store are a valid `ResultMeta`: by the committed schema
/// `schemas/result-1.schema.json` and by `ResultMeta::check`, also before the first output.
#[test]
fn result_metadata_is_schema_valid() {
    c_cache();
    let petsc = Petsc::init(&[]).unwrap();
    let out = scratch("result_metadata_is_schema_valid", None);
    let schema = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../schemas/result-1.schema.json");
    let schema: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(schema).unwrap()).unwrap();
    let validator = jsonschema::validator_for(&schema).unwrap();
    let attributes = || {
        let text = std::fs::read_to_string(out.join(RESULT_STORE).join("zarr.json")).unwrap();
        let root: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(root["node_type"], "group");
        assert_eq!(root["zarr_format"], 3);
        root["attributes"].clone()
    };
    let mut simulation = Simulation::new(&petsc, inputs(false), Instant::now()).unwrap();
    let mut writer = ResultWriter::create(&out, &simulation, &spt()).unwrap();
    let mut written = 0;
    loop {
        let doc = attributes();
        assert!(validator.is_valid(&doc), "{doc}");
        let meta: ResultMeta = serde_json::from_value(doc).unwrap();
        meta.check().unwrap();
        assert_eq!(meta.written, written);
        let Some(index) = simulation.next_output().unwrap() else {
            break;
        };
        writer.write(&simulation, index).unwrap();
        written += 1;
    }
    assert!(read_result(&out.join(RESULT_STORE)).unwrap().meta.complete);
}

/// Each output time of a field and of a cell value is one chunk (`[1, V]` and `[1, C]`, zstd),
/// readable alone as soon as it is written: the later ones read as the fill value, NaN, until
/// they are written.
#[test]
fn one_output_time_reads_alone() {
    c_cache();
    let petsc = Petsc::init(&[]).unwrap();
    let out = scratch("one_output_time_reads_alone", None);
    let store = out.join(RESULT_STORE);
    let mut simulation = Simulation::new(&petsc, inputs(false), Instant::now()).unwrap();
    let mut writer = ResultWriter::create(&out, &simulation, &spt()).unwrap();
    let open = |path: &str| {
        let storage = Arc::new(FilesystemStore::new(&store).unwrap());
        Array::open(storage, path).unwrap()
    };
    let (v, c) = {
        let field = open("/fields/S");
        let cells = open("/cells/element");
        (field.shape()[1], cells.shape()[0])
    };
    assert!(v > 0 && c > 0);
    let t_all = TIMES.len() as u64;
    for path in ["/fields/S", "/fields/P", "/cells/protein"] {
        let array = open(path);
        let n = if path.starts_with("/fields") { v } else { c };
        assert_eq!(array.shape(), [t_all, n], "{path}");
        assert_eq!(array.chunk_grid_shape(), [t_all, 1], "{path}");
        let shape: Vec<u64> = array
            .chunk_shape(&[0, 0])
            .unwrap()
            .iter()
            .map(|s| s.get())
            .collect();
        assert_eq!(shape, [1, n], "{path}");
        let metadata = std::fs::read_to_string(store.join(&path[1..]).join("zarr.json")).unwrap();
        assert!(metadata.contains("\"zstd\""), "{path}: {metadata}");
    }
    while let Some(index) = simulation.next_output().unwrap() {
        writer.write(&simulation, index).unwrap();
        let (fields, _, columns) = gathered(&simulation).unwrap();
        let t = index as u64;
        for (path, expected) in [("/fields/S", &fields[0]), ("/fields/P", &fields[1])] {
            let chunk: Vec<f64> = open(path).retrieve_chunk(&[t, 0]).unwrap();
            assert!(bitwise(&chunk, expected), "{path} at {index}");
        }
        let names = simulation.cell_value_names();
        let k = names.iter().position(|n| n == "protein").unwrap();
        let chunk: Vec<f64> = open("/cells/protein").retrieve_chunk(&[t, 0]).unwrap();
        assert!(bitwise(&chunk, &columns[k]), "protein at {index}");
        if t + 1 < t_all {
            let later: Vec<f64> = open("/fields/S").retrieve_chunk(&[t + 1, 0]).unwrap();
            assert!(later.iter().all(|x| x.is_nan()), "fields/S at {}", t + 1);
        }
    }
}

/// A writer dropped after 2 of the 4 output times leaves a store marked unfinished: `written`
/// 2 and `complete` false, the 2 output times readable.
#[test]
fn unfinished_result_is_marked() {
    c_cache();
    let petsc = Petsc::init(&[]).unwrap();
    let out = scratch("unfinished_result_is_marked", None);
    let (_, values) = run(&petsc, inputs(false), &out, 2);
    assert_eq!(values.len(), 2);
    let data = read_result(&out.join(RESULT_STORE)).unwrap();
    assert_eq!((data.meta.written, data.meta.complete), (2, false));
    assert_eq!(data.meta.times, TIMES);
    let n = data.mesh.vertices.len();
    let (fields, _, _) = values[1].as_ref().unwrap();
    assert!(bitwise(row(&data.fields[0].1, 1, n), &fields[0]));
    assert!(row(&data.fields[0].1, 2, n).iter().all(|x| x.is_nan()));
}

/// The writer refuses an output time the simulation is not at, and an output time out of turn.
#[test]
fn writer_follows_the_simulation() {
    c_cache();
    let petsc = Petsc::init(&[]).unwrap();
    let out = scratch("writer_follows_the_simulation", None);
    let mut simulation = Simulation::new(&petsc, inputs(false), Instant::now()).unwrap();
    let mut writer = ResultWriter::create(&out, &simulation, &spt()).unwrap();
    assert_eq!(simulation.next_output().unwrap(), Some(0));
    let e = writer.write(&simulation, 1).unwrap_err();
    assert!(format!("{e:#}").contains("output time 1"), "{e:#}");
    assert!(e.downcast_ref::<OnEveryProcess>().is_some(), "{e:#}");
    writer.write(&simulation, 0).unwrap();
    let e = writer.write(&simulation, 0).unwrap_err();
    assert!(format!("{e:#}").contains("output time 0"), "{e:#}");
    assert_eq!(simulation.next_output().unwrap(), Some(1));
    writer.write(&simulation, 1).unwrap();
}

/// A run which writes its store has the same results as one which does not: the summary and
/// the fields at the last output time, bitwise.
#[test]
fn writing_does_not_change_the_run() {
    c_cache();
    let petsc = Petsc::init(&[]).unwrap();
    let out = scratch("writing_does_not_change_the_run", None);
    let mut plain = Simulation::new(&petsc, inputs(false), Instant::now()).unwrap();
    while plain.next_output().unwrap().is_some() {}
    let without: Vec<Vec<f64>> = (0..2).map(|f| plain.field_values(f).unwrap()).collect();
    let mut summary = serde_json::to_value(plain.summary()).unwrap();
    drop(plain);
    let mut simulation = Simulation::new(&petsc, inputs(false), Instant::now()).unwrap();
    let mut writer = ResultWriter::create(&out, &simulation, &spt()).unwrap();
    while let Some(index) = simulation.next_output().unwrap() {
        writer.write(&simulation, index).unwrap();
    }
    let with: Vec<Vec<f64>> = (0..2)
        .map(|f| simulation.field_values(f).unwrap())
        .collect();
    let mut written = serde_json::to_value(simulation.summary()).unwrap();
    for s in [&mut summary, &mut written] {
        let s = s.as_object_mut().unwrap();
        s.remove("phases");
        s["cells"].as_object_mut().unwrap().remove("seconds");
    }
    assert_eq!(summary, written);
    for (a, b) in without.iter().zip(&with) {
        assert!(bitwise(a, b));
    }
}

/// The store of a run on 2 and 3 processes is that of one process: the mesh arrays and the
/// cell element ids bitwise, the fields and the cell values within 1e-10 relative to the
/// largest value (at the tight settings of [`inputs`]).
#[test]
fn result_is_partition_independent() {
    c_cache();
    match std::env::var(REFERENCE) {
        Err(_) => {
            let dir = scratch("result_is_partition_independent", None);
            {
                let petsc = Petsc::init(&[]).unwrap();
                run(&petsc, inputs(true), &dir.join("1"), TIMES.len());
                // MPI is finalized before mpiexec starts.
            }
            for ranks in [2, 3] {
                test_support::parallel_with_env(
                    ranks,
                    &["result_is_partition_independent"],
                    &[(REFERENCE, dir.to_str().unwrap())],
                );
            }
        }
        Ok(dir) => {
            let petsc = Petsc::init(&[]).unwrap();
            let ranks = petsc.comm().size();
            let out = Path::new(&dir).join(ranks.to_string());
            run(&petsc, inputs(true), &out, TIMES.len());
            if petsc.comm().rank() == 0 {
                let one = read_result(&Path::new(&dir).join("1").join(RESULT_STORE)).unwrap();
                let many = read_result(&out.join(RESULT_STORE)).unwrap();
                compare(&one, &many, ranks);
            }
        }
    }
}

/// [`result_is_partition_independent`]: `many` of `ranks` processes against `one`.
fn compare(one: &ResultData, many: &ResultData, ranks: usize) {
    assert_eq!(one.meta, many.meta);
    assert_eq!(one.time, many.time);
    assert!(one.mesh == many.mesh, "the mesh of {ranks} processes");
    assert_eq!(one.cell_elements, many.cell_elements);
    let relative = |a: &[f64], b: &[f64]| {
        assert_eq!(a.len(), b.len());
        let scale = a.iter().chain(b).fold(0.0_f64, |m, x| m.max(x.abs()));
        let largest = a
            .iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f64::max);
        if scale > 0.0 {
            largest / scale
        } else {
            largest
        }
    };
    let mut differences = Vec::new();
    for ((id, a), (_, b)) in one.fields.iter().zip(&many.fields) {
        let d = relative(a, b);
        assert!(d < 1e-10, "field {id} on {ranks} processes: {d:e}");
        differences.push(format!("{id} {d:.1e}"));
    }
    for ((name, a), (_, b)) in one.cells.iter().zip(&many.cells) {
        let d = relative(a, b);
        assert!(d < 1e-10, "cell value {name} on {ranks} processes: {d:e}");
        differences.push(format!("{name} {d:.1e}"));
    }
    eprintln!(
        "result on {ranks} processes against 1, relative differences: {}",
        differences.join(", ")
    );
}
