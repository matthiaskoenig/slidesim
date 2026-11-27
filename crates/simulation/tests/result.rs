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
use contracts::{RasterOptions, ResultOptions};
use mesh::CanonicalMesh;
use petsc::Petsc;
use simulation::raster::Raster;
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
        // The partition independence includes the raster.
        inputs.study.result = raster_options(3.0, 2);
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
    let mut simulation = Simulation::new(petsc, inputs, Instant::now(), &mut |_| {}).unwrap();
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
    let mut simulation =
        Simulation::new(&petsc, inputs(false), Instant::now(), &mut |_| {}).unwrap();
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
    let mut simulation =
        Simulation::new(&petsc, inputs(false), Instant::now(), &mut |_| {}).unwrap();
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
    let mut simulation =
        Simulation::new(&petsc, inputs(false), Instant::now(), &mut |_| {}).unwrap();
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
    let mut plain = Simulation::new(&petsc, inputs(false), Instant::now(), &mut |_| {}).unwrap();
    while plain.next_output().unwrap().is_some() {}
    let without: Vec<Vec<f64>> = (0..2).map(|f| plain.field_values(f).unwrap()).collect();
    let mut summary = serde_json::to_value(plain.summary()).unwrap();
    drop(plain);
    let mut simulation =
        Simulation::new(&petsc, inputs(false), Instant::now(), &mut |_| {}).unwrap();
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
    assert_eq!(one.rasters.len(), one.fields.len());
    for ((id, a), (_, b)) in one.rasters.iter().zip(&many.rasters) {
        for (k, (a, b)) in a.iter().zip(b).enumerate() {
            assert_eq!(a.len(), b.len());
            let same = a.iter().zip(b).all(|(x, y)| {
                (x.is_nan() && y.is_nan()) || (x - y).abs() <= 1e-6 * x.abs().max(1e-3)
            });
            assert!(same, "raster of {id} level {k} on {ranks} processes");
        }
    }
    eprintln!(
        "result on {ranks} processes against 1, relative differences: {}",
        differences.join(", ")
    );
}

/// An L shaped mesh of three squares of side 2 um ([0, 2] x [0, 2], [2, 4] x [0, 2] and
/// [0, 2] x [2, 4]; the square [2, 4] x [2, 4] is missing), each split into two triangles,
/// counterclockwise.
fn l_mesh() -> CanonicalMesh {
    let mut vertices = Vec::new();
    let mut index = |x: f64, y: f64| {
        let k = vertices
            .iter()
            .position(|v| *v == [x, y])
            .unwrap_or_else(|| {
                vertices.push([x, y]);
                vertices.len() - 1
            });
        k as u32
    };
    let mut triangles = Vec::new();
    for (x, y) in [(0.0, 0.0), (2.0, 0.0), (0.0, 2.0)] {
        let a = index(x, y);
        let b = index(x + 2.0, y);
        let c = index(x + 2.0, y + 2.0);
        let d = index(x, y + 2.0);
        triangles.push([a, b, c]);
        triangles.push([a, c, d]);
    }
    CanonicalMesh {
        vertices,
        element: vec![0; triangles.len()],
        triangles,
        tags: Vec::new(),
    }
}

/// `x + 2 y` at the vertices.
fn linear(mesh: &CanonicalMesh) -> Vec<f64> {
    mesh.vertices.iter().map(|v| v[0] + 2.0 * v[1]).collect()
}

/// A level 0 pixel of a linear field is the field at its centre (within f32 rounding), NaN
/// outside the mesh; row 0 is at the smallest y; the shapes follow the bounding box.
#[test]
fn raster_of_a_linear_field_is_exact() {
    let mesh = l_mesh();
    let raster = Raster::new(&mesh, 0.5, None, 3).unwrap();
    let meta = raster.meta();
    assert_eq!(
        (meta.um_per_px, meta.origin_um, meta.levels),
        (0.5, [0.0, 0.0], 3)
    );
    assert_eq!(meta.shapes, [[8, 8], [4, 4], [2, 2]]);
    let level0 = &raster.pyramid(&linear(&mesh))[0];
    let mut inside = 0;
    for r in 0..8 {
        for c in 0..8 {
            let (x, y) = ((c as f64 + 0.5) * 0.5, (r as f64 + 0.5) * 0.5);
            let v = level0[r * 8 + c];
            if x > 2.0 && y > 2.0 {
                assert!(v.is_nan(), "({r}, {c}) is outside: {v}");
            } else {
                inside += 1;
                assert!(
                    (f64::from(v) - (x + 2.0 * y)).abs() < 1e-6,
                    "({r}, {c}): {v}"
                );
            }
        }
    }
    assert_eq!(inside, 48);
    // An extent that is no multiple of the pixel: ceil, and a pixel size beyond the mesh is one
    // pixel.
    assert_eq!(
        Raster::new(&mesh, 0.7, None, 1).unwrap().meta().shapes,
        [[6, 6]]
    );
    assert_eq!(
        Raster::new(&mesh, 100.0, None, 3).unwrap().meta().shapes,
        [[1, 1]; 3]
    );
    // The origin given: the pixels start there, pixel centres at origin + (c + 0.5) h.
    let shifted = Raster::new(&mesh, 0.5, Some([1.0, -1.0]), 1).unwrap();
    assert_eq!(shifted.meta().origin_um, [1.0, -1.0]);
    assert_eq!(shifted.meta().shapes, [[10, 6]]);
    let level0 = &shifted.pyramid(&linear(&mesh))[0];
    for r in 0..10 {
        for c in 0..6 {
            let (x, y) = (1.0 + (c as f64 + 0.5) * 0.5, -1.0 + (r as f64 + 0.5) * 0.5);
            let v = level0[r * 6 + c];
            if y < 0.0 || (x > 2.0 && y > 2.0) {
                assert!(v.is_nan(), "({r}, {c}): {v}");
            } else {
                assert!(
                    (f64::from(v) - (x + 2.0 * y)).abs() < 1e-6,
                    "({r}, {c}): {v}"
                );
            }
        }
    }
}

/// Level k + 1 is the mean of the valid pixels of 2 x 2 of level k (the edge blocks of an odd
/// shape the pixels they have, a block without a valid pixel is NaN).
#[test]
fn pyramid_levels_are_means() {
    let mesh = l_mesh();
    let raster = Raster::new(&mesh, 0.6, None, 4).unwrap();
    let shapes = raster.meta().shapes;
    assert_eq!(shapes, [[7, 7], [4, 4], [2, 2], [1, 1]]);
    let levels = raster.pyramid(&linear(&mesh));
    assert_eq!(levels.len(), 4);
    let mut nan_blocks = 0;
    let mut partial_blocks = 0;
    for k in 0..3 {
        let [rows, cols] = shapes[k].map(|n| n as usize);
        let [next_rows, next_cols] = shapes[k + 1].map(|n| n as usize);
        assert_eq!(levels[k].len(), rows * cols);
        assert_eq!(levels[k + 1].len(), next_rows * next_cols);
        for r in 0..next_rows {
            for c in 0..next_cols {
                let mut values = Vec::new();
                for (dr, dc) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                    let (rr, cc) = (2 * r + dr, 2 * c + dc);
                    if rr < rows && cc < cols && !levels[k][rr * cols + cc].is_nan() {
                        values.push(f64::from(levels[k][rr * cols + cc]));
                    }
                }
                let got = levels[k + 1][r * next_cols + c];
                if values.is_empty() {
                    nan_blocks += 1;
                    assert!(got.is_nan(), "level {} ({r}, {c}): {got}", k + 1);
                } else {
                    partial_blocks += usize::from(values.len() < 4);
                    let mean = values.iter().sum::<f64>() / values.len() as f64;
                    assert!(
                        (f64::from(got) - mean).abs() < 1e-6,
                        "level {} ({r}, {c}): {got} against {mean}",
                        k + 1
                    );
                }
            }
        }
    }
    assert!(nan_blocks > 0 && partial_blocks > 0);
}

fn raster_options(um_per_px: f64, levels: u32) -> Option<ResultOptions> {
    Some(ResultOptions {
        raster: Some(RasterOptions {
            um_per_px,
            origin_um: None,
            levels,
        }),
    })
}

/// A study without `result.raster` writes no `raster` group and no `raster` attribute.
#[test]
fn no_raster_without_the_option() {
    c_cache();
    let petsc = Petsc::init(&[]).unwrap();
    let out = scratch("no_raster_without_the_option", None);
    run(&petsc, inputs(false), &out, 2);
    let store = out.join(RESULT_STORE);
    assert!(!store.join("raster").exists());
    let data = read_result(&store).unwrap();
    assert!(data.meta.raster.is_none() && data.rasters.is_empty());
    let text = std::fs::read_to_string(store.join("zarr.json")).unwrap();
    assert!(!text.contains("raster"), "{text}");
}

/// With `result.raster` every field has its pyramid in the store, an output time a chunk of each
/// 256 x 256 block, written before `written` is raised; the arrays are those of
/// [`Raster::pyramid`] of the fields of the store, and the attributes validate.
#[test]
fn store_holds_the_raster_pyramid() {
    c_cache();
    let petsc = Petsc::init(&[]).unwrap();
    let out = scratch("store_holds_the_raster_pyramid", None);
    let mut inputs = inputs(false);
    inputs.study.result = raster_options(3.0, 3);
    run(&petsc, inputs, &out, 3);
    let store = out.join(RESULT_STORE);
    let data = read_result(&store).unwrap();
    let meta = data.meta.raster.clone().expect("raster attributes");
    assert_eq!(meta.um_per_px, 3.0);
    assert_eq!((meta.levels, meta.shapes.len()), (3, 3));
    let [rows, cols] = meta.shapes[0];
    assert!(
        rows > 256 || cols > 256 || rows * cols > 1000,
        "{rows} x {cols}"
    );
    let raster = Raster::new(&data.mesh, 3.0, None, 3).unwrap();
    assert_eq!(raster.meta(), meta);
    assert_eq!(data.rasters.len(), data.fields.len());
    let storage = Arc::new(FilesystemStore::new(&store).unwrap());
    for ((id, nodal), (rid, levels)) in data.fields.iter().zip(&data.rasters) {
        assert_eq!(id, rid);
        assert_eq!(levels.len(), 3);
        let v = data.mesh.vertices.len();
        for t in 0..data.meta.written {
            let expected = raster.pyramid(&nodal[t * v..(t + 1) * v]);
            for (k, level) in levels.iter().enumerate() {
                let [r, c] = meta.shapes[k].map(|n| n as usize);
                let got = &level[t * r * c..(t + 1) * r * c];
                assert!(
                    got.iter()
                        .zip(&expected[k])
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "field {id} level {k} time {t}"
                );
            }
        }
        let array = Array::open(storage.clone(), &format!("/raster/{id}/0")).unwrap();
        assert_eq!(array.shape(), [4, rows, cols]);
        assert_eq!(array.chunk_shape(&[0, 0, 0]).unwrap()[0].get(), 1);
        assert_eq!(
            array.chunk_grid_shape(),
            [4, rows.div_ceil(256), cols.div_ceil(256)]
        );
    }
    // The time not reached is NaN everywhere, the others have values inside the mesh.
    let first = &data.rasters[0].1[0];
    let [r, c] = meta.shapes[0].map(|n| n as usize);
    assert!(first[..r * c].iter().any(|v| v.is_finite()));
    assert!(first[3 * r * c..].iter().all(|v| v.is_nan()));
}

/// A raster whose level 0 would have more than 2^28 pixels is refused by name of the option,
/// before anything is written to the store.
#[test]
fn huge_raster_is_refused() {
    c_cache();
    let petsc = Petsc::init(&[]).unwrap();
    let out = scratch("huge_raster_is_refused", None);
    let mut inputs = inputs(false);
    inputs.study.result = raster_options(1e-3, 1);
    let simulation = Simulation::new(&petsc, inputs, Instant::now(), &mut |_| {}).unwrap();
    let e = ResultWriter::create(&out, &simulation, &spt())
        .err()
        .expect("refused");
    assert!(
        format!("{e:#}").contains("result.raster.um_per_px"),
        "{e:#}"
    );
}
