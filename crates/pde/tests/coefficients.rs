//! Tests of coefficients at the quadrature points of a discretization: tables over maps, and map
//! errors which name the map, the point and the area tag, on every process. nextest runs each
//! test in its own process, so every test initializes PETSc once. `parallel_on_2_processes` and
//! `parallel_on_3_processes` run some of them again under `mpiexec`.
//!
//! The maps are test Zarr arrays over `unit_square.msh` with pixels of 0.01 um, the centre of
//! pixel (row 0, col 0) at (-0.05, -0.05): 111 pixels cover [-0.05, 1.05] in each direction.
//! Every process writes its own copy into a directory of its own.

use std::path::{Path, PathBuf};

use ceed::{Ceed, DEFAULT_RESOURCE};
use contracts::{
    Coefficient, LengthUnit, MapSpec, MeshFormat, MeshRef, SCENE_VERSION, Scene, Table, Units,
};
use maps::Maps;
use mesh::Mesh;
use pde::{Discretization, PdeError, coefficient_values};
use petsc::{Comm, Petsc};

/// The pixel size of the maps, in um.
const PIXEL: f64 = 0.01;
/// The centre of pixel (row 0, col 0), in x and in y.
const ORIGIN: f64 = -0.05;
/// The pixels which cover [-0.05, 1.05].
const PIXELS: usize = 111;

fn fixture(file: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../testdata/meshes")
        .join(file)
}

fn unit_square(p: &Petsc, refinements: u32) -> Mesh {
    let scene = Scene {
        version: SCENE_VERSION.to_string(),
        dimension: 2,
        units: Units {
            length: LengthUnit::Um,
        },
        mesh: MeshRef {
            format: MeshFormat::Gmsh,
            path: fixture("unit_square.msh"),
        },
        maps: Vec::new(),
    };
    Mesh::load(p, &scene, refinements).unwrap()
}

fn ceed() -> Ceed {
    Ceed::new(DEFAULT_RESOURCE).unwrap()
}

/// A directory of this process for the maps of a test, removed when dropped.
struct Scratch(PathBuf);

impl Scratch {
    fn new(test: &str) -> Scratch {
        let dir = std::env::temp_dir().join(format!("slidesim-pde-{test}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        Scratch(dir)
    }

    /// The map `id` of `f` at the pixel centres, `PIXELS` rows (y) and `cols` columns (x) from
    /// the origin.
    fn map(&self, id: &str, cols: usize, f: impl Fn([f64; 2]) -> f64) -> MapSpec {
        self.map_of_shape(id, PIXELS, cols, f)
    }

    /// The map `id` of `f` at the pixel centres, `rows` rows (y) and `cols` columns (x) from
    /// the origin.
    fn map_of_shape(
        &self,
        id: &str,
        rows: usize,
        cols: usize,
        f: impl Fn([f64; 2]) -> f64,
    ) -> MapSpec {
        let values: Vec<f64> = (0..rows)
            .flat_map(|r| (0..cols).map(move |c| (r, c)))
            .map(|(r, c)| f([ORIGIN + c as f64 * PIXEL, ORIGIN + r as f64 * PIXEL]))
            .collect();
        let path = self.0.join(format!("{id}.zarr"));
        let mut spec = maps::write_test_map(&path, &values, rows, cols, PIXEL, [ORIGIN, ORIGIN]);
        spec.id = id.to_string();
        spec
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// The maps `specs`, read for the cells of this process.
fn read(specs: &[MapSpec], mesh: &Mesh) -> Maps {
    let ids: Vec<&str> = specs.iter().map(|s| s.id.as_str()).collect();
    Maps::read(
        specs,
        &ids,
        mesh.local_bounds().expect("cells on every process"),
    )
    .unwrap()
}

fn table(map: &str, points: &[[f64; 2]]) -> Coefficient {
    Coefficient::Table(Table {
        map: map.to_string(),
        points: points.to_vec(),
    })
}

#[test]
fn table_over_map_at_points() {
    let p = Petsc::init(&[]).unwrap();
    let mesh = unit_square(&p, 1);
    let disc = Discretization::new(&mesh, 1, &ceed(), 1).unwrap();
    let scratch = Scratch::new("table_over_map_at_points");
    let maps = read(
        &[scratch.map("position", PIXELS, |[x, y]| x + 2.0 * y)],
        &mesh,
    );
    let c = table("position", &[[0.0, 1.0], [3.0, 4.0]]);
    let values = coefficient_values(&disc, &c, &maps).unwrap();
    assert_eq!(values.len(), disc.num_quadrature_points());
    assert!(!values.is_empty());
    let map = maps.get("position").unwrap();
    for (&v, &x) in values.iter().zip(disc.quadrature_points()) {
        assert_eq!(v, c.eval(map.sample(x).unwrap()), "at {x:?}");
        // x + 2 y in [0, 3]: the table is 1 + x + 2 y there.
        assert!(
            (v - (1.0 + x[0] + 2.0 * x[1])).abs() < 1e-12,
            "at {x:?}: {v}"
        );
    }
    // A constant needs no map.
    let constant = coefficient_values(&disc, &Coefficient::Constant(2.5), &Maps::default());
    let constant = constant.unwrap();
    assert_eq!(constant.len(), disc.num_quadrature_points());
    assert!(constant.iter().all(|&v| v == 2.5));
}

#[test]
fn values_beyond_table_hold_end_values() {
    let p = Petsc::init(&[]).unwrap();
    let mesh = unit_square(&p, 1);
    let disc = Discretization::new(&mesh, 1, &ceed(), 1).unwrap();
    let scratch = Scratch::new("values_beyond_table_hold_end_values");
    let maps = read(
        &[scratch.map("position", PIXELS, |[x, y]| x + 2.0 * y)],
        &mesh,
    );
    // x + 2 y runs over [0, 3]: below 0.5 and above 1 the end values hold.
    let c = table("position", &[[0.5, 1.0], [1.0, 2.0]]);
    let values = coefficient_values(&disc, &c, &maps).unwrap();
    let (mut below, mut above) = (0, 0);
    for (&v, &[x, y]) in values.iter().zip(disc.quadrature_points()) {
        let s = x + 2.0 * y;
        if s < 0.5 - 1e-9 {
            assert_eq!(v, 1.0, "at ({x}, {y})");
            below += 1;
        } else if s > 1.0 + 1e-9 {
            assert_eq!(v, 2.0, "at ({x}, {y})");
            above += 1;
        } else {
            assert!(
                (v - (1.0 + 2.0 * (s - 0.5))).abs() < 1e-12,
                "at ({x}, {y}): {v}"
            );
        }
    }
    let comm = p.comm();
    assert!(comm.sum_int(below).unwrap() > 0);
    assert!(comm.sum_int(above).unwrap() > 0);
}

/// The columns of a map over x in [-0.05, 0.5]: the left half of the unit square.
const LEFT_HALF: usize = 56;

/// The point a map error names: `(x, y)` after "point ".
fn named_point(message: &str) -> [f64; 2] {
    let start = message.find("point (").expect("a point") + "point (".len();
    let end = start + message[start..].find(')').unwrap();
    let xy: Vec<f64> = message[start..end]
        .split(", ")
        .map(|v| v.parse().unwrap())
        .collect();
    [xy[0], xy[1]]
}

#[test]
fn map_error_names_map_point_and_tag() {
    let p = Petsc::init(&[]).unwrap();
    let mesh = unit_square(&p, 0);
    let disc = Discretization::new(&mesh, 0, &ceed(), 1).unwrap();
    let scratch = Scratch::new("map_error_names_map_point_and_tag");
    let maps = read(&[scratch.map("position", LEFT_HALF, |[x, _]| x)], &mesh);
    let e = coefficient_values(&disc, &table("position", &[[0.0, 1.0], [1.0, 2.0]]), &maps)
        .unwrap_err();
    let PdeError::Map { message } = &e else {
        panic!("expected a map error, got {e}");
    };
    println!("{message}");
    assert!(message.contains("'position'"), "{message}");
    assert!(message.contains("outside"), "{message}");
    assert!(message.contains("area tag 'domain'"), "{message}");
    // A point of the right half, where the map has no data.
    let [x, y] = named_point(message);
    assert!(
        x > 0.5 && (0.0..=1.0).contains(&x) && (0.0..=1.0).contains(&y),
        "{message}"
    );
    // And how many points fail: those right of x = 0.5, at least one per cell there.
    assert!(message.contains("quadrature points"), "{message}");
    assert_eq!(e.to_string(), *message);
}

#[test]
fn no_value_pixel_is_error() {
    let p = Petsc::init(&[]).unwrap();
    let mesh = unit_square(&p, 0);
    let disc = Discretization::new(&mesh, 0, &ceed(), 1).unwrap();
    let scratch = Scratch::new("no_value_pixel_is_error");
    // NaN right of x = 0.5.
    let spec = scratch.map(
        "position",
        PIXELS,
        |[x, _]| if x > 0.5 { f64::NAN } else { x },
    );
    let maps = read(&[spec], &mesh);
    let e = coefficient_values(&disc, &table("position", &[[0.0, 1.0], [1.0, 2.0]]), &maps)
        .unwrap_err();
    let PdeError::Map { message } = &e else {
        panic!("expected a map error, got {e}");
    };
    println!("{message}");
    assert!(message.contains("'position'"), "{message}");
    assert!(message.contains("no-value"), "{message}");
    assert!(message.contains("area tag 'domain'"), "{message}");
    let [x, _] = named_point(message);
    assert!(x > 0.49, "{message}");
}

/// The rows and columns of a map from the origin which covers the points of `disc` on process 0
/// and not all of the others: the left half of the unit square on one process; on several, the
/// unit square up to the largest x (or else y) of the points of process 0, if that is less than
/// 0.9. Collective.
fn shape_of_process_0(comm: &Comm, disc: &Discretization) -> [usize; 2] {
    if comm.size() == 1 {
        return [PIXELS, LEFT_HALF];
    }
    // The largest coordinates of the points of process 0, on every process.
    let largest = |axis: usize| {
        let local = disc
            .quadrature_points()
            .iter()
            .map(|x| x[axis])
            .fold(f64::NEG_INFINITY, f64::max);
        comm.sum_real(if comm.rank() == 0 { local } else { 0.0 })
            .unwrap()
    };
    // The pixels from the origin to the first centre at or beyond `end`.
    let pixels = |end: f64| ((end - ORIGIN) / PIXEL).ceil() as usize + 1;
    let [x, y] = [largest(0), largest(1)];
    if x < 0.9 {
        [PIXELS, pixels(x)]
    } else if y < 0.9 {
        [pixels(y), PIXELS]
    } else {
        panic!("the points of process 0 reach ({x}, {y})")
    }
}

/// The number of points of `disc` outside a map of `rows` and `cols` from the origin.
fn points_outside(disc: &Discretization, [rows, cols]: [usize; 2]) -> i32 {
    let end = |n: usize| ORIGIN + (n - 1) as f64 * PIXEL;
    disc.quadrature_points()
        .iter()
        .filter(|&&[x, y]| x > end(cols) || y > end(rows))
        .count() as i32
}

/// `x` is the same on every process of `comm`.
fn same_everywhere(comm: &Comm, x: i32) -> bool {
    comm.max_int(x).unwrap() == x && -comm.max_int(-x).unwrap() == x
}

#[test]
fn map_error_on_one_process_is_error_on_all() {
    let p = Petsc::init(&[]).unwrap();
    let comm = p.comm();
    let mesh = unit_square(&p, 1);
    let disc = Discretization::new(&mesh, 1, &ceed(), 1).unwrap();
    let scratch = Scratch::new("map_error_on_one_process_is_error_on_all");
    let [rows, cols] = shape_of_process_0(&comm, &disc);
    let maps = read(
        &[scratch.map_of_shape("position", rows, cols, |[x, _]| x)],
        &mesh,
    );
    let failing = points_outside(&disc, [rows, cols]);
    println!(
        "process {} of {}: {failing} of {} points outside the map of {rows} x {cols} pixels",
        comm.rank(),
        comm.size(),
        disc.num_quadrature_points()
    );
    // Some processes have points outside the map, and process 0 has none: it learns of the
    // error from the others.
    assert!(comm.sum_int(failing).unwrap() > 0);
    if comm.size() > 1 && comm.rank() == 0 {
        assert_eq!(failing, 0);
    }
    let e = coefficient_values(&disc, &table("position", &[[0.0, 1.0], [1.0, 2.0]]), &maps)
        .unwrap_err();
    let PdeError::Map { message } = &e else {
        panic!("expected a map error, got {e}");
    };
    println!("{message}");
    assert!(message.contains("'position'"), "{message}");
    assert!(message.contains("area tag 'domain'"), "{message}");
    // The same message on every process, with the count of all of them.
    assert!(same_everywhere(&comm, message.len() as i32), "{message}");
    let checksum = message.bytes().map(i32::from).sum::<i32>();
    assert!(same_everywhere(&comm, checksum), "{message}");
    let total = comm.sum_int(failing).unwrap();
    assert!(
        message.contains(&format!("{total} quadrature points")),
        "{message}"
    );
}

/// The tests `parallel_on_*_processes` run on several processes: tables at the points of each
/// process, and map errors found on some processes only.
const PARALLEL_TESTS: [&str; 6] = [
    "mpi_world_has_the_expected_size",
    "table_over_map_at_points",
    "values_beyond_table_hold_end_values",
    "map_error_names_map_point_and_tag",
    "no_value_pixel_is_error",
    "map_error_on_one_process_is_error_on_all",
];

#[test]
fn mpi_world_has_the_expected_size() {
    let p = Petsc::init(&[]).unwrap();
    assert_eq!(p.comm().size(), test_support::ranks());
}

#[test]
fn parallel_on_2_processes() {
    test_support::parallel(2, &PARALLEL_TESTS);
}

#[test]
fn parallel_on_3_processes() {
    test_support::parallel(3, &PARALLEL_TESTS);
}
