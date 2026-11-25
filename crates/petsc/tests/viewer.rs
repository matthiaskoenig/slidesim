//! Tests of the VTK viewer: a P1 field of a DMPlex written as a `.vtu` file, on one process and
//! (by `parallel_on_2_processes`) on two, where process 0 writes the pieces of every process into
//! one file. nextest runs each test in its own process, so every test initializes PETSc once.

use std::path::{Path, PathBuf};

use petsc::{Dm, Fe, Petsc, Vector, Viewer};

/// The gmsh square [0, 1] x [0, 1] of `testdata/meshes` (31 vertices), distributed, with the
/// P1 field, affine coordinates and an essential boundary on one side (`Face Sets` value 11).
fn p1_square(p: &Petsc) -> Dm {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/meshes/unit_square.msh");
    let mut mesh = Dm::plex_from_file(&p.comm(), &path, "square", true).unwrap();
    if let Some(distributed) = mesh.distribute(0).unwrap() {
        mesh = distributed;
    }
    let mut dm = mesh.clone_mesh().unwrap();
    dm.set_coordinate_disc(None, false).unwrap();
    dm.add_field(&Fe::lagrange(2, 1, true, 1, 2).unwrap())
        .unwrap();
    dm.create_ds().unwrap();
    dm.add_essential_boundary("wall", "Face Sets", &[11], 0)
        .unwrap();
    dm
}

/// The output file of `test` on `ranks` processes, the same on every process.
fn output(test: &str, ranks: usize) -> PathBuf {
    Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("{test}-{ranks}.vtu"))
}

/// The values of the attribute `name` in `text`, as numbers.
fn attribute_values(text: &str, name: &str) -> Vec<usize> {
    let key = format!("{name}=\"");
    text.match_indices(&key)
        .map(|(i, _)| {
            let rest = &text[i + key.len()..];
            rest[..rest.find('"').unwrap()].parse().unwrap()
        })
        .collect()
}

#[test]
fn vtk_viewer_writes_a_named_p1_field() {
    let p = Petsc::init(&[]).unwrap();
    let comm = p.comm();
    let mut dm = p1_square(&p);
    // PETSc names the field after its element, and the component by number.
    let (field, components) = dm.field_names(0).unwrap();
    assert_eq!(
        (field.as_str(), components.as_slice()),
        ("P1", ["0".to_string()].as_slice())
    );
    dm.set_field_names(0, "oxygen", &["mM"]).unwrap();
    assert_eq!(
        dm.field_names(0).unwrap(),
        ("oxygen".to_string(), vec!["mM".to_string()])
    );
    let path = output("vtk_viewer_writes_a_named_p1_field", comm.size());
    if comm.rank() == 0 {
        // An existing file is overwritten.
        std::fs::write(&path, "old").unwrap();
    }

    // Every entry of the local vector, the constrained ones included, is 2.5.
    let mut u = Vector::create_local(&dm).unwrap();
    u.set(2.5).unwrap();
    u.set_name("").unwrap();
    assert_eq!(u.name().unwrap(), "");
    let mut viewer = Viewer::vtk(&comm, &path).unwrap();
    u.view(&mut viewer).unwrap();
    viewer.flush().unwrap();
    drop(viewer);

    if comm.rank() == 0 {
        let bytes = std::fs::read(&path).unwrap();
        let text = String::from_utf8_lossy(&bytes);
        assert!(
            text.contains("<VTKFile type=\"UnstructuredGrid\""),
            "{text}"
        );
        let points = attribute_values(&text, "NumberOfPoints");
        assert_eq!(points.len(), comm.size(), "one piece per process");
        let points: usize = points.iter().sum();
        if comm.size() == 1 {
            assert_eq!(points, 31);
        }
        // The point data is named `<vector><field>.<component>`.
        let point_data = point_data_names(&text);
        assert_eq!(point_data, ["oxygen.mM"], "{text}");
        // In double precision, every point has the value, the constrained ones included.
        let value = 2.5_f64.to_le_bytes();
        let count = bytes.windows(8).filter(|w| *w == value).count();
        assert_eq!(count, points);
    }
}

#[test]
fn field_names_of_a_missing_field_or_component_are_error() {
    let p = Petsc::init(&[]).unwrap();
    let mut dm = p1_square(&p);
    let e = dm.set_field_names(1, "v", &["0"]).unwrap_err();
    assert!(e.message.contains("Invalid field number 1"), "{e}");
    let e = dm.set_field_names(0, "u", &["x", "y"]).unwrap_err();
    assert!(e.message.contains("2 component names"), "{e}");
    assert!(dm.field_names(1).is_err());
}

#[test]
fn vector_names() {
    let p = Petsc::init(&[]).unwrap();
    let mut u = Vector::create_seq(&p, 2).unwrap();
    // PETSc names an unnamed vector.
    assert!(u.name().unwrap().starts_with("Vec_"));
    u.set_name("u").unwrap();
    assert_eq!(u.name().unwrap(), "u");
    assert!(u.set_name("u\0v").unwrap_err().message.contains("NUL"));
}

#[test]
fn vtk_viewer_on_a_path_with_nul_is_error() {
    let p = Petsc::init(&[]).unwrap();
    let e = Viewer::vtk(&p.comm(), Path::new("out\0.vtu")).unwrap_err();
    assert!(e.message.contains("NUL"), "{e}");
}

#[test]
fn vtk_viewer_into_a_missing_directory_fails_to_flush() {
    let p = Petsc::init(&[]).unwrap();
    let dm = p1_square(&p);
    let path = Path::new(env!("CARGO_TARGET_TMPDIR")).join("no/such/dir/u.vtu");
    let mut u = Vector::create_local(&dm).unwrap();
    u.set(1.0).unwrap();
    let mut viewer = Viewer::vtk(&p.comm(), &path).unwrap();
    u.view(&mut viewer).unwrap();
    let e = viewer.flush().unwrap_err();
    assert!(e.message.contains("u.vtu"), "{e}");
}

/// The names of the arrays of the `<PointData>` element of the VTU text `text`.
fn point_data_names(text: &str) -> Vec<String> {
    let start = text.find("<PointData>").expect("point data");
    let end = start
        + text[start..]
            .find("</PointData>")
            .expect("end of point data");
    let key = "Name=\"";
    text[start..end]
        .match_indices(key)
        .map(|(i, _)| {
            let rest = &text[start + i + key.len()..end];
            rest[..rest.find('"').unwrap()].to_string()
        })
        .collect()
}

/// The tests `parallel_on_*_processes` run on several processes: one file of the pieces of all.
const PARALLEL_TESTS: [&str; 2] = [
    "mpi_world_has_the_expected_size",
    "vtk_viewer_writes_a_named_p1_field",
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
