//! Tests of the metric adaptation of a DMPlex with Mmg (`DMAdaptMetric`). nextest runs each test
//! in its own process, so every test initializes PETSc once.

use petsc::sys::PetscInt;
use petsc::{Dm, MetricOptions, Petsc, Point};

/// The unit square of two triangles, with the diagonal from (0, 0) to (1, 1).
fn unit_square(p: &Petsc) -> Dm {
    let cells = [0, 1, 2, 0, 2, 3];
    let coordinates = [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0];
    Dm::plex_from_cells(&p.comm_self(), "square", 2, 3, &cells, 2, &coordinates).unwrap()
}

/// The coordinates of the vertices of `dm`, in the order of the depth-0 stratum.
fn vertex_coordinates(dm: &Dm) -> Vec<[f64; 2]> {
    let xy = dm.local_coordinates().unwrap();
    assert_eq!(xy.len(), 2 * dm.depth_stratum(0).unwrap().len());
    xy.chunks(2).map(|c| [c[0], c[1]]).collect()
}

/// The end points of the edge `edge` of `dm`.
fn edge_ends(dm: &Dm, xy: &[[f64; 2]], edge: Point) -> [[f64; 2]; 2] {
    let v0 = dm.depth_stratum(0).unwrap().start;
    let cone = dm.cone(edge).unwrap();
    assert_eq!(cone.len(), 2);
    [xy[(cone[0] - v0) as usize], xy[(cone[1] - v0) as usize]]
}

fn length([a, b]: [[f64; 2]; 2]) -> f64 {
    (b[0] - a[0]).hypot(b[1] - a[1])
}

/// The square adapted to the uniform size `h`, with the boundary refined too (`no_surf` off:
/// the two triangles have boundary edges of length 1, which `no_surf` would keep).
fn adapt_square(p: &Petsc, dm: &Dm, h: f64, faces: Option<&str>, cells: Option<&str>) -> Dm {
    let n = dm.depth_stratum(0).unwrap().len();
    let options = MetricOptions {
        no_surf: false,
        ..MetricOptions::default()
    };
    let metric = dm.isotropic_metric(&vec![h; n], &options).unwrap();
    let adapted = dm.adapt_metric(&metric, faces, cells).unwrap();
    assert_eq!(adapted.comm().size(), p.comm_self().size());
    adapted
}

#[test]
fn mmg_is_available() {
    let p = Petsc::init(&[]).unwrap();
    assert!(p.has_external_package("mmg").unwrap());
}

#[test]
fn metric_options_defaults_are_petscs() {
    let o = MetricOptions::default();
    assert!(o.no_surf);
    assert_eq!(o.gradation, 1.3);
    assert_eq!(o.hausdorff, 0.01);
    assert_eq!(o.verbosity, -1);
}

#[test]
fn uniform_metric_on_unit_square() {
    let p = Petsc::init(&[]).unwrap();
    let dm = unit_square(&p);
    let adapted = adapt_square(&p, &dm, 0.1, None, None);
    let cells = adapted.height_stratum(0).unwrap().len();
    assert!((150..=320).contains(&cells), "{cells} cells");
    let xy = vertex_coordinates(&adapted);
    for e in adapted.depth_stratum(1).unwrap() {
        let l = length(edge_ends(&adapted, &xy, e));
        assert!((0.04..=0.2).contains(&l), "edge {e} has the length {l}");
    }
}

#[test]
fn labels_survive_adaptation() {
    let p = Petsc::init(&[]).unwrap();
    let mut dm = unit_square(&p);
    // Side k (1 bottom, 2 right, 3 top, 4 left) and the test whether a point lies on it.
    let on_side = |k: PetscInt, [x, y]: [f64; 2]| match k {
        1 => y.abs() < 1e-12,
        2 => (x - 1.0).abs() < 1e-12,
        3 => (y - 1.0).abs() < 1e-12,
        4 => x.abs() < 1e-12,
        _ => unreachable!(),
    };
    let xy = vertex_coordinates(&dm);
    for e in dm.boundary_faces().unwrap() {
        let ends = edge_ends(&dm, &xy, e);
        let k = (1..=4)
            .find(|&k| ends.iter().all(|&q| on_side(k, q)))
            .unwrap();
        dm.set_label_value("sides", e, k).unwrap();
    }
    for c in dm.height_stratum(0).unwrap() {
        dm.set_label_value("area", c, 7).unwrap();
    }

    let adapted = adapt_square(&p, &dm, 0.1, Some("sides"), Some("area"));
    assert_eq!(adapted.label_values("sides").unwrap(), [1, 2, 3, 4]);
    let xy = vertex_coordinates(&adapted);
    let boundary = adapted.boundary_faces().unwrap();
    let mut values = vec![0; boundary.len()];
    for k in 1..=4 {
        let edges = adapted.stratum_points("sides", k).unwrap();
        let mut sum = 0.0;
        for &e in &edges {
            let i = boundary
                .binary_search(&e)
                .unwrap_or_else(|_| panic!("edge {e} of side {k} is not on the boundary"));
            values[i] += 1;
            let ends = edge_ends(&adapted, &xy, e);
            assert!(
                ends.iter().all(|&q| on_side(k, q)),
                "edge {e} of side {k} lies at {ends:?}"
            );
            sum += length(ends);
        }
        assert!(edges.len() > 1, "side {k} was not refined");
        assert!((sum - 1.0).abs() <= 1e-12, "side {k} has the length {sum}");
    }
    assert!(
        values.iter().all(|&n| n == 1),
        "boundary edges with other than one value"
    );

    let cells = adapted.height_stratum(0).unwrap();
    assert_eq!(
        adapted.stratum_points("area", 7).unwrap().len(),
        cells.len()
    );
    assert_eq!(adapted.label_values("area").unwrap(), [7]);
}

#[test]
fn no_surf_keeps_the_boundary() {
    let p = Petsc::init(&[]).unwrap();
    let dm = unit_square(&p);
    let n = dm.depth_stratum(0).unwrap().len();
    let metric = dm
        .isotropic_metric(&vec![0.1; n], &MetricOptions::default())
        .unwrap();
    let adapted = dm.adapt_metric(&metric, None, None).unwrap();
    assert!(adapted.height_stratum(0).unwrap().len() > 2);
    let xy = vertex_coordinates(&adapted);
    let boundary = adapted.boundary_faces().unwrap();
    assert_eq!(boundary.len(), 4);
    for e in boundary {
        assert_eq!(length(edge_ends(&adapted, &xy, e)), 1.0);
    }
}

#[test]
fn metric_sizes_must_match_the_vertices() {
    let p = Petsc::init(&[]).unwrap();
    let dm = unit_square(&p);
    let options = MetricOptions::default();
    let e = dm.isotropic_metric(&[0.1; 3], &options).unwrap_err();
    assert!(e.to_string().contains("3 sizes for 4 vertices"), "{e}");
    for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        let e = dm
            .isotropic_metric(&[0.1, 0.1, bad, 0.1], &options)
            .unwrap_err();
        assert!(e.to_string().contains("vertex 2"), "{e}");
    }
}

#[test]
fn adapt_needs_existing_labels() {
    let p = Petsc::init(&[]).unwrap();
    let dm = unit_square(&p);
    let metric = dm
        .isotropic_metric(&[0.5; 4], &MetricOptions::default())
        .unwrap();
    let e = dm.adapt_metric(&metric, Some("walls"), None).unwrap_err();
    assert!(e.to_string().contains("no label \"walls\""), "{e}");
    let e = dm.adapt_metric(&metric, None, Some("areas")).unwrap_err();
    assert!(e.to_string().contains("no label \"areas\""), "{e}");
}

#[test]
fn adapt_leaves_the_input_mesh_without_fields() {
    let p = Petsc::init(&[]).unwrap();
    let dm = unit_square(&p);
    let metric = dm
        .isotropic_metric(&[0.5; 4], &MetricOptions::default())
        .unwrap();
    dm.adapt_metric(&metric, None, None).unwrap();
    let mut fields: PetscInt = -1;
    // SAFETY: a valid DM; `fields` receives the number of fields.
    assert_eq!(
        unsafe { petsc::sys::DMGetNumFields(dm.as_raw(), &mut fields) },
        0
    );
    assert_eq!(fields, 0);
}
