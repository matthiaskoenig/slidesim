//! Tests of the DMPlex queries, the options database and the reductions over a communicator.
//! nextest runs each test in its own process, so every test initializes PETSc once.

use std::path::Path;
use std::ptr::{null, null_mut};

use petsc::sys;
use petsc::{Comm, Dm, Petsc};

/// A box mesh of 2 x 2 quadrilaterals of [0, 2] x [0, 1], with the boundary label `marker`.
fn quad_box(comm: &Comm) -> Dm {
    box_mesh(comm, [2, 2], [2.0, 1.0])
}

/// A box mesh of `faces` quadrilaterals of [0, upper[0]] x [0, upper[1]], with the boundary label
/// `marker`.
fn box_mesh(comm: &Comm, faces: [sys::PetscInt; 2], upper: [f64; 2]) -> Dm {
    let mut dm: sys::DM = null_mut();
    let lower = [0.0, 0.0];
    // SAFETY: valid arguments; `dm` receives a new DM whose reference `Dm` takes.
    unsafe {
        assert_eq!(
            sys::DMPlexCreateBoxMesh(
                comm.as_raw(),
                2,
                false,
                faces.as_ptr(),
                lower.as_ptr(),
                upper.as_ptr(),
                null(),
                true,
                0,
                false,
                &mut dm
            ),
            0
        );
        Dm::from_raw(dm)
    }
}

#[test]
fn plex_strata_and_dimension() {
    let p = Petsc::init(&[]).unwrap();
    let dm = quad_box(&p.comm());
    assert_eq!(dm.dimension().unwrap(), 2);
    assert_eq!(dm.coordinate_dimension().unwrap(), 2);
    // Cells first, then vertices, then edges.
    assert_eq!(dm.height_stratum(0).unwrap(), 0..4);
    assert_eq!(dm.depth_stratum(0).unwrap(), 4..13);
    assert_eq!(dm.depth_stratum(1).unwrap(), 13..25);
    assert_eq!(dm.height_stratum(1).unwrap(), dm.depth_stratum(1).unwrap());
}

#[test]
fn cell_volumes_and_numbering() {
    let p = Petsc::init(&[]).unwrap();
    let dm = quad_box(&p.comm());
    for c in dm.height_stratum(0).unwrap() {
        assert!((dm.cell_volume(c).unwrap() - 0.5).abs() < 1e-14);
    }
    // One process owns every cell: the global numbers are 0..4.
    let mut numbers = dm.cell_numbering().unwrap();
    numbers.sort();
    assert_eq!(numbers, [0, 1, 2, 3]);
}

#[test]
fn cell_volume_of_a_vertex_is_error() {
    let p = Petsc::init(&[]).unwrap();
    let dm = quad_box(&p.comm());
    let vertex = dm.depth_stratum(0).unwrap().start;
    assert!(dm.cell_volume(vertex).is_err());
}

#[test]
fn labels_and_strata() {
    let p = Petsc::init(&[]).unwrap();
    let dm = quad_box(&p.comm());
    let names = dm.label_names().unwrap();
    for name in ["depth", "marker", "Face Sets"] {
        assert!(names.iter().any(|n| n == name), "{name} in {names:?}");
    }
    assert!(dm.has_label("marker").unwrap());
    assert!(!dm.has_label("nothing").unwrap());
    assert_eq!(dm.label_values("marker").unwrap(), [1]);
    // The boundary: 8 edges and 8 vertices.
    let edges = dm.depth_stratum(1).unwrap();
    let points = dm.stratum_points("marker", 1).unwrap();
    assert_eq!(points.len(), 16);
    assert_eq!(points.iter().filter(|p| edges.contains(p)).count(), 8);
    // Absent labels and values have no points.
    assert!(dm.label_values("nothing").unwrap().is_empty());
    assert!(dm.stratum_points("nothing", 1).unwrap().is_empty());
    assert!(dm.stratum_points("marker", 7).unwrap().is_empty());
}

#[test]
fn label_name_with_nul_is_error() {
    let p = Petsc::init(&[]).unwrap();
    let dm = quad_box(&p.comm());
    let e = dm.has_label("a\0b").unwrap_err();
    assert_eq!(e.code, sys::PETSC_ERR_ARG_WRONG);
}

#[test]
fn refine_quadruples_cells_and_keeps_labels() {
    let p = Petsc::init(&[]).unwrap();
    let coarse = quad_box(&p.comm());
    let mut fine = coarse.refine().unwrap();
    assert_eq!(fine.height_stratum(0).unwrap().len(), 16);
    // The boundary is twice as finely divided: 16 edges and 16 vertices.
    assert_eq!(fine.stratum_points("marker", 1).unwrap().len(), 32);
    assert!(fine.coarse_dm().unwrap().is_none());
    fine.set_coarse_dm(&coarse).unwrap();
    let c = fine.coarse_dm().unwrap().unwrap();
    assert_eq!(c.as_raw(), coarse.as_raw());
}

#[test]
fn distribute_on_one_process_is_none() {
    let p = Petsc::init(&[]).unwrap();
    let dm = quad_box(&p.comm());
    assert!(dm.distribute(0).unwrap().is_none());
    assert_eq!(dm.comm().size(), 1);
}

#[test]
fn plex_from_missing_file_is_error() {
    let p = Petsc::init(&[]).unwrap();
    let e = Dm::plex_from_file(&p.comm(), Path::new("/nonexistent/mesh.msh"), "mesh", true)
        .unwrap_err();
    assert!(e.message.contains("/nonexistent/mesh.msh"), "{e}");
}

#[test]
fn plex_from_file_with_unknown_extension_is_error() {
    let p = Petsc::init(&[]).unwrap();
    let e = Dm::plex_from_file(&p.comm(), Path::new("/nonexistent/mesh.xyz"), "mesh", true)
        .unwrap_err();
    assert!(e.message.contains("mesh.xyz"), "{e}");
}

#[test]
fn options_get_set_clear() {
    let p = Petsc::init(&["-from_init", "3"]).unwrap();
    assert_eq!(p.option("-from_init").unwrap().as_deref(), Some("3"));
    assert_eq!(p.option("-unset").unwrap(), None);
    p.set_option("-flag", None).unwrap();
    assert_eq!(p.option("-flag").unwrap().as_deref(), Some(""));
    p.set_option("-flag", Some("yes")).unwrap();
    assert_eq!(p.option("-flag").unwrap().as_deref(), Some("yes"));
    p.clear_option("-flag").unwrap();
    assert_eq!(p.option("-flag").unwrap(), None);
    assert_eq!(
        p.set_option("no_dash", None).unwrap_err().code,
        sys::PETSC_ERR_ARG_WRONG
    );
}

#[test]
fn with_option_restores_the_previous_state() {
    let p = Petsc::init(&["-kept", "1"]).unwrap();
    let seen = p
        .with_option("-kept", Some("2"), || p.option("-kept"))
        .unwrap();
    assert_eq!(seen.as_deref(), Some("2"));
    assert_eq!(p.option("-kept").unwrap().as_deref(), Some("1"));
    let seen = p.with_option("-new", None, || p.option("-new")).unwrap();
    assert_eq!(seen.as_deref(), Some(""));
    assert_eq!(p.option("-new").unwrap(), None);
    // Also when the body fails.
    let e = p
        .with_option("-new", Some("x"), || -> petsc::Result<()> {
            Err(Dm::plex_from_file(&p.comm(), Path::new("/x.msh"), "m", true).unwrap_err())
        })
        .unwrap_err();
    assert!(e.message.contains("/x.msh"), "{e}");
    assert_eq!(p.option("-new").unwrap(), None);
}

#[test]
fn reductions_over_one_process() {
    let p = Petsc::init(&[]).unwrap();
    let comm = p.comm();
    assert_eq!(comm.sum_int(3).unwrap(), 3);
    assert_eq!(comm.max_int(-1).unwrap(), -1);
    assert_eq!(comm.sum_real(0.25).unwrap(), 0.25);
    let one = p.comm_self();
    assert_eq!((one.rank(), one.size()), (0, 1));
    assert_eq!(one.sum_int(5).unwrap(), 5);
}

#[test]
fn refined_cell_volumes_need_local_coordinates() {
    let p = Petsc::init(&[]).unwrap();
    let fine = quad_box(&p.comm()).refine().unwrap();
    let cells = fine.height_stratum(0).unwrap();
    // The refined DM has global coordinates only: a noncollective query cannot make local ones.
    let e = fine.cell_volume(cells.start).unwrap_err();
    assert_eq!(e.code, sys::PETSC_ERR_ARG_WRONGSTATE);
    fine.set_up_local_coordinates().unwrap();
    let total: f64 = cells.map(|c| fine.cell_volume(c).unwrap()).sum();
    assert!((total - 2.0).abs() < 1e-14);
}

#[test]
fn ghost_points_of_a_distributed_mesh() {
    let p = Petsc::init(&[]).unwrap();
    let comm = p.comm();
    let mut dm = quad_box(&comm);
    if let Some(distributed) = dm.distribute(0).unwrap() {
        dm = distributed;
    }
    let ghosts = dm.ghost_points().unwrap();
    assert!(ghosts.is_sorted());
    if comm.size() == 1 {
        assert!(ghosts.is_empty());
    }
    // Each point is owned by one process: 4 cells, 9 vertices and 12 edges in all.
    let owned = |range: std::ops::Range<sys::PetscInt>| {
        let ghost = ghosts.iter().filter(|g| range.contains(g)).count();
        comm.sum_int((range.len() - ghost) as sys::PetscInt)
            .unwrap()
    };
    assert_eq!(owned(dm.height_stratum(0).unwrap()), 4);
    assert_eq!(owned(dm.depth_stratum(0).unwrap()), 9);
    assert_eq!(owned(dm.depth_stratum(1).unwrap()), 12);
}

#[test]
fn boundary_faces_and_supports() {
    let p = Petsc::init(&[]).unwrap();
    let comm = p.comm();
    let mut dm = quad_box(&comm);
    if let Some(distributed) = dm.distribute(0).unwrap() {
        dm = distributed;
    }
    let cells = dm.height_stratum(0).unwrap();
    let edges = dm.depth_stratum(1).unwrap();
    let boundary = dm.boundary_faces().unwrap();
    assert!(boundary.is_sorted());
    // A boundary edge has one cell, a local one.
    for &edge in &boundary {
        assert!(edges.contains(&edge), "{edge}");
        let support = dm.support(edge).unwrap();
        assert_eq!(support.len(), 1, "edge {edge}: {support:?}");
        assert!(cells.contains(&support[0]));
    }
    // The 2 x 2 box has 8 boundary edges, each on one process; the edges between the parts of
    // processes are not among them.
    let count = comm.sum_int(boundary.len() as sys::PetscInt).unwrap();
    assert_eq!(count, 8);
    // A cell has no support in 2D.
    assert!(dm.support(cells.start).unwrap().is_empty());
}

#[test]
fn partitioner_type_is_set_and_read_from_options() {
    let p = Petsc::init(&[]).unwrap();
    let dm = quad_box(&p.comm());
    let partitioner = dm.partitioner().unwrap();
    partitioner.set_type("simple").unwrap();
    assert_eq!(partitioner.type_name().unwrap(), "simple");
    // Without the option, the type stays as set.
    partitioner.set_from_options().unwrap();
    assert_eq!(partitioner.type_name().unwrap(), "simple");
    p.with_option("-petscpartitioner_type", Some("gather"), || {
        partitioner.set_from_options()
    })
    .unwrap();
    assert_eq!(partitioner.type_name().unwrap(), "gather");
    // The DM hands out the same partitioner.
    assert_eq!(dm.partitioner().unwrap().type_name().unwrap(), "gather");
    let e = partitioner.set_type("no_such_partitioner").unwrap_err();
    assert!(e.message.contains("no_such_partitioner"), "{e}");
    assert_eq!(partitioner.type_name().unwrap(), "gather");
}

/// The native builds of PETSc have the graph partitioner PT-Scotch (scripts/deps/build-native.sh).
#[test]
fn has_external_package() {
    let p = Petsc::init(&[]).unwrap();
    assert!(p.has_external_package("ptscotch").unwrap());
    assert!(p.has_external_package("PTScotch").unwrap());
    assert!(!p.has_external_package("no_such_package").unwrap());
    let e = p.has_external_package("a\0b").unwrap_err();
    assert!(e.message.contains("NUL"), "{e}");
}

/// The global cell numbers and the local coordinates of a 24 x 24 box mesh distributed by
/// PT-Scotch, after a reset of its random generator.
fn ptscotch_partition(p: &Petsc) -> (Vec<sys::PetscInt>, Vec<f64>) {
    let mut dm = box_mesh(&p.comm(), [24, 24], [1.0, 1.0]);
    dm.partitioner().unwrap().set_type("ptscotch").unwrap();
    p.reset_partitioner_random();
    if let Some(distributed) = dm.distribute(0).unwrap() {
        dm = distributed;
    }
    (
        dm.cell_numbering().unwrap(),
        dm.local_coordinates().unwrap(),
    )
}

/// After a reset of PT-Scotch's random generator, distributing the same mesh again gives the same
/// partition.
#[test]
fn partitioner_random_reset_repeats_the_partition() {
    let p = Petsc::init(&[]).unwrap();
    let first = ptscotch_partition(&p);
    for _ in 0..2 {
        // Not assert_eq: the coordinates are long.
        assert!(
            ptscotch_partition(&p) == first,
            "process {}: the partition differs",
            p.comm().rank()
        );
    }
}

/// The tests `parallel_on_*_processes` run on several processes: points shared between them, and
/// partitions.
const PARALLEL_TESTS: [&str; 5] = [
    "mpi_world_has_the_expected_size",
    "barrier_waits_for_every_process",
    "ghost_points_of_a_distributed_mesh",
    "boundary_faces_and_supports",
    "partitioner_random_reset_repeats_the_partition",
];

#[test]
fn mpi_world_has_the_expected_size() {
    let p = Petsc::init(&[]).unwrap();
    assert_eq!(p.comm().size(), test_support::ranks());
}

#[test]
fn barrier_waits_for_every_process() {
    let p = Petsc::init(&[]).unwrap();
    let comm = p.comm();
    // One file per number of processes: the runs on 2 and 3 processes may overlap.
    let file = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("barrier")
        .join(comm.size().to_string());
    if comm.rank() == 0 {
        if file.exists() {
            std::fs::remove_file(&file).unwrap();
        }
        // Late, after the others reached the barrier.
        std::thread::sleep(std::time::Duration::from_millis(200));
        std::fs::create_dir_all(file.parent().unwrap()).unwrap();
        std::fs::write(&file, "").unwrap();
    }
    comm.barrier().unwrap();
    assert!(
        file.exists(),
        "process {} passed the barrier early",
        comm.rank()
    );
    comm.barrier().unwrap();
}

#[test]
fn parallel_on_2_processes() {
    test_support::parallel(2, &PARALLEL_TESTS);
}

#[test]
fn parallel_on_3_processes() {
    test_support::parallel(3, &PARALLEL_TESTS);
}
