//! Tests of loading gmsh meshes. nextest runs each test in its own process, so every test
//! initializes PETSc once. `parallel_on_*_processes` run some of them again under `mpiexec`.

use std::f64::consts::PI;
use std::path::{Path, PathBuf};

use contracts::{LengthUnit, MeshFormat, MeshRef, SCENE_VERSION, Scene, Units};
use mesh::{Mesh, MeshError, Tag};
use petsc::{Dm, Petsc};

/// The radius in um of the two holes of `tissue_small.msh` (`HOLE_RADIUS` of
/// `testdata/meshes/make_fixtures.py`).
const HOLE_RADIUS: f64 = 20.0;

fn fixture(file: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../testdata/meshes")
        .join(file)
}

fn scene_at(path: PathBuf, dimension: u8) -> Scene {
    Scene {
        version: SCENE_VERSION.to_string(),
        dimension,
        units: Units {
            length: LengthUnit::Um,
        },
        mesh: MeshRef {
            format: MeshFormat::Gmsh,
            path,
        },
        maps: Vec::new(),
    }
}

fn scene(file: &str) -> Scene {
    scene_at(fixture(file), 2)
}

/// The number of cells over all processes (each counted by its owner).
fn global_cells(dm: &Dm) -> i32 {
    let owned = dm
        .cell_numbering()
        .unwrap()
        .iter()
        .filter(|&&g| g >= 0)
        .count();
    dm.comm().sum_int(owned as i32).unwrap()
}

/// The integral of 1 over the mesh: the sum of the volumes of the owned cells over all processes.
fn area(dm: &Dm) -> f64 {
    let cells = dm.height_stratum(0).unwrap();
    let numbers = dm.cell_numbering().unwrap();
    let local: f64 = cells
        .zip(&numbers)
        .filter(|&(_, &g)| g >= 0)
        .map(|(c, _)| dm.cell_volume(c).unwrap())
        .sum();
    dm.comm().sum_real(local).unwrap()
}

fn tag(name: &str, dim: u8) -> Tag {
    Tag {
        name: name.to_string(),
        dim,
    }
}

#[test]
fn unit_square_has_edges() {
    let p = Petsc::init(&[]).unwrap();
    let mesh = Mesh::load(&p, &scene("unit_square.msh"), 0).unwrap();
    for name in ["left", "right", "top", "bottom"] {
        assert_eq!(mesh.tag(name).unwrap(), tag(name, 1));
    }
    assert_eq!(mesh.tag("domain").unwrap(), tag("domain", 2));
}

#[test]
fn refinement_quadruples_cells() {
    let p = Petsc::init(&[]).unwrap();
    let mesh = Mesh::load(&p, &scene("unit_square.msh"), 1).unwrap();
    let levels = mesh.levels();
    assert_eq!(levels.len(), 2);
    let coarse = global_cells(&levels[0]);
    assert!(coarse > 0);
    assert_eq!(global_cells(&levels[1]), 4 * coarse);
}

#[test]
fn unknown_tag_lists_available() {
    let p = Petsc::init(&[]).unwrap();
    let mesh = Mesh::load(&p, &scene("tissue_small.msh"), 0).unwrap();
    let e = mesh.tag("portal_wall").unwrap_err();
    assert!(matches!(e, MeshError::UnknownTag { .. }), "{e:?}");
    assert_eq!(
        e.to_string(),
        "mesh has no tag 'portal_wall'; available: section_edge, tissue, vessel_wall"
    );
}

#[test]
fn tissue_area() {
    let p = Petsc::init(&[]).unwrap();
    let mesh = Mesh::load(&p, &scene("tissue_small.msh"), 1).unwrap();
    let exact = 40000.0 - 2.0 * PI * HOLE_RADIUS * HOLE_RADIUS;
    let a = area(mesh.finest());
    assert!(
        ((a - exact) / exact).abs() < 1e-2,
        "area {a}, exact {exact}"
    );
    // Refinement does not change the polygonal domain.
    let coarse = area(&mesh.levels()[0]);
    assert!(((a - coarse) / coarse).abs() < 1e-12, "{a} != {coarse}");
}

#[test]
fn tags_are_the_physical_names() {
    let p = Petsc::init(&[]).unwrap();
    let mesh = Mesh::load(&p, &scene("tissue_small.msh"), 0).unwrap();
    assert_eq!(
        mesh.tags(),
        [
            tag("section_edge", 1),
            tag("tissue", 2),
            tag("vessel_wall", 1)
        ]
    );
}

#[test]
fn tags_survive_refinement() {
    let p = Petsc::init(&[]).unwrap();
    let mesh = Mesh::load(&p, &scene("tissue_small.msh"), 2).unwrap();
    assert_eq!(mesh.levels().len(), 3);
    let finest = mesh.finest();
    let edges = finest.depth_stratum(1).unwrap();
    let cells = finest.height_stratum(0).unwrap();
    for t in mesh.tags() {
        for level in mesh.levels() {
            assert!(level.has_label(&t.name).unwrap(), "{} lost", t.name);
        }
        let marked = finest.label_values(&t.name).unwrap();
        let points: Vec<_> = marked
            .iter()
            .flat_map(|&v| finest.stratum_points(&t.name, v).unwrap())
            .collect();
        let of_dim = match t.dim {
            1 => &edges,
            _ => &cells,
        };
        let n = points.iter().filter(|p| of_dim.contains(p)).count();
        assert!(n > 0, "{} marks no points of dimension {}", t.name, t.dim);
    }
    // The tissue label marks every cell of the finest level.
    let tissue = finest.stratum_points("tissue", 3).unwrap();
    let n = tissue.iter().filter(|p| cells.contains(p)).count();
    assert_eq!(n, cells.len());
}

#[test]
fn levels_are_coarse_first_and_linked() {
    let p = Petsc::init(&[]).unwrap();
    let mesh = Mesh::load(&p, &scene("unit_square.msh"), 2).unwrap();
    let levels = mesh.levels();
    assert_eq!(mesh.finest().as_raw(), levels[2].as_raw());
    assert!(levels[0].coarse_dm().unwrap().is_none());
    for pair in levels.windows(2) {
        let coarse = pair[1].coarse_dm().unwrap().unwrap();
        assert_eq!(coarse.as_raw(), pair[0].as_raw());
        assert_eq!(global_cells(&pair[1]), 4 * global_cells(&pair[0]));
    }
}

#[test]
fn zero_refinements_has_one_level() {
    let p = Petsc::init(&[]).unwrap();
    let mesh = Mesh::load(&p, &scene("unit_square.msh"), 0).unwrap();
    assert_eq!(mesh.levels().len(), 1);
    assert_eq!(mesh.finest().as_raw(), mesh.levels()[0].as_raw());
    assert_eq!(mesh.finest().dimension().unwrap(), 2);
    assert_eq!(mesh.finest().coordinate_dimension().unwrap(), 2);
    assert!((area(mesh.finest()) - 1.0).abs() < 1e-12);
}

#[test]
fn load_leaves_the_options_unchanged() {
    let p = Petsc::init(&[]).unwrap();
    Mesh::load(&p, &scene("unit_square.msh"), 0).unwrap();
    assert_eq!(p.option("-dm_plex_gmsh_use_regions").unwrap(), None);
}

#[test]
fn missing_mesh_file_is_error() {
    let p = Petsc::init(&[]).unwrap();
    let path = fixture("no_such_mesh.msh");
    let e = Mesh::load(&p, &scene_at(path.clone(), 2), 0).err().unwrap();
    assert!(matches!(e, MeshError::NotFound { .. }), "{e:?}");
    assert_eq!(
        e.to_string(),
        format!("mesh file not found: {}", path.display())
    );
}

#[test]
fn malformed_mesh_file_is_error() {
    let p = Petsc::init(&[]).unwrap();
    let dir = std::env::temp_dir().join(format!("slidesim-mesh-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("broken.msh");
    std::fs::write(&path, "$MeshFormat\nnot a mesh\n").unwrap();
    let e = Mesh::load(&p, &scene_at(path.clone(), 2), 0).err().unwrap();
    std::fs::remove_dir_all(&dir).unwrap();
    assert!(matches!(e, MeshError::Read { .. }), "{e:?}");
    let text = e.to_string();
    assert!(
        text.starts_with(&format!("cannot read mesh {}: PETSc error", path.display())),
        "{text}"
    );
}

#[test]
fn dimension_mismatch_is_error() {
    let p = Petsc::init(&[]).unwrap();
    let path = fixture("unit_square.msh");
    let e = Mesh::load(&p, &scene_at(path.clone(), 3), 0).err().unwrap();
    assert_eq!(
        e.to_string(),
        format!(
            "mesh {} has dimension 2, but the scene has dimension 3",
            path.display()
        )
    );
}

/// The edge cut of the distributed mesh `dm` (distributed without overlap): the number of interior
/// faces whose two cells are owned by different processes, over all processes. Without overlap,
/// such a face is on both processes and is a ghost on exactly one of them; every other face is
/// on one process only. Collective.
fn edge_cut(dm: &Dm) -> i32 {
    let faces = dm.height_stratum(1).unwrap();
    let ghosts = dm.ghost_points().unwrap();
    let cut = ghosts.iter().filter(|p| faces.contains(p)).count();
    dm.comm().sum_int(cut as i32).unwrap()
}

/// The number of cells this process owns.
fn owned_cells(dm: &Dm) -> usize {
    let numbers = dm.cell_numbering().unwrap();
    numbers.iter().filter(|&&g| g >= 0).count()
}

/// Distributed meshes are partitioned compactly: the edge cut of the default partitioner is at
/// most a third of the cut of PETSc's `simple` partitioner (which splits the cells by their number
/// in the file), and every process owns cells.
#[test]
fn partition_is_compact() {
    let p = Petsc::init(&[]).unwrap();
    let scene = scene("tissue_small.msh");
    let mesh = Mesh::load(&p, &scene, 0).unwrap();
    let simple = p
        .with_option("-petscpartitioner_type", Some("simple"), || {
            Mesh::load(&p, &scene, 0)
        })
        .unwrap();
    let (dm, dm_simple) = (mesh.finest(), simple.finest());
    let (cut, cut_simple) = (edge_cut(dm), edge_cut(dm_simple));
    let ranks = p.comm().size();
    if p.comm().rank() == 0 {
        println!("edge cut on {ranks} processes: default {cut}, simple {cut_simple}");
    }
    assert!(
        3 * cut <= cut_simple,
        "edge cut {cut} is more than a third of the simple partitioner's {cut_simple}"
    );
    assert!(
        owned_cells(dm) > 0,
        "process {} owns no cell",
        p.comm().rank()
    );
    assert_eq!(global_cells(dm), global_cells(dm_simple));
}

/// Loading a mesh again gives the same partition: it depends on the mesh and the number of
/// processes only, not on the partitionings before in this process.
#[test]
fn partition_is_reproducible() {
    let p = Petsc::init(&[]).unwrap();
    let scene = scene("tissue_small.msh");
    let first = Mesh::load(&p, &scene, 0).unwrap();
    for _ in 0..2 {
        let again = Mesh::load(&p, &scene, 0).unwrap();
        let (a, b) = (first.finest(), again.finest());
        // Not assert_eq: the coordinates are long.
        assert!(
            a.cell_numbering().unwrap() == b.cell_numbering().unwrap()
                && a.local_coordinates().unwrap() == b.local_coordinates().unwrap(),
            "process {}: the partition differs",
            p.comm().rank()
        );
    }
}

/// The tests `parallel_on_*_processes` run on several processes: the paths a single process does
/// not take (the distributed mesh, its partition, reductions over processes, the readability check
/// of process 0).
const PARALLEL_TESTS: [&str; 10] = [
    "mpi_world_has_the_expected_size",
    "tissue_area",
    "refinement_quadruples_cells",
    "tags_are_the_physical_names",
    "tags_survive_refinement",
    "levels_are_coarse_first_and_linked",
    "missing_mesh_file_is_error",
    "malformed_mesh_file_is_error",
    "partition_is_compact",
    "partition_is_reproducible",
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
