//! Tests of loading gmsh meshes. nextest runs each test in its own process, so every test
//! initializes PETSc once. `parallel_on_*_processes` run some of them again under `mpiexec`.

use std::f64::consts::PI;
use std::path::{Path, PathBuf};

use contracts::{LengthUnit, MeshSource, SCENE_VERSION, Scene, Units};
use mesh::{Mesh, MeshError, Tag};
use mesher::TaggedMesh;
use petsc::{Dm, Petsc, Point};

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
        mesh: MeshSource::Gmsh { path },
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

#[test]
fn local_bounds_cover_the_local_cells() {
    let p = Petsc::init(&[]).unwrap();
    let mesh = Mesh::load(&p, &scene("tissue_small.msh"), 1).unwrap();
    let [lo, hi] = mesh.local_bounds().expect("cells on every process");
    // Every vertex of every level of this process is inside, and the box is in the section
    // (200 x 200 um).
    for level in mesh.levels() {
        for x in level.local_coordinates().unwrap().as_chunks::<2>().0 {
            assert!((0..2).all(|i| lo[i] <= x[i] && x[i] <= hi[i]), "{x:?}");
        }
    }
    assert!(
        0.0 <= lo[0] && lo[0] < hi[0] && hi[0] <= 200.0,
        "{lo:?} {hi:?}"
    );
    assert!(
        0.0 <= lo[1] && lo[1] < hi[1] && hi[1] <= 200.0,
        "{lo:?} {hi:?}"
    );
    if p.comm().size() == 1 {
        assert_eq!([lo, hi], [[0.0, 0.0], [200.0, 200.0]]);
    }
}

/// The mesh file `file` as PETSc reads it, on process 0 (with the labels of the physical names),
/// before `Mesh::load` changes anything.
fn read_raw(p: &Petsc, file: &str) -> Dm {
    p.with_option("-dm_plex_gmsh_use_regions", Some("true"), || {
        Dm::plex_from_file(&p.comm(), &fixture(file), "mesh", true)
    })
    .unwrap()
}

/// The numbers of cells, edges and vertices of `dm` over all processes, each point counted by
/// its owner, and for each of its tags the numbers of the cells, edges and vertices it marks.
fn census(dm: &Dm, tags: &[Tag]) -> Vec<(String, [i32; 3])> {
    let ghosts = dm.ghost_points().unwrap();
    let strata = [2, 1, 0].map(|d| dm.depth_stratum(d).unwrap());
    let count = |points: &mut dyn Iterator<Item = petsc::Point>| -> [i32; 3] {
        let mut n = [0; 3];
        for point in points {
            if ghosts.binary_search(&point).is_ok() {
                continue;
            }
            let d = strata.iter().position(|s| s.contains(&point)).unwrap();
            n[d] += 1;
        }
        n.map(|k| dm.comm().sum_int(k).unwrap())
    };
    let all = strata.clone().into_iter().flatten();
    let mut census = vec![("mesh".to_string(), count(&mut all.into_iter()))];
    for tag in tags {
        let mut points = Vec::new();
        for value in dm.label_values(&tag.name).unwrap() {
            points.extend(dm.stratum_points(&tag.name, value).unwrap());
        }
        census.push((tag.name.clone(), count(&mut points.into_iter())));
    }
    census
}

/// `pinched.msh` has a vertex at (2, 1) where two holes touch, with two fans of cells around it:
/// loaded, it is two vertices at (2, 1), one per fan, so no vertex has more than two edges of the
/// boundary. The cells, the edges and the tags stay: each tag marks the same numbers of cells,
/// edges and vertices.
#[test]
fn pinched_vertex_is_split_per_fan() {
    let p = Petsc::init(&[]).unwrap();
    let mesh = Mesh::load(&p, &scene("pinched.msh"), 0).unwrap();
    let names: Vec<&str> = mesh.tags().iter().map(|t| t.name.as_str()).collect();
    assert_eq!(
        names,
        ["domain", "hole_wall", "inner_line", "left", "right", "rim"]
    );
    let raw = read_raw(&p, "pinched.msh");
    let before = census(&raw, mesh.tags());
    let after = census(&mesh.levels()[0], mesh.tags());
    for ((name, b), (_, a)) in before.iter().zip(&after) {
        println!("{name}: cells, edges, vertices {b:?} as read, {a:?} loaded");
        // One more vertex in the mesh and on `hole_wall`, which marks the edges around it.
        let extra = i32::from(name == "mesh");
        assert_eq!(a, &[b[0], b[1], b[2] + extra], "{name}");
    }
    // Two owned vertices at (2, 1), and a vertex with at most two edges of the boundary on
    // each process.
    let dm = &mesh.levels()[0];
    let vertices = dm.depth_stratum(0).unwrap();
    let ghosts = dm.ghost_points().unwrap();
    let dim = dm.coordinate_dimension().unwrap();
    let coordinates = dm.local_coordinates().unwrap();
    let at_pinch = vertices
        .clone()
        .filter(|v| ghosts.binary_search(v).is_err())
        .filter(|&v| {
            let i = (v - vertices.start) as usize * dim;
            (coordinates[i] - 2.0).abs() < 1e-12 && (coordinates[i + 1] - 1.0).abs() < 1e-12
        })
        .count();
    assert_eq!(dm.comm().sum_int(at_pinch as i32).unwrap(), 2);
    let mut boundary_edges = std::collections::BTreeMap::new();
    for face in dm.boundary_faces().unwrap() {
        for v in dm.cone(face).unwrap() {
            *boundary_edges.entry(v).or_insert(0) += 1;
        }
    }
    assert!(
        boundary_edges.values().all(|&n| n <= 2),
        "{boundary_edges:?}"
    );
}

/// A mesh without a vertex where the boundary touches itself is loaded as read: the same numbers
/// of cells, edges and vertices, and of the points of each tag.
#[test]
fn mesh_without_pinched_vertex_is_unchanged() {
    let p = Petsc::init(&[]).unwrap();
    for file in ["unit_square.msh", "tissue_small.msh", "lobule_toy.msh"] {
        let mesh = Mesh::load(&p, &scene(file), 0).unwrap();
        let raw = read_raw(&p, file);
        assert_eq!(
            census(&raw, mesh.tags()),
            census(&mesh.levels()[0], mesh.tags()),
            "{file}"
        );
    }
}

/// The vertices of cell `cell` of `dm` (its local coordinates, set up).
fn cell_vertices(dm: &Dm, cell: Point) -> Vec<[f64; 2]> {
    let vertices = dm.depth_stratum(0).unwrap();
    let dim = dm.coordinate_dimension().unwrap();
    let coordinates = dm.local_coordinates().unwrap();
    dm.closure(cell)
        .unwrap()
        .into_iter()
        .filter(|p| vertices.contains(p))
        .map(|v| {
            let i = (v - vertices.start) as usize * dim;
            [coordinates[i], coordinates[i + 1]]
        })
        .collect()
}

/// Whether `x` lies in the triangle `t`, up to `tolerance` (um) outside its edges.
fn in_triangle(x: [f64; 2], t: &[[f64; 2]], tolerance: f64) -> bool {
    let cross = |a: [f64; 2], b: [f64; 2], p: [f64; 2]| {
        let (ex, ey) = (b[0] - a[0], b[1] - a[1]);
        ((p[0] - a[0]) * ey - (p[1] - a[1]) * ex) / ex.hypot(ey)
    };
    let sides: Vec<f64> = (0..3).map(|i| cross(t[i], t[(i + 1) % 3], x)).collect();
    sides.iter().all(|&s| s <= tolerance) || sides.iter().all(|&s| s >= -tolerance)
}

/// The coarse cell of every cell of every level contains its centroid: the numbering of the
/// regular refinement, on the distributed mesh.
#[test]
fn coarse_cell_contains_fine_centroids() {
    let p = Petsc::init(&[]).unwrap();
    let mesh = Mesh::load(&p, &scene("lobule_toy.msh"), 2).unwrap();
    let coarse = &mesh.levels()[0];
    let coarse_cells = coarse.height_stratum(0).unwrap();
    for (level, dm) in mesh.levels().iter().enumerate() {
        let cells = dm.height_stratum(0).unwrap();
        assert_eq!(cells.len(), coarse_cells.len() << (2 * level));
        for cell in cells.clone() {
            let fine = cell_vertices(dm, cell);
            assert_eq!(fine.len(), 3);
            let centroid = [0, 1].map(|i| fine.iter().map(|v| v[i]).sum::<f64>() / 3.0);
            let parent = mesh.coarse_cell(level, (cell - cells.start) as usize);
            assert!(parent < coarse_cells.len());
            if level == 0 {
                assert_eq!(parent, (cell - cells.start) as usize);
            }
            let triangle = cell_vertices(coarse, coarse_cells.start + parent as Point);
            assert!(
                in_triangle(centroid, &triangle, 1e-9),
                "level {level} cell {cell}: centroid {centroid:?} outside its coarse cell \
                 {parent} {triangle:?}"
            );
        }
    }
}

/// The tests `parallel_on_*_processes` run on several processes: the paths a single process does
/// not take (the distributed mesh, its partition, reductions over processes, the readability check
/// of process 0).
const PARALLEL_TESTS: [&str; 17] = [
    "mpi_world_has_the_expected_size",
    "from_tagged_matches_load_of_the_same_mesh",
    "lobule_scene_loads",
    "mesher_error_names_the_polygons_file",
    "coarse_cell_contains_fine_centroids",
    "pinched_vertex_is_split_per_fan",
    "mesh_without_pinched_vertex_is_unchanged",
    "local_bounds_cover_the_local_cells",
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

/// The `TaggedMesh` of `lobule_toy.msh` (read on process 0 alone, converted with
/// `TaggedMesh::from_dm`) through `Mesh::from_tagged` gives the mesh `Mesh::load` gives: the same
/// cells, tags, area and points per tag, on every level.
#[test]
fn from_tagged_matches_load_of_the_same_mesh() {
    let p = Petsc::init(&[]).unwrap();
    let tagged = (p.comm().rank() == 0).then(|| {
        let dm = p
            .with_option("-dm_plex_gmsh_use_regions", Some("true"), || {
                Dm::plex_from_file(&p.comm_self(), &fixture("lobule_toy.msh"), "mesh", true)
            })
            .unwrap();
        TaggedMesh::from_dm(&dm).unwrap()
    });
    for refinements in [0, 1] {
        let generated = Mesh::from_tagged(&p, tagged.as_ref(), refinements).unwrap();
        let loaded = Mesh::load(&p, &scene("lobule_toy.msh"), refinements).unwrap();
        assert_eq!(generated.tags(), loaded.tags());
        assert_eq!(generated.levels().len(), loaded.levels().len());
        for (g, l) in generated.levels().iter().zip(loaded.levels()) {
            assert_eq!(global_cells(g), global_cells(l));
            let (a, b) = (area(g), area(l));
            assert!(((a - b) / b).abs() < 1e-12, "{a} != {b}");
            assert_eq!(census(g, generated.tags()), census(l, loaded.tags()));
        }
    }
}

/// The lobule scene of the example studies (one lobule, rings 0) meshed on process 0 and
/// distributed: its tags, the area of the analytic geometry, and 4 times the coarse cells after
/// one refinement.
#[test]
fn lobule_scene_loads() {
    let p = Petsc::init(&[]).unwrap();
    let scene = Scene::load(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/studies/lobules/scene.json"),
    )
    .unwrap();
    let mesh = Mesh::load(&p, &scene, 1).unwrap();
    assert_eq!(
        mesh.tags(),
        [
            tag("central_wall", 1),
            tag("field_cut", 1),
            tag("portal_wall", 1),
            tag("tissue", 2)
        ]
    );
    let exact = 227072.43481658038;
    let a = area(mesh.finest());
    assert!(
        ((a - exact) / exact).abs() < 1e-3,
        "area {a}, exact {exact}"
    );
    let coarse = global_cells(&mesh.levels()[0]);
    assert_eq!(global_cells(mesh.finest()), 4 * coarse);
    // Process 0 meshed, and holds the mesh and its quality.
    assert_eq!(mesh.generated().is_some(), p.comm().rank() == 0);
    if let Some((tagged, quality)) = mesh.generated() {
        assert_eq!(tagged.triangles.len() as i32, coarse);
        assert_eq!(quality.triangles, tagged.triangles.len());
        assert!(quality.min_angle_deg >= 20.0, "{quality:?}");
    }
}

/// A polygons scene the mesher cannot mesh (a ring crossing itself): an error on every process
/// which names the polygons file and the crossing.
#[test]
fn mesher_error_names_the_polygons_file() {
    let p = Petsc::init(&[]).unwrap();
    let contracts = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/contracts");
    let scene = Scene::load(contracts.join("scene-polygons-self-intersecting.json")).unwrap();
    let err = Mesh::load(&p, &scene, 0).unwrap_err();
    let MeshError::Mesher { message } = &err else {
        panic!("{err:?}");
    };
    let file = std::path::absolute(contracts.join("polygons-self-intersecting.geojson")).unwrap();
    assert!(message.contains(&file.display().to_string()), "{message}");
    assert!(message.contains("cross at (50, 50) um"), "{message}");
    assert_eq!(err.to_string(), *message);
}
