//! Tests of loading gmsh meshes. nextest runs each test in its own process, so every test
//! initializes PETSc once. `parallel_on_*_processes` run some of them again under `mpiexec`.

use std::f64::consts::PI;
use std::io::{ErrorKind, Read};
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Stdio};
use std::time::{Duration, Instant};

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

/// Set in the runs [`parallel`] starts under `mpiexec`: the number of processes.
const MPI_RANKS: &str = "SLIDESIM_MESH_MPI_RANKS";

/// The tests [`parallel`] runs on several processes: the paths a single process does not take
/// (the distributed mesh, reductions over processes, the readability check of process 0).
const PARALLEL_TESTS: [&str; 8] = [
    "mpi_world_has_the_expected_size",
    "tissue_area",
    "refinement_quadruples_cells",
    "tags_are_the_physical_names",
    "tags_survive_refinement",
    "levels_are_coarse_first_and_linked",
    "missing_mesh_file_is_error",
    "malformed_mesh_file_is_error",
];

/// How long one run under `mpiexec` may take: a hang (a process waiting for another which left a
/// collective call with an error) fails the test.
const PARALLEL_TIMEOUT: Duration = Duration::from_secs(120);

/// Runs each of [`PARALLEL_TESTS`] of this test binary under `mpiexec -n ranks`, one run per
/// test (PETSc is initialized once per process). Skipped if there is no `mpiexec`.
fn parallel(ranks: usize) {
    if std::env::var_os(MPI_RANKS).is_some() {
        // A run under mpiexec: do not start runs from it.
        return;
    }
    let version = match Command::new("mpiexec").arg("--version").output() {
        Ok(out) => String::from_utf8_lossy(&out.stdout).into_owned(),
        Err(e) if e.kind() == ErrorKind::NotFound => {
            eprintln!("skipped: no mpiexec");
            return;
        }
        Err(e) => panic!("cannot run mpiexec: {e}"),
    };
    let open_mpi = version.contains("Open MPI") || version.contains("OpenRTE");
    let exe = std::env::current_exe().unwrap();
    for test in PARALLEL_TESTS {
        let mut cmd = Command::new("mpiexec");
        if open_mpi {
            // More processes than cores is fine for these small meshes.
            cmd.arg("--oversubscribe");
        }
        cmd.args(["-n", &ranks.to_string()])
            .arg(&exe)
            .args(["--exact", test, "--test-threads", "1", "--color", "never"])
            .env(MPI_RANKS, ranks.to_string())
            // Its own process group, so that a timeout kills mpiexec and every process.
            .process_group(0)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let (status, stdout, stderr) = run_with_timeout(cmd, PARALLEL_TIMEOUT);
        let output = format!("{stdout}\n{stderr}");
        let status = status.unwrap_or_else(|| {
            panic!("{test} on {ranks} processes timed out after {PARALLEL_TIMEOUT:?}:\n{output}")
        });
        assert!(
            status.success(),
            "{test} on {ranks} processes failed ({status}):\n{output}"
        );
        // Each process ran the test (and not zero tests): output of the processes may be
        // interleaved, so count the summaries in the whole text.
        let ran = stdout.matches("1 passed; 0 failed").count();
        assert_eq!(
            ran, ranks,
            "{test} ran on {ran} of {ranks} processes:\n{output}"
        );
    }
}

/// Runs `cmd` (in its own process group) for at most `timeout`: its exit status, or `None` if it
/// was killed (with its whole process group) for taking longer; and its stdout and stderr.
fn run_with_timeout(mut cmd: Command, timeout: Duration) -> (Option<ExitStatus>, String, String) {
    let mut child = cmd.spawn().unwrap();
    let read = |mut pipe: Box<dyn Read + Send>| {
        std::thread::spawn(move || {
            let mut text = String::new();
            pipe.read_to_string(&mut text).unwrap();
            text
        })
    };
    let stdout = read(Box::new(child.stdout.take().unwrap()));
    let stderr = read(Box::new(child.stderr.take().unwrap()));
    let start = Instant::now();
    let status = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break Some(status);
        }
        if start.elapsed() > timeout {
            // The group id is the pid of mpiexec (`process_group(0)`).
            let group = format!("-{}", child.id());
            Command::new("kill")
                .args(["-KILL", "--", &group])
                .status()
                .unwrap();
            child.wait().unwrap();
            break None;
        }
        std::thread::sleep(Duration::from_millis(50));
    };
    (status, stdout.join().unwrap(), stderr.join().unwrap())
}

#[test]
fn mpi_world_has_the_expected_size() {
    let p = Petsc::init(&[]).unwrap();
    let ranks = std::env::var(MPI_RANKS).map_or(1, |n| n.parse().unwrap());
    assert_eq!(p.comm().size(), ranks);
}

#[test]
fn parallel_on_2_processes() {
    parallel(2);
}

#[test]
fn parallel_on_3_processes() {
    parallel(3);
}
