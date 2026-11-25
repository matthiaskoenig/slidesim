//! Tests of the P1 discretization and its libCEED operators on gmsh meshes. nextest runs each
//! test in its own process, so every test initializes PETSc once. `parallel_on_2_processes` runs
//! some of them again under `mpiexec`, as does `parallel_on_3_processes`.

use std::io::{ErrorKind, Read};
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Stdio};
use std::time::{Duration, Instant};

use ceed::{Ceed, DEFAULT_RESOURCE};
use contracts::{LengthUnit, MeshFormat, MeshRef, SCENE_VERSION, Scene, Units};
use mesh::{Mesh, MeshError};
use pde::{DiffusionReaction, Discretization, PdeError, SourceField};
use petsc::{Petsc, Vector};

fn fixture(file: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../testdata/meshes")
        .join(file)
}

fn scene(file: &str, dimension: u8) -> Scene {
    Scene {
        version: SCENE_VERSION.to_string(),
        dimension,
        units: Units {
            length: LengthUnit::Um,
        },
        mesh: MeshRef {
            format: MeshFormat::Gmsh,
            path: fixture(file),
        },
    }
}

fn load(p: &Petsc, file: &str, refinements: u32) -> Mesh {
    Mesh::load(p, &scene(file, 2), refinements).unwrap()
}

fn ceed() -> Ceed {
    Ceed::new(DEFAULT_RESOURCE).unwrap()
}

/// The four sides of `unit_square.msh`.
const SIDES: [&str; 4] = ["left", "right", "top", "bottom"];

/// A global vector of `disc` with every entry `value`.
fn constant(disc: &Discretization, value: f64) -> Vector {
    let mut v = disc.global_vector().unwrap();
    v.set(value).unwrap();
    v
}

/// `y = A x` with the shell matrix of `op`.
fn apply(disc: &Discretization, op: DiffusionReaction, x: &Vector) -> Vector {
    let a = disc.shell_matrix(op).unwrap();
    let mut y = disc.global_vector().unwrap();
    a.mult(x, &mut y).unwrap();
    y
}

/// `|a - b|_2` for global vectors of `disc`.
fn distance(disc: &Discretization, a: &Vector, b: &Vector) -> f64 {
    let mut d = disc.global_vector().unwrap();
    d.set(0.0).unwrap();
    d.axpy(1.0, a).unwrap();
    d.axpy(-1.0, b).unwrap();
    d.norm2().unwrap()
}

#[test]
fn mass_integrates_area() {
    let p = Petsc::init(&[]).unwrap();
    let mesh = load(&p, "unit_square.msh", 1);
    let ceed = ceed();
    for level in 0..2 {
        let disc = Discretization::new(&mesh, level, &ceed, 1).unwrap();
        let one = constant(&disc, 1.0);
        let mass = DiffusionReaction::new(&disc, 0.0, 1.0, 0.0).unwrap();
        let m1 = apply(&disc, mass, &one);
        let area = one.dot(&m1).unwrap();
        assert!((area - 1.0).abs() < 1e-12, "level {level}: {area}");
    }
}

#[test]
fn shift_and_decay_both_scale_the_mass() {
    let p = Petsc::init(&[]).unwrap();
    let mesh = load(&p, "unit_square.msh", 0);
    let disc = Discretization::new(&mesh, 0, &ceed(), 1).unwrap();
    let one = constant(&disc, 1.0);
    let op = DiffusionReaction::new(&disc, 0.0, 0.5, 2.0).unwrap();
    let area = one.dot(&apply(&disc, op, &one)).unwrap();
    assert!((area - 2.5).abs() < 1e-12, "{area}");
}

#[test]
fn diffusion_kills_constants() {
    let p = Petsc::init(&[]).unwrap();
    let ceed = ceed();
    for (file, refinements) in [("unit_square.msh", 1), ("tissue_small.msh", 0)] {
        let mesh = load(&p, file, refinements);
        let disc = Discretization::new(&mesh, refinements as usize, &ceed, 1).unwrap();
        let one = constant(&disc, 1.0);
        let stiffness = DiffusionReaction::new(&disc, 1.0, 0.0, 0.0).unwrap();
        let k1 = apply(&disc, stiffness, &one).norm2().unwrap();
        assert!(k1 < 1e-12, "{file}: |K 1| = {k1}");
    }
}

/// The operator of the diagonal and assembly tests: all three terms, essential dofs on the
/// vessel walls, on the refined tissue mesh.
fn tissue_operator(p: &Petsc) -> (Discretization, DiffusionReaction) {
    let mesh = load(p, "tissue_small.msh", 1);
    let disc = Discretization::with_essential(&mesh, 1, &ceed(), 1, &["vessel_wall"]).unwrap();
    let op = DiffusionReaction::new(&disc, 0.3, 0.7, 2.0).unwrap();
    (disc, op)
}

#[test]
fn diagonal_matches_assembled() {
    let p = Petsc::init(&[]).unwrap();
    let (disc, op) = tissue_operator(&p);
    let assembled = op.assemble().unwrap();
    let mut expected = disc.global_vector().unwrap();
    assembled.diagonal(&mut expected).unwrap();
    let shell = disc.shell_matrix(op).unwrap();
    let mut diagonal = disc.global_vector().unwrap();
    shell.diagonal(&mut diagonal).unwrap();
    // Both sum the same element contributions, in different orders: relative to the diagonal
    // (entries up to about 50 on this mesh), they agree to rounding.
    let error = distance(&disc, &diagonal, &expected) / expected.norm2().unwrap();
    assert!(error < 1e-13, "relative |diag - diag(A)| = {error}");
}

#[test]
fn assembled_matches_shell() {
    let p = Petsc::init(&[]).unwrap();
    let (disc, op) = tissue_operator(&p);
    let assembled = op.assemble().unwrap();
    // A vector with distinct entries on each process.
    let mut x = disc.global_vector().unwrap();
    let rank = p.comm().rank();
    x.with_array_mut(|a| {
        for (i, v) in a.iter_mut().enumerate() {
            *v = (i as f64 + 1000.0 * rank as f64).sin();
        }
    })
    .unwrap();
    let mut expected = disc.global_vector().unwrap();
    assembled.mult(&x, &mut expected).unwrap();
    let y = apply(&disc, op, &x);
    let error = distance(&disc, &y, &expected) / expected.norm2().unwrap();
    assert!(error < 1e-14, "relative |A x - A_coo x| = {error}");
}

#[test]
fn essential_dofs_leave_global_vectors() {
    let p = Petsc::init(&[]).unwrap();
    let mesh = load(&p, "unit_square.msh", 0);
    let ceed = ceed();
    let natural = Discretization::new(&mesh, 0, &ceed, 1).unwrap();
    assert_eq!(natural.global_vector().unwrap().size().unwrap(), 31);
    let left = Discretization::with_essential(&mesh, 0, &ceed, 1, &["left"]).unwrap();
    assert_eq!(left.global_vector().unwrap().size().unwrap(), 31 - 5);
    let all = Discretization::with_essential(&mesh, 0, &ceed, 1, &SIDES).unwrap();
    assert_eq!(all.global_vector().unwrap().size().unwrap(), 31 - 16);
}

#[test]
fn essential_values_are_set_in_local_vectors() {
    let p = Petsc::init(&[]).unwrap();
    let mesh = load(&p, "unit_square.msh", 0);
    let disc = Discretization::with_essential(&mesh, 0, &ceed(), 1, &["left", "top"]).unwrap();
    let mut local = disc.local_vector().unwrap();
    local.set(0.0).unwrap();
    disc.set_essential_values(&mut local, "left", 2.0).unwrap();
    disc.set_essential_values(&mut local, "top", 3.0).unwrap();
    let values = local.with_array(<[f64]>::to_vec).unwrap();
    // The corner (0, 1) is on both sides: the later value wins.
    assert_eq!(values.iter().filter(|&&v| v == 2.0).count(), 4);
    assert_eq!(values.iter().filter(|&&v| v == 3.0).count(), 5);
    assert_eq!(values.iter().filter(|&&v| v == 0.0).count(), 31 - 9);
    let e = disc
        .set_essential_values(&mut local, "right", 1.0)
        .unwrap_err();
    assert!(e.to_string().contains("right"), "{e}");
}

#[test]
fn apply_local_uses_essential_values() {
    let p = Petsc::init(&[]).unwrap();
    let mesh = load(&p, "unit_square.msh", 1);
    let disc = Discretization::with_essential(&mesh, 1, &ceed(), 1, &SIDES).unwrap();
    let one = constant(&disc, 1.0);
    // As a Jacobian, the operator sees zero essential values: K 1 is not zero next to them.
    let jacobian = apply(
        &disc,
        DiffusionReaction::new(&disc, 1.0, 0.0, 0.0).unwrap(),
        &one,
    );
    assert!(jacobian.norm2().unwrap() > 0.1);
    // With the essential values 1 in the local vector, it applies to the constant function.
    let mut local = disc.local_vector().unwrap();
    local.set(0.0).unwrap();
    disc.dm().global_to_local(&one, &mut local).unwrap();
    for side in SIDES {
        disc.set_essential_values(&mut local, side, 1.0).unwrap();
    }
    let stiffness = DiffusionReaction::new(&disc, 1.0, 0.0, 0.0).unwrap();
    let mut y = disc.global_vector().unwrap();
    stiffness.apply_local(&local, &mut y).unwrap();
    let k1 = y.norm2().unwrap();
    assert!(k1 < 1e-12, "|K 1| = {k1}");
}

#[test]
fn coefficients_take_effect_when_changed() {
    let p = Petsc::init(&[]).unwrap();
    let mesh = load(&p, "unit_square.msh", 0);
    let disc = Discretization::new(&mesh, 0, &ceed(), 1).unwrap();
    let mut op = DiffusionReaction::new(&disc, 0.0, 1.0, 0.0).unwrap();
    let diagonal = |op: &DiffusionReaction| {
        let mut d = disc.global_vector().unwrap();
        op.assemble().unwrap().diagonal(&mut d).unwrap();
        d
    };
    let mass = diagonal(&op);
    op.k = 0.0;
    op.shift = 3.0;
    let mut scaled = diagonal(&op);
    scaled.axpy(-3.0, &mass).unwrap();
    assert!(scaled.norm2().unwrap() < 1e-14);
    op.d = 1.0;
    op.shift = 0.0;
    let stiffness = diagonal(&op);
    assert!(distance(&disc, &stiffness, &mass) > 1.0);
}

#[test]
fn source_integrates_over_the_domain() {
    let p = Petsc::init(&[]).unwrap();
    let mesh = load(&p, "unit_square.msh", 1);
    let disc = Discretization::new(&mesh, 1, &ceed(), 1).unwrap();
    let points = disc.quadrature_points();
    assert_eq!(points.len(), disc.num_quadrature_points());
    assert!(!points.is_empty() || p.comm().size() > 1);
    let inside = |&[x, y]: &[f64; 2]| 0.0 < x && x < 1.0 && 0.0 < y && y < 1.0;
    assert!(points.iter().all(inside));
    let one = constant(&disc, 1.0);
    let mut source = SourceField::new(&disc).unwrap();
    assert_eq!(source.values().len(), points.len());
    // f = 1 + x: its integral is 1.5, exact for a linear f.
    for (f, [x, _]) in source.values_mut().iter_mut().zip(points) {
        *f = 1.0 + x;
    }
    let mut b = disc.global_vector().unwrap();
    b.set(f64::NAN).unwrap();
    source.apply(&mut b).unwrap();
    let integral = one.dot(&b).unwrap();
    assert!((integral - 1.5).abs() < 1e-12, "{integral}");
}

#[test]
fn quadrature_points_cover_every_cell() {
    let p = Petsc::init(&[]).unwrap();
    let mesh = load(&p, "unit_square.msh", 0);
    let disc = Discretization::new(&mesh, 0, &ceed(), 1).unwrap();
    let total = p
        .comm()
        .sum_int(disc.num_quadrature_points() as i32)
        .unwrap();
    // 44 triangles with the same number of points each.
    assert_eq!(total % 44, 0);
    assert!(total >= 3 * 44);
    // The points are where the quadrature weights belong: f = x y is integrated exactly (the
    // quadrature is exact for quadratics).
    let one = constant(&disc, 1.0);
    let mut source = SourceField::new(&disc).unwrap();
    for (f, [x, y]) in source.values_mut().iter_mut().zip(disc.quadrature_points()) {
        *f = x * y;
    }
    let mut b = disc.global_vector().unwrap();
    source.apply(&mut b).unwrap();
    let integral = one.dot(&b).unwrap();
    assert!((integral - 0.25).abs() < 1e-14, "{integral}");
}

#[test]
fn inverted_element_is_error() {
    let p = Petsc::init(&[]).unwrap();
    let mesh = load(&p, "inverted_triangle.msh", 0);
    let e = Discretization::new(&mesh, 0, &ceed(), 1).unwrap_err();
    let PdeError::InvertedElement { cell, det } = e else {
        panic!("expected InvertedElement, got {e}");
    };
    // Area 1/2 against the reference triangle of area 2, and clockwise.
    assert!((det + 0.25).abs() < 1e-14, "{det}");
    if p.comm().size() == 1 {
        assert_eq!(cell, 1);
    } else {
        assert!((0..2).contains(&cell), "{cell}");
    }
    assert!(e.to_string().contains("inverted"), "{e}");
}

#[test]
fn unknown_essential_tag_is_error() {
    let p = Petsc::init(&[]).unwrap();
    let mesh = load(&p, "unit_square.msh", 0);
    let e = Discretization::with_essential(&mesh, 0, &ceed(), 1, &["left", "nowhere"]).unwrap_err();
    assert!(
        matches!(&e, PdeError::Mesh(MeshError::UnknownTag { name, .. }) if name == "nowhere"),
        "{e}"
    );
}

#[test]
fn degree_other_than_one_is_unsupported() {
    let p = Petsc::init(&[]).unwrap();
    let mesh = load(&p, "unit_square.msh", 0);
    for degree in [0, 2] {
        let e = Discretization::new(&mesh, 0, &ceed(), degree).unwrap_err();
        assert!(matches!(e, PdeError::Unsupported(_)), "{e}");
        assert!(e.to_string().contains(&format!("degree {degree}")), "{e}");
    }
}

#[test]
fn three_dimensions_are_unsupported() {
    let p = Petsc::init(&[]).unwrap();
    let mesh = Mesh::load(&p, &scene("unit_tetrahedron.msh", 3), 0).unwrap();
    let e = Discretization::new(&mesh, 0, &ceed(), 1).unwrap_err();
    assert!(matches!(e, PdeError::Unsupported(_)), "{e}");
    assert!(e.to_string().contains("dimension 3"), "{e}");
}

#[test]
fn level_out_of_range_is_error() {
    let p = Petsc::init(&[]).unwrap();
    let mesh = load(&p, "unit_square.msh", 1);
    let e = Discretization::new(&mesh, 2, &ceed(), 1).unwrap_err();
    assert!(
        matches!(
            e,
            PdeError::NoLevel {
                level: 2,
                levels: 2
            }
        ),
        "{e}"
    );
}

/// Set in the runs [`parallel`] starts under `mpiexec`: the number of processes.
const MPI_RANKS: &str = "SLIDESIM_PDE_MPI_RANKS";

/// The tests [`parallel`] runs on several processes: operators and assembly on a distributed
/// mesh (ghost dofs shared between processes, global numbering by owner), and errors found on
/// some processes only. On 3 processes, `inverted_triangle.msh` (2 cells) leaves one without
/// cells.
const PARALLEL_TESTS: [&str; 10] = [
    "mpi_world_has_the_expected_size",
    "mass_integrates_area",
    "diffusion_kills_constants",
    "diagonal_matches_assembled",
    "assembled_matches_shell",
    "essential_dofs_leave_global_vectors",
    "apply_local_uses_essential_values",
    "source_integrates_over_the_domain",
    "quadrature_points_cover_every_cell",
    "inverted_element_is_error",
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
