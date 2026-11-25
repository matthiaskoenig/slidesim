//! Tests of geometric multigrid over the refinement hierarchy of a mesh: iterations which stay
//! bounded as the mesh is refined, and the preconditioner of [`Problem`] kept in step with the
//! shift of its time steps. nextest runs each test in its own process, so every test
//! initializes PETSc once. `parallel_on_2_processes` and `parallel_on_3_processes` run some of
//! them again under `mpiexec`.
//!
//! The problem: the steady problem of `spatial_order_p1` (`tests/verification.rs`),
//! `-d lap u + k u = f` with `f = (2 pi^2 d + k) sin(pi x) sin(pi y)`, here on
//! `tissue_small.msh` with zero Dirichlet values on all its boundary tags, and with the
//! coefficients of tissue: `d = 2000` um^2/s, `k = 0.1` 1/s.

use std::cell::RefCell;
use std::f64::consts::PI;
use std::path::{Path, PathBuf};
use std::rc::Rc;

use ceed::{Ceed, DEFAULT_RESOURCE};
use contracts::{
    BoundarySpec, Condition, FieldSpec, LengthUnit, MeshFormat, MeshRef, SCENE_VERSION, Scene,
    TimeSpec, Units,
};
use mesh::Mesh;
use pde::{
    DiffusionReaction, Discretization, Multigrid, PdeError, Problem, SharedOperator, SourceField,
    configure_multigrid,
};
use petsc::{Ksp, Matrix, Petsc, Vector};

/// Diffusion dominates on these meshes (`k h^2 / d` at most 5e-3 for `h` at most 10 um), the
/// regime slidesim runs in, where Jacobi alone needs more iterations on every refinement (137,
/// 269, 467, 780 at refinements 1 to 4). The values of `spatial_order_p1` (`d = 0.01`,
/// `k = 0.5`) are dominated by the mass term on `tissue_small` (`k h^2 / d` at least 19), where
/// Jacobi alone stays bounded and the test could not tell multigrid from Jacobi.
const D: f64 = 2000.0;
const K: f64 = 0.1;
/// `-d lap phi + k phi = LAMBDA phi` for `phi = sin(pi x) sin(pi y)`.
const LAMBDA: f64 = 2.0 * PI * PI * D + K;

/// The boundary tags of `tissue_small.msh`: the outer edge of the section and the walls of the
/// two vessels, all of its boundary.
const BOUNDARY: [&str; 2] = ["section_edge", "vessel_wall"];

fn tissue_small(p: &Petsc, refinements: u32) -> Mesh {
    let scene = Scene {
        version: SCENE_VERSION.to_string(),
        dimension: 2,
        units: Units {
            length: LengthUnit::Um,
        },
        mesh: MeshRef {
            format: MeshFormat::Gmsh,
            path: fixture("tissue_small.msh"),
        },
    };
    Mesh::load(p, &scene, refinements).unwrap()
}

fn fixture(file: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../testdata/meshes")
        .join(file)
}

fn ceed() -> Ceed {
    Ceed::new(DEFAULT_RESOURCE).unwrap()
}

fn phi([x, y]: [f64; 2]) -> f64 {
    (PI * x).sin() * (PI * y).sin()
}

/// The steady problem on `tissue_small` with the operator `shift M + d K + k M`, and CG
/// preconditioned by multigrid to `rtol = 1e-10`.
struct Steady {
    ksp: Ksp,
    multigrid: Multigrid,
    /// The operator of the finest level, the operator of `ksp`.
    a: Matrix,
    b: Vector,
    fine: Discretization,
}

impl Steady {
    fn new(p: &Petsc, refinements: u32, shift: f64) -> Steady {
        let mesh = tissue_small(p, refinements);
        let ceed = ceed();
        let discs = Discretization::hierarchy(&mesh, &ceed, 1, &BOUNDARY).unwrap();
        let levels = operators(&discs, shift);
        let fine = discs.last().unwrap().clone();
        let mut a = fine.shell_matrix(levels.last().unwrap().clone()).unwrap();
        a.set_spd().unwrap();
        let mut source = SourceField::new(&fine).unwrap();
        let points = fine.quadrature_points().to_vec();
        for (f, &x) in source.values_mut().iter_mut().zip(&points) {
            *f = LAMBDA * phi(x);
        }
        let mut b = fine.global_vector().unwrap();
        source.apply(&mut b).unwrap();
        let mut ksp = Ksp::create(&fine.dm().comm()).unwrap();
        ksp.set_operators(&a, &a).unwrap();
        ksp.set_type("cg").unwrap();
        ksp.set_tolerances(Some(1e-10), None, Some(1000)).unwrap();
        let multigrid = configure_multigrid(&mut ksp, &mesh, &levels).unwrap();
        Steady {
            ksp,
            multigrid,
            a,
            b,
            fine,
        }
    }

    /// The solution, and the number of iterations to it.
    fn solve(&mut self) -> (Vector, usize) {
        let mut u = self.fine.global_vector().unwrap();
        self.ksp.solve(&self.b, &mut u).unwrap();
        let reason = self.ksp.converged_reason().unwrap();
        assert!(!reason.diverged(), "{reason}");
        (u, self.ksp.iterations().unwrap())
    }
}

/// The operators `shift M + D K + K M` on `discs`.
fn operators(discs: &[Discretization], shift: f64) -> Vec<SharedOperator> {
    discs
        .iter()
        .map(|disc| {
            let op = DiffusionReaction::new(disc, D, K, shift).unwrap();
            SharedOperator(Rc::new(RefCell::new(op)))
        })
        .collect()
}

/// The iterations of CG with multigrid at each of `refinements`, printed.
fn iterations(p: &Petsc, refinements: &[u32]) -> Vec<usize> {
    let counts: Vec<usize> = refinements
        .iter()
        .map(|&r| Steady::new(p, r, 0.0).solve().1)
        .collect();
    println!(
        "CG + multigrid on {} processes, refinements {refinements:?}: iterations {counts:?}",
        test_support::ranks()
    );
    counts
}

/// At most 20 iterations on the finest mesh, and at most 1.5 times those on the coarsest.
fn assert_bounded(counts: &[usize]) {
    let (first, last) = (counts[0], counts[counts.len() - 1]);
    assert!(last <= 20, "iterations {counts:?}");
    assert!(last as f64 <= 1.5 * first as f64, "iterations {counts:?}");
}

#[test]
fn iterations_bounded() {
    let p = Petsc::init(&[]).unwrap();
    assert_bounded(&iterations(&p, &[1, 2, 3, 4]));
}

/// A copy of the global vector `v` of `disc`.
fn copy(disc: &Discretization, v: &Vector) -> Vector {
    let mut c = disc.global_vector().unwrap();
    c.set(0.0).unwrap();
    c.axpy(1.0, v).unwrap();
    c
}

/// `|a - b| / |b|` for global vectors of `disc`.
fn relative_distance(disc: &Discretization, a: &Vector, b: &Vector) -> f64 {
    let mut d = copy(disc, a);
    d.axpy(-1.0, b).unwrap();
    d.norm2().unwrap() / b.norm2().unwrap()
}

/// The shift of [`refreshed_preconditioner_matches_a_new_one`]: `shift h^2 / d` about 1 on the
/// coarse level (`h` about 10 um), so the coarse operator changes much.
const SHIFT: f64 = 20.0;

#[test]
fn refreshed_preconditioner_matches_a_new_one() {
    // GAMG keeps its interpolation when its matrix changes, by default: here it builds it
    // again, as for a new matrix.
    let p = Petsc::init(&["-mg_coarse_pc_gamg_reuse_interpolation", "false"]).unwrap();
    let mut steady = Steady::new(&p, 2, 0.0);
    steady.solve();
    // The coefficients change on every level; the operator of the solver is marked changed.
    steady.multigrid.set_coefficients(D, K, SHIFT).unwrap();
    steady.a.mark_changed().unwrap();
    let mut fresh = Steady::new(&p, 2, SHIFT);
    let (refreshed_u, refreshed_iterations) = steady.solve();
    let (fresh_u, fresh_iterations) = fresh.solve();
    println!("iterations: refreshed {refreshed_iterations}, new {fresh_iterations}");
    assert_eq!(refreshed_iterations, fresh_iterations);
    let distance = relative_distance(&steady.fine, &refreshed_u, &fresh_u);
    assert!(distance < 1e-14, "{distance}");
    // One application of each preconditioner (a V-cycle): the same.
    for s in [&mut steady, &mut fresh] {
        s.ksp.set_type("preonly").unwrap();
    }
    let (refreshed_v, _) = steady.solve();
    let (fresh_v, _) = fresh.solve();
    let distance = relative_distance(&steady.fine, &refreshed_v, &fresh_v);
    assert!(distance < 1e-14, "{distance}");
}

#[test]
fn one_level_is_the_coarse_solve() {
    let p = Petsc::init(&[]).unwrap();
    let mut steady = Steady::new(&p, 0, 0.0);
    let (_, iterations) = steady.solve();
    // Algebraic multigrid alone.
    assert!(iterations <= 20, "{iterations}");
    let pc = steady.ksp.pc().unwrap();
    assert_eq!(pc.type_name().unwrap(), "mg");
    assert_eq!(pc.mg_levels().unwrap(), 1);
    let coarse = pc.mg_smoother(0).unwrap();
    assert_eq!(coarse.pc().unwrap().type_name().unwrap(), "gamg");
    assert_eq!(coarse.pc().unwrap().options_prefix().unwrap(), "mg_coarse_");
}

#[test]
fn levels_must_be_a_hierarchy() {
    let p = Petsc::init(&[]).unwrap();
    let mesh = tissue_small(&p, 1);
    let ceed = ceed();
    let mut ksp = Ksp::create(&p.comm()).unwrap();
    // One operator for two levels.
    let discs = Discretization::hierarchy(&mesh, &ceed, 1, &BOUNDARY).unwrap();
    let e = configure_multigrid(&mut ksp, &mesh, &operators(&discs[1..], 0.0)).unwrap_err();
    assert!(matches!(e, PdeError::NotHierarchy(_)), "{e}");
    // Discretizations of each level, not linked.
    let discs: Vec<Discretization> = (0..2)
        .map(|level| Discretization::with_essential(&mesh, level, &ceed, 1, &BOUNDARY).unwrap())
        .collect();
    let e = configure_multigrid(&mut ksp, &mesh, &operators(&discs, 0.0)).unwrap_err();
    assert!(matches!(e, PdeError::NotHierarchy(_)), "{e}");
    assert!(e.to_string().contains("level 0"), "{e}");
}

/// A field on `tissue_small` as in a study of oxygen: `d = D`, `k = K`, a fixed value on the
/// vessel walls, no flux through the edge of the section.
fn tissue_field() -> FieldSpec {
    FieldSpec {
        id: "u".to_string(),
        diffusion: D,
        decay: K,
        source: 0.0,
        initial: 0.0,
        boundaries: vec![BoundarySpec {
            tag: "vessel_wall".to_string(),
            condition: Condition::Dirichlet { value: 0.1 },
        }],
    }
}

/// The solution at the times `times` and the linear iterations of all steps, stepped as `time`
/// says on `tissue_small` at 1 refinement, with the options `options` and the solves to
/// `rtol = 1e-12`.
fn run(
    p: &Petsc,
    time: &TimeSpec,
    times: &[f64],
    options: &str,
) -> (Discretization, Vec<Vector>, usize) {
    let mesh = tissue_small(p, 1);
    let options = format!("-ksp_rtol 1e-12 {options}");
    let mut problem = Problem::new(&mesh, &tissue_field(), time, &ceed(), &options).unwrap();
    let solutions = times
        .iter()
        .map(|&t| {
            problem.step_to(t).unwrap();
            assert_eq!(problem.time(), t);
            copy(problem.discretization(), problem.solution())
        })
        .collect();
    let disc = problem.discretization().clone();
    (disc, solutions, problem.linear_iterations().unwrap())
}

/// Multigrid (the default) against Jacobi, in runs whose steps change size.
fn assert_matches_jacobi(time: TimeSpec, times: &[f64]) {
    let p = Petsc::init(&[]).unwrap();
    let (disc, multigrid, multigrid_iterations) = run(&p, &time, times, "");
    let (_, jacobi, jacobi_iterations) = run(&p, &time, times, "-pc_type jacobi");
    println!("linear iterations: multigrid {multigrid_iterations}, Jacobi {jacobi_iterations}");
    for ((m, j), t) in multigrid.iter().zip(&jacobi).zip(times) {
        let distance = relative_distance(&disc, m, j);
        assert!(distance < 1e-8, "t = {t}: {distance}");
    }
    // The default is multigrid, which needs fewer iterations once the steps are long enough
    // for diffusion to dominate (short steps are dominated by the mass matrix, where Jacobi
    // alone does well).
    assert!(
        2 * multigrid_iterations < jacobi_iterations,
        "{multigrid_iterations} {jacobi_iterations}"
    );
}

#[test]
fn problem_matches_jacobi_after_step_changes() {
    // Steps of 0.3 s shortened to end at 0.5 s, then 0.3 s again: the shift changes.
    let time = TimeSpec {
        end: 2.0,
        dt: 0.3,
        adaptive: false,
        outputs: Vec::new(),
    };
    assert_matches_jacobi(time, &[0.5, 2.0]);
}

#[test]
fn adaptive_problem_matches_jacobi() {
    let time = TimeSpec {
        end: 2.0,
        dt: 0.01,
        adaptive: true,
        outputs: Vec::new(),
    };
    assert_matches_jacobi(time, &[0.2, 0.5]);
}

/// The results of this process in [`two_processes_match_one`]: the L2 norm and the total mass
/// of the solution, for the runs on two processes.
const REFERENCE: &str = "SLIDESIM_TEST_REFERENCE";

/// The L2 norm and the total mass of the solution of the tissue field at t = 2 s, after 4 BDF2
/// steps on `tissue_small` at 2 refinements, with the default options of [`Problem`].
fn tissue_results(p: &Petsc) -> [f64; 2] {
    let mesh = tissue_small(p, 2);
    let time = TimeSpec {
        end: 2.0,
        dt: 0.5,
        adaptive: false,
        outputs: Vec::new(),
    };
    let mut problem = Problem::new(&mesh, &tissue_field(), &time, &ceed(), "").unwrap();
    problem.step_to(2.0).unwrap();
    assert_eq!(problem.steps().unwrap(), 4);
    [
        problem.l2_error(|_, _| 0.0).unwrap(),
        problem.total_mass().unwrap(),
    ]
}

/// The results of a [`Problem`] on two processes match those on one up to the tolerance of the
/// linear solves: multigrid depends on the partition. Run alone, the test computes the results
/// on one process and starts itself on two with them (in [`REFERENCE`]) to compare.
#[test]
fn two_processes_match_one() {
    let p = Petsc::init(&[]).unwrap();
    let results = tissue_results(&p);
    match std::env::var(REFERENCE) {
        Err(_) => {
            assert_eq!(test_support::ranks(), 1);
            // MPI is finalized before mpiexec starts.
            drop(p);
            let reference = format!("{:e} {:e}", results[0], results[1]);
            println!("on one process: L2 norm and total mass {reference}");
            test_support::parallel_with_env(
                2,
                &["two_processes_match_one"],
                &[(REFERENCE, &reference)],
            );
        }
        Ok(reference) => {
            let reference: Vec<f64> = reference.split(' ').map(|v| v.parse().unwrap()).collect();
            for (name, (r, one)) in ["L2 norm", "total mass"]
                .iter()
                .zip(results.iter().zip(&reference))
            {
                let difference = ((r - one) / one).abs();
                println!(
                    "{name} on {} processes: {r:e}, on one: {one:e}, relative difference \
                     {difference:e}",
                    test_support::ranks()
                );
                assert!(difference <= 1e-8, "{name}: {r} against {one}");
            }
        }
    }
}

/// The tests `parallel_on_*_processes` run on several processes: multigrid on a distributed
/// mesh hierarchy, with the coarse level solved by algebraic multigrid.
const PARALLEL_TESTS: [&str; 6] = [
    "mpi_world_has_the_expected_size",
    "iterations_bounded",
    "refreshed_preconditioner_matches_a_new_one",
    "one_level_is_the_coarse_solve",
    "problem_matches_jacobi_after_step_changes",
    "adaptive_problem_matches_jacobi",
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
