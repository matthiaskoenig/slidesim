//! Verification of the time integration of [`Problem`]: convergence orders against manufactured
//! solutions, conservation of mass, boundary fluxes, and failures as errors. nextest runs each
//! test in its own process, so every test initializes PETSc once. `parallel_on_2_processes` and
//! `parallel_on_3_processes` run the cheap ones again under `mpiexec`.
//!
//! The problem of the convergence tests: `u_t = d lap u - k u + f` on the unit square with
//! `d = 0.01`, `k = 0.5` and zero Dirichlet values on all four sides, for which
//! `phi = sin(pi x) sin(pi y)` is an eigenfunction: `-d lap phi + k phi = (2 pi^2 d + k) phi`.
//! The linear solvers run to tolerances far below the errors measured, which are those of the
//! discretization ([`ACCURATE`]), or the conservation of mass ([`CONSERVATIVE`]).

use std::f64::consts::PI;
use std::path::{Path, PathBuf};

use ceed::{Ceed, DEFAULT_RESOURCE};
use contracts::{
    BoundarySpec, Condition, FieldSpec, LengthUnit, MeshFormat, MeshRef, SCENE_VERSION, Scene,
    TimeSpec, Units,
};
use mesh::{Mesh, MeshError};
use pde::{DiffusionReaction, PdeError, Problem};
use petsc::{Petsc, Vector};

const D: f64 = 0.01;
const K: f64 = 0.5;
/// Options of the convergence tests: solver errors far below the discretization errors. The
/// absolute tolerance ends the solves once the solution is steady (the residual of the predictor
/// is rounding), which makes the steps to t = 200 of [`spatial_order_p1`] cheap.
const ACCURATE: &str = "-ksp_rtol 1e-8 -ksp_atol 1e-12";
/// Options of the tests of conservation: mass changes by the residuals of the linear solves.
const CONSERVATIVE: &str = "-ksp_rtol 1e-12";

/// The four sides of `unit_square.msh`, each of length 1.
const SIDES: [&str; 4] = ["left", "right", "top", "bottom"];

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
    };
    Mesh::load(p, &scene, refinements).unwrap()
}

fn ceed() -> Ceed {
    Ceed::new(DEFAULT_RESOURCE).unwrap()
}

fn phi([x, y]: [f64; 2]) -> f64 {
    (PI * x).sin() * (PI * y).sin()
}

/// The eigenvalue of `phi`: `-d lap phi + k phi = LAMBDA phi`.
const LAMBDA: f64 = 2.0 * PI * PI * D + K;

/// The field of the convergence tests: zero Dirichlet values on all four sides.
fn field(diffusion: f64, decay: f64) -> FieldSpec {
    FieldSpec {
        id: "u".to_string(),
        diffusion,
        decay,
        source: 0.0,
        initial: 0.0,
        boundaries: SIDES
            .iter()
            .map(|tag| BoundarySpec {
                tag: tag.to_string(),
                condition: Condition::Dirichlet { value: 0.0 },
            })
            .collect(),
    }
}

/// A field with natural (zero flux) boundaries everywhere, and `boundaries`.
fn natural_field(decay: f64, boundaries: Vec<BoundarySpec>) -> FieldSpec {
    FieldSpec {
        id: "u".to_string(),
        diffusion: D,
        decay,
        source: 0.0,
        initial: 0.0,
        boundaries,
    }
}

fn fixed(end: f64, dt: f64) -> TimeSpec {
    TimeSpec {
        end,
        dt,
        adaptive: false,
        outputs: Vec::new(),
    }
}

/// The observed orders of the errors `e`, refined by halving: `log2(e_l / e_(l+1))`.
fn orders(e: &[f64]) -> Vec<f64> {
    e.windows(2).map(|w| (w[0] / w[1]).log2()).collect()
}

/// The L2 errors of the steady manufactured solution `phi` (source `LAMBDA phi`) at t = 200 on
/// the refinements `levels` of the unit square.
fn spatial_errors(p: &Petsc, levels: &[u32]) -> Vec<f64> {
    levels
        .iter()
        .map(|&refinements| {
            let mesh = unit_square(p, refinements);
            let mut problem =
                Problem::new(&mesh, &field(D, K), &fixed(200.0, 1.0), &ceed(), ACCURATE).unwrap();
            let points = problem.discretization().quadrature_points().to_vec();
            for (f, &x) in problem.source_mut().iter_mut().zip(&points) {
                *f = LAMBDA * phi(x);
            }
            problem.step_to(200.0).unwrap();
            assert_eq!(problem.time(), 200.0);
            problem.l2_error(|x, _| phi(x)).unwrap()
        })
        .collect()
}

fn assert_orders(errors: &[f64], range: std::ops::RangeInclusive<f64>) {
    let observed = orders(errors);
    println!("errors {errors:?}, orders {observed:?}");
    for order in &observed {
        assert!(
            range.contains(order),
            "order {order} outside {range:?}: errors {errors:?}, orders {observed:?}"
        );
    }
}

#[test]
fn spatial_order_p1() {
    let p = Petsc::init(&[]).unwrap();
    assert_orders(&spatial_errors(&p, &[2, 3, 4]), 1.9..=2.1);
}

/// [`spatial_order_p1`] on the two coarser meshes only, cheap enough to run on several
/// processes.
#[test]
fn spatial_order_p1_on_refinements_2_and_3() {
    let p = Petsc::init(&[]).unwrap();
    assert_orders(&spatial_errors(&p, &[2, 3]), 1.9..=2.1);
}

/// A copy of the solution of `problem`.
fn solution(problem: &Problem) -> Vector {
    let mut u = problem.discretization().global_vector().unwrap();
    u.set(0.0).unwrap();
    u.axpy(1.0, problem.solution()).unwrap();
    u
}

/// The end time of [`temporal_order_bdf2`]: `LAMBDA T_END` about 2.1. `TSBDF` starts with a BDF1
/// half step; for the decaying mode `phi` its error and the BDF2 error have opposite signs and
/// nearly cancel at `LAMBDA t` about 0.7 (t = 1), which leaves the `dt^3` terms visible at these
/// steps (observed orders 2.6 and 2.4 at t = 1; the scalar `y' = -lambda y` through `TSBDF` shows
/// the same, and orders 1.95, 1.99 at `lambda t` = 2.1).
const T_END: f64 = 3.0;

#[test]
fn temporal_order_bdf2() {
    let p = Petsc::init(&[]).unwrap();
    let mesh = unit_square(&p, 3);
    let run = |dt: f64| {
        let mut problem =
            Problem::new(&mesh, &field(D, K), &fixed(T_END, dt), &ceed(), ACCURATE).unwrap();
        problem.set_initial_condition(phi).unwrap();
        problem.step_to(T_END).unwrap();
        assert_eq!(problem.time(), T_END);
        problem
    };
    let (disc, reference) = {
        let reference = run(0.003125);
        (reference.discretization().clone(), solution(&reference))
    };
    // The L2 norm of differences of solutions on the same mesh: sqrt(e^T M e).
    let mass = disc
        .shell_matrix(DiffusionReaction::new(&disc, 0.0, 1.0, 0.0).unwrap())
        .unwrap();
    let errors: Vec<f64> = [0.1, 0.05, 0.025]
        .into_iter()
        .map(|dt| {
            let mut e = solution(&run(dt));
            e.axpy(-1.0, &reference).unwrap();
            let mut me = disc.global_vector().unwrap();
            mass.mult(&e, &mut me).unwrap();
            e.dot(&me).unwrap().sqrt()
        })
        .collect();
    assert_orders(&errors, 1.8..=2.2);
}

/// A bump of mass about 0.1 in the middle of the unit square.
fn bump([x, y]: [f64; 2]) -> f64 {
    let r2 = (x - 0.5).powi(2) + (y - 0.5).powi(2);
    (-r2 / 0.02).exp()
}

#[test]
fn mass_conserved_with_zero_flux() {
    let p = Petsc::init(&[]).unwrap();
    let mesh = unit_square(&p, 2);
    let field = natural_field(0.0, Vec::new());
    let mut problem =
        Problem::new(&mesh, &field, &fixed(10.0, 0.5), &ceed(), CONSERVATIVE).unwrap();
    problem.set_initial_condition(bump).unwrap();
    let initial = problem.total_mass().unwrap();
    assert!((initial - 0.02 * PI).abs() < 1e-3, "{initial}");
    problem.step_to(10.0).unwrap();
    let error = (problem.total_mass().unwrap() - initial).abs() / initial;
    assert!(error < 1e-10, "relative mass error {error}");
    // It did diffuse: the bump flattened.
    let peak_error = problem.l2_error(|x, _| bump(x)).unwrap();
    assert!(peak_error > 1e-2, "{peak_error}");
}

#[test]
fn flux_balances_mass() {
    let p = Petsc::init(&[]).unwrap();
    let mesh = unit_square(&p, 2);
    // An inward flux on the left side (length 1), zero flux elsewhere.
    let q = 0.3;
    let field = natural_field(
        0.0,
        vec![BoundarySpec {
            tag: "left".to_string(),
            condition: Condition::Flux { value: q },
        }],
    );
    let mut problem =
        Problem::new(&mesh, &field, &fixed(10.0, 0.5), &ceed(), CONSERVATIVE).unwrap();
    problem.set_initial_condition(bump).unwrap();
    let initial = problem.total_mass().unwrap();
    for t in [2.5, 10.0] {
        problem.step_to(t).unwrap();
        let expected = initial + q * 1.0 * t;
        let error = (problem.total_mass().unwrap() - expected).abs() / (q * 1.0 * t);
        assert!(error < 1e-10, "t = {t}: relative error {error}");
    }
}

#[test]
fn divergence_is_error() {
    let p = Petsc::init(&[]).unwrap();
    let mesh = unit_square(&p, 1);
    let mut problem = Problem::new(
        &mesh,
        &field(D, K),
        &fixed(1e9, 1e9),
        &ceed(),
        "-ksp_max_it 1",
    )
    .unwrap();
    problem.set_initial_condition(phi).unwrap();
    let e = problem.step_to(1e9).unwrap_err();
    let PdeError::Diverged { reason, t } = &e else {
        panic!("expected Diverged, got {e}");
    };
    assert!(reason.contains("DIVERGED_ITS"), "{reason}");
    assert_eq!(*t, 0.0);
    assert!(e.to_string().contains("DIVERGED_ITS"), "{e}");
}

#[test]
fn options_are_scoped_to_the_problem() {
    let p = Petsc::init(&[]).unwrap();
    let mesh = unit_square(&p, 1);
    let problem = Problem::new(
        &mesh,
        &field(D, K),
        &fixed(1.0, 0.1),
        &ceed(),
        "-ksp_max_it 7",
    )
    .unwrap();
    // Under the prefix of the field ("u_"), and nowhere else.
    assert_eq!(p.option("-u_ksp_max_it").unwrap().as_deref(), Some("7"));
    assert_eq!(
        p.option("-u_ksp_error_if_not_converged").unwrap(),
        Some(String::new())
    );
    assert_eq!(p.option("-ksp_max_it").unwrap(), None);
    assert_eq!(p.option("-ksp_error_if_not_converged").unwrap(), None);
    // One problem per prefix at a time.
    let e = Problem::new(&mesh, &field(D, K), &fixed(1.0, 0.1), &ceed(), "").unwrap_err();
    assert!(e.to_string().contains("u_"), "{e}");
    // Dropping the problem removes its options.
    drop(problem);
    assert_eq!(p.option("-u_ksp_max_it").unwrap(), None);
    assert_eq!(p.option("-u_ksp_error_if_not_converged").unwrap(), None);
    Problem::new(&mesh, &field(D, K), &fixed(1.0, 0.1), &ceed(), "").unwrap();
}

#[test]
fn user_options_override_the_defaults() {
    let p = Petsc::init(&[]).unwrap();
    let mesh = unit_square(&p, 1);
    // Without the error, a linear solve that fails makes the step fail: still an error.
    let mut problem = Problem::new(
        &mesh,
        &field(D, K),
        &fixed(1e9, 1e9),
        &ceed(),
        "-ksp_max_it 1 -ksp_error_if_not_converged false",
    )
    .unwrap();
    assert_eq!(
        p.option("-u_ksp_error_if_not_converged")
            .unwrap()
            .as_deref(),
        Some("false")
    );
    problem.set_initial_condition(phi).unwrap();
    let e = problem.step_to(1e9).unwrap_err();
    assert!(matches!(e, PdeError::Diverged { .. }), "{e}");
}

#[test]
fn steps_end_at_the_requested_times() {
    let p = Petsc::init(&[]).unwrap();
    let mesh = unit_square(&p, 2);
    let mut problem =
        Problem::new(&mesh, &field(D, K), &fixed(1.0, 0.1), &ceed(), ACCURATE).unwrap();
    problem.set_initial_condition(phi).unwrap();
    for t in [0.15, 0.3, 0.3, 1.0] {
        problem.step_to(t).unwrap();
        assert_eq!(problem.time(), t);
    }
    // phi decays with exp(-LAMBDA t), up to the spatial error.
    let error = problem
        .l2_error(|x, t| (-LAMBDA * t).exp() * phi(x))
        .unwrap();
    assert!(error < 2e-3, "{error}");
    let e = problem.step_to(0.5).unwrap_err();
    assert!(e.to_string().contains("0.5"), "{e}");
}

#[test]
fn adaptive_steps_reach_the_requested_times() {
    let p = Petsc::init(&[]).unwrap();
    let mesh = unit_square(&p, 2);
    let time = TimeSpec {
        adaptive: true,
        ..fixed(2.0, 0.01)
    };
    let mut problem = Problem::new(&mesh, &field(D, K), &time, &ceed(), ACCURATE).unwrap();
    problem.set_initial_condition(phi).unwrap();
    for t in [0.5, 2.0] {
        problem.step_to(t).unwrap();
        assert_eq!(problem.time(), t);
    }
    let error = problem
        .l2_error(|x, t| (-LAMBDA * t).exp() * phi(x))
        .unwrap();
    assert!(error < 2e-3, "{error}");
}

#[test]
fn uniform_initial_and_source_and_dirichlet_values() {
    let p = Petsc::init(&[]).unwrap();
    let mesh = unit_square(&p, 1);
    // u = 2 everywhere, on the boundary too, with k u = f: a steady state from the start.
    let mut field = field(D, K);
    field.initial = 2.0;
    field.source = 2.0 * K;
    for b in &mut field.boundaries {
        b.condition = Condition::Dirichlet { value: 2.0 };
    }
    let mut problem = Problem::new(&mesh, &field, &fixed(1.0, 0.25), &ceed(), ACCURATE).unwrap();
    assert!(problem.source_mut().iter().all(|&f| f == 2.0 * K));
    assert!((problem.total_mass().unwrap() - 2.0).abs() < 1e-12);
    problem.step_to(1.0).unwrap();
    let error = problem.l2_error(|_, _| 2.0).unwrap();
    assert!(error < 1e-10, "{error}");
}

#[test]
fn unknown_boundary_tag_is_error() {
    let p = Petsc::init(&[]).unwrap();
    let mesh = unit_square(&p, 0);
    for condition in [
        Condition::Dirichlet { value: 1.0 },
        Condition::Flux { value: 1.0 },
    ] {
        let field = natural_field(
            0.0,
            vec![BoundarySpec {
                tag: "nowhere".to_string(),
                condition,
            }],
        );
        let e = Problem::new(&mesh, &field, &fixed(1.0, 0.1), &ceed(), "").unwrap_err();
        assert!(
            matches!(&e, PdeError::Mesh(MeshError::UnknownTag { name, .. }) if name == "nowhere"),
            "{e}"
        );
    }
}

#[test]
fn flux_on_a_region_is_error() {
    let p = Petsc::init(&[]).unwrap();
    let mesh = unit_square(&p, 0);
    let field = natural_field(
        0.0,
        vec![BoundarySpec {
            tag: "domain".to_string(),
            condition: Condition::Flux { value: 1.0 },
        }],
    );
    let e = Problem::new(&mesh, &field, &fixed(1.0, 0.1), &ceed(), "").unwrap_err();
    assert!(matches!(e, PdeError::Unsupported(_)), "{e}");
    assert!(e.to_string().contains("domain"), "{e}");
}

#[test]
fn field_id_must_be_an_options_prefix() {
    let p = Petsc::init(&[]).unwrap();
    let mesh = unit_square(&p, 0);
    for id in ["", "2u", "my field", "u-v"] {
        let field = FieldSpec {
            id: id.to_string(),
            ..natural_field(0.0, Vec::new())
        };
        let e = Problem::new(&mesh, &field, &fixed(1.0, 0.1), &ceed(), "").unwrap_err();
        assert!(matches!(e, PdeError::Unsupported(_)), "{id:?}: {e}");
    }
}

/// The tests `parallel_on_*_processes` run on several processes: the integrals over cells and
/// boundary edges of each process, the Dirichlet values and the solvers on a distributed mesh.
const PARALLEL_TESTS: [&str; 6] = [
    "mpi_world_has_the_expected_size",
    "mass_conserved_with_zero_flux",
    "flux_balances_mass",
    "divergence_is_error",
    "spatial_order_p1_on_refinements_2_and_3",
    "uniform_initial_and_source_and_dirichlet_values",
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
