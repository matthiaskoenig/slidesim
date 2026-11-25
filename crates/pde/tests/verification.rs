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

mod common;

use std::f64::consts::PI;
use std::rc::Rc;

use common::{PIXELS, Scratch, ceed, unit_square};
use contracts::{BoundarySpec, Coefficient, Condition, FieldSpec, TimeSpec};
use maps::Maps;
use mesh::MeshError;
use pde::{Inputs, PdeError, PointData, Problem, Terms, Transport};
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

/// The inputs of a [`Problem`] of constant coefficients: no maps.
fn no_maps() -> Inputs<'static> {
    static NONE: std::sync::LazyLock<Maps> = std::sync::LazyLock::new(Maps::default);
    Inputs { maps: &NONE }
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
        diffusion: Coefficient::Constant(diffusion),
        decay: Coefficient::Constant(decay),
        source: Coefficient::Constant(0.0),
        porosity: Coefficient::Constant(1.0),
        advection: false,
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
        diffusion: Coefficient::Constant(D),
        decay: Coefficient::Constant(decay),
        source: Coefficient::Constant(0.0),
        porosity: Coefficient::Constant(1.0),
        advection: false,
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
            let mut problem = Problem::new(
                &mesh,
                &field(D, K),
                &fixed(200.0, 1.0),
                &ceed(),
                ACCURATE,
                &no_maps(),
            )
            .unwrap();
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
        let mut problem = Problem::new(
            &mesh,
            &field(D, K),
            &fixed(T_END, dt),
            &ceed(),
            ACCURATE,
            &no_maps(),
        )
        .unwrap();
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
    let n = disc.num_quadrature_points();
    let data = PointData::isotropic(&disc, &vec![1.0; n], &vec![0.0; n], &vec![0.0; n]);
    let mass = disc
        .shell_matrix(Transport::new(&disc, Rc::new(data), Terms::Mass).unwrap())
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
    let mut problem = Problem::new(
        &mesh,
        &field,
        &fixed(10.0, 0.5),
        &ceed(),
        CONSERVATIVE,
        &no_maps(),
    )
    .unwrap();
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
    let mut problem = Problem::new(
        &mesh,
        &field,
        &fixed(10.0, 0.5),
        &ceed(),
        CONSERVATIVE,
        &no_maps(),
    )
    .unwrap();
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
        &no_maps(),
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
    // The solvers may be left in any state: the problem does not step again.
    for e in [
        problem.step_to(1e9).unwrap_err(),
        problem.step_to(2e9).unwrap_err(),
        problem.set_initial_condition(phi).unwrap_err(),
    ] {
        let PdeError::Poisoned { failure, t } = &e else {
            panic!("expected Poisoned, got {e}");
        };
        assert!(failure.contains("DIVERGED_ITS"), "{failure}");
        assert_eq!(*t, 0.0);
    }
    // What it reached stays readable.
    assert_eq!(problem.time(), 0.0);
    let mass = problem.total_mass().unwrap();
    assert!((mass - 4.0 / (PI * PI)).abs() < 1e-2, "{mass}");
    assert!(problem.l2_error(|x, _| phi(x)).unwrap() < 1e-1);
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
        &no_maps(),
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
    let e = Problem::new(
        &mesh,
        &field(D, K),
        &fixed(1.0, 0.1),
        &ceed(),
        "",
        &no_maps(),
    )
    .unwrap_err();
    assert!(e.to_string().contains("u_"), "{e}");
    // Dropping the problem removes its options.
    drop(problem);
    assert_eq!(p.option("-u_ksp_max_it").unwrap(), None);
    assert_eq!(p.option("-u_ksp_error_if_not_converged").unwrap(), None);
    Problem::new(
        &mesh,
        &field(D, K),
        &fixed(1.0, 0.1),
        &ceed(),
        "",
        &no_maps(),
    )
    .unwrap();
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
        &no_maps(),
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
fn command_line_options_override_the_defaults_and_the_caller() {
    // A loose tolerance from the command line, against the caller's 1e-12 and the default 1e-10.
    let p = Petsc::init(&["-u_ksp_rtol", "1e-1"]).unwrap();
    let mesh = unit_square(&p, 2);
    let iterations = |rtol: &str| {
        let mut problem = Problem::new(
            &mesh,
            &field(D, K),
            &fixed(0.1, 0.1),
            &ceed(),
            "-ksp_rtol 1e-12",
            &no_maps(),
        )
        .unwrap();
        assert_eq!(p.option("-u_ksp_rtol").unwrap().as_deref(), Some(rtol));
        problem.set_initial_condition(phi).unwrap();
        problem.step_to(0.1).unwrap();
        problem.linear_iterations().unwrap()
    };
    let loose = iterations("1e-1");
    // The problem leaves the command line option as it was.
    assert_eq!(p.option("-u_ksp_rtol").unwrap().as_deref(), Some("1e-1"));
    // Without it, the caller's option holds over the default.
    p.clear_option("-u_ksp_rtol").unwrap();
    let tight = iterations("1e-12");
    assert_eq!(p.option("-u_ksp_rtol").unwrap(), None);
    assert!(
        loose < tight,
        "{loose} iterations at 1e-1, {tight} at 1e-12"
    );
}

#[test]
fn steps_end_at_the_requested_times() {
    let p = Petsc::init(&[]).unwrap();
    let mesh = unit_square(&p, 2);
    let mut problem = Problem::new(
        &mesh,
        &field(D, K),
        &fixed(1.0, 0.1),
        &ceed(),
        ACCURATE,
        &no_maps(),
    )
    .unwrap();
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
fn step_to_a_time_which_is_not_finite_is_error() {
    let p = Petsc::init(&[]).unwrap();
    let mesh = unit_square(&p, 1);
    let mut problem = Problem::new(
        &mesh,
        &field(D, K),
        &fixed(1.0, 0.1),
        &ceed(),
        ACCURATE,
        &no_maps(),
    )
    .unwrap();
    problem.set_initial_condition(phi).unwrap();
    problem.step_to(0.1).unwrap();
    for t in [f64::INFINITY, f64::NAN, f64::NEG_INFINITY] {
        let e = problem.step_to(t).unwrap_err();
        let PdeError::InvalidTime { time, .. } = &e else {
            panic!("expected InvalidTime for {t}, got {e}");
        };
        assert_eq!(*time, 0.1);
    }
    // Rejected before any step: the problem is not poisoned.
    assert_eq!(problem.time(), 0.1);
    problem.step_to(0.2).unwrap();
    assert_eq!(problem.time(), 0.2);
}

#[test]
fn fixed_steps_keep_their_size_after_requested_times() {
    let p = Petsc::init(&[]).unwrap();
    let mesh = unit_square(&p, 1);
    let mut problem = Problem::new(
        &mesh,
        &field(D, K),
        &fixed(1.0, 0.1),
        &ceed(),
        ACCURATE,
        &no_maps(),
    )
    .unwrap();
    problem.set_initial_condition(phi).unwrap();
    // 0.1, then 0.05 to end at 0.15.
    problem.step_to(0.15).unwrap();
    assert_eq!(problem.steps().unwrap(), 2);
    assert_eq!(problem.time_step(), 0.1);
    // Steps of 0.1 to 0.85, then the remaining 0.15 in two (TSAdapt's match of the end time):
    // 7 + 2 steps, where steps of 0.05 would take 17.
    problem.step_to(1.0).unwrap();
    assert_eq!(problem.steps().unwrap(), 11);
    assert_eq!(problem.time(), 1.0);
    assert_eq!(problem.time_step(), 0.1);
}

#[test]
fn adaptive_steps_keep_the_choice_of_the_adaptor() {
    let p = Petsc::init(&[]).unwrap();
    let mesh = unit_square(&p, 1);
    let time = TimeSpec {
        adaptive: true,
        ..fixed(2.0, 0.01)
    };
    let mut problem =
        Problem::new(&mesh, &field(D, K), &time, &ceed(), ACCURATE, &no_maps()).unwrap();
    problem.set_initial_condition(phi).unwrap();
    problem.step_to(0.5).unwrap();
    // The adaptor grew the step from 0.01; the next step_to starts with its choice.
    let h = problem.time_step();
    assert!(h > 0.02, "{h}");
    let steps = problem.steps().unwrap();
    problem.step_to(0.5 + h).unwrap();
    assert_eq!(problem.steps().unwrap(), steps + 1);
}

#[test]
fn adaptive_steps_reach_the_requested_times() {
    let p = Petsc::init(&[]).unwrap();
    let mesh = unit_square(&p, 2);
    let time = TimeSpec {
        adaptive: true,
        ..fixed(2.0, 0.01)
    };
    let mut problem =
        Problem::new(&mesh, &field(D, K), &time, &ceed(), ACCURATE, &no_maps()).unwrap();
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
    field.source = Coefficient::Constant(2.0 * K);
    for b in &mut field.boundaries {
        b.condition = Condition::Dirichlet { value: 2.0 };
    }
    let mut problem = Problem::new(
        &mesh,
        &field,
        &fixed(1.0, 0.25),
        &ceed(),
        ACCURATE,
        &no_maps(),
    )
    .unwrap();
    assert!(problem.source_mut().iter().all(|&f| f == 2.0 * K));
    assert!((problem.total_mass().unwrap() - 2.0).abs() < 1e-12);
    problem.step_to(1.0).unwrap();
    let error = problem.l2_error(|_, _| 2.0).unwrap();
    assert!(error < 1e-10, "{error}");
    // On the unit square: the norm of 2 is 2.
    let norm = problem.l2_norm().unwrap();
    assert!((norm - 2.0).abs() < 1e-10, "{norm}");
}

#[test]
fn local_solution_has_the_dirichlet_values() {
    let p = Petsc::init(&[]).unwrap();
    let mesh = unit_square(&p, 1);
    // u = 1 inside, Dirichlet 3 on the left side only.
    let mut field = natural_field(
        0.0,
        vec![BoundarySpec {
            tag: "left".to_string(),
            condition: Condition::Dirichlet { value: 3.0 },
        }],
    );
    field.initial = 1.0;
    let problem = Problem::new(
        &mesh,
        &field,
        &fixed(1.0, 0.25),
        &ceed(),
        ACCURATE,
        &no_maps(),
    )
    .unwrap();
    let local = problem.local_solution().unwrap();
    // Named for viewers: the point data of a VTK viewer is `u.mM`.
    assert_eq!(local.name().unwrap(), "");
    assert_eq!(
        problem.discretization().dm().field_names(0).unwrap(),
        ("u".to_string(), vec!["mM".to_string()])
    );
    let disc = problem.discretization();
    let values = local.with_array(<[f64]>::to_vec).unwrap();
    assert_eq!(values.len(), disc.node_coordinates().len());
    for (u, [x, _]) in values.iter().zip(disc.node_coordinates()) {
        let expected = if *x == 0.0 { 3.0 } else { 1.0 };
        assert_eq!(*u, expected, "at x = {x}");
    }
    // The norm with the Dirichlet values: a P1 ramp from 3 to 1 in the cells along the side.
    let norm = problem.l2_norm().unwrap();
    assert!(norm > 1.0 && norm < 3.0, "{norm}");
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
        let e = Problem::new(&mesh, &field, &fixed(1.0, 0.1), &ceed(), "", &no_maps()).unwrap_err();
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
    let e = Problem::new(&mesh, &field, &fixed(1.0, 0.1), &ceed(), "", &no_maps()).unwrap_err();
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
        let e = Problem::new(&mesh, &field, &fixed(1.0, 0.1), &ceed(), "", &no_maps()).unwrap_err();
        assert!(matches!(e, PdeError::Unsupported(_)), "{id:?}: {e}");
    }
}

/// Advection is not supported yet (milestone 2, Task 5).
#[test]
fn advection_is_unsupported() {
    let p = Petsc::init(&[]).unwrap();
    let mesh = unit_square(&p, 0);
    let field = FieldSpec {
        advection: true,
        ..natural_field(0.0, Vec::new())
    };
    let e = Problem::new(&mesh, &field, &fixed(1.0, 0.1), &ceed(), "", &no_maps()).unwrap_err();
    assert!(matches!(e, PdeError::Unsupported(_)), "{e}");
    assert!(e.to_string().contains("advection"), "{e}");
}

#[test]
fn table_of_a_missing_map_is_error() {
    let p = Petsc::init(&[]).unwrap();
    let mesh = unit_square(&p, 0);
    let field = FieldSpec {
        porosity: table("nowhere", [0.0, 0.5], [1.0, 1.0]),
        ..natural_field(0.0, Vec::new())
    };
    let e = Problem::new(&mesh, &field, &fixed(1.0, 0.1), &ceed(), "", &no_maps()).unwrap_err();
    let PdeError::Map { message } = &e else {
        panic!("expected a map error, got {e}");
    };
    // The map, the field and its coefficient.
    for name in ["'nowhere'", "'u'", "porosity"] {
        assert!(message.contains(name), "{name}: {message}");
    }
}

/// A table over the map `map` from `(v0, c0)` to `(v1, c1)`.
fn table(map: &str, [v0, c0]: [f64; 2], [v1, c1]: [f64; 2]) -> Coefficient {
    Coefficient::Table(contracts::Table {
        map: map.to_string(),
        points: vec![[v0, c0], [v1, c1]],
    })
}

/// The diffusion coefficient of [`spatial_order_variable_coefficients`]: `0.01 (1 + x)`.
fn variable_d([x, _]: [f64; 2]) -> f64 {
    0.01 * (1.0 + x)
}

/// The porosity of [`spatial_order_variable_coefficients`]: `0.5 + 0.25 y`.
fn variable_phi([_, y]: [f64; 2]) -> f64 {
    0.5 + 0.25 * y
}

/// The source `f` of `phi u_t - div(phi d grad u) + phi k u = phi f` for `u = phi(x) e^-t`
/// ([`phi`], the eigenfunction of the convergence tests), with [`variable_d`],
/// [`variable_phi`] and `k = K`, at `x` and `t = 0`: it scales with `e^-t`.
///
/// `div(phi d grad u) = phi d lap u + grad(phi d) . grad u`, `lap u = -2 pi^2 u`, and
/// `grad(phi d) = (0.01 phi, 0.25 d)`, so
/// `f = (K - 1 + 2 pi^2 d) u - 0.01 du/dx - (0.25 d / phi) du/dy`.
fn variable_source(x: [f64; 2]) -> f64 {
    let (d, porosity) = (variable_d(x), variable_phi(x));
    let [px, py] = [PI * x[0], PI * x[1]];
    let du_dx = PI * px.cos() * py.sin();
    let du_dy = PI * px.sin() * py.cos();
    (K - 1.0 + 2.0 * PI * PI * d) * phi(x) - 0.01 * du_dx - 0.25 * d / porosity * du_dy
}

/// The time step of [`spatial_order_variable_coefficients`]: a power of 2, so that the times
/// `n DT` are exact. The temporal error at it is below 1% of the spatial errors of refinements
/// 1 to 3 (halving it changes them by at most 0.6%).
const DT: f64 = 1.0 / 64.0;

/// The L2 errors at t = 1 of `u = phi(x) e^-t` with the variable coefficients of
/// [`variable_source`] on the refinements `levels` of the unit square, the coefficients from
/// maps of x and y. The source is set before each step to its value at the end of the step,
/// where the BDF2 residual takes it.
fn variable_errors(p: &Petsc, levels: &[u32]) -> Vec<f64> {
    let scratch = Scratch::new("spatial_order_variable_coefficients");
    let specs = [
        scratch.map("position_x", PIXELS, |[x, _]| x),
        scratch.map("position_y", PIXELS, |[_, y]| y),
    ];
    let field = FieldSpec {
        diffusion: table("position_x", [0.0, 0.01], [1.0, 0.02]),
        porosity: table("position_y", [0.0, 0.5], [1.0, 0.75]),
        ..field(D, K)
    };
    levels
        .iter()
        .map(|&refinements| {
            let mesh = unit_square(p, refinements);
            let bounds = mesh.local_bounds().expect("cells on every process");
            let maps = Maps::read(&specs, &["position_x", "position_y"], bounds).unwrap();
            let inputs = Inputs { maps: &maps };
            let mut problem =
                Problem::new(&mesh, &field, &fixed(1.0, DT), &ceed(), ACCURATE, &inputs).unwrap();
            problem.set_initial_condition(phi).unwrap();
            let points = problem.discretization().quadrature_points().to_vec();
            let source: Vec<f64> = points.iter().map(|&x| variable_source(x)).collect();
            let steps = (1.0 / DT) as u32;
            for n in 1..=steps {
                let t = f64::from(n) * DT;
                for (f, g) in problem.source_mut().iter_mut().zip(&source) {
                    *f = g * (-t).exp();
                }
                problem.step_to(t).unwrap();
            }
            assert_eq!(problem.time(), 1.0);
            assert_eq!(problem.steps().unwrap(), steps as usize);
            problem.l2_error(|x, t| phi(x) * (-t).exp()).unwrap()
        })
        .collect()
}

#[test]
fn spatial_order_variable_coefficients() {
    let p = Petsc::init(&[]).unwrap();
    assert_orders(&variable_errors(&p, &[1, 2, 3]), 1.9..=2.1);
}

/// [`spatial_order_variable_coefficients`] on the two coarser meshes only, cheap enough to run
/// on several processes: the maps read for the cells of each process.
#[test]
fn spatial_order_variable_coefficients_on_refinements_1_and_2() {
    let p = Petsc::init(&[]).unwrap();
    assert_orders(&variable_errors(&p, &[1, 2]), 1.9..=2.1);
}

/// The tests `parallel_on_*_processes` run on several processes: the integrals over cells and
/// boundary edges of each process, the Dirichlet values and the solvers on a distributed mesh.
const PARALLEL_TESTS: [&str; 10] = [
    "mpi_world_has_the_expected_size",
    "mass_conserved_with_zero_flux",
    "flux_balances_mass",
    "divergence_is_error",
    "fixed_steps_keep_their_size_after_requested_times",
    "spatial_order_p1_on_refinements_2_and_3",
    "spatial_order_variable_coefficients_on_refinements_1_and_2",
    "table_of_a_missing_map_is_error",
    "uniform_initial_and_source_and_dirichlet_values",
    "local_solution_has_the_dirichlet_values",
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
