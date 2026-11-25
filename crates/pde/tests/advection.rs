//! Tests of advection with SUPG stabilization in [`Problem`], carried by the Darcy velocity of a
//! [`Flow`] on the unit square: convergence orders against a manufactured solution from a cell
//! Peclet number below 1 to an advection-dominated one, a front without oscillations, the
//! inflow rule, the mass balance, and the agreement of results on several processes. nextest
//! runs each test in its own process, so every test initializes PETSc once.
//! `parallel_on_2_processes` and `parallel_on_3_processes` run some of them again under
//! `mpiexec`, and compare the results of the front with those of one process.
//!
//! The flows are uniform, which P1 pressures represent exactly (as `uniform_flow` of the tests
//! of the flow), scaled to the cell Peclet number wanted: `Pe = |u| h / (2 d)` for the
//! diffusion coefficient `d` and the nominal mesh size `h = 0.25 / 2^r` of `unit_square.msh`
//! refined `r` times. The direction maps are test Zarr arrays over the unit square (see
//! `common`).

mod common;

use std::f64::consts::PI;
use std::rc::Rc;

use common::{PIXELS, Scratch, ceed, read_maps, unit_square};
use contracts::{
    BoundarySpec, Coefficient, Condition, Direction, FieldSpec, Perfusion, PressureBoundary,
    TimeSpec,
};
use maps::Maps;
use mesh::Mesh;
use pde::{
    BoundaryInflow, Discretization, Flow, Inputs, PdeError, PointData, Problem, Terms, Transport,
};
use petsc::Petsc;

/// The four sides of `unit_square.msh`, each of length 1.
const SIDES: [&str; 4] = ["left", "right", "top", "bottom"];

/// The diffusion coefficient of the convergence tests (um^2/s).
const D: f64 = 0.01;
/// The decay rate of the convergence tests (1/s).
const K: f64 = 0.5;
/// The porosity of the convergence tests.
const POROSITY: f64 = 0.5;

/// Options of the convergence tests: solver errors far below the discretization errors.
const ACCURATE: &str = "-ksp_rtol 1e-8 -ksp_atol 1e-12";
/// Options of the tests of the mass balance: it closes up to the residuals of the solves.
const CONSERVATIVE: &str = "-ksp_rtol 1e-12";

/// Set in the runs of `parallel_on_*_processes` under `mpiexec`: the results on one process.
const REFERENCE: &str = "SLIDESIM_TEST_ADVECTION_REFERENCE";

/// The nominal mesh size of `unit_square.msh` refined `refinements` times.
fn mesh_size(refinements: u32) -> f64 {
    0.25 / f64::from(1 << refinements)
}

/// The speed for the cell Peclet number `peclet` with the diffusion coefficient `d` on the mesh
/// refined `refinements` times.
fn speed(peclet: f64, d: f64, refinements: u32) -> f64 {
    2.0 * d * peclet / mesh_size(refinements)
}

/// The perfusion of viscosity 1 Pa*s, permeabilities `k_along`, `k_across` (um^2) along the
/// gradient of the map `direction`, and the pressures (Pa) of `boundaries`.
fn perfusion([k_along, k_across]: [f64; 2], boundaries: &[(&str, f64)]) -> Perfusion {
    Perfusion {
        viscosity: 1.0,
        k_along: Coefficient::Constant(k_along),
        k_across: Coefficient::Constant(k_across),
        direction: Direction {
            map: "direction".to_string(),
            g0: 1e-6,
        },
        boundaries: boundaries
            .iter()
            .map(|&(tag, pressure)| PressureBoundary {
                tag: tag.to_string(),
                pressure,
            })
            .collect(),
    }
}

/// The direction map of the flows, `x + 0.5 y`, read for the cells of this process.
fn direction_maps(scratch: &Scratch, mesh: &Mesh) -> Maps {
    read_maps(
        &[scratch.map("direction", PIXELS, |[x, y]| x + 0.5 * y)],
        mesh,
    )
}

/// The uniform flow `a (1, 0.5)` (um/s): isotropic `k = a`, the pressure `-(x + 0.5 y)` fixed
/// on all four sides.
fn diagonal_flow(mesh: &Mesh, maps: &Maps, a: f64) -> Flow {
    let boundaries: Vec<(&str, f64)> = SIDES.iter().map(|&side| (side, 0.0)).collect();
    let pressure = |[x, y]: [f64; 2]| -(x + 0.5 * y);
    Flow::solve_manufactured(
        mesh,
        &perfusion([a, a], &boundaries),
        maps,
        &ceed(),
        "",
        &pressure,
        &|_| 0.0,
    )
    .unwrap()
}

/// The uniform flow `(a, 0)` (um/s) from `left` (pressure `a`) to `right` (0), isotropic
/// `k = 1`: no flow through `top` and `bottom`.
fn flow_along_x(mesh: &Mesh, maps: &Maps, a: f64) -> Flow {
    let perfusion = perfusion([1.0, 1.0], &[("left", a), ("right", 0.0)]);
    Flow::solve(mesh, &perfusion, maps, &ceed(), "").unwrap()
}

fn dirichlet(tag: &str, value: f64) -> BoundarySpec {
    BoundarySpec {
        tag: tag.to_string(),
        condition: Condition::Dirichlet { value },
    }
}

/// A field `c` with advection, diffusion `d`, decay `k`, porosity `phi`, no source and initial
/// value 0, and the conditions `boundaries`.
fn field(d: f64, k: f64, phi: f64, boundaries: Vec<BoundarySpec>) -> FieldSpec {
    FieldSpec {
        id: "c".to_string(),
        diffusion: Coefficient::Constant(d),
        decay: Coefficient::Constant(k),
        source: Coefficient::Constant(0.0),
        porosity: Coefficient::Constant(phi),
        advection: true,
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

fn phi([x, y]: [f64; 2]) -> f64 {
    (PI * x).sin() * (PI * y).sin()
}

/// The source `f` of `phi c_t + u . grad c - div(phi d grad c) + phi k c = phi f` for
/// `c = sin(pi x) sin(pi y) e^-t` with the velocity `u`, the porosity [`POROSITY`], `d = D` and
/// `k = K`, at `x` and `t = 0`: it scales with `e^-t`.
fn manufactured_source(u: [f64; 2], x: [f64; 2]) -> f64 {
    let [px, py] = [PI * x[0], PI * x[1]];
    let grad = [PI * px.cos() * py.sin(), PI * px.sin() * py.cos()];
    let advection = u[0] * grad[0] + u[1] * grad[1];
    (K - 1.0 + 2.0 * PI * PI * D) * phi(x) + advection / POROSITY
}

/// The time step of the convergence tests: a power of 2, so that the times `n DT` are exact.
const DT: f64 = 1.0 / 64.0;

/// The refinements of the convergence tests.
const REFINEMENTS: [u32; 3] = [1, 2, 3];

/// The L2 errors at t = 1 of `c = sin(pi x) sin(pi y) e^-t` (zero on the four sides) on each of
/// [`REFINEMENTS`], carried by the uniform flow `a (1, 0.5)` with `a` for the cell Peclet
/// number `peclet` on the mesh refined `at` times (the speed is the same on every mesh, so the
/// Peclet number halves with each refinement). The source is set before each step to its value
/// at the end of the step, where the BDF2 residual takes it.
fn advection_errors(p: &Petsc, test: &str, peclet: f64, at: u32) -> Vec<f64> {
    let a = speed(peclet, D, at) / 1.25_f64.sqrt();
    let u = [a, 0.5 * a];
    let scratch = Scratch::new(test);
    let boundaries = SIDES.iter().map(|side| dirichlet(side, 0.0)).collect();
    let field = field(D, K, POROSITY, boundaries);
    REFINEMENTS
        .iter()
        .map(|&refinements| {
            let mesh = unit_square(p, refinements);
            let maps = direction_maps(&scratch, &mesh);
            let flow = diagonal_flow(&mesh, &maps, a);
            let inputs = Inputs {
                maps: &maps,
                flow: Some(&flow),
            };
            let mut problem =
                Problem::new(&mesh, &field, &fixed(1.0, DT), &ceed(), ACCURATE, &inputs).unwrap();
            problem.set_initial_condition(phi).unwrap();
            let points = problem.discretization().quadrature_points().to_vec();
            let source: Vec<f64> = points.iter().map(|&x| manufactured_source(u, x)).collect();
            let steps = (1.0 / DT) as u32;
            for n in 1..=steps {
                let t = f64::from(n) * DT;
                for (f, g) in problem.source_mut().iter_mut().zip(&source) {
                    *f = g * (-t).exp();
                }
                problem.step_to(t).unwrap();
            }
            assert_eq!(problem.time(), 1.0);
            problem.l2_error(|x, t| phi(x) * (-t).exp()).unwrap()
        })
        .collect()
}

/// Asserts the orders of `errors` between successive refinements are at least `min`.
fn assert_orders(errors: &[f64], min: f64) {
    let orders: Vec<f64> = errors.windows(2).map(|w| (w[0] / w[1]).log2()).collect();
    println!("errors {errors:?}, orders {orders:?}");
    for order in &orders {
        assert!(
            *order >= min,
            "order {order} below {min}: errors {errors:?}"
        );
    }
}

/// Cell Peclet numbers 0.5, 0.25, 0.125: below 1 on every mesh, where the rate is that of the
/// Galerkin method, 2 (SUPG's `tau` scales with `h^2`). With 0.5 on the finest mesh instead,
/// the coarser ones are at 1 and 2, between the regimes, where the stabilization adds an error
/// of order `h`: the orders were 1.77 and 1.81 (1.79 and 1.89 with 4 times smaller steps).
#[test]
fn orders_low_peclet() {
    let p = Petsc::init(&[]).unwrap();
    assert_orders(&advection_errors(&p, "orders_low_peclet", 0.5, 1), 1.9);
}

/// Cell Peclet numbers 400, 200, 100. SUPG's rate in the L2 norm for advection-dominated flow
/// is between 1 and 2 (1.5 in theory for general meshes); for this smooth solution the orders
/// were 2.19 and 2.11. The bound 1.4 sits below the theoretical 1.5 and catches a loss of the
/// stabilization's consistency (order 1).
#[test]
fn orders_high_peclet() {
    let p = Petsc::init(&[]).unwrap();
    assert_orders(&advection_errors(&p, "orders_high_peclet", 100.0, 3), 1.4);
}

/// The refinements of the front.
const FRONT_REFINEMENTS: u32 = 3;

/// The time step of the front: the Courant number `|u| dt / h` is 0.25.
const FRONT_DT: f64 = 1.0 / 128.0;

/// The front of [`front_without_oscillations`] at t = 0.5: a concentration of 1 entering
/// through `left` into a concentration of 0, carried by the flow `(1, 0)` with the cell Peclet
/// number 100, porosity 1, no decay; returns the problem and its mesh.
fn front(p: &Petsc, test: &str) -> (Problem, Mesh) {
    let mesh = unit_square(p, FRONT_REFINEMENTS);
    let scratch = Scratch::new(test);
    let maps = direction_maps(&scratch, &mesh);
    let a = 1.0;
    let flow = flow_along_x(&mesh, &maps, a);
    let d = a * mesh_size(FRONT_REFINEMENTS) / (2.0 * 100.0);
    let field = field(d, 0.0, 1.0, vec![dirichlet("left", 1.0)]);
    let inputs = Inputs {
        maps: &maps,
        flow: Some(&flow),
    };
    let mut problem =
        Problem::new(&mesh, &field, &fixed(0.5, FRONT_DT), &ceed(), "", &inputs).unwrap();
    problem.step_to(0.5).unwrap();
    (problem, mesh)
}

/// The front of SUPG with BDF2: its oscillations at the cell Peclet number 100, where the
/// physical width of the front is below the mesh size. SUPG reduces them (the Galerkin method
/// overshoots by 18% at the Courant number 0.5) but, being linear, does not remove them; they
/// trail the front (the phase lag of BDF2) and lead it at small steps (the phase lead of the
/// consistent mass). Measured at t = 0.5 on this mesh, min and max: Courant number 0.5:
/// -0.0004, 1.130; 0.33: -0.011, 1.092; 0.25: -0.033, 1.066; 0.2: -0.048, 1.053; 0.125: -0.066,
/// 1.041. No step keeps both within 5%; the bounds are 5% below and 10% above.
#[test]
fn front_without_oscillations() {
    let p = Petsc::init(&[]).unwrap();
    let (problem, _mesh) = front(&p, "front_without_oscillations");
    let values = problem
        .local_solution()
        .unwrap()
        .with_array(<[f64]>::to_vec)
        .unwrap();
    let min = values.iter().copied().fold(f64::INFINITY, f64::min);
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    println!("front at t = 0.5: min {min}, max {max}");
    assert!(min >= -0.05, "undershoot {min}");
    assert!(max <= 1.10, "overshoot {max}");
    // The front is at x = 0.5: half of the unit square is filled.
    let mass = problem.total_mass().unwrap();
    assert!((mass - 0.5).abs() < 0.02, "{mass}");
}

#[test]
fn inflow_without_dirichlet_is_error() {
    let p = Petsc::init(&[]).unwrap();
    let mesh = unit_square(&p, 1);
    let scratch = Scratch::new("inflow_without_dirichlet_is_error");
    let maps = direction_maps(&scratch, &mesh);
    let flow = flow_along_x(&mesh, &maps, 1.0);
    let inputs = Inputs {
        maps: &maps,
        flow: Some(&flow),
    };
    // A condition at the outflow does not count.
    let field = field(D, 0.0, 1.0, vec![dirichlet("right", 0.0)]);
    let e = Problem::new(&mesh, &field, &fixed(1.0, 0.1), &ceed(), "", &inputs).unwrap_err();
    let PdeError::Inflow { tag, field: id } = &e else {
        panic!("expected an inflow error, got {e}");
    };
    assert_eq!((tag.as_str(), id.as_str()), ("left", "c"));
    assert_eq!(
        e.to_string(),
        "tag 'left' has inflow; field 'c' with advection needs a dirichlet condition there"
    );
    // A flux is not a dirichlet condition.
    let flux = BoundarySpec {
        tag: "left".to_string(),
        condition: Condition::Flux { value: 1.0 },
    };
    let field = FieldSpec {
        boundaries: vec![flux],
        ..field
    };
    let e = Problem::new(&mesh, &field, &fixed(1.0, 0.1), &ceed(), "", &inputs).unwrap_err();
    assert!(matches!(e, PdeError::Inflow { .. }), "{e}");
    // Without advection the flow does not matter.
    let field = FieldSpec {
        advection: false,
        ..field
    };
    Problem::new(&mesh, &field, &fixed(1.0, 0.1), &ceed(), "", &inputs).unwrap();
}

/// A tag whose net flow is rounding (here `top` and `bottom` of the flow along x, with the
/// pressure fixed on all four sides) has no inflow, whatever the sign of its rounding: only
/// `left` needs the dirichlet condition. (The sign of the rounding depends on the build and the
/// solver; the unit test of `Problem`'s inflow check covers a positive residue for certain.)
#[test]
fn rounding_flow_is_not_inflow() {
    let p = Petsc::init(&[]).unwrap();
    let mesh = unit_square(&p, 1);
    let scratch = Scratch::new("rounding_flow_is_not_inflow");
    let maps = direction_maps(&scratch, &mesh);
    let boundaries: Vec<(&str, f64)> = SIDES.iter().map(|&side| (side, 0.0)).collect();
    let pressure = |[x, _]: [f64; 2]| 1.0 - x;
    let flow = Flow::solve_manufactured(
        &mesh,
        &perfusion([1.0, 1.0], &boundaries),
        &maps,
        &ceed(),
        "",
        &pressure,
        &|_| 0.0,
    )
    .unwrap();
    let flows = flow.tag_flows();
    println!("{flows:?}");
    let rounding = flows
        .iter()
        .filter(|(tag, _)| tag == "top" || tag == "bottom")
        .map(|&(_, q)| q);
    assert!(rounding.clone().all(|q| q.abs() < 1e-10), "{flows:?}");
    let inputs = Inputs {
        maps: &maps,
        flow: Some(&flow),
    };
    let field = field(D, 0.0, 1.0, vec![dirichlet("left", 1.0)]);
    Problem::new(&mesh, &field, &fixed(1.0, 0.1), &ceed(), "", &inputs).unwrap();
}

#[test]
fn advection_without_flow_is_unsupported() {
    let p = Petsc::init(&[]).unwrap();
    let mesh = unit_square(&p, 0);
    let maps = Maps::default();
    let inputs = Inputs {
        maps: &maps,
        flow: None,
    };
    let field = field(D, 0.0, 1.0, vec![dirichlet("left", 1.0)]);
    let e = Problem::new(&mesh, &field, &fixed(1.0, 0.1), &ceed(), "", &inputs).unwrap_err();
    assert!(matches!(e, PdeError::Unsupported(_)), "{e}");
    for word in ["advection", "'c'", "flow"] {
        assert!(e.to_string().contains(word), "{word}: {e}");
    }
}

/// `tau` of SUPG, from the velocity and the cells: zero where the velocity is zero, and for
/// pure advection half the length of the cell along the flow over the speed, which halves with
/// each uniform refinement (the cells of a refinement are similar to their parent, at half the
/// size).
#[test]
fn supg_tau_of_pure_advection() {
    let p = Petsc::init(&[]).unwrap();
    let ceed = ceed();
    let mut extremes = Vec::new();
    for refinements in [0, 1] {
        let mesh = unit_square(&p, refinements);
        let level = mesh.levels().len() - 1;
        let disc = Discretization::new(&mesh, level, &ceed, 1).unwrap();
        let n = disc.num_quadrature_points();
        // Zero velocity in the left half of the square.
        let velocity: Vec<[f64; 2]> = disc
            .quadrature_points()
            .iter()
            .map(|x| if x[0] < 0.5 { [0.0; 2] } else { [2.0, 1.0] })
            .collect();
        let advection = |velocity: Vec<[f64; 2]>| {
            PointData::new(vec![1.0; n], vec![0.0; n], vec![[0.0; 3]; n])
                .with_velocity(velocity)
                .stabilized(&disc)
        };
        let half = advection(velocity.clone());
        for (t, u) in half.tau().iter().zip(&velocity) {
            if u[0] == 0.0 {
                assert_eq!(*t, 0.0);
            } else {
                assert!(*t > 0.0);
            }
        }
        let tau = advection(vec![[2.0, 1.0]; n]).tau().to_vec();
        // The length along the flow, 2 |u| tau, is at most the size of the cells.
        let speed = 5.0_f64.sqrt();
        let longest = tau.iter().fold(0.0_f64, |m, &t| m.max(2.0 * speed * t));
        assert!(longest <= 2.0 * mesh_size(refinements), "{longest}");
        // On one process, all the cells: the extremes over the mesh.
        if p.comm().size() == 1 {
            let shortest = tau.iter().fold(f64::INFINITY, |m, &t| m.min(t));
            extremes.push([shortest, tau.iter().fold(0.0_f64, |m, &t| m.max(t))]);
        }
    }
    if let [coarse, fine] = extremes.as_slice() {
        for (c, f) in coarse.iter().zip(fine) {
            assert!((c / f - 2.0).abs() < 1e-12, "{coarse:?} -> {fine:?}");
        }
    }
}

/// `storage_rate - source - sum(boundary) + advection_defect`, relative to the largest of the
/// terms.
fn imbalance(balance: &pde::MassBalance) -> f64 {
    let boundary: f64 = balance.boundary.iter().map(BoundaryInflow::total).sum();
    let residual = balance.storage_rate - balance.source - boundary + balance.advection_defect;
    let scale = balance
        .boundary
        .iter()
        .flat_map(|b| [b.advective.abs(), b.diffusive.abs()])
        .chain([
            balance.storage_rate.abs(),
            balance.source.abs(),
            balance.advection_defect.abs(),
        ])
        .fold(0.0, f64::max);
    residual.abs() / scale
}

#[test]
fn mass_balance_closes() {
    let p = Petsc::init(&[]).unwrap();
    let mesh = unit_square(&p, 2);
    let scratch = Scratch::new("mass_balance_closes");
    let maps = direction_maps(&scratch, &mesh);
    // Anisotropic permeability, no flow through top and bottom: a flow which is not uniform,
    // whose normal component jumps between the cells.
    let perfusion = perfusion([3.0, 1.0], &[("left", 1.0), ("right", 0.0)]);
    let flow = Flow::solve(&mesh, &perfusion, &maps, &ceed(), "").unwrap();
    let inputs = Inputs {
        maps: &maps,
        flow: Some(&flow),
    };
    let flux = BoundarySpec {
        tag: "top".to_string(),
        condition: Condition::Flux { value: 0.1 },
    };
    let field = FieldSpec {
        source: Coefficient::Constant(0.3),
        ..field(D, 0.2, POROSITY, vec![dirichlet("left", 1.0), flux])
    };
    let mut problem = Problem::new(
        &mesh,
        &field,
        &fixed(0.5, 1.0 / 32.0),
        &ceed(),
        CONSERVATIVE,
        &inputs,
    )
    .unwrap();
    // Before the first step there is no rate of change.
    let e = problem.mass_balance().unwrap_err();
    assert!(matches!(e, PdeError::NotStepped { .. }), "{e}");
    for t in [1.0 / 32.0, 0.25, 0.5] {
        problem.step_to(t).unwrap();
        let balance = problem.mass_balance().unwrap();
        println!(
            "t = {t}: {balance:?}, advection_defect {:e}",
            balance.advection_defect
        );
        let imbalance = imbalance(&balance);
        assert!(imbalance < 1e-8, "t = {t}: imbalance {imbalance:e}");
        // The tags: those of the conditions first, then the other edges of the boundary.
        let mut tags: Vec<&str> = balance.boundary.iter().map(|b| b.tag.as_str()).collect();
        assert_eq!(tags[..2], ["left", "top"]);
        tags.sort_unstable();
        assert_eq!(tags, ["bottom", "left", "right", "top"]);
        let inflow = |tag: &str| balance.boundary.iter().find(|b| b.tag == tag).unwrap();
        assert!(inflow("left").total() > 0.0, "{balance:?}");
        assert!(inflow("left").advective > 0.0, "{balance:?}");
        assert!(inflow("right").total() < 0.0, "{balance:?}");
        // The diffusive parts: the consistent flux at the Dirichlet tag, the flux of the
        // condition, none at the free outflow; the advective part of a flux tag is outflow
        // only.
        assert_eq!(inflow("right").diffusive, 0.0, "{balance:?}");
        assert_eq!(inflow("bottom").diffusive, 0.0, "{balance:?}");
        assert!((inflow("top").diffusive - 0.1).abs() < 1e-12, "{balance:?}");
        for tag in ["right", "top", "bottom"] {
            assert!(inflow(tag).advective <= 0.0, "{tag}: {balance:?}");
        }
        // The P1 velocity has a normal component on the sides without flow (the condition is
        // natural, so weak), and jumps of it between the cells.
        assert!(balance.advection_defect.abs() > 1e-3, "{balance:?}");
    }
    // A new initial condition: no rate of change until the next step.
    problem.set_initial_condition(|_| 0.0).unwrap();
    assert!(matches!(
        problem.mass_balance().unwrap_err(),
        PdeError::NotStepped { .. }
    ));
}

/// The mass balance of a uniform flow, which P1 represents exactly: no advection defect, no
/// flow through `top` and `bottom` (whose terms are the flux of the condition, and zero).
#[test]
fn mass_balance_of_a_uniform_flow() {
    let p = Petsc::init(&[]).unwrap();
    let mesh = unit_square(&p, 2);
    let scratch = Scratch::new("mass_balance_of_a_uniform_flow");
    let maps = direction_maps(&scratch, &mesh);
    let flow = flow_along_x(&mesh, &maps, 1.0);
    let inputs = Inputs {
        maps: &maps,
        flow: Some(&flow),
    };
    let flux = BoundarySpec {
        tag: "top".to_string(),
        condition: Condition::Flux { value: 0.1 },
    };
    let field = field(D, 0.2, POROSITY, vec![flux, dirichlet("left", 1.0)]);
    let mut problem = Problem::new(
        &mesh,
        &field,
        &fixed(0.25, 1.0 / 32.0),
        &ceed(),
        CONSERVATIVE,
        &inputs,
    )
    .unwrap();
    problem.step_to(0.25).unwrap();
    let balance = problem.mass_balance().unwrap();
    println!("{balance:?}");
    assert!(imbalance(&balance) < 1e-8, "{balance:?}");
    let scale = balance.storage_rate.abs();
    assert!(
        balance.advection_defect.abs() < 1e-10 * scale,
        "{balance:?}"
    );
    let tags: Vec<&str> = balance.boundary.iter().map(|b| b.tag.as_str()).collect();
    assert_eq!(tags[..2], ["top", "left"]);
    let small = |x: f64| x.abs() < 1e-10 * scale;
    for b in &balance.boundary {
        let q = b.total();
        match b.tag.as_str() {
            "top" => assert!(small(q - 0.1) && small(b.advective), "{balance:?}"),
            "bottom" => assert!(small(q), "{balance:?}"),
            "left" => assert!(q > 0.0 && b.advective > 0.0, "{balance:?}"),
            "right" => assert!(q < 0.0 && b.diffusive == 0.0, "{balance:?}"),
            tag => panic!("tag {tag}: {balance:?}"),
        }
    }
}

/// `beta` of [`backflow_flow`].
const BETA: f64 = 2.0;

/// The divergence free flow of the harmonic pressure
/// `p = -x - beta sinh(pi x) cos(pi y) / (pi cosh pi)` with `k = mu = 1`:
/// `u = (1 + beta cosh(pi x) cos(pi y) / cosh pi, beta sinh(pi x) sin(pi y) / cosh pi)`, the
/// pressure fixed on `left` and `right` (no flow through `top` and `bottom`, as `u_y` is zero
/// there). It enters through `left`, and leaves through `right` (net flow 1) except where
/// `cos(pi y) < -1 / beta` (`y > 2/3`), where it comes back in.
fn backflow_flow(mesh: &Mesh, maps: &Maps) -> Flow {
    let pressure =
        |[x, y]: [f64; 2]| -x - BETA * (PI * x).sinh() * (PI * y).cos() / (PI * PI.cosh());
    Flow::solve_manufactured(
        mesh,
        &perfusion([1.0, 1.0], &[("left", 0.0), ("right", 0.0)]),
        maps,
        &ceed(),
        "",
        &pressure,
        &|_| 0.0,
    )
    .unwrap()
}

/// Backflow through a side without condition: at the Gauss points of `right` with
/// `u . n < 0` the boundary term lets no advective flux in, so the fluid which comes back in
/// carries no substance. With the concentration 1 at the start and at the inflow, only the
/// backflow changes it: the discrete energy `int c^2` falls step by step (by the flux of the
/// backflow, `int (u . n)^- c^2`; without the term it would not fall), the concentration stays
/// bounded, and the mass balance closes.
#[test]
fn backflow_without_condition() {
    let p = Petsc::init(&[]).unwrap();
    let mesh = unit_square(&p, 2);
    let scratch = Scratch::new("backflow_without_condition");
    let maps = direction_maps(&scratch, &mesh);
    let flow = backflow_flow(&mesh, &maps);
    let finest = mesh.levels().len() - 1;
    // The flow comes back in through part of `right`.
    let back = flow
        .edge_velocity(finest)
        .iter()
        .flat_map(|(_, u)| u.iter())
        .filter(|u| u[0] < -0.1)
        .count();
    let back = p.comm().sum_int(back as petsc::sys::PetscInt).unwrap();
    assert!(back > 0, "no backflow");
    let flows = flow.tag_flows();
    assert!(flows[0].1 > 0.0 && flows[1].1 < 0.0, "{flows:?}");
    let inputs = Inputs {
        maps: &maps,
        flow: Some(&flow),
    };
    let field = FieldSpec {
        initial: 1.0,
        ..field(D, 0.0, 1.0, vec![dirichlet("left", 1.0)])
    };
    let dt = 1.0 / 32.0;
    let mut problem = Problem::new(
        &mesh,
        &field,
        &fixed(1.0, dt),
        &ceed(),
        CONSERVATIVE,
        &inputs,
    )
    .unwrap();
    let initial = problem.l2_norm().unwrap();
    let mut energy = initial;
    let (mut min, mut max) = (f64::INFINITY, f64::NEG_INFINITY);
    for n in 1..=32 {
        problem.step_to(f64::from(n) * dt).unwrap();
        let next = problem.l2_norm().unwrap();
        assert!(next <= energy, "step {n}: energy {next} after {energy}");
        energy = next;
        let balance = problem.mass_balance().unwrap();
        let imbalance = imbalance(&balance);
        assert!(imbalance < 1e-8, "step {n}: {imbalance:e}: {balance:?}");
        let values = problem
            .local_solution()
            .unwrap()
            .with_array(<[f64]>::to_vec)
            .unwrap();
        for v in values {
            min = min.min(v);
            max = max.max(v);
        }
    }
    println!("L2 norm at t = 1: {energy}, at t = 0: {initial}; min {min}, max {max}");
    assert!(energy < 0.99 * initial, "{energy}");
    // Bounded: the backflow makes a layer at `right` (the substance-free fluid coming in), where
    // SUPG oscillates as at the front (measured min -0.104, max 1.043).
    assert!(min >= -0.15 && max <= 1.1, "min {min}, max {max}");
}

/// The backflow term is in the operator as PETSc sees it: its products, its diagonal and its
/// assembled matrix (the coarse level of the multigrid) agree.
#[test]
fn backflow_term_in_mult_diagonal_and_assembly() {
    let p = Petsc::init(&[]).unwrap();
    let mesh = unit_square(&p, 1);
    let scratch = Scratch::new("backflow_term_in_mult_diagonal_and_assembly");
    let maps = direction_maps(&scratch, &mesh);
    let flow = backflow_flow(&mesh, &maps);
    let level = mesh.levels().len() - 1;
    let disc = Discretization::with_essential(&mesh, level, &ceed(), 1, &["left"]).unwrap();
    let n = disc.num_quadrature_points();
    let data = PointData::isotropic(&disc, &vec![1.0; n], &vec![0.1; n], &vec![D; n])
        .with_velocity(flow.velocity(level).to_vec())
        .stabilized(&disc)
        .with_backflow(&disc, flow.edge_velocity(level), &["left"])
        .unwrap();
    let entries = p
        .comm()
        .sum_int(data.backflow().len() as petsc::sys::PetscInt)
        .unwrap();
    assert!(entries > 0);
    let mut op = Transport::new(&disc, Rc::new(data), Terms::Jacobian).unwrap();
    op.shift = 3.0;
    let assembled = op.assemble().unwrap();
    let shell = disc.shell_matrix(op).unwrap();
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
    let mut y = disc.global_vector().unwrap();
    shell.mult(&x, &mut y).unwrap();
    y.axpy(-1.0, &expected).unwrap();
    let error = y.norm2().unwrap() / expected.norm2().unwrap();
    assert!(error < 1e-13, "relative |A x - A_coo x| = {error}");
    let mut diagonal = disc.global_vector().unwrap();
    shell.diagonal(&mut diagonal).unwrap();
    let mut expected = disc.global_vector().unwrap();
    assembled.diagonal(&mut expected).unwrap();
    diagonal.axpy(-1.0, &expected).unwrap();
    let error = diagonal.norm2().unwrap() / expected.norm2().unwrap();
    assert!(error < 1e-13, "relative |diag - diag(A)| = {error}");
}

/// The total mass is the amount of substance, `int phi c`: the porosity weighs it.
#[test]
fn total_mass_weighs_the_porosity() {
    let p = Petsc::init(&[]).unwrap();
    let mesh = unit_square(&p, 1);
    let field = FieldSpec {
        initial: 2.0,
        advection: false,
        ..field(D, 0.0, 0.25, Vec::new())
    };
    let maps = Maps::default();
    let inputs = Inputs {
        maps: &maps,
        flow: None,
    };
    let problem = Problem::new(&mesh, &field, &fixed(1.0, 0.1), &ceed(), "", &inputs).unwrap();
    assert!((problem.total_mass().unwrap() - 0.5).abs() < 1e-12);
}

/// The results `parallel_on_*_processes` compares between process counts: the L2 norm and the
/// total mass of the front.
fn parity_results(p: &Petsc) -> Vec<f64> {
    let (problem, _mesh) = front(p, "parity");
    vec![problem.l2_norm().unwrap(), problem.total_mass().unwrap()]
}

/// Runs [`parity_results`] on one process, then the test `test` itself on `ranks` processes
/// with them in [`REFERENCE`], where it compares its results with them: within `1e-10`
/// relative. Then other tests of advection on `ranks` processes.
fn parity(ranks: usize, test: &str) {
    let p = Petsc::init(&[]).unwrap();
    let results = parity_results(&p);
    match std::env::var(REFERENCE) {
        Err(_) => {
            assert_eq!(test_support::ranks(), 1);
            // MPI is finalized before mpiexec starts.
            drop(p);
            let reference: Vec<String> = results.iter().map(|r| format!("{r:e}")).collect();
            let reference = reference.join(" ");
            println!("on one process: {reference}");
            test_support::parallel_with_env(ranks, &[test], &[(REFERENCE, &reference)]);
            test_support::parallel(
                ranks,
                &[
                    "front_without_oscillations",
                    "inflow_without_dirichlet_is_error",
                    "mass_balance_closes",
                    "mass_balance_of_a_uniform_flow",
                    "backflow_without_condition",
                    "backflow_term_in_mult_diagonal_and_assembly",
                    "supg_tau_of_pure_advection",
                ],
            );
        }
        Ok(reference) => {
            let reference: Vec<f64> = reference.split(' ').map(|v| v.parse().unwrap()).collect();
            assert_eq!(results.len(), reference.len());
            for (i, (r, one)) in results.iter().zip(&reference).enumerate() {
                let difference = ((r - one) / one).abs();
                println!(
                    "result {i} on {} processes: {r:e}, on one: {one:e}, relative difference \
                     {difference:e}",
                    test_support::ranks()
                );
                assert!(difference <= 1e-10, "result {i}: {r} against {one}");
            }
        }
    }
}

#[test]
fn parallel_on_2_processes() {
    parity(2, "parallel_on_2_processes");
}

#[test]
fn parallel_on_3_processes() {
    parity(3, "parallel_on_3_processes");
}
