//! Tests of the steady Darcy perfusion [`Flow`] on the unit square: a uniform flow, which P1
//! represents exactly; convergence orders of pressure and velocity against manufactured
//! solutions, with isotropic and anisotropic permeability; the permeability where the direction
//! map has no gradient; and errors. nextest runs each test in its own process, so every test
//! initializes PETSc once. `parallel_on_2_processes` and `parallel_on_3_processes` run some of
//! them again under `mpiexec`, and compare the results with those of one process.
//!
//! The direction maps are test Zarr arrays over `unit_square.msh` (see `common`).

mod common;

use std::f64::consts::PI;

use common::{PIXELS, Scratch, ceed, read_maps, unit_square};
use contracts::{Coefficient, Direction, Perfusion, PressureBoundary};
use mesh::{Mesh, MeshError};
use pde::{Discretization, Flow, PdeError};
use petsc::Petsc;

/// The four sides of `unit_square.msh`.
const SIDES: [&str; 4] = ["left", "right", "top", "bottom"];

/// The refinements of the convergence tests: 4 levels, h from 1/8 to 1/64 (the mesh of
/// `unit_square.msh` has size 0.25).
const REFINEMENTS: [u32; 4] = [1, 2, 3, 4];

/// Set in the runs of `parallel_on_*_processes` under `mpiexec`: the results on one process.
const REFERENCE: &str = "SLIDESIM_TEST_FLOW_REFERENCE";

/// The perfusion with viscosity `viscosity` (Pa*s), constant permeabilities `k_along` and
/// `k_across` (um^2) along the gradient of the map `direction` with `g0` (1/um), and the
/// pressures (Pa) of `boundaries`.
fn perfusion(
    viscosity: f64,
    [k_along, k_across]: [f64; 2],
    g0: f64,
    boundaries: &[(&str, f64)],
) -> Perfusion {
    Perfusion {
        viscosity,
        k_along: Coefficient::Constant(k_along),
        k_across: Coefficient::Constant(k_across),
        direction: Direction {
            map: "direction".to_string(),
            g0,
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

/// The flow of `perfusion` on `mesh` with the direction map `direction`, solved with the
/// default options.
fn solve(
    test: &str,
    mesh: &Mesh,
    perfusion: &Perfusion,
    direction: impl Fn([f64; 2]) -> f64,
) -> Result<Flow, PdeError> {
    let scratch = Scratch::new(test);
    let maps = read_maps(&[scratch.map("direction", PIXELS, direction)], mesh);
    Flow::solve(mesh, perfusion, &maps, &ceed(), "")
}

/// The L2 norm of `velocity - exact` at the quadrature points of level `level` of `mesh`, by
/// the quadrature of a discretization of the level. Collective.
fn velocity_l2_error(
    mesh: &Mesh,
    level: usize,
    velocity: &[[f64; 2]],
    exact: impl Fn([f64; 2]) -> [f64; 2],
) -> f64 {
    let disc = Discretization::new(mesh, level, &ceed(), 1).unwrap();
    assert_eq!(velocity.len(), disc.num_quadrature_points());
    let local: f64 = velocity
        .iter()
        .zip(disc.quadrature_points())
        .zip(disc.quadrature_weights())
        .map(|((u, &x), w)| {
            let e = exact(x);
            w * ((u[0] - e[0]).powi(2) + (u[1] - e[1]).powi(2))
        })
        .sum();
    disc.dm().comm().sum_real(local).unwrap().sqrt()
}

/// Asserts that the velocity of `flow` is `expected` at every quadrature point of every level
/// of `mesh`, within `1e-10`.
fn assert_uniform_velocity(mesh: &Mesh, flow: &Flow, expected: [f64; 2]) {
    for level in 0..mesh.levels().len() {
        let disc = Discretization::new(mesh, level, &ceed(), 1).unwrap();
        let velocity = flow.velocity(level);
        assert_eq!(
            velocity.len(),
            disc.num_quadrature_points(),
            "level {level}"
        );
        for (u, x) in velocity.iter().zip(disc.quadrature_points()) {
            for c in 0..2 {
                assert!(
                    (u[c] - expected[c]).abs() < 1e-10,
                    "level {level} at {x:?}: {u:?}, expected {expected:?}"
                );
            }
        }
    }
}

/// The flow from `left` (pressure 1 Pa) to `right` (0) with isotropic `k = 2` um^2 and
/// `mu = 1` Pa*s on the unit square at 2 refinements: `p = 1 - x`, `u = (2, 0)` um/s.
fn uniform(p: &Petsc) -> (Mesh, Flow) {
    let mesh = unit_square(p, 2);
    let perfusion = perfusion(1.0, [2.0, 2.0], 1e-6, &[("left", 1.0), ("right", 0.0)]);
    let flow = solve("uniform_flow", &mesh, &perfusion, |[x, y]| x + 0.5 * y).unwrap();
    (mesh, flow)
}

#[test]
fn uniform_flow() {
    let p = Petsc::init(&[]).unwrap();
    let (mesh, flow) = uniform(&p);
    assert_uniform_velocity(&mesh, &flow, [2.0, 0.0]);

    // Through the left side 2 um^2/s flow in, through the right side out.
    let flows = flow.tag_flows();
    let tags: Vec<&str> = flows.iter().map(|(tag, _)| tag.as_str()).collect();
    assert_eq!(tags, ["left", "right"]);
    for ((tag, q), expected) in flows.iter().zip([2.0, -2.0]) {
        assert!((q - expected).abs() < 1e-10, "{tag}: {q}");
    }

    // The pressure at the nodes, the Dirichlet values included.
    let disc = flow.discretization();
    let nodes = disc.node_coordinates();
    let pressure = flow.local_pressure().unwrap();
    pressure
        .with_array(|values| {
            assert_eq!(values.len(), nodes.len());
            for (v, x) in values.iter().zip(nodes) {
                assert!((v - (1.0 - x[0])).abs() < 1e-10, "at {x:?}: {v}");
            }
        })
        .unwrap();
    assert!(flow.pressure_l2_error(|[x, _]| 1.0 - x).unwrap() < 1e-10);

    // The velocity projected to the nodes: 2 components per node.
    let velocity = flow.nodal_velocity().unwrap();
    velocity
        .with_array(|values| {
            assert_eq!(values.len(), 2 * nodes.len());
            for u in values.as_chunks::<2>().0 {
                assert!((u[0] - 2.0).abs() < 1e-10 && u[1].abs() < 1e-10, "{u:?}");
            }
        })
        .unwrap();
}

/// A manufactured pressure on the unit square, Dirichlet values on all four sides.
struct Manufactured {
    viscosity: f64,
    /// `k_along`, `k_across`.
    k: [f64; 2],
    /// The direction map.
    direction: fn([f64; 2]) -> f64,
    /// The constant permeability tensor of `k` and `direction`, `[K00, K01, K11]`.
    tensor: [f64; 3],
    pressure: fn([f64; 2]) -> f64,
    gradient: fn([f64; 2]) -> [f64; 2],
    /// The second derivatives `[p_xx, p_xy, p_yy]`.
    hessian: fn([f64; 2]) -> [f64; 3],
}

/// `g0` of the manufactured solutions: far below the gradient of their direction maps.
const G0: f64 = 1e-6;

impl Manufactured {
    /// The harmonic `p = sin(pi x / 2) cosh(pi y / 2)` with isotropic `k = 2`, `mu = 1`.
    fn isotropic() -> Manufactured {
        Manufactured {
            viscosity: 1.0,
            k: [2.0, 2.0],
            direction: |[x, y]| x + 0.5 * y,
            tensor: [2.0, 0.0, 2.0],
            pressure: |[x, y]| (PI * x / 2.0).sin() * (PI * y / 2.0).cosh(),
            gradient: |[x, y]| {
                let (a, b) = (PI * x / 2.0, PI * y / 2.0);
                [PI / 2.0 * a.cos() * b.cosh(), PI / 2.0 * a.sin() * b.sinh()]
            },
            hessian: |[x, y]| {
                let (a, b) = (PI * x / 2.0, PI * y / 2.0);
                let c = PI * PI / 4.0;
                [
                    -c * a.sin() * b.cosh(),
                    c * a.cos() * b.sinh(),
                    c * a.sin() * b.cosh(),
                ]
            },
        }
    }

    /// `p = sin(pi x) sin(pi y) + x` with `k_along = 3`, `k_across = 1` along the gradient
    /// `g = (1, 0.5)` of the map `x + 0.5 y`, `mu = 0.5`:
    /// `K = I + 2 g g^T / (|g|^2 + g0^2)`.
    fn anisotropic() -> Manufactured {
        let g = [1.0, 0.5];
        let s = 2.0 / (g[0] * g[0] + g[1] * g[1] + G0 * G0);
        Manufactured {
            viscosity: 0.5,
            k: [3.0, 1.0],
            direction: |[x, y]| x + 0.5 * y,
            tensor: [
                1.0 + s * g[0] * g[0],
                s * g[0] * g[1],
                1.0 + s * g[1] * g[1],
            ],
            pressure: |[x, y]| (PI * x).sin() * (PI * y).sin() + x,
            gradient: |[x, y]| {
                let (a, b) = (PI * x, PI * y);
                [PI * a.cos() * b.sin() + 1.0, PI * a.sin() * b.cos()]
            },
            hessian: |[x, y]| {
                let (a, b) = (PI * x, PI * y);
                let c = PI * PI;
                [
                    -c * a.sin() * b.sin(),
                    c * a.cos() * b.cos(),
                    -c * a.sin() * b.sin(),
                ]
            },
        }
    }

    /// The velocity `-K grad p / mu`.
    fn velocity(&self, x: [f64; 2]) -> [f64; 2] {
        let [k00, k01, k11] = self.tensor;
        let [px, py] = (self.gradient)(x);
        [
            -(k00 * px + k01 * py) / self.viscosity,
            -(k01 * px + k11 * py) / self.viscosity,
        ]
    }

    /// The source `-div(K grad p) / mu` (K constant).
    fn source(&self, x: [f64; 2]) -> f64 {
        let [k00, k01, k11] = self.tensor;
        let [pxx, pxy, pyy] = (self.hessian)(x);
        -(k00 * pxx + 2.0 * k01 * pxy + k11 * pyy) / self.viscosity
    }

    /// The flow on `mesh`, with Dirichlet values from the pressure on all four sides.
    fn solve(&self, test: &str, mesh: &Mesh) -> Flow {
        let boundaries: Vec<(&str, f64)> = SIDES.iter().map(|&side| (side, 0.0)).collect();
        let perfusion = perfusion(self.viscosity, self.k, G0, &boundaries);
        let scratch = Scratch::new(test);
        let maps = read_maps(&[scratch.map("direction", PIXELS, self.direction)], mesh);
        Flow::solve_manufactured(mesh, &perfusion, &maps, &ceed(), "", &self.pressure, &|x| {
            self.source(x)
        })
        .unwrap()
    }

    /// The L2 errors of pressure and velocity on the finest level at each of `REFINEMENTS`, and
    /// the results which `parallel_on_*_processes` compares between process counts: the L2
    /// norms of pressure and velocity, and the flows through the sides.
    fn errors(&self, p: &Petsc, test: &str) -> (Vec<[f64; 2]>, Vec<f64>) {
        let mut errors = Vec::new();
        let mut results = Vec::new();
        for refinements in REFINEMENTS {
            let mesh = unit_square(p, refinements);
            let flow = self.solve(test, &mesh);
            let finest = mesh.levels().len() - 1;
            let velocity = flow.velocity(finest);
            errors.push([
                flow.pressure_l2_error(self.pressure).unwrap(),
                velocity_l2_error(&mesh, finest, velocity, |x| self.velocity(x)),
            ]);
            results.push(flow.pressure_l2_error(|_| 0.0).unwrap());
            results.push(velocity_l2_error(&mesh, finest, velocity, |_| [0.0; 2]));
            let flows: Vec<f64> = flow.tag_flows().iter().map(|&(_, q)| q).collect();
            self.assert_conserved(&flow, &flows);
            results.extend(flows);
        }
        (errors, results)
    }

    /// Asserts that the flows `flows` through the four sides balance the source: the inflow
    /// through the whole boundary is `-int f`, by the quadrature of the load, up to the
    /// residual of the solve.
    fn assert_conserved(&self, flow: &Flow, flows: &[f64]) {
        let disc = flow.discretization();
        let local: f64 = disc
            .quadrature_points()
            .iter()
            .zip(disc.quadrature_weights())
            .map(|(&x, w)| w * self.source(x))
            .sum();
        let source = disc.dm().comm().sum_real(local).unwrap();
        let inflow: f64 = flows.iter().sum();
        let scale = flows.iter().fold(source.abs(), |m, q| m.max(q.abs()));
        assert!(
            (inflow + source).abs() <= 1e-8 * scale,
            "inflow {inflow}, source {source}"
        );
    }
}

/// Asserts the orders of the pressure errors (`errors[i][0]`) and the velocity errors
/// (`errors[i][1]`) between successive refinements are at least `2 - 0.1` and `1 - 0.1`.
fn assert_orders(errors: &[[f64; 2]]) {
    for pair in errors.windows(2) {
        let order = |i: usize| (pair[0][i] / pair[1][i]).log2();
        println!(
            "errors {:?} -> {:?}: orders {:.3} (pressure), {:.3} (velocity)",
            pair[0],
            pair[1],
            order(0),
            order(1)
        );
        assert!(order(0) >= 1.9, "pressure order {}", order(0));
        assert!(order(1) >= 0.9, "velocity order {}", order(1));
    }
}

#[test]
fn pressure_order_isotropic() {
    let p = Petsc::init(&[]).unwrap();
    let (errors, _) = Manufactured::isotropic().errors(&p, "pressure_order_isotropic");
    assert_orders(&errors);
}

#[test]
fn pressure_order_anisotropic() {
    let p = Petsc::init(&[]).unwrap();
    let (errors, _) = Manufactured::anisotropic().errors(&p, "pressure_order_anisotropic");
    assert_orders(&errors);
}

/// The velocity of every level is at the quadrature points of that level, in the layout of any
/// discretization of it (here one with the essential tags of a species): against the exact
/// velocity of the anisotropic manufactured flow there, the error falls with the mesh size
/// level by level, and is small, which a velocity of other points would not be.
#[test]
fn velocity_of_every_level_at_its_points() {
    let p = Petsc::init(&[]).unwrap();
    let manufactured = Manufactured::anisotropic();
    let mesh = unit_square(&p, 3);
    let flow = manufactured.solve("velocity_of_every_level_at_its_points", &mesh);
    let ceed = ceed();
    let mut errors = Vec::new();
    for level in 0..mesh.levels().len() {
        let species = Discretization::with_essential(&mesh, level, &ceed, 1, &["left"]).unwrap();
        let velocity = flow.velocity(level);
        assert_eq!(velocity.len(), species.num_quadrature_points());
        let local: f64 = velocity
            .iter()
            .zip(species.quadrature_points())
            .zip(species.quadrature_weights())
            .map(|((u, &x), w)| {
                let e = manufactured.velocity(x);
                w * ((u[0] - e[0]).powi(2) + (u[1] - e[1]).powi(2))
            })
            .sum();
        let error = species.dm().comm().sum_real(local).unwrap().sqrt();
        let norm = velocity_l2_error(&mesh, level, velocity, |_| [0.0; 2]);
        errors.push(error / norm);
    }
    println!("relative velocity errors by level: {errors:?}");
    for pair in errors.windows(2) {
        let order = (pair[0] / pair[1]).log2();
        assert!(order >= 0.9, "{errors:?}");
    }
    assert!(errors[1] < 0.15, "{errors:?}");
}

#[test]
fn isotropic_where_gradient_vanishes() {
    let p = Petsc::init(&[]).unwrap();
    let mesh = unit_square(&p, 1);
    let test = "isotropic_where_gradient_vanishes";
    let (k_along, k_across) = (3.0, 1.0);
    let k = [k_along, k_across];
    let along_x = perfusion(1.0, k, G0, &[("left", 1.0), ("right", 0.0)]);
    let along_y = perfusion(1.0, k, G0, &[("bottom", 1.0), ("top", 0.0)]);
    // A constant map: K = k_across I, whichever way the pressure falls.
    let flow = solve(test, &mesh, &along_x, |_| 0.5).unwrap();
    assert_uniform_velocity(&mesh, &flow, [k_across, 0.0]);
    let flow = solve(test, &mesh, &along_y, |_| 0.5).unwrap();
    assert_uniform_velocity(&mesh, &flow, [0.0, k_across]);
    // A map of x: k_along along x (w = 1 / (1 + g0^2)), k_across along y.
    let w = 1.0 / (1.0 + G0 * G0);
    let flow = solve(test, &mesh, &along_x, |[x, _]| x).unwrap();
    assert_uniform_velocity(&mesh, &flow, [k_across + w * (k_along - k_across), 0.0]);
    let flow = solve(test, &mesh, &along_y, |[x, _]| x).unwrap();
    assert_uniform_velocity(&mesh, &flow, [0.0, k_across]);
}

#[test]
fn divergence_is_error() {
    let p = Petsc::init(&[]).unwrap();
    let mesh = unit_square(&p, 2);
    let perfusion = perfusion(1.0, [3.0, 1.0], G0, &[("left", 1.0), ("right", 0.0)]);
    let scratch = Scratch::new("divergence_is_error");
    let maps = read_maps(
        &[scratch.map("direction", PIXELS, |[x, y]| x + 0.5 * y)],
        &mesh,
    );
    // One iteration does not reach the tolerance: the options are read under the prefix.
    let e = Flow::solve(&mesh, &perfusion, &maps, &ceed(), "-ksp_max_it 1").unwrap_err();
    let PdeError::SolveFailed { solve, reason } = &e else {
        panic!("expected a failed solve, got {e}");
    };
    assert_eq!(solve, "perfusion");
    assert!(reason.contains("DIVERGED_ITS"), "{e}");
    let message = e.to_string();
    assert!(
        message.starts_with("perfusion solve failed: KSP DIVERGED_ITS"),
        "{message}"
    );
    assert!(!message.contains("time integration"), "{message}");
    // The options lived as long as the solve.
    Flow::solve(&mesh, &perfusion, &maps, &ceed(), "").unwrap();
}

#[test]
fn unknown_pressure_tag_lists_tags() {
    let p = Petsc::init(&[]).unwrap();
    let mesh = unit_square(&p, 0);
    let perfusion = perfusion(1.0, [1.0, 1.0], G0, &[("left", 1.0), ("nowhere", 0.0)]);
    let e = solve(
        "unknown_pressure_tag_lists_tags",
        &mesh,
        &perfusion,
        |[x, _]| x,
    )
    .unwrap_err();
    assert!(
        matches!(&e, PdeError::Mesh(MeshError::UnknownTag { name, .. }) if name == "nowhere"),
        "{e}"
    );
    let message = e.to_string();
    for tag in ["nowhere", "left", "right", "top", "bottom", "domain"] {
        assert!(message.contains(tag), "{tag}: {message}");
    }
}

/// The results `parallel_on_*_processes` compares between process counts: of [`uniform`], the
/// L2 norms of the pressure and of the velocity on every level, and the flows through the
/// tags; and those of the anisotropic manufactured solution (see [`Manufactured::errors`]).
fn parity_results(p: &Petsc) -> Vec<f64> {
    let (mesh, flow) = uniform(p);
    let mut results = vec![flow.pressure_l2_error(|_| 0.0).unwrap()];
    for level in 0..mesh.levels().len() {
        results.push(velocity_l2_error(
            &mesh,
            level,
            flow.velocity(level),
            |_| [0.0; 2],
        ));
    }
    results.extend(flow.tag_flows().iter().map(|(_, q)| q));
    let (_, anisotropic) = Manufactured::anisotropic().errors(p, "parity_anisotropic");
    results.extend(anisotropic);
    results
}

/// Runs [`parity_results`] on one process, then the test `test` itself on `ranks` processes
/// with them in [`REFERENCE`], where it compares its results with them: within `1e-10`
/// relative. Then the other tests of the flow on `ranks` processes.
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
                    "uniform_flow",
                    "velocity_of_every_level_at_its_points",
                    "isotropic_where_gradient_vanishes",
                    "divergence_is_error",
                    "unknown_pressure_tag_lists_tags",
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
