//! Tests of the steady Darcy perfusion [`Flow`] on the unit square: a uniform flow, which P1
//! represents exactly; convergence orders of pressure and velocity (primal and conservative)
//! against manufactured solutions, with isotropic and anisotropic permeability, and on the tagged
//! mesh with holes `lobule_toy.msh` with a permeability which varies over the mesh; the
//! conservation of the edge fluxes; the permeability where the direction map has no gradient;
//! and errors. nextest runs each test in its own process, so
//! every test initializes PETSc once. `parallel_on_2_processes` and `parallel_on_3_processes`
//! run some of them again under `mpiexec`, and compare the results with those of one process.
//!
//! The direction maps are test Zarr arrays over `unit_square.msh` (see `common`), and over
//! `lobule_toy.msh` for its test.

mod common;

use std::f64::consts::PI;

use common::{PIXELS, Scratch, ceed, read_maps, unit_square};
use contracts::{Coefficient, Direction, Perfusion, PressureBoundary};
use mesh::{Mesh, MeshError};
use pde::{Discretization, Flow, PdeError};
use petsc::{Petsc, Point};

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

/// Asserts that the velocity of `flow`, conservative and primal, is `expected` at every
/// quadrature point of every level of `mesh`, within `1e-10`.
fn assert_uniform_velocity(mesh: &Mesh, flow: &Flow, expected: [f64; 2]) {
    for level in 0..mesh.levels().len() {
        let disc = Discretization::new(mesh, level, &ceed(), 1).unwrap();
        for velocity in [flow.velocity(level), flow.primal_velocity(level)] {
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
}

/// The edges of one level of a mesh, to read the edge fluxes of a flow
/// ([`Flow::edge_fluxes`]): each in the direction of the reference normal of its edge, the
/// vector from its first vertex to its second (in the order of its cone) turned by -90 degrees.
struct Edges {
    disc: Discretization,
    level: usize,
}

impl Edges {
    fn new(mesh: &Mesh, level: usize) -> Edges {
        Edges {
            disc: Discretization::new(mesh, level, &ceed(), 1).unwrap(),
            level,
        }
    }

    fn at(&self, vertex: Point) -> [f64; 2] {
        let dm = self.disc.dm();
        self.disc.node_coordinates()[dm.local_dofs(vertex).unwrap().start]
    }

    /// 1 if the reference normal of `edge` points out of `cell`, else -1.
    fn outward(&self, cell: Point, edge: Point) -> f64 {
        let dm = self.disc.dm();
        let vertices = dm.depth_stratum(0).unwrap();
        let corners: Vec<[f64; 2]> = dm
            .closure(cell)
            .unwrap()
            .into_iter()
            .filter(|p| vertices.contains(p))
            .map(|v| self.at(v))
            .collect();
        assert_eq!(corners.len(), 3);
        let centroid = [0, 1].map(|c| corners.iter().map(|x| x[c]).sum::<f64>() / 3.0);
        let &[a, b] = dm.cone(edge).unwrap().as_slice() else {
            panic!("edge {edge}");
        };
        let ([xa, ya], [xb, yb]) = (self.at(a), self.at(b));
        let normal = [yb - ya, xa - xb];
        let middle = [0.5 * (xa + xb), 0.5 * (ya + yb)];
        let out = normal[0] * (middle[0] - centroid[0]) + normal[1] * (middle[1] - centroid[1]);
        if out > 0.0 { 1.0 } else { -1.0 }
    }

    /// The edge fluxes of `flow` on the level, checked to be one per local edge.
    fn fluxes<'a>(&self, flow: &'a Flow) -> &'a [f64] {
        let fluxes = flow.edge_fluxes(self.level);
        let edges = self.disc.dm().depth_stratum(1).unwrap();
        assert_eq!(fluxes.len(), edges.len(), "level {}", self.level);
        fluxes
    }

    /// The fluxes of `flow` out of each local cell through its three edges.
    fn cell_fluxes(&self, flow: &Flow) -> Vec<[f64; 3]> {
        let fluxes = self.fluxes(flow);
        let dm = self.disc.dm();
        let edges = dm.depth_stratum(1).unwrap();
        dm.height_stratum(0)
            .unwrap()
            .map(|cell| {
                let cone = dm.cone(cell).unwrap();
                assert_eq!(cone.len(), 3);
                [0, 1, 2]
                    .map(|i| self.outward(cell, cone[i]) * fluxes[(cone[i] - edges.start) as usize])
            })
            .collect()
    }

    /// The local edges the tag `tag` marks.
    fn tagged(&self, tag: &str) -> Vec<Point> {
        let dm = self.disc.dm();
        let edges = dm.depth_stratum(1).unwrap();
        let mut points = Vec::new();
        for value in dm.label_values(tag).unwrap() {
            points.extend(dm.stratum_points(tag, value).unwrap());
        }
        points.retain(|p| edges.contains(p));
        assert!(
            dm.comm()
                .sum_int(points.len() as petsc::sys::PetscInt)
                .unwrap()
                > 0,
            "{tag}"
        );
        points
    }

    /// The flow of `flow` into the domain through the edges of the tag `tag`, each edge counted
    /// on the process which owns it. Collective.
    fn inflow(&self, flow: &Flow, tag: &str) -> f64 {
        let fluxes = self.fluxes(flow);
        let dm = self.disc.dm();
        let edges = dm.depth_stratum(1).unwrap();
        let ghosts = dm.ghost_points().unwrap();
        let local: f64 = self
            .tagged(tag)
            .into_iter()
            .filter(|edge| ghosts.binary_search(edge).is_err())
            .map(|edge| {
                let &[cell] = dm.support(edge).unwrap().as_slice() else {
                    panic!("edge {edge} of {tag} inside the domain");
                };
                -self.outward(cell, edge) * fluxes[(edge - edges.start) as usize]
            })
            .sum();
        dm.comm().sum_real(local).unwrap()
    }

    /// The sum of the absolute edge fluxes of `flow` over the edges, each counted on the process
    /// which owns it. Collective.
    fn total(&self, flow: &Flow) -> f64 {
        let fluxes = self.fluxes(flow);
        let dm = self.disc.dm();
        let edges = dm.depth_stratum(1).unwrap();
        let ghosts = dm.ghost_points().unwrap();
        let local: f64 = edges
            .clone()
            .filter(|edge| ghosts.binary_search(edge).is_err())
            .map(|edge| fluxes[(edge - edges.start) as usize].abs())
            .sum();
        dm.comm().sum_real(local).unwrap()
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

    // The velocities projected to the nodes: 2 components of each of the conservative and the
    // primal velocity per node.
    let velocity = flow.nodal_velocity().unwrap();
    velocity
        .with_array(|values| {
            assert_eq!(values.len(), 4 * nodes.len());
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

    /// The L2 errors of the pressure, the primal velocity and the conservative velocity on the
    /// finest level at each of `REFINEMENTS`, and the results which `parallel_on_*_processes`
    /// compares between process counts: the L2 norms of pressure and velocities, the flows
    /// through the sides, and the sums of the edge fluxes (through each side, and of their
    /// absolute values).
    fn errors(&self, p: &Petsc, test: &str) -> (Vec<[f64; 3]>, Vec<f64>) {
        let mut errors = Vec::new();
        let mut results = Vec::new();
        for refinements in REFINEMENTS {
            let mesh = unit_square(p, refinements);
            let flow = self.solve(test, &mesh);
            let finest = mesh.levels().len() - 1;
            let (primal, velocity) = (flow.primal_velocity(finest), flow.velocity(finest));
            errors.push([
                flow.pressure_l2_error(self.pressure).unwrap(),
                velocity_l2_error(&mesh, finest, primal, |x| self.velocity(x)),
                velocity_l2_error(&mesh, finest, velocity, |x| self.velocity(x)),
            ]);
            results.push(flow.pressure_l2_error(|_| 0.0).unwrap());
            results.push(velocity_l2_error(&mesh, finest, primal, |_| [0.0; 2]));
            results.push(velocity_l2_error(&mesh, finest, velocity, |_| [0.0; 2]));
            let flows: Vec<f64> = flow.tag_flows().iter().map(|&(_, q)| q).collect();
            self.assert_conserved(&flow, &flows);
            results.extend(flows);
            let edges = Edges::new(&mesh, finest);
            results.extend(SIDES.map(|side| edges.inflow(&flow, side)));
            results.push(edges.total(&flow));
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

/// The orders of the errors `errors` (of the pressure, the primal velocity and the conservative
/// velocity, see [`Manufactured::errors`]) between successive refinements.
fn orders(errors: &[[f64; 3]]) -> Vec<[f64; 3]> {
    errors
        .windows(2)
        .map(|pair| {
            let order = [0, 1, 2].map(|i| (pair[0][i] / pair[1][i]).log2());
            println!(
                "errors {:?} -> {:?}: orders {:.3} (pressure), {:.3} (primal velocity), {:.3} \
                 (conservative velocity)",
                pair[0], pair[1], order[0], order[1], order[2]
            );
            order
        })
        .collect()
}

/// Asserts the orders of the pressure errors and the primal velocity errors between successive
/// refinements are at least `2 - 0.1` and `1 - 0.1`.
fn assert_orders(errors: &[[f64; 3]]) {
    for [pressure, primal, _] in orders(errors) {
        assert!(pressure >= 1.9, "pressure order {pressure}");
        assert!(primal >= 0.9, "primal velocity order {primal}");
    }
}

/// Asserts the orders of the conservative velocity errors between successive refinements are at
/// least `1 - 0.1`.
fn assert_conservative_orders(errors: &[[f64; 3]]) {
    for [_, _, conservative] in orders(errors) {
        assert!(
            conservative >= 0.9,
            "conservative velocity order {conservative}"
        );
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

/// The edge of the square of `lobule_toy.msh` ([0, `LOBULE`] x [0, `LOBULE`] um, holes of radius
/// 20 um at (90, 90) and (310, 310) tagged `portal_wall` and `central_wall`, the outer boundary
/// `section_edge`).
const LOBULE: f64 = 400.0;

/// The tags of the boundary of `lobule_toy.msh`.
const LOBULE_TAGS: [&str; 3] = ["portal_wall", "central_wall", "section_edge"];

/// `k_along`, `k_across` (um^2) of the manufactured flow on `lobule_toy.msh`, with `mu = 1`.
const LOBULE_K: [f64; 2] = [3.0, 1.0];

/// `g0` (1/um) of the manufactured flow on `lobule_toy.msh`: as large as the gradient of its
/// direction map, so that the blend `w = |g|^2 / (|g|^2 + g0^2)` varies over the mesh (from
/// 1/2 at x = 0 to 5/6 at x = `LOBULE`).
const LOBULE_G0: f64 = 1.0 / LOBULE;

/// A manufactured flow on a tagged mesh with holes, `lobule_toy.msh`, with a permeability which
/// varies in direction and in strength: the direction map `m = (x/L)^2 + y/L` (L = `LOBULE`)
/// has the gradient `g = (2x/L^2, 1/L)`, which turns with x, and
/// `K = k_across I + w (k_along - k_across) t t^T = k_across I + b g g^T` with
/// `b = (k_along - k_across) / (|g|^2 + g0^2)`. The map's central differences on the raster
/// are exact for a quadratic, and their bilinear interpolation is exact for the linear `g`, so
/// slidesim's `K` is this one at every point.
mod lobule {
    use std::f64::consts::PI;

    use super::{LOBULE as L, LOBULE_G0 as G0, LOBULE_K};

    pub fn map([x, y]: [f64; 2]) -> f64 {
        (x / L).powi(2) + y / L
    }

    fn map_gradient([x, _]: [f64; 2]) -> [f64; 2] {
        [2.0 * x / (L * L), 1.0 / L]
    }

    /// The Jacobian of `map_gradient`, `[[g0_x, g0_y], [g1_x, g1_y]]`.
    const MAP_HESSIAN: [[f64; 2]; 2] = [[2.0 / (L * L), 0.0], [0.0, 0.0]];

    /// `p = 100 (sin(pi x/L) sin(pi y/L) + x/L)` (Pa).
    pub fn pressure([x, y]: [f64; 2]) -> f64 {
        let (s, t) = (x / L, y / L);
        100.0 * ((PI * s).sin() * (PI * t).sin() + s)
    }

    fn pressure_gradient([x, y]: [f64; 2]) -> [f64; 2] {
        let (a, b) = (PI * x / L, PI * y / L);
        let c = 100.0 * PI / L;
        [c * a.cos() * b.sin() + 100.0 / L, c * a.sin() * b.cos()]
    }

    fn pressure_hessian([x, y]: [f64; 2]) -> [[f64; 2]; 2] {
        let (a, b) = (PI * x / L, PI * y / L);
        let c = 100.0 * (PI / L).powi(2);
        [
            [-c * a.sin() * b.sin(), c * a.cos() * b.cos()],
            [c * a.cos() * b.cos(), -c * a.sin() * b.sin()],
        ]
    }

    fn dot(a: [f64; 2], b: [f64; 2]) -> f64 {
        a[0] * b[0] + a[1] * b[1]
    }

    fn times(m: [[f64; 2]; 2], v: [f64; 2]) -> [f64; 2] {
        [dot(m[0], v), dot(m[1], v)]
    }

    /// `b` of `K = k_across I + b g g^T` and the gradient `g`.
    fn blend(x: [f64; 2]) -> (f64, [f64; 2]) {
        let g = map_gradient(x);
        let [k_along, k_across] = LOBULE_K;
        ((k_along - k_across) / (dot(g, g) + G0 * G0), g)
    }

    /// The velocity `-K grad p` (`mu = 1`).
    pub fn velocity(x: [f64; 2]) -> [f64; 2] {
        let (b, g) = blend(x);
        let gp = pressure_gradient(x);
        let k_across = LOBULE_K[1];
        let along = b * dot(g, gp);
        [
            -(k_across * gp[0] + along * g[0]),
            -(k_across * gp[1] + along * g[1]),
        ]
    }

    /// The source `-div(K grad p)` (`mu = 1`): with `F = k_across grad p + b (g . grad p) g`,
    /// `div F = k_across lap p + (grad b . g)(g . grad p)
    ///   + b (g . H g + g . J grad p + (g . grad p) tr J)`, `H` the Hessian of `p`, `J` that of
    /// the map, and `grad b = -2 (k_along - k_across) J g / (|g|^2 + g0^2)^2`.
    pub fn source(x: [f64; 2]) -> f64 {
        let (b, g) = blend(x);
        let [k_along, k_across] = LOBULE_K;
        let gp = pressure_gradient(x);
        let h = pressure_hessian(x);
        let j = MAP_HESSIAN;
        let denominator = dot(g, g) + G0 * G0;
        let grad_b = times(j, g).map(|v| -2.0 * (k_along - k_across) * v / denominator.powi(2));
        let div = k_across * (h[0][0] + h[1][1])
            + dot(grad_b, g) * dot(g, gp)
            + b * (dot(g, times(h, g)) + dot(g, times(j, gp)) + dot(g, gp) * (j[0][0] + j[1][1]));
        -div
    }
}

/// The errors of the manufactured flow of [`lobule`] on `lobule_toy.msh` at 0 to 3 refinements
/// (as [`Manufactured::errors`]), with the pressure fixed on both holes and the outer boundary;
/// asserts that the flows through the tags balance the source.
fn tagged_mesh_errors(p: &Petsc, test: &str) -> Vec<[f64; 3]> {
    let boundaries: Vec<(&str, f64)> = LOBULE_TAGS.iter().map(|&tag| (tag, 0.0)).collect();
    let perfusion = perfusion(1.0, LOBULE_K, LOBULE_G0, &boundaries);
    let mut errors = Vec::new();
    for refinements in 0..4 {
        let mesh = common::fixture_mesh(p, "lobule_toy.msh", refinements);
        let scratch = Scratch::new(test);
        // 2 um pixels over [-6, 410] um.
        let direction =
            scratch.map_on_grid("direction", [209, 209], 2.0, [-6.0, -6.0], lobule::map);
        let maps = read_maps(&[direction], &mesh);
        let flow = Flow::solve_manufactured(
            &mesh,
            &perfusion,
            &maps,
            &ceed(),
            "",
            &lobule::pressure,
            &lobule::source,
        )
        .unwrap();
        let finest = mesh.levels().len() - 1;
        errors.push([
            flow.pressure_l2_error(lobule::pressure).unwrap(),
            velocity_l2_error(
                &mesh,
                finest,
                flow.primal_velocity(finest),
                lobule::velocity,
            ),
            velocity_l2_error(&mesh, finest, flow.velocity(finest), lobule::velocity),
        ]);
        // The flows through the three tags balance the source, `-int f`.
        let flows = flow.tag_flows();
        let tags: Vec<&str> = flows.iter().map(|(tag, _)| tag.as_str()).collect();
        assert_eq!(tags, LOBULE_TAGS);
        let disc = flow.discretization();
        let local: f64 = disc
            .quadrature_points()
            .iter()
            .zip(disc.quadrature_weights())
            .map(|(&x, w)| w * lobule::source(x))
            .sum();
        let source = disc.dm().comm().sum_real(local).unwrap();
        let inflow: f64 = flows.iter().map(|&(_, q)| q).sum();
        let scale = flows.iter().fold(source.abs(), |m, (_, q)| m.max(q.abs()));
        assert!(
            (inflow + source).abs() <= 1e-8 * scale,
            "inflow {inflow}, source {source}"
        );
    }
    errors
}

/// Darcy on a slideviz-like tagged mesh with holes (spec, verification item 2): pressure
/// Dirichlet values on both holes and the outer boundary, a permeability which turns and changes
/// its anisotropy over the mesh (see [`lobule`]): pressure O(h^2) and primal velocity O(h) on 4
/// levels (h about 20, 10, 5, 2.5 um), and the flows through the tags balance the source.
#[test]
fn pressure_order_on_tagged_mesh() {
    let p = Petsc::init(&[]).unwrap();
    assert_orders(&tagged_mesh_errors(&p, "pressure_order_on_tagged_mesh"));
}

/// The conservative velocity converges at order 1 as the primal one, on the manufactured flows
/// of the `pressure_order_*` tests.
#[test]
fn conservative_velocity_order() {
    let p = Petsc::init(&[]).unwrap();
    let test = "conservative_velocity_order";
    for manufactured in [Manufactured::isotropic(), Manufactured::anisotropic()] {
        assert_conservative_orders(&manufactured.errors(&p, test).0);
    }
    assert_conservative_orders(&tagged_mesh_errors(&p, test));
}

/// The pressures of the perfusion through `lobule_toy.msh` (Pa): from the portal wall to the
/// central wall, no flow through the section edge.
const LOBULE_PRESSURES: [(&str, f64); 2] = [("portal_wall", 390.0), ("central_wall", 0.0)];

/// The flow through `lobule_toy.msh` refined `refinements` times, from the portal wall to the
/// central wall ([`LOBULE_PRESSURES`]), with the permeability of [`lobule`] (`mu = 1`): no
/// flow through the section edge.
fn lobule_flow(p: &Petsc, test: &str, refinements: u32) -> (Mesh, Flow) {
    let mesh = common::fixture_mesh(p, "lobule_toy.msh", refinements);
    let scratch = Scratch::new(test);
    let direction = scratch.map_on_grid("direction", [209, 209], 2.0, [-6.0, -6.0], lobule::map);
    let maps = read_maps(&[direction], &mesh);
    let perfusion = perfusion(1.0, LOBULE_K, LOBULE_G0, &LOBULE_PRESSURES);
    let flow = Flow::solve(&mesh, &perfusion, &maps, &ceed(), "").unwrap();
    (mesh, flow)
}

/// The anisotropic flow from `left` to `right` on the unit square refined twice, whose axes are
/// oblique to the sides: not uniform, no flow through `top` and `bottom`.
fn oblique_flow(p: &Petsc, test: &str) -> (Mesh, Flow) {
    let mesh = unit_square(p, 2);
    let perfusion = perfusion(1.0, [3.0, 1.0], G0, &[("left", 1.0), ("right", 0.0)]);
    let flow = solve(test, &mesh, &perfusion, |[x, y]| x + 0.5 * y).unwrap();
    (mesh, flow)
}

/// The flows of the tests of the edge fluxes, with the tags without pressure condition (walls).
fn conservation_cases(p: &Petsc, test: &str) -> Vec<(Mesh, Flow, Vec<&'static str>)> {
    let (mesh, flow) = lobule_flow(p, test, 1);
    let (square, oblique) = oblique_flow(p, test);
    vec![
        (mesh, flow, vec!["section_edge"]),
        (square, oblique, vec!["top", "bottom"]),
    ]
}

/// On every level, the fluxes through the three edges of each cell add up to zero (no sources
/// in the Darcy problem), to rounding.
#[test]
fn element_fluxes_sum_to_zero() {
    let p = Petsc::init(&[]).unwrap();
    for (mesh, flow, _) in conservation_cases(&p, "element_fluxes_sum_to_zero") {
        for level in 0..mesh.levels().len() {
            let cells = Edges::new(&mesh, level).cell_fluxes(&flow);
            let scale = cells.iter().flatten().fold(0.0_f64, |m, f| m.max(f.abs()));
            assert!(scale > 0.0);
            let worst = cells
                .iter()
                .map(|f| (f[0] + f[1] + f[2]).abs())
                .fold(0.0, f64::max);
            println!("level {level}: largest flux {scale:e}, largest cell sum {worst:e}");
            assert!(
                worst <= 1e-12 * scale,
                "level {level}: {worst:e} of {scale:e}"
            );
        }
    }
}

/// On every level, the flux through each edge of a tag without pressure condition is zero.
#[test]
fn no_flux_through_walls_without_pressure() {
    let p = Petsc::init(&[]).unwrap();
    for (mesh, flow, walls) in conservation_cases(&p, "no_flux_through_walls_without_pressure") {
        for level in 0..mesh.levels().len() {
            let edges = Edges::new(&mesh, level);
            let fluxes = edges.fluxes(&flow);
            let start = edges.disc.dm().depth_stratum(1).unwrap().start;
            for wall in &walls {
                for edge in edges.tagged(wall) {
                    let flux = fluxes[(edge - start) as usize];
                    assert_eq!(flux, 0.0, "level {level}, {wall}, edge {edge}");
                }
            }
        }
    }
}

/// The edge fluxes through each pressure tag add up to its flow from the residual
/// ([`Flow::tag_flows`]): on the flows of [`conservation_cases`], and on the manufactured flow
/// with a source, whose four sides meet at corners (a corner counts for the side listed last).
#[test]
fn tag_flows_equal_edge_flux_sums() {
    let p = Petsc::init(&[]).unwrap();
    let test = "tag_flows_equal_edge_flux_sums";
    let mut cases: Vec<(Mesh, Flow)> = conservation_cases(&p, test)
        .into_iter()
        .map(|(mesh, flow, _)| (mesh, flow))
        .collect();
    let square = unit_square(&p, 2);
    let manufactured = Manufactured::anisotropic().solve(test, &square);
    cases.push((square, manufactured));
    for (mesh, flow) in &cases {
        let edges = Edges::new(mesh, mesh.levels().len() - 1);
        for (tag, q) in flow.tag_flows() {
            let sum = edges.inflow(flow, tag);
            println!("{tag}: tag flow {q:e}, edge flux sum {sum:e}");
            assert!(
                (sum - q).abs() <= 1e-10 * q.abs(),
                "{tag}: {sum} against {q}"
            );
        }
    }
}

/// `pinched.msh` refined once: the rectangle [0, 4] x [0, 2] um with two triangular holes
/// (`hole_wall`) which touch at (2, 1), where the boundary touches itself: around that vertex two
/// fans of cells, above and below, which share no edge. The anisotropic permeability of the
/// oblique flow (map `x + 0.5 y`), with the pressures `pressures`.
fn pinched_flow(
    p: &Petsc,
    test: &str,
    pressures: &[(&str, f64)],
) -> Result<(Mesh, Flow), PdeError> {
    let mesh = common::fixture_mesh(p, "pinched.msh", 1);
    let scratch = Scratch::new(test);
    let direction = scratch.map_on_grid("direction", [45, 85], 0.05, [-0.1, -0.1], |[x, y]| {
        x + 0.5 * y
    });
    let maps = read_maps(&[direction], &mesh);
    let perfusion = perfusion(1.0, [3.0, 1.0], G0, pressures);
    let flow = Flow::solve(&mesh, &perfusion, &maps, &ceed(), "")?;
    Ok((mesh, flow))
}

/// Asserts on every level of `mesh`: the flux through each edge of the tags `walls` is zero, the
/// edge fluxes through each pressure tag add up to its flow, and the fluxes out of each cell add
/// up to zero, but for the cells at the pinch vertex (2, 1) if `pinch_transfer`: there the
/// cells of each fan add up to the flow through the vertex from one fan to the other (which the
/// P1 pressure has, its function of the vertex spanning both fans), returned per level, and
/// all of them to zero.
fn assert_pinched_conservation(
    mesh: &Mesh,
    flow: &Flow,
    walls: &[&str],
    pinch_transfer: bool,
) -> Vec<f64> {
    let mut transfers = Vec::new();
    for level in 0..mesh.levels().len() {
        let edges = Edges::new(mesh, level);
        let fluxes = edges.fluxes(flow);
        let start = edges.disc.dm().depth_stratum(1).unwrap().start;
        for wall in walls {
            for edge in edges.tagged(wall) {
                assert_eq!(
                    fluxes[(edge - start) as usize],
                    0.0,
                    "level {level}, {wall}"
                );
            }
        }
        if level == mesh.levels().len() - 1 {
            for (tag, q) in flow.tag_flows() {
                let sum = edges.inflow(flow, tag);
                assert!(
                    (sum - q).abs() <= 1e-10 * q.abs(),
                    "{tag}: {sum} against {q}"
                );
            }
        }
        let dm = edges.disc.dm();
        let vertices = dm.depth_stratum(0).unwrap();
        let at_pinch = |cell: Point| {
            dm.closure(cell).unwrap().into_iter().any(|v| {
                let x = vertices.contains(&v).then(|| edges.at(v));
                x.is_some_and(|[x, y]| (x - 2.0).abs() < 1e-12 && (y - 1.0).abs() < 1e-12)
            })
        };
        let cells = edges.cell_fluxes(flow);
        let scale = cells.iter().flatten().fold(0.0_f64, |m, f| m.max(f.abs()));
        let mut pinch = Vec::new();
        for (cell, f) in dm.height_stratum(0).unwrap().zip(&cells) {
            let sum = f[0] + f[1] + f[2];
            if pinch_transfer && at_pinch(cell) {
                pinch.push(sum);
            } else {
                assert!(
                    sum.abs() <= 1e-12 * scale,
                    "level {level}: cell {cell} sum {sum:e}"
                );
            }
        }
        let comm = dm.comm();
        let total = comm.sum_real(pinch.iter().sum()).unwrap();
        let transfer = comm
            .sum_real(pinch.iter().filter(|s| **s > 0.0).sum())
            .unwrap();
        println!("level {level}: largest flux {scale:e}, flow through the pinch {transfer:e}");
        assert!(total.abs() <= 1e-12 * scale, "level {level}: {pinch:?}");
        transfers.push(transfer);
    }
    transfers
}

/// A vertex where the boundary touches itself (two holes touching): each fan of cells around it
/// is a patch of its own. With a pressure condition on both holes, each fan lets the flow at the
/// vertex out through its own edges of the holes: every cell conserves mass. With walls as holes,
/// the flow which the P1 pressure passes through the vertex from one fan to the other stays a
/// source and a sink in a cell of each fan at the vertex (no edge joins them); every other cell
/// conserves mass. In both cases the walls have no flux, and the pressure tags their flows.
#[test]
fn pinched_boundary() {
    let p = Petsc::init(&[]).unwrap();
    let test = "pinched_boundary";
    let pressures = [("hole_wall", 1.0), ("left", 0.0), ("right", 0.0)];
    let (mesh, flow) = pinched_flow(&p, test, &pressures).unwrap();
    assert_pinched_conservation(&mesh, &flow, &["rim"], false);

    let pressures = [("left", 1.0), ("right", 0.0)];
    let (mesh, flow) = pinched_flow(&p, test, &pressures).unwrap();
    let transfers = assert_pinched_conservation(&mesh, &flow, &["rim", "hole_wall"], true);
    // Small against the flow through the domain (measured 5 % and 4 % on the two levels).
    let q = flow.tag_flows()[0].1;
    println!("flow {q:e}, through the pinch by level {transfers:?}");
    assert!(
        transfers.iter().all(|t| *t < 0.1 * q),
        "{transfers:?} of {q}"
    );
}

/// A pressure tag inside the domain (here on edges embedded in the mesh) is an error, naming the
/// tag: the equilibrated fluxes of the conservative velocity need the pressure tags on edges of
/// the boundary.
#[test]
fn pressure_tag_inside_the_domain_is_error() {
    let p = Petsc::init(&[]).unwrap();
    let pressures = [("left", 1.0), ("right", 0.0), ("inner_line", 0.5)];
    let e = pinched_flow(&p, "pressure_tag_inside_the_domain_is_error", &pressures).unwrap_err();
    let message = e.to_string();
    assert!(matches!(e, PdeError::Unsupported(_)), "{message}");
    assert!(message.contains("'inner_line'"), "{message}");
    assert!(message.contains("edges of the boundary"), "{message}");
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

/// The results `parallel_on_*_processes` compares between process counts: of [`uniform`] and
/// of the flow through `lobule_toy.msh` ([`lobule_flow`]), the L2 norm of the pressure, the
/// L2 norms of the velocity (conservative and primal) and the sum of the absolute edge fluxes
/// on every level, the flows through the tags, and the sums of the edge fluxes through them on
/// the finest level; and those of the anisotropic manufactured solution (see
/// [`Manufactured::errors`]).
fn parity_results(p: &Petsc) -> Vec<f64> {
    let mut results = Vec::new();
    for (mesh, flow) in [uniform(p), lobule_flow(p, "parity_lobule", 1)] {
        results.push(flow.pressure_l2_error(|_| 0.0).unwrap());
        for level in 0..mesh.levels().len() {
            for velocity in [flow.velocity(level), flow.primal_velocity(level)] {
                results.push(velocity_l2_error(&mesh, level, velocity, |_| [0.0; 2]));
            }
            results.push(Edges::new(&mesh, level).total(&flow));
        }
        let edges = Edges::new(&mesh, mesh.levels().len() - 1);
        for (tag, q) in flow.tag_flows() {
            results.push(*q);
            results.push(edges.inflow(&flow, tag));
        }
    }
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
                    "element_fluxes_sum_to_zero",
                    "no_flux_through_walls_without_pressure",
                    "tag_flows_equal_edge_flux_sums",
                    "pinched_boundary",
                    "pressure_tag_inside_the_domain_is_error",
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
