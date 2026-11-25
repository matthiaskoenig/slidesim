//! Tests of the P1 discretization and its libCEED operators on gmsh meshes. nextest runs each
//! test in its own process, so every test initializes PETSc once. `parallel_on_2_processes` runs
//! some of them again under `mpiexec`, as does `parallel_on_3_processes`.

use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::rc::Rc;

use ceed::{Ceed, DEFAULT_RESOURCE};
use contracts::{LengthUnit, MeshFormat, MeshRef, SCENE_VERSION, Scene, Units};
use mesh::{Mesh, MeshError};
use pde::{Discretization, PdeError, PointData, SharedOperator, SourceField, Terms, Transport};
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
        maps: Vec::new(),
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

/// The operator of milestone 1, `shift M + d K + k M` (`M` the mass matrix, `K` the stiffness
/// matrix): the Jacobian of [`Transport`] with mass 1, reaction `k` and diffusion `d` at every
/// point.
fn diffusion_reaction(disc: &Discretization, d: f64, k: f64, shift: f64) -> Transport {
    let n = disc.num_quadrature_points();
    let data = PointData::isotropic(disc, &vec![1.0; n], &vec![k; n], &vec![d; n]);
    let mut op = Transport::new(disc, Rc::new(data), Terms::Jacobian).unwrap();
    op.shift = shift;
    op
}

/// `y = A x` with the shell matrix of `op`.
fn apply(disc: &Discretization, op: Transport, x: &Vector) -> Vector {
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
        let mass = diffusion_reaction(&disc, 0.0, 1.0, 0.0);
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
    let op = diffusion_reaction(&disc, 0.0, 0.5, 2.0);
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
        let stiffness = diffusion_reaction(&disc, 1.0, 0.0, 0.0);
        let k1 = apply(&disc, stiffness, &one).norm2().unwrap();
        assert!(k1 < 1e-12, "{file}: |K 1| = {k1}");
    }
}

/// The operator of the diagonal and assembly tests: all three terms, essential dofs on the
/// vessel walls, on the refined tissue mesh.
fn tissue_operator(p: &Petsc) -> (Discretization, Transport) {
    let mesh = load(p, "tissue_small.msh", 1);
    let disc = Discretization::with_essential(&mesh, 1, &ceed(), 1, &["vessel_wall"]).unwrap();
    let op = diffusion_reaction(&disc, 0.3, 0.7, 2.0);
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
fn changed_shift_reaches_mult_and_diagonal() {
    let p = Petsc::init(&[]).unwrap();
    let (disc, op) = tissue_operator(&p);
    let op = Rc::new(RefCell::new(op));
    let shell = disc.shell_matrix(SharedOperator(Rc::clone(&op))).unwrap();
    let mut x = disc.global_vector().unwrap();
    let rank = p.comm().rank();
    x.with_array_mut(|a| {
        for (i, v) in a.iter_mut().enumerate() {
            *v = (i as f64 + 1000.0 * rank as f64).cos();
        }
    })
    .unwrap();
    // Applied once with the first shift, then changed.
    let mut before = disc.global_vector().unwrap();
    shell.mult(&x, &mut before).unwrap();
    let mut diagonal_before = disc.global_vector().unwrap();
    shell.diagonal(&mut diagonal_before).unwrap();
    op.borrow_mut().shift = 10.0;
    let mut y = disc.global_vector().unwrap();
    shell.mult(&x, &mut y).unwrap();
    let mut diagonal = disc.global_vector().unwrap();
    shell.diagonal(&mut diagonal).unwrap();
    // The assembled matrix of the new shift.
    let assembled = op.borrow().assemble().unwrap();
    let mut expected = disc.global_vector().unwrap();
    assembled.mult(&x, &mut expected).unwrap();
    let error = distance(&disc, &y, &expected) / expected.norm2().unwrap();
    assert!(error < 1e-14, "relative |A x - A_coo x| = {error}");
    assembled.diagonal(&mut expected).unwrap();
    let error = distance(&disc, &diagonal, &expected) / expected.norm2().unwrap();
    assert!(error < 1e-13, "relative |diag - diag(A)| = {error}");
    // And they differ from those of the first shift.
    assert!(distance(&disc, &y, &before) > 0.1 * expected.norm2().unwrap());
    assert!(distance(&disc, &diagonal, &diagonal_before) > 0.1 * expected.norm2().unwrap());
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
    let jacobian = apply(&disc, diffusion_reaction(&disc, 1.0, 0.0, 0.0), &one);
    assert!(jacobian.norm2().unwrap() > 0.1);
    // With the essential values 1 in the local vector, it applies to the constant function.
    let mut local = disc.local_vector().unwrap();
    local.set(0.0).unwrap();
    disc.dm().global_to_local(&one, &mut local).unwrap();
    for side in SIDES {
        disc.set_essential_values(&mut local, side, 1.0).unwrap();
    }
    let stiffness = diffusion_reaction(&disc, 1.0, 0.0, 0.0);
    let mut y = disc.global_vector().unwrap();
    stiffness.apply_local(&local, &mut y).unwrap();
    let k1 = y.norm2().unwrap();
    assert!(k1 < 1e-12, "|K 1| = {k1}");
}

#[test]
fn shift_takes_effect_when_changed() {
    let p = Petsc::init(&[]).unwrap();
    let mesh = load(&p, "unit_square.msh", 0);
    let disc = Discretization::new(&mesh, 0, &ceed(), 1).unwrap();
    let mut op = diffusion_reaction(&disc, 0.0, 1.0, 0.0);
    let diagonal = |op: &Transport| {
        let mut d = disc.global_vector().unwrap();
        op.assemble().unwrap().diagonal(&mut d).unwrap();
        d
    };
    let mass = diagonal(&op);
    // (shift + k) M.
    op.shift = 2.0;
    let mut scaled = diagonal(&op);
    scaled.axpy(-3.0, &mass).unwrap();
    assert!(scaled.norm2().unwrap() < 1e-14);
    let stiffness = diagonal(&diffusion_reaction(&disc, 1.0, 0.0, 0.0));
    assert!(distance(&disc, &stiffness, &mass) > 1.0);
}

/// `a x - sum_i c_i (b_i x)`, relative to `|a x|`, for global vectors of `disc`.
fn combination_error(
    disc: &Discretization,
    a: Transport,
    combination: Vec<(f64, Transport)>,
    x: &Vector,
) -> f64 {
    let ax = apply(disc, a, x);
    let mut d = disc.global_vector().unwrap();
    d.set(0.0).unwrap();
    d.axpy(1.0, &ax).unwrap();
    for (c, b) in combination {
        d.axpy(-c, &apply(disc, b, x)).unwrap();
    }
    d.norm2().unwrap() / ax.norm2().unwrap()
}

#[test]
fn transport_matches_diffusion_reaction() {
    let p = Petsc::init(&[]).unwrap();
    let mesh = load(&p, "tissue_small.msh", 1);
    let disc = Discretization::with_essential(&mesh, 1, &ceed(), 1, &["vessel_wall"]).unwrap();
    let n = disc.num_quadrature_points();
    let mut x = disc.global_vector().unwrap();
    let rank = p.comm().rank();
    x.with_array_mut(|a| {
        for (i, v) in a.iter_mut().enumerate() {
            *v = (i as f64 + 1000.0 * rank as f64).sin();
        }
    })
    .unwrap();
    let transport = |mass: f64, reaction: f64, diffusion: f64, terms: Terms| {
        let data = PointData::isotropic(
            &disc,
            &vec![mass; n],
            &vec![reaction; n],
            &vec![diffusion; n],
        );
        Transport::new(&disc, Rc::new(data), terms).unwrap()
    };
    // The mass matrix M and the stiffness matrix K.
    let m = || transport(1.0, 0.0, 0.0, Terms::Mass);
    let k = || transport(0.0, 0.0, 1.0, Terms::Steady);
    // The Jacobian of milestone 1: shift M + d K + k M, here 2 M + 0.3 K + 0.7 M.
    let error = combination_error(
        &disc,
        diffusion_reaction(&disc, 0.3, 0.7, 2.0),
        vec![(2.7, m()), (0.3, k())],
        &x,
    );
    assert!(error < 1e-14, "Jacobian: {error}");
    // Its steady part, d K + k M; and the mass term with the mass factor 0.5, 0.5 M.
    let steady = transport(1.0, 0.7, 0.3, Terms::Steady);
    let error = combination_error(&disc, steady, vec![(0.7, m()), (0.3, k())], &x);
    assert!(error < 1e-14, "steady: {error}");
    let mass = transport(0.5, 0.7, 0.3, Terms::Mass);
    let error = combination_error(&disc, mass, vec![(0.5, m())], &x);
    assert!(error < 1e-14, "mass: {error}");
    // A Jacobian with the mass factor 0.5: shift 0.5 M + steady.
    let mut jacobian = transport(0.5, 0.7, 0.3, Terms::Jacobian);
    jacobian.shift = 4.0;
    let error = combination_error(&disc, jacobian, vec![(2.7, m()), (0.3, k())], &x);
    assert!(error < 1e-14, "Jacobian with mass 0.5: {error}");
}

#[test]
fn anisotropic_tensor_integrates_exactly() {
    let p = Petsc::init(&[]).unwrap();
    let mesh = load(&p, "unit_square.msh", 1);
    let disc = Discretization::new(&mesh, 1, &ceed(), 1).unwrap();
    let n = disc.num_quadrature_points();
    // D = [[2, 1], [1, 3]] as (D00, D01, D11).
    let data = PointData::new(vec![0.0; n], vec![0.0; n], vec![[2.0, 1.0, 3.0]; n]);
    let op = Rc::new(RefCell::new(
        Transport::new(&disc, Rc::new(data), Terms::Steady).unwrap(),
    ));
    let a = disc.shell_matrix(SharedOperator(Rc::clone(&op))).unwrap();
    let mut u = disc.global_vector().unwrap();
    disc.interpolate(|[x, _]| x, &mut u).unwrap();
    let mut v = disc.global_vector().unwrap();
    disc.interpolate(|[_, y]| y, &mut v).unwrap();
    let mut au = disc.global_vector().unwrap();
    a.mult(&u, &mut au).unwrap();
    let mut av = disc.global_vector().unwrap();
    a.mult(&v, &mut av).unwrap();
    // int D grad u . grad v over the unit square, for grad u = (1, 0) and grad v = (0, 1): D_10,
    // D_00 and D_11.
    let uv = v.dot(&au).unwrap();
    assert!((uv - 1.0).abs() < 1e-12, "{uv}");
    let vu = u.dot(&av).unwrap();
    assert!((vu - 1.0).abs() < 1e-12, "{vu}");
    let uu = u.dot(&au).unwrap();
    assert!((uu - 2.0).abs() < 1e-12, "{uu}");
    let vv = v.dot(&av).unwrap();
    assert!((vv - 3.0).abs() < 1e-12, "{vv}");
}

#[test]
fn point_data_must_match_the_discretization() {
    let p = Petsc::init(&[]).unwrap();
    let mesh = load(&p, "unit_square.msh", 0);
    let disc = Discretization::new(&mesh, 0, &ceed(), 1).unwrap();
    let n = disc.num_quadrature_points() + 1;
    let data = PointData::new(vec![1.0; n], vec![0.0; n], vec![[1.0, 0.0, 1.0]; n]);
    let e = Transport::new(&disc, Rc::new(data), Terms::Steady).unwrap_err();
    assert!(e.to_string().contains(&n.to_string()), "{e}");
}

#[test]
fn cells_have_their_area_tag() {
    let p = Petsc::init(&[]).unwrap();
    let ceed = ceed();
    for (file, tag) in [
        ("unit_square.msh", "domain"),
        ("tissue_small.msh", "tissue"),
    ] {
        let mesh = load(&p, file, 1);
        for level in 0..2 {
            let disc = Discretization::new(&mesh, level, &ceed, 1).unwrap();
            let cells = disc.dm().height_stratum(0).unwrap().len();
            for cell in 0..cells {
                assert_eq!(disc.cell_area_tag(cell).as_deref(), Some(tag), "{file}");
            }
            assert_eq!(disc.cell_area_tag(cells), None);
        }
    }
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
fn quadrature_weights_sum_to_the_area() {
    let p = Petsc::init(&[]).unwrap();
    let mesh = load(&p, "unit_square.msh", 1);
    let disc = Discretization::new(&mesh, 1, &ceed(), 1).unwrap();
    let weights = disc.quadrature_weights();
    assert_eq!(weights.len(), disc.num_quadrature_points());
    assert!(weights.iter().all(|&w| w > 0.0));
    let area = p.comm().sum_real(weights.iter().sum()).unwrap();
    assert!((area - 1.0).abs() < 1e-14, "{area}");
}

#[test]
fn interpolation_takes_the_values_at_the_nodes() {
    let p = Petsc::init(&[]).unwrap();
    let mesh = load(&p, "unit_square.msh", 1);
    let disc = Discretization::with_essential(&mesh, 1, &ceed(), 1, &["left"]).unwrap();
    let f = |[x, y]: [f64; 2]| 1.0 + 2.0 * x + 3.0 * y;
    let mut u = disc.global_vector().unwrap();
    disc.interpolate(f, &mut u).unwrap();
    let mut local = disc.local_vector().unwrap();
    local.set(f64::NAN).unwrap();
    disc.dm().global_to_local(&u, &mut local).unwrap();
    let nodes = disc.node_coordinates();
    let values = local.with_array(<[f64]>::to_vec).unwrap();
    assert_eq!(values.len(), nodes.len());
    // Every entry but the essential ones (on x = 0, left NaN) is f at its node.
    for (&v, &x) in values.iter().zip(nodes) {
        if x[0] == 0.0 {
            assert!(v.is_nan(), "{x:?}: {v}");
        } else {
            assert!((v - f(x)).abs() < 1e-14, "{x:?}: {v}");
        }
    }
    let total = p.comm().sum_int(nodes.len() as i32).unwrap();
    assert!(total >= 31 * 3, "{total}");
}

#[test]
fn boundary_load_integrates_the_fluxes() {
    let p = Petsc::init(&[]).unwrap();
    let mesh = load(&p, "unit_square.msh", 1);
    let disc = Discretization::new(&mesh, 1, &ceed(), 1).unwrap();
    let one = constant(&disc, 1.0);
    // Sides of length 1.
    let load = disc.boundary_load(&[("left", 2.0), ("top", 0.5)]).unwrap();
    let total = one.dot(&load).unwrap();
    assert!((total - 2.5).abs() < 1e-13, "{total}");
    // int_left x v = 0 for P1, which is exact for linear x.
    let mut x = disc.global_vector().unwrap();
    disc.interpolate(|[x, _]| x, &mut x).unwrap();
    let left = disc.boundary_load(&[("left", 1.0)]).unwrap();
    assert!(x.dot(&left).unwrap().abs() < 1e-14);
    // int_bottom y v = 0, int_top y v = 1.
    let mut y = disc.global_vector().unwrap();
    disc.interpolate(|[_, y]| y, &mut y).unwrap();
    let top_bottom = disc
        .boundary_load(&[("top", 1.0), ("bottom", 1.0)])
        .unwrap();
    assert!((y.dot(&top_bottom).unwrap() - 1.0).abs() < 1e-14);
    // A region is not a boundary.
    let e = disc.boundary_load(&[("domain", 1.0)]).unwrap_err();
    assert!(matches!(e, PdeError::Unsupported(_)), "{e}");
    assert!(e.to_string().contains("domain"), "{e}");
    // Essential dofs are not in the load: the corners of "left" are lost to "top".
    let essential = Discretization::with_essential(&mesh, 1, &ceed(), 1, &["left"]).unwrap();
    let one = constant(&essential, 1.0);
    let top = essential.boundary_load(&[("top", 1.0)]).unwrap();
    let total = one.dot(&top).unwrap();
    // The top side has 8 edges after one refinement: the corner on the left side lacks half of
    // the first one.
    assert!((total - 15.0 / 16.0).abs() < 1e-14, "{total}");
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

/// The tests `parallel_on_*_processes` run on several processes: operators and assembly on a distributed
/// mesh (ghost dofs shared between processes, global numbering by owner), and errors found on
/// some processes only. On 3 processes, `inverted_triangle.msh` (2 cells) leaves one without
/// cells.
const PARALLEL_TESTS: [&str; 17] = [
    "mpi_world_has_the_expected_size",
    "mass_integrates_area",
    "diffusion_kills_constants",
    "diagonal_matches_assembled",
    "assembled_matches_shell",
    "changed_shift_reaches_mult_and_diagonal",
    "transport_matches_diffusion_reaction",
    "anisotropic_tensor_integrates_exactly",
    "cells_have_their_area_tag",
    "essential_dofs_leave_global_vectors",
    "apply_local_uses_essential_values",
    "source_integrates_over_the_domain",
    "quadrature_points_cover_every_cell",
    "quadrature_weights_sum_to_the_area",
    "interpolation_takes_the_values_at_the_nodes",
    "boundary_load_integrates_the_fluxes",
    "inverted_element_is_error",
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
