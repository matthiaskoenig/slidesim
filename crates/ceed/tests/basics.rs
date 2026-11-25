//! Tests of the safe libCEED wrappers: mass operators on simplex meshes, assembly, and errors.

use std::ffi::{c_int, c_void};
use std::slice;

use ceed::sys::{CeedInt, CeedScalar};
use ceed::{
    Basis, CVector, Ceed, DEFAULT_RESOURCE, ElemRestriction, EvalMode, FieldVector, Operator,
    QFunction, Topology,
};

/// Mass QFunction: `v = c u w |det(dx/dX)|`, with the context `[dim, c]`.
/// Inputs: `dx` (`dim * dim` values per point), `w` (quadrature weights), `u`; output: `v`.
unsafe extern "C" fn mass_qf(
    ctx: *mut c_void,
    q: CeedInt,
    input: *const *const CeedScalar,
    output: *const *mut CeedScalar,
) -> c_int {
    let q = q as usize;
    let ctx = unsafe { slice::from_raw_parts(ctx as *const f64, 2) };
    let (dim, scale) = (ctx[0] as usize, ctx[1]);
    let (dx, w, u, v) = unsafe {
        (
            slice::from_raw_parts(*input, dim * dim * q),
            slice::from_raw_parts(*input.add(1), q),
            slice::from_raw_parts(*input.add(2), q),
            slice::from_raw_parts_mut(*output, q),
        )
    };
    for i in 0..q {
        let a = |r: usize, c: usize| dx[(r * dim + c) * q + i];
        let det = match dim {
            1 => a(0, 0),
            2 => a(0, 0) * a(1, 1) - a(0, 1) * a(1, 0),
            _ => {
                a(0, 0) * (a(1, 1) * a(2, 2) - a(1, 2) * a(2, 1))
                    - a(0, 1) * (a(1, 0) * a(2, 2) - a(1, 2) * a(2, 0))
                    + a(0, 2) * (a(1, 0) * a(2, 1) - a(1, 1) * a(2, 0))
            }
        };
        v[i] = scale * u[i] * w[i] * det.abs();
    }
    0
}

/// A QFunction which always fails, without a libCEED message.
unsafe extern "C" fn failing_qf(
    _ctx: *mut c_void,
    _q: CeedInt,
    _input: *const *const CeedScalar,
    _output: *const *mut CeedScalar,
) -> c_int {
    1
}

/// Copies its input to its output (one value per point): `u` (values at points) to `q`.
unsafe extern "C" fn copy_qf(
    _ctx: *mut c_void,
    q: CeedInt,
    input: *const *const CeedScalar,
    output: *const *mut CeedScalar,
) -> c_int {
    let q = q as usize;
    let (u, v) = unsafe {
        (
            slice::from_raw_parts(*input, q),
            slice::from_raw_parts_mut(*output, q),
        )
    };
    v.copy_from_slice(u);
    0
}

/// A simplex mesh with coordinates of degree `x_degree`: node coordinates (interleaved) and
/// cells (the nodes of the Lagrange basis of `x_degree` each).
struct Mesh {
    dim: usize,
    x_degree: usize,
    coords: Vec<f64>,
    cells: Vec<i32>,
}

impl Mesh {
    /// `n` intervals on [0, 1].
    fn interval(n: usize) -> Mesh {
        Mesh {
            dim: 1,
            x_degree: 1,
            coords: (0..=n).map(|i| i as f64 / n as f64).collect(),
            cells: (0..n as i32).flat_map(|e| [e, e + 1]).collect(),
        }
    }
}

/// The mass operator (scaled by `scale`) of `mesh` with the basis `degree` (nodes per cell given
/// by `u_offsets`, `u_size` nodes in all). Only the operator is returned: it keeps the
/// restrictions, bases, coordinates and QFunction alive.
fn mass_operator(
    ceed: &Ceed,
    mesh: &Mesh,
    degree: usize,
    u_offsets: &[i32],
    u_size: usize,
    scale: f64,
) -> Operator {
    let dim = mesh.dim;
    let q_order = 2 * degree;
    let b_x = Basis::h1_lagrange_simplex(ceed, dim, dim, mesh.x_degree, q_order).unwrap();
    let ncell = mesh.cells.len() / b_x.num_nodes();
    let x_offsets: Vec<i32> = mesh.cells.iter().map(|v| v * dim as i32).collect();
    let r_x = ElemRestriction::new(
        ceed,
        ncell,
        b_x.num_nodes(),
        dim,
        1,
        mesh.coords.len(),
        &x_offsets,
    )
    .unwrap();
    let x = CVector::from_slice(ceed, &mesh.coords).unwrap();
    let b_u = Basis::h1_lagrange_simplex(ceed, dim, 1, degree, q_order).unwrap();
    let r_u = ElemRestriction::new(ceed, ncell, b_u.num_nodes(), 1, 1, u_size, u_offsets).unwrap();

    // SAFETY: the fields below and the context [dim, scale] match what mass_qf reads.
    let mut qf =
        unsafe { QFunction::interior(ceed, 1, Some(mass_qf), concat!(file!(), ":mass_qf")) }
            .unwrap();
    qf.add_input("dx", dim * dim, EvalMode::Grad).unwrap();
    qf.add_input("w", 1, EvalMode::Weight).unwrap();
    qf.add_input("u", 1, EvalMode::Interp).unwrap();
    qf.add_output("v", 1, EvalMode::Interp).unwrap();
    qf.set_context(&[dim as f64, scale]).unwrap();

    let mut op = Operator::new(ceed, &qf).unwrap();
    op.set_field("dx", Some(&r_x), Some(&b_x), FieldVector::Passive(&x))
        .unwrap();
    op.set_field("w", None, Some(&b_x), FieldVector::None)
        .unwrap();
    op.set_field("u", Some(&r_u), Some(&b_u), FieldVector::Active)
        .unwrap();
    op.set_field("v", Some(&r_u), Some(&b_u), FieldVector::Active)
        .unwrap();
    op
}

/// The P1 mass operator of `n` intervals on [0, 1].
fn mass_1d(ceed: &Ceed, n: usize, scale: f64) -> Operator {
    let mesh = Mesh::interval(n);
    let offsets = mesh.cells.clone();
    mass_operator(ceed, &mesh, 1, &offsets, n + 1, scale)
}

fn ceed() -> Ceed {
    Ceed::new(DEFAULT_RESOURCE).unwrap()
}

/// `1^T M u` for the mass operator `op`.
fn integral(ceed: &Ceed, op: &Operator, u: &[f64]) -> f64 {
    let u = CVector::from_slice(ceed, u).unwrap();
    let mut v = CVector::new(ceed, u.len()).unwrap();
    op.apply(&u, &mut v).unwrap();
    v.with_array(|a| a.iter().sum()).unwrap()
}

#[test]
fn mass_operator_integrates_one() {
    let ceed = ceed();
    let op = mass_1d(&ceed, 4, 1.0);
    let total = integral(&ceed, &op, &[1.0; 5]);
    assert!((total - 1.0).abs() < 1e-14, "{total}");
}

#[test]
fn error_carries_message() {
    let e = Ceed::new("/nonexistent").unwrap_err();
    assert!(e.message.contains("/nonexistent"), "{e}");
}

#[test]
fn default_resource_resolves_exactly() {
    assert_eq!(DEFAULT_RESOURCE, "/cpu/self/opt/blocked");
    assert_eq!(ceed().resource().unwrap(), DEFAULT_RESOURCE);
}

#[test]
fn prefix_resources_are_errors_not_aborts() {
    // libCEED itself would pick a backend for these and abort in its initialization
    // ("/cpu/self/o"), or pick a backend which differs between targets ("/cpu/self").
    for resource in [
        "/cpu/self",
        "/cpu/self/o",
        "/cpu/self/opt/blocked:x",
        "help",
    ] {
        let e = Ceed::new(resource).unwrap_err();
        assert!(e.message.contains(resource), "{e}");
        assert!(e.message.contains(DEFAULT_RESOURCE), "lists backends: {e}");
    }
    assert!(Ceed::new("/cpu/self/opt\0").is_err());
}

#[test]
fn resources_lists_default() {
    assert!(
        ceed::resources()
            .unwrap()
            .iter()
            .any(|r| r == DEFAULT_RESOURCE)
    );
}

#[test]
fn libceed_error_has_message_and_call() {
    let ceed = ceed();
    // SAFETY: never applied (setting the field fails), so mass_qf is never called.
    let mut qf = unsafe { QFunction::interior(&ceed, 1, Some(mass_qf), "qf.h:mass_qf") }.unwrap();
    qf.add_input("u", 2, EvalMode::Interp).unwrap();
    qf.add_output("v", 1, EvalMode::Interp).unwrap();
    let mut op = Operator::new(&ceed, &qf).unwrap();
    let r = ElemRestriction::new(&ceed, 1, 2, 1, 1, 2, &[0, 1]).unwrap();
    let b = Basis::h1_lagrange_simplex(&ceed, 1, 1, 1, 2).unwrap();
    // The QFunction field has 2 components, the restriction and basis 1.
    let e = op
        .set_field("u", Some(&r), Some(&b), FieldVector::Active)
        .unwrap_err();
    assert_eq!(e.code, ceed::sys::CEED_ERROR_DIMENSION, "{e}");
    assert!(e.message.contains("Field 'u' of size 2"), "{e}");
    assert!(
        e.message.contains("called as sys::CeedOperatorSetField("),
        "{e}"
    );
}

#[test]
fn stale_message_never_attaches() {
    let ceed = ceed();
    // An error with a libCEED message: a source without a function name.
    // SAFETY: creation fails, so mass_qf is never called.
    let e = unsafe { QFunction::interior(&ceed, 1, Some(mass_qf), "no-function") }.unwrap_err();
    assert!(e.message.contains("no-function"), "{e}");

    // An error without one: a QFunction returning an error code.
    let mesh = Mesh::interval(2);
    let r = ElemRestriction::new(&ceed, 2, 2, 1, 1, 3, &mesh.cells).unwrap();
    let b = Basis::h1_lagrange_simplex(&ceed, 1, 1, 1, 2).unwrap();
    // SAFETY: failing_qf accesses nothing.
    let mut qf =
        unsafe { QFunction::interior(&ceed, 1, Some(failing_qf), "qf.h:failing_qf") }.unwrap();
    qf.add_input("u", 1, EvalMode::Interp).unwrap();
    qf.add_output("v", 1, EvalMode::Interp).unwrap();
    let mut op = Operator::new(&ceed, &qf).unwrap();
    op.set_field("u", Some(&r), Some(&b), FieldVector::Active)
        .unwrap();
    op.set_field("v", Some(&r), Some(&b), FieldVector::Active)
        .unwrap();
    let u = CVector::from_slice(&ceed, &[1.0; 3]).unwrap();
    let mut v = CVector::new(&ceed, 3).unwrap();
    let e = op.apply(&u, &mut v).unwrap_err();
    assert!(!e.message.contains("no-function"), "{e}");
    assert!(e.message.contains("no message"), "{e}");
    assert!(
        e.message.contains("called as sys::CeedOperatorApply("),
        "{e}"
    );
}

#[test]
fn vector_roundtrip() {
    let ceed = ceed();
    let mut v = CVector::from_slice(&ceed, &[1.0, 2.0, 3.0]).unwrap();
    assert_eq!(v.len(), 3);
    v.with_array_mut(|a| a[1] = 5.0).unwrap();
    assert_eq!(v.with_array(|a| a.to_vec()).unwrap(), [1.0, 5.0, 3.0]);
    v.set_value(2.0).unwrap();
    assert_eq!(v.with_array(|a| a.to_vec()).unwrap(), [2.0; 3]);
    let z = CVector::new(&ceed, 2).unwrap();
    assert_eq!(z.with_array(|a| a.to_vec()).unwrap(), [0.0; 2]);
    let empty = CVector::new(&ceed, 0).unwrap();
    assert!(empty.is_empty());
    assert_eq!(empty.with_array(|a| a.len()).unwrap(), 0);
    let empty = CVector::from_slice(&ceed, &[]).unwrap();
    assert_eq!(empty.to_vec().unwrap(), [0.0; 0]);
}

#[test]
fn vector_stays_usable_after_panic_in_closure() {
    let ceed = ceed();
    let mut v = CVector::from_slice(&ceed, &[1.0, 2.0]).unwrap();
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        v.with_array_mut(|_| panic!("in closure")).unwrap()
    }));
    assert!(r.is_err());
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        v.with_array(|_| panic!("in closure")).unwrap()
    }));
    assert!(r.is_err());
    // The array was restored: it can be accessed again.
    v.with_array_mut(|a| a[0] = 3.0).unwrap();
    assert_eq!(v.with_array(|a| a.to_vec()).unwrap(), [3.0, 2.0]);
}

#[test]
fn diagonal_matches_coo() {
    let ceed = ceed();
    let n = 4;
    let h = 1.0 / n as f64;
    let op = mass_1d(&ceed, n, 1.0);

    let mut diag = CVector::new(&ceed, n + 1).unwrap();
    op.assemble_diagonal(&mut diag).unwrap();
    let diag = diag.with_array(|a| a.to_vec()).unwrap();
    for (i, d) in diag.iter().enumerate() {
        let expected = if i == 0 || i == n {
            h / 3.0
        } else {
            2.0 * h / 3.0
        };
        assert!((d - expected).abs() < 1e-15, "{i}: {d} != {expected}");
    }

    let (rows, cols, values) = op.assemble_coo().unwrap();
    let values = values.with_array(|a| a.to_vec()).unwrap();
    assert_eq!(rows.len(), values.len());
    assert_eq!(cols.len(), values.len());
    assert_eq!(values.len(), n * 4, "2x2 entries per element");
    let mut coo_diag = vec![0.0; n + 1];
    for ((&i, &j), &a) in rows.iter().zip(&cols).zip(&values) {
        assert!((0..=n as i64).contains(&i) && (0..=n as i64).contains(&j));
        if i == j {
            coo_diag[i as usize] += a;
        }
    }
    for (a, b) in coo_diag.iter().zip(&diag) {
        assert!((a - b).abs() < 1e-15, "{a} != {b}");
    }
    let total: f64 = values.iter().sum();
    assert!((total - 1.0).abs() < 1e-14, "{total}");
}

#[test]
fn context_scales_operator() {
    let ceed = ceed();
    let op = mass_1d(&ceed, 3, 2.5);
    let total = integral(&ceed, &op, &[1.0; 4]);
    assert!((total - 2.5).abs() < 1e-14, "{total}");
}

#[test]
fn p2_triangle_integrates_x_squared() {
    let ceed = ceed();
    // P2 nodes in lattice order: (0,0) (1/2,0) (1,0) (0,1/2) (1/2,1/2) (0,1).
    let p2 = [0.0, 0.0, 0.5, 0.0, 1.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.0, 1.0];
    let meshes = [
        Mesh {
            dim: 2,
            x_degree: 1,
            coords: vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            cells: vec![0, 1, 2],
        },
        // The same triangle with P2 coordinates: the Jacobian comes from the P2 gradients.
        Mesh {
            dim: 2,
            x_degree: 2,
            coords: p2.to_vec(),
            cells: (0..6).collect(),
        },
    ];
    for mesh in &meshes {
        let op = mass_operator(&ceed, mesh, 2, &[0, 1, 2, 3, 4, 5], 6, 1.0);
        let total = integral(&ceed, &op, &[0.0, 0.25, 1.0, 0.0, 0.25, 0.0]);
        assert!((total - 1.0 / 12.0).abs() < 1e-15, "{total}");
    }
}

#[test]
fn p1_tetrahedron_volume() {
    let ceed = ceed();
    let mesh = Mesh {
        dim: 3,
        x_degree: 1,
        coords: vec![0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 3.0],
        cells: vec![0, 1, 2, 3],
    };
    let op = mass_operator(&ceed, &mesh, 1, &[0, 1, 2, 3], 4, 1.0);
    let total = integral(&ceed, &op, &[1.0; 4]);
    assert!((total - 1.0).abs() < 1e-15, "{total}");
}

#[test]
fn simplex_quadrature_is_exact() {
    let ceed = ceed();
    // Reference volumes 1, 1/2, 1/6; weights sum to them for every order.
    for (dim, volume) in [(1, 1.0), (2, 0.5), (3, 1.0 / 6.0)] {
        for q_order in 0..6 {
            let b = Basis::h1_lagrange_simplex(&ceed, dim, 1, 1, q_order).unwrap();
            assert_eq!(b.dim(), dim);
            let w: f64 = b.quadrature_weights().unwrap().iter().sum();
            assert!((w - volume).abs() < 1e-15, "dim {dim} order {q_order}: {w}");
        }
    }
    // A P3 basis: (3+1)(3+2)/2 nodes in 2D.
    let b = Basis::h1_lagrange_simplex(&ceed, 2, 1, 3, 6).unwrap();
    assert_eq!(b.num_nodes(), 10);
}

#[test]
fn apply_length_mismatch_is_error() {
    let ceed = ceed();
    let op = mass_1d(&ceed, 4, 1.0);
    let u = CVector::from_slice(&ceed, &[1.0; 4]).unwrap();
    let mut v = CVector::new(&ceed, 5).unwrap();
    let e = op.apply(&u, &mut v).unwrap_err();
    assert!(e.message.contains("length 4"), "{e}");
    let mut d = CVector::new(&ceed, 3).unwrap();
    assert!(op.assemble_diagonal(&mut d).is_err());
}

#[test]
fn objects_of_another_ceed_are_error() {
    let a = ceed();
    let b = ceed();
    let op = mass_1d(&a, 2, 1.0);
    let u = CVector::from_slice(&b, &[1.0; 3]).unwrap();
    let mut v = CVector::new(&a, 3).unwrap();
    let e = op.apply(&u, &mut v).unwrap_err();
    assert!(e.message.contains("another Ceed"), "{e}");
}

#[test]
fn restriction_is_validated() {
    let ceed = ceed();
    // Too few offsets.
    assert!(ElemRestriction::new(&ceed, 2, 2, 1, 1, 3, &[0, 1, 1]).is_err());
    // An offset out of range (the opt backends do not check offsets).
    let e = ElemRestriction::new(&ceed, 2, 2, 1, 1, 3, &[0, 1, 1, 3]).unwrap_err();
    assert!(e.message.contains("out of range"), "{e}");
    // The last component out of range.
    assert!(ElemRestriction::new(&ceed, 1, 2, 2, 2, 3, &[0, 1]).is_err());
    let r = ElemRestriction::strided(&ceed, 2, 3, 1, 6).unwrap();
    assert_eq!(r.l_size(), 6);
}

#[test]
fn basis_arrays_are_validated() {
    let ceed = ceed();
    let e = Basis::h1(
        &ceed,
        Topology::Triangle,
        1,
        3,
        1,
        &[1.0 / 3.0; 3],
        &[0.0; 5],
        &[1.0 / 3.0; 2],
        &[0.5],
    )
    .unwrap_err();
    assert!(e.message.contains("grad"), "{e}");
    assert!(Basis::h1_lagrange_simplex(&ceed, 4, 1, 1, 2).is_err());
    assert!(Basis::h1_lagrange_simplex(&ceed, 2, 1, 0, 2).is_err());
}

#[test]
fn qfunction_is_validated() {
    let ceed = ceed();
    // SAFETY (both): creation fails, so no function is ever called.
    assert!(unsafe { QFunction::interior(&ceed, 1, None, "qf.h:none") }.is_err());
    assert!(unsafe { QFunction::interior(&ceed, 0, Some(mass_qf), "qf.h:mass_qf") }.is_err());
}

#[test]
fn apply_slices_matches_apply() {
    let ceed = ceed();
    let op = mass_1d(&ceed, 4, 2.0);
    let u = [0.0, 1.0, 4.0, 9.0, 16.0];
    let mut expected = CVector::new(&ceed, 5).unwrap();
    op.apply(&CVector::from_slice(&ceed, &u).unwrap(), &mut expected)
        .unwrap();
    // The output is overwritten, and the arrays can be used again.
    let mut v = [f64::NAN; 5];
    for _ in 0..2 {
        op.apply_slices(&u, &mut v).unwrap();
        assert_eq!(v.as_slice(), expected.to_vec().unwrap());
    }
    assert_eq!(u, [0.0, 1.0, 4.0, 9.0, 16.0]);
}

#[test]
fn apply_slices_length_mismatch_is_error() {
    let ceed = ceed();
    let op = mass_1d(&ceed, 4, 1.0);
    let e = op.apply_slices(&[1.0; 4], &mut [0.0; 5]).unwrap_err();
    assert!(e.message.contains("input array has length 4"), "{e}");
    let e = op.apply_slices(&[1.0; 5], &mut [0.0; 6]).unwrap_err();
    assert!(e.message.contains("output array has length 6"), "{e}");
}

#[test]
fn strided_with_lays_out_values_as_given() {
    let ceed = ceed();
    // x at the quadrature points of 3 intervals on [0, 1], stored at e * q + i.
    let n = 3;
    let mesh = Mesh::interval(n);
    let b = Basis::h1_lagrange_simplex(&ceed, 1, 1, 1, 4).unwrap();
    let q = b.num_quadrature_points();
    let r_x = ElemRestriction::new(&ceed, n, 2, 1, 1, n + 1, &mesh.cells).unwrap();
    let r_q = ElemRestriction::strided_with(&ceed, n, q, 1, n * q, [1, q, q]).unwrap();
    // SAFETY: one input and one output of one value per point, as copy_qf reads and writes.
    let mut qf =
        unsafe { QFunction::interior(&ceed, 1, Some(copy_qf), concat!(file!(), ":copy_qf")) }
            .unwrap();
    qf.add_input("u", 1, EvalMode::Interp).unwrap();
    qf.add_output("q", 1, EvalMode::None).unwrap();
    let mut op = Operator::new(&ceed, &qf).unwrap();
    op.set_field("u", Some(&r_x), Some(&b), FieldVector::Active)
        .unwrap();
    op.set_field("q", Some(&r_q), None, FieldVector::Active)
        .unwrap();
    let mut xq = vec![0.0; n * q];
    op.apply_slices(&mesh.coords, &mut xq).unwrap();
    for (k, x) in xq.iter().enumerate() {
        let (e, n) = ((k / q) as f64, n as f64);
        assert!(e / n < *x && *x < (e + 1.0) / n, "{xq:?}");
    }
    assert!(xq.windows(2).all(|w| w[0] < w[1]), "{xq:?}");
}

#[test]
fn strided_with_is_validated() {
    let ceed = ceed();
    // Element 1 reaches entry 2 * 3 + 2 * 1 = 8.
    let e = ElemRestriction::strided_with(&ceed, 2, 3, 1, 8, [1, 3, 6]).unwrap_err();
    assert!(e.message.contains("entry 8"), "{e}");
    assert!(ElemRestriction::strided_with(&ceed, 2, 3, 1, 9, [1, 3, 6]).is_ok());
    // Component 1 of node 2 of element 1: 2 + 3 + 6.
    assert!(ElemRestriction::strided_with(&ceed, 2, 3, 2, 11, [1, 3, 6]).is_err());
    assert!(ElemRestriction::strided_with(&ceed, 2, 3, 2, 12, [1, 3, 6]).is_ok());
    assert!(ElemRestriction::strided_with(&ceed, 2, 0, 1, 12, [1, 3, 6]).is_err());
    assert!(ElemRestriction::strided_with(&ceed, 2, 3, 1, 12, [usize::MAX, 3, 6]).is_err());
}
