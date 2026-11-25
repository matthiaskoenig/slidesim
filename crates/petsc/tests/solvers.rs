//! Tests of linear solvers (`KSP`), preconditioners (`PC`, multigrid `PCMG`) and the matrix
//! options they rely on. nextest runs each test in its own process, so every test initializes
//! PETSc once.

use petsc::sys;
use petsc::{Comm, Ksp, Matrix, Petsc, Result, ShellOperator, VecMut, VecRef, Vector};

fn values(v: &Vector) -> Vec<f64> {
    v.with_array(|a| a.to_vec()).unwrap()
}

/// `scale * tridiag(-1, 2, -1)` of size `n`: the 1D Laplacian of linear elements of length
/// `1 / scale` with zero Dirichlet values at both ends.
fn laplacian(comm: &Comm, n: usize, scale: f64) -> Matrix {
    let (mut i, mut j, mut v) = (Vec::new(), Vec::new(), Vec::new());
    for r in 0..n as i64 {
        for (c, value) in [(r - 1, -1.0), (r, 2.0), (r + 1, -1.0)] {
            if (0..n as i64).contains(&c) {
                i.push(r);
                j.push(c);
                v.push(scale * value);
            }
        }
    }
    let mut a = Matrix::aij_coo(comm, n, n, &i, &j).unwrap();
    a.set_values_coo(&v).unwrap();
    a
}

/// The linear interpolation from `m` coarse nodes to the `2 m + 1` fine nodes of the same
/// interval, coarse node `c` at fine node `2 c + 1`.
fn interpolation_1d(comm: &Comm, m: usize) -> Matrix {
    let (mut i, mut j, mut v) = (Vec::new(), Vec::new(), Vec::new());
    for c in 0..m as sys::PetscInt {
        for (f, value) in [(2 * c, 0.5), (2 * c + 1, 1.0), (2 * c + 2, 0.5)] {
            i.push(f);
            j.push(c);
            v.push(value);
        }
    }
    // A rectangular matrix: rows of the fine nodes, columns of the coarse ones.
    let n = 2 * m + 1;
    let mut raw = std::ptr::null_mut();
    unsafe {
        assert_eq!(sys::MatCreate(comm.as_raw(), &mut raw), 0);
        let (n, m) = (n as sys::PetscInt, m as sys::PetscInt);
        assert_eq!(sys::MatSetSizes(raw, n, m, n, m), 0);
        assert_eq!(sys::MatSetType(raw, sys::MATAIJ.as_ptr().cast()), 0);
        assert_eq!(
            sys::MatSetPreallocationCOO(
                raw,
                i.len() as sys::PetscCount,
                i.as_mut_ptr(),
                j.as_mut_ptr()
            ),
            0
        );
        assert_eq!(sys::MatSetValuesCOO(raw, v.as_ptr(), sys::INSERT_VALUES), 0);
    }
    // SAFETY: a new matrix, whose reference the wrapper takes.
    unsafe { Matrix::from_raw(raw) }
}

fn rhs(p: &Petsc, n: usize) -> Vector {
    let mut b = Vector::create_seq(p, n).unwrap();
    b.set(1.0).unwrap();
    b
}

#[test]
fn ksp_solves_to_its_tolerances() {
    let p = Petsc::init(&[]).unwrap();
    let n = 50;
    let a = laplacian(&p.comm(), n, 1.0);
    let mut ksp = Ksp::create(&p.comm()).unwrap();
    ksp.set_operators(&a, &a).unwrap();
    ksp.set_type("cg").unwrap();
    ksp.set_pc_type("jacobi").unwrap();
    ksp.set_tolerances(Some(1e-10), None, None).unwrap();
    let b = rhs(&p, n);
    let mut x = Vector::create_seq(&p, n).unwrap();
    ksp.solve(&b, &mut x).unwrap();
    assert_eq!(ksp.type_name().unwrap(), "cg");
    assert!(!ksp.converged_reason().unwrap().diverged());
    // CG on the 1D Laplacian: at most n iterations, about n / 2 for this symmetric load.
    let iterations = ksp.iterations().unwrap();
    assert!((1..=n).contains(&iterations), "{iterations}");
    let mut r = Vector::create_seq(&p, n).unwrap();
    a.mult(&x, &mut r).unwrap();
    r.axpy(-1.0, &b).unwrap();
    assert!(r.norm2().unwrap() < 1e-9 * b.norm2().unwrap());
    // At most 3 iterations: not converged.
    ksp.set_tolerances(None, None, Some(3)).unwrap();
    ksp.solve(&b, &mut x).unwrap();
    assert_eq!(ksp.iterations().unwrap(), 3);
    assert_eq!(ksp.converged_reason().unwrap().name, "DIVERGED_ITS");
}

#[test]
fn pc_type_and_options_prefix() {
    let p = Petsc::init(&[]).unwrap();
    let mut ksp = Ksp::create(&p.comm()).unwrap();
    ksp.set_options_prefix("u_").unwrap();
    let mut pc = ksp.pc().unwrap();
    assert_eq!(pc.options_prefix().unwrap(), "u_");
    pc.set_type("jacobi").unwrap();
    assert_eq!(ksp.pc().unwrap().type_name().unwrap(), "jacobi");
    assert_eq!(pc.as_raw(), ksp.pc().unwrap().as_raw());
}

/// Whether PETSc knows `a` to be symmetric positive definite (`MatIsSPDKnown`).
fn spd_known(a: &Matrix) -> bool {
    let (mut set, mut spd) = (false, false);
    assert_eq!(
        unsafe { sys::MatIsSPDKnown(a.as_raw(), &mut set, &mut spd) },
        0
    );
    set && spd
}

#[test]
fn spd_survives_new_values() {
    let p = Petsc::init(&[]).unwrap();
    let mut a = laplacian(&p.comm(), 4, 1.0);
    assert!(!spd_known(&a));
    a.set_spd().unwrap();
    assert!(spd_known(&a));
    a.set_values_coo(&[2.0; 10]).unwrap();
    a.mark_changed().unwrap();
    assert!(spd_known(&a));
}

/// `y = 2 x`.
struct Double;

impl ShellOperator for Double {
    fn mult(&self, x: &VecRef<'_>, y: &mut VecMut<'_>) -> Result<()> {
        y.set(0.0)?;
        y.axpy(2.0, x)
    }
    fn diagonal(&self, d: &mut VecMut<'_>) -> Result<()> {
        d.set(2.0)
    }
}

fn state(a: &Matrix) -> sys::PetscObjectState {
    let mut s = 0;
    assert_eq!(
        unsafe { sys::PetscObjectStateGet(a.as_raw() as sys::PetscObject, &mut s) },
        0
    );
    s
}

#[test]
fn mark_changed_advances_the_state() {
    let p = Petsc::init(&[]).unwrap();
    let mut a = Matrix::shell(&p.comm(), 3, 3, Box::new(Double)).unwrap();
    let before = state(&a);
    a.mark_changed().unwrap();
    assert!(state(&a) > before);
}

/// The 1D Laplacian `tridiag(-1, 2, -1)` as a shell operator on sequential vectors.
struct ShellLaplacian;

impl ShellOperator for ShellLaplacian {
    fn mult(&self, x: &VecRef<'_>, y: &mut VecMut<'_>) -> Result<()> {
        let x = x.with_array(|x| x.to_vec())?;
        y.with_array_mut(|y| {
            let n = x.len();
            for i in 0..n {
                let left = if i > 0 { x[i - 1] } else { 0.0 };
                let right = if i + 1 < n { x[i + 1] } else { 0.0 };
                y[i] = 2.0 * x[i] - left - right;
            }
        })
    }
    fn diagonal(&self, d: &mut VecMut<'_>) -> Result<()> {
        d.set(2.0)
    }
}

#[test]
fn chebyshev_estimates_the_eigenvalues_of_a_shell() {
    let p = Petsc::init(&[]).unwrap();
    // Small enough for the estimator (10 iterations) to find the extreme eigenvalues.
    let n = 8;
    let mut a = Matrix::shell(&p.comm(), n, n, Box::new(ShellLaplacian)).unwrap();
    a.set_spd().unwrap();
    let mut ksp = Ksp::create(&p.comm()).unwrap();
    ksp.set_operators(&a, &a).unwrap();
    ksp.set_type("chebyshev").unwrap();
    ksp.set_pc_type("jacobi").unwrap();
    // The whole spectrum, [emin, 1.1 emax]: Chebyshev converges as a solver.
    ksp.chebyshev_est_eig_set(1.0, 0.0, 0.0, 1.1).unwrap();
    ksp.set_tolerances(Some(1e-6), None, Some(100)).unwrap();
    let b = rhs(&p, n);
    let mut x = Vector::create_seq(&p, n).unwrap();
    ksp.solve(&b, &mut x).unwrap();
    // The estimator is CG (the matrix is SPD), and its estimates made Chebyshev converge.
    let estimator = ksp.chebyshev_est_eig_ksp().unwrap().unwrap();
    assert_eq!(estimator.type_name().unwrap(), "cg");
    assert!(!ksp.converged_reason().unwrap().diverged());
    // Not a Chebyshev solver: no estimator.
    let mut cg = Ksp::create(&p.comm()).unwrap();
    cg.set_type("cg").unwrap();
    assert!(cg.chebyshev_est_eig_ksp().unwrap().is_none());
}

#[test]
fn two_level_multigrid() {
    let p = Petsc::init(&[]).unwrap();
    let comm = p.comm();
    let m = 31;
    let n = 2 * m + 1;
    // Linear elements on [0, 1]: n fine nodes of spacing 1 / (n + 1), m coarse ones of twice it.
    let fine = laplacian(&comm, n, (n + 1) as f64);
    let coarse = laplacian(&comm, m, (n + 1) as f64 / 2.0);
    let mut ksp = Ksp::create(&comm).unwrap();
    ksp.set_operators(&fine, &fine).unwrap();
    ksp.set_type("cg").unwrap();
    ksp.set_tolerances(Some(1e-10), None, None).unwrap();
    let mut pc = ksp.pc().unwrap();
    pc.set_type("mg").unwrap();
    assert_eq!(pc.mg_levels().unwrap(), 0);
    pc.mg_set_levels(2).unwrap();
    assert_eq!(pc.mg_levels().unwrap(), 2);
    pc.mg_set_interpolation(1, &interpolation_1d(&comm, m))
        .unwrap();
    let mut coarse_solve = pc.mg_smoother(0).unwrap();
    coarse_solve.set_operators(&coarse, &coarse).unwrap();
    assert_eq!(coarse_solve.type_name().unwrap(), "preonly");
    let smoother = pc.mg_smoother(1).unwrap();
    assert_eq!(smoother.type_name().unwrap(), "chebyshev");
    // The load of f = 1: h at every node.
    let mut b = rhs(&p, n);
    b.set(1.0 / (n + 1) as f64).unwrap();
    let mut x = Vector::create_seq(&p, n).unwrap();
    ksp.solve(&b, &mut x).unwrap();
    assert!(!ksp.converged_reason().unwrap().diverged());
    // Two-grid with an exact coarse solve: a few iterations, where Jacobi takes about n / 2.
    let iterations = ksp.iterations().unwrap();
    assert!(iterations <= 8, "{iterations}");
    let u = values(&x);
    // The exact solution of -u'' = 1, u(0) = u(1) = 0 at the nodes: x (1 - x) / 2.
    for (i, u) in u.iter().enumerate() {
        let x = (i + 1) as f64 / (n + 1) as f64;
        assert!((u - x * (1.0 - x) / 2.0).abs() < 1e-8, "{i}: {u}");
    }
}
