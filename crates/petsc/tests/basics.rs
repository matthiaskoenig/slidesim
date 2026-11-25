//! Tests of the safe PETSc wrappers. nextest runs each test in its own process, so every test
//! initializes PETSc once.

use std::cell::RefCell;
use std::ptr::{null, null_mut};
use std::sync::atomic::{AtomicUsize, Ordering};

use petsc::sys;
use petsc::{Comm, Dm, Error, Ksp, Matrix, Petsc, Result, ShellOperator, Ts, Vector};

fn values(v: &Vector) -> Vec<f64> {
    v.with_array(|a| a.to_vec()).unwrap()
}

fn seq(p: &Petsc, a: &[f64]) -> Vector {
    let mut v = Vector::create_seq(p, a.len()).unwrap();
    v.with_array_mut(|x| x.copy_from_slice(a)).unwrap();
    v
}

#[test]
fn init_and_finalize() {
    let p = Petsc::init(&[]).unwrap();
    assert_eq!(p.comm().size(), 1);
}

#[test]
fn error_carries_message() {
    let p = Petsc::init(&[]).unwrap();
    let e = Vector::create_seq(&p, 3)
        .unwrap()
        .axpy(1.0, &Vector::create_seq(&p, 4).unwrap())
        .unwrap_err();
    // The generic text of PETSC_ERR_ARG_INCOMP (PETSc 3.26 checks vector sizes with it).
    assert!(e.message.contains("Arguments are incompatible"), "{e}");
}

#[test]
fn error_has_specific_message_and_traceback() {
    let p = Petsc::init(&[]).unwrap();
    let e = Vector::create_seq(&p, 3)
        .unwrap()
        .axpy(1.0, &Vector::create_seq(&p, 4).unwrap())
        .unwrap_err();
    assert_eq!(e.code, sys::PETSC_ERR_ARG_INCOMP);
    assert!(
        e.message
            .contains("parameter # 3 global size 4 != parameter # 1 global size 3"),
        "{e}"
    );
    assert!(e.message.contains("\n  VecAXPY() at "), "{e}");
    assert!(e.message.contains("called as sys::VecAXPY("), "{e}");
}

#[test]
fn mat_mult_size_mismatch_is_error() {
    let p = Petsc::init(&[]).unwrap();
    let a = Matrix::shell(&p.comm(), 3, 3, Box::new(Double)).unwrap();
    let x = Vector::create_seq(&p, 4).unwrap();
    let mut y = Vector::create_seq(&p, 3).unwrap();
    let e = a.mult(&x, &mut y).unwrap_err();
    assert_eq!(e.code, sys::PETSC_ERR_ARG_SIZ);
    assert!(e.message.contains("Nonconforming object sizes"), "{e}");
}

#[test]
fn init_twice_is_error() {
    let _p = Petsc::init(&[]).unwrap();
    let e = Petsc::init(&[]).err().unwrap();
    assert!(e.message.contains("once per process"), "{e}");
}

#[test]
fn rank_and_size() {
    let p = Petsc::init(&[]).unwrap();
    assert_eq!(p.comm().rank(), 0);
    assert_eq!(p.comm().size(), 1);
}

#[test]
fn vector_operations() {
    let p = Petsc::init(&[]).unwrap();
    let mut x = seq(&p, &[1.0, 2.0, 2.0]);
    assert_eq!(x.size().unwrap(), 3);
    assert_eq!(x.local_size().unwrap(), 3);
    assert_eq!(x.norm2().unwrap(), 3.0);
    let mut y = Vector::create_seq(&p, 3).unwrap();
    y.set(0.5).unwrap();
    assert_eq!(x.dot(&y).unwrap(), 2.5);
    x.axpy(2.0, &y).unwrap();
    assert_eq!(values(&x), [2.0, 3.0, 3.0]);
    assert_eq!(values(&y), [0.5, 0.5, 0.5]);
}

#[test]
fn panic_in_with_array_mut_restores_array() {
    let p = Petsc::init(&[]).unwrap();
    let mut x = seq(&p, &[1.0, 2.0]);
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        x.with_array_mut(|_| panic!("boom")).unwrap();
    }));
    assert!(r.is_err());
    // The array was restored: the vector is usable again.
    x.set(1.0).unwrap();
    assert_eq!(values(&x), [1.0, 1.0]);
}

/// y = 2 x, diagonal 2.
struct Double;

impl ShellOperator for Double {
    fn mult(&self, x: &Vector, y: &mut Vector) -> Result<()> {
        y.set(0.0)?;
        y.axpy(2.0, x)
    }
    fn diagonal(&self, d: &mut Vector) -> Result<()> {
        d.set(2.0)
    }
}

#[test]
fn shell_matrix_applies() {
    let p = Petsc::init(&[]).unwrap();
    let a = Matrix::shell(&p.comm(), 3, 3, Box::new(Double)).unwrap();
    let x = seq(&p, &[1.0, -2.0, 3.0]);
    let mut y = Vector::create_seq(&p, 3).unwrap();
    a.mult(&x, &mut y).unwrap();
    assert_eq!(values(&y), [2.0, -4.0, 6.0]);
}

/// Swaps the output vector it is given with a vector of its own.
struct Swapping(RefCell<Vector>);

impl ShellOperator for Swapping {
    fn mult(&self, _x: &Vector, y: &mut Vector) -> Result<()> {
        std::mem::swap(y, &mut *self.0.borrow_mut());
        Ok(())
    }
    fn diagonal(&self, _d: &mut Vector) -> Result<()> {
        Ok(())
    }
}

/// A callback cannot destroy the vectors PETSc lends it, even by swapping them out.
#[test]
fn shell_cannot_destroy_lent_vectors() {
    let p = Petsc::init(&[]).unwrap();
    let mine = Vector::create_seq(&p, 3).unwrap();
    let a = Matrix::shell(&p.comm(), 3, 3, Box::new(Swapping(RefCell::new(mine)))).unwrap();
    let x = Vector::create_seq(&p, 3).unwrap();
    let mut y = seq(&p, &[1.0, 2.0, 3.0]);
    a.mult(&x, &mut y).unwrap();
    // y is alive and untouched; the operator now holds (and drops) the lent handle.
    assert_eq!(values(&y), [1.0, 2.0, 3.0]);
    drop(a);
    y.set(1.0).unwrap();
    assert_eq!(values(&y), [1.0, 1.0, 1.0]);
}

#[test]
fn shell_matrix_diagonal() {
    let p = Petsc::init(&[]).unwrap();
    let a = Matrix::shell(&p.comm(), 3, 3, Box::new(Double)).unwrap();
    let mut d = Vector::create_seq(&p, 3).unwrap();
    a.diagonal(&mut d).unwrap();
    assert_eq!(values(&d), [2.0, 2.0, 2.0]);
}

/// Fails in `mult` with a Rust error, or panics in `diagonal`.
struct Failing;

impl ShellOperator for Failing {
    fn mult(&self, _x: &Vector, _y: &mut Vector) -> Result<()> {
        Err(Error {
            code: sys::PETSC_ERR_USER_INPUT,
            message: "negative diffusion coefficient".into(),
        })
    }
    fn diagonal(&self, _d: &mut Vector) -> Result<()> {
        panic!("diagonal exploded")
    }
}

#[test]
fn shell_error_surfaces_from_mat_mult() {
    let p = Petsc::init(&[]).unwrap();
    let a = Matrix::shell(&p.comm(), 3, 3, Box::new(Failing)).unwrap();
    let x = Vector::create_seq(&p, 3).unwrap();
    let mut y = Vector::create_seq(&p, 3).unwrap();
    let e = a.mult(&x, &mut y).unwrap_err();
    assert_eq!(e.code, sys::PETSC_ERR_USER_INPUT);
    assert!(e.message.contains("negative diffusion coefficient"), "{e}");
    assert!(e.message.contains("MatMult() at "), "{e}");
}

#[test]
fn shell_panic_surfaces_as_error() {
    let p = Petsc::init(&[]).unwrap();
    let a = Matrix::shell(&p.comm(), 3, 3, Box::new(Failing)).unwrap();
    let mut d = Vector::create_seq(&p, 3).unwrap();
    let e = a.diagonal(&mut d).unwrap_err();
    assert_eq!(e.code, sys::PETSC_ERR_LIB);
    assert!(e.message.contains("panic"), "{e}");
    assert!(e.message.contains("diagonal exploded"), "{e}");
}

/// Counts its drops.
struct Counted(&'static AtomicUsize);

impl ShellOperator for Counted {
    fn mult(&self, x: &Vector, y: &mut Vector) -> Result<()> {
        Double.mult(x, y)
    }
    fn diagonal(&self, d: &mut Vector) -> Result<()> {
        Double.diagonal(d)
    }
}

impl Drop for Counted {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn shell_context_dropped_with_matrix() {
    static DROPS: AtomicUsize = AtomicUsize::new(0);
    let p = Petsc::init(&[]).unwrap();
    let a = Matrix::shell(&p.comm(), 3, 3, Box::new(Counted(&DROPS))).unwrap();
    assert_eq!(DROPS.load(Ordering::SeqCst), 0);
    drop(a);
    assert_eq!(DROPS.load(Ordering::SeqCst), 1);
}

/// The shell context lives as long as PETSc holds a reference to the matrix.
#[test]
fn shell_context_outlives_wrapper() {
    static DROPS: AtomicUsize = AtomicUsize::new(0);
    let p = Petsc::init(&[]).unwrap();
    let a = Matrix::shell(&p.comm(), 3, 3, Box::new(Counted(&DROPS))).unwrap();
    let mut raw = a.as_raw();
    unsafe { sys::PetscObjectReference(raw as sys::PetscObject) };
    drop(a);
    assert_eq!(DROPS.load(Ordering::SeqCst), 0);
    let x = seq(&p, &[1.0, 1.0, 1.0]);
    let y = Vector::create_seq(&p, 3).unwrap();
    assert_eq!(unsafe { sys::MatMult(raw, x.as_raw(), y.as_raw()) }, 0);
    assert_eq!(values(&y), [2.0, 2.0, 2.0]);
    assert_eq!(unsafe { sys::MatDestroy(&mut raw) }, 0);
    assert_eq!(DROPS.load(Ordering::SeqCst), 1);
}

#[test]
fn aij_coo_assembles_and_sums_repeated_entries() {
    let p = Petsc::init(&[]).unwrap();
    // [[2, -1, 0], [-1, 2, -1], [0, -1, 2]], the (1, 1) entry given as 1 + 1.
    let i = [0, 0, 1, 1, 1, 1, 2, 2];
    let j = [0, 1, 0, 1, 1, 2, 1, 2];
    let mut a = Matrix::aij_coo(&p.comm(), 3, 3, &i, &j).unwrap();
    a.set_values_coo(&[2.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 2.0])
        .unwrap();
    let x = seq(&p, &[1.0, 2.0, 3.0]);
    let mut y = Vector::create_seq(&p, 3).unwrap();
    a.mult(&x, &mut y).unwrap();
    assert_eq!(values(&y), [0.0, 0.0, 4.0]);
    let mut d = Vector::create_seq(&p, 3).unwrap();
    a.diagonal(&mut d).unwrap();
    assert_eq!(values(&d), [2.0, 2.0, 2.0]);
    // New values on the same pattern replace the old ones.
    a.set_values_coo(&[1.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 1.0])
        .unwrap();
    a.mult(&x, &mut y).unwrap();
    assert_eq!(values(&y), [1.0, 2.0, 3.0]);
}

#[test]
fn aij_coo_checks_lengths() {
    let p = Petsc::init(&[]).unwrap();
    let e = Matrix::aij_coo(&p.comm(), 2, 2, &[0, 1], &[0])
        .err()
        .unwrap();
    assert_eq!(e.code, sys::PETSC_ERR_ARG_SIZ);
    let mut a = Matrix::aij_coo(&p.comm(), 2, 2, &[0, 1], &[0, 1]).unwrap();
    let e = a.set_values_coo(&[1.0]).unwrap_err();
    assert_eq!(e.code, sys::PETSC_ERR_ARG_SIZ);
    let e = Matrix::aij_coo(&p.comm(), 2, 2, &[0, 1 << 40], &[0, 1])
        .err()
        .unwrap();
    assert_eq!(e.code, sys::PETSC_ERR_ARG_OUTOFRANGE);
}

#[test]
fn set_values_coo_on_shell_is_error() {
    let p = Petsc::init(&[]).unwrap();
    let mut a = Matrix::shell(&p.comm(), 3, 3, Box::new(Double)).unwrap();
    let e = a.set_values_coo(&[1.0]).unwrap_err();
    assert_eq!(e.code, sys::PETSC_ERR_ARG_WRONGSTATE);
}

/// P1 on a DMPlex box mesh of 2 x 2 quadrilaterals: 9 vertices, one dof each.
fn p1_box(comm: &Comm) -> Dm {
    let mut dm: sys::DM = null_mut();
    let mut fe: sys::PetscFE = null_mut();
    let faces = [2, 2];
    let (lower, upper) = ([0.0, 0.0], [1.0, 1.0]);
    unsafe {
        assert_eq!(
            sys::DMPlexCreateBoxMesh(
                comm.as_raw(),
                2,
                false,
                faces.as_ptr(),
                lower.as_ptr(),
                upper.as_ptr(),
                null(),
                true,
                0,
                false,
                &mut dm
            ),
            0
        );
        assert_eq!(
            sys::PetscFECreateLagrange(
                comm.as_raw(),
                2,
                1,
                false,
                1,
                sys::PETSC_DETERMINE,
                &mut fe
            ),
            0
        );
        assert_eq!(
            sys::DMSetField(dm, 0, null_mut(), fe as sys::PetscObject),
            0
        );
        assert_eq!(sys::DMCreateDS(dm), 0);
        assert_eq!(sys::PetscFEDestroy(&mut fe), 0);
        Dm::from_raw(dm)
    }
}

#[test]
fn dm_vectors() {
    let p = Petsc::init(&[]).unwrap();
    let dm = p1_box(&p.comm());
    let g = Vector::create_global(&dm).unwrap();
    let l = Vector::create_local(&dm).unwrap();
    assert_eq!(g.size().unwrap(), 9);
    assert_eq!(l.local_size().unwrap(), 9);
}

#[test]
fn dm_clone_shares_the_handle() {
    let p = Petsc::init(&[]).unwrap();
    let dm = p1_box(&p.comm());
    let copy = dm.clone();
    assert_eq!(copy.as_raw(), dm.as_raw());
    drop(dm);
    // The clone holds its own reference.
    let mut copy = copy;
    copy.set_from_options().unwrap();
    assert_eq!(Vector::create_global(&copy).unwrap().size().unwrap(), 9);
}

#[test]
fn options_reach_ksp_and_ts() {
    let p = Petsc::init(&["-ksp_type", "cg", "-ts_type", "bdf"]).unwrap();
    let mut ksp = Ksp::create(&p.comm()).unwrap();
    ksp.set_from_options().unwrap();
    let mut ts = Ts::create(&p.comm()).unwrap();
    ts.set_from_options().unwrap();
    let mut name: sys::KSPType = null();
    assert_eq!(unsafe { sys::KSPGetType(ksp.as_raw(), &mut name) }, 0);
    assert_eq!(unsafe { std::ffi::CStr::from_ptr(name) }, c"cg");
    let mut name: sys::TSType = null();
    assert_eq!(unsafe { sys::TSGetType(ts.as_raw(), &mut name) }, 0);
    assert_eq!(unsafe { std::ffi::CStr::from_ptr(name) }, c"bdf");
}

#[test]
fn unknown_type_in_options_is_error() {
    let p = Petsc::init(&["-ksp_type", "nonsense"]).unwrap();
    let mut ksp = Ksp::create(&p.comm()).unwrap();
    let e = ksp.set_from_options().unwrap_err();
    assert!(e.message.contains("nonsense"), "{e}");
}

/// Objects dropped after `Petsc` are leaked instead of destroyed after `PetscFinalize`.
#[test]
fn objects_outliving_petsc_are_safe() {
    let p = Petsc::init(&[]).unwrap();
    let v = Vector::create_seq(&p, 3).unwrap();
    let a = Matrix::shell(&p.comm(), 3, 3, Box::new(Double)).unwrap();
    drop(p);
    drop(v);
    drop(a);
}
