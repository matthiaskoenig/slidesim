//! Tests of the safe PETSc wrappers. nextest runs each test in its own process, so every test
//! initializes PETSc once.

use std::cell::RefCell;
use std::ptr::{null, null_mut};
use std::sync::atomic::{AtomicUsize, Ordering};

use petsc::sys;
use petsc::{
    AsVecMut, Comm, Dm, Error, Ksp, Matrix, Petsc, Result, ShellOperator, Ts, VecMut, VecRef,
    Vector,
};

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
    x.zero().unwrap();
    assert_eq!(values(&x), [0.0; 3]);
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
    fn mult(&self, x: &VecRef<'_>, y: &mut VecMut<'_>) -> Result<()> {
        y.set(0.0)?;
        y.axpy(2.0, x)
    }
    fn diagonal(&self, d: &mut VecMut<'_>) -> Result<()> {
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
    fn mult(&self, _x: &VecRef<'_>, _y: &mut VecMut<'_>) -> Result<()> {
        Err(Error {
            code: sys::PETSC_ERR_USER_INPUT,
            message: "negative diffusion coefficient".into(),
        })
    }
    fn diagonal(&self, _d: &mut VecMut<'_>) -> Result<()> {
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
    fn mult(&self, x: &VecRef<'_>, y: &mut VecMut<'_>) -> Result<()> {
        Double.mult(x, y)
    }
    fn diagonal(&self, d: &mut VecMut<'_>) -> Result<()> {
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

/// After PetscFinalize every wrapper call is an error, not a call into a finalized PETSc.
#[test]
fn use_after_finalize_is_error() {
    let p = Petsc::init(&[]).unwrap();
    // Owned: the communicator outlives `p`.
    let owned = p.comm().duplicate().unwrap();
    let comm = owned.comm();
    let mut v = Vector::create_seq(&p, 3).unwrap();
    drop(p);
    let e = v.set(1.0).unwrap_err();
    assert_eq!(e.code, sys::PETSC_ERR_ORDER);
    assert!(e.message.contains("finalized"), "{e}");
    assert!(e.message.contains("VecSet"), "{e}");
    let e = v.norm2().unwrap_err();
    assert_eq!(e.code, sys::PETSC_ERR_ORDER);
    let e = Ksp::create(&comm).err().unwrap();
    assert_eq!(e.code, sys::PETSC_ERR_ORDER);
    assert!(e.message.contains("KSPCreate"), "{e}");
    let e = Matrix::shell(&comm, 3, 3, Box::new(Double)).err().unwrap();
    assert_eq!(e.code, sys::PETSC_ERR_ORDER);
}

/// Owns `Petsc`, so dropping the operator drops `Petsc` inside a PETSc callback.
struct OwnsPetsc(RefCell<Option<Petsc>>);

impl ShellOperator for OwnsPetsc {
    /// Drops `Petsc` during the call, then uses PETSc.
    fn mult(&self, x: &VecRef<'_>, y: &mut VecMut<'_>) -> Result<()> {
        drop(self.0.borrow_mut().take());
        y.set(0.0)?;
        y.axpy(3.0, x)
    }
    fn diagonal(&self, d: &mut VecMut<'_>) -> Result<()> {
        d.set(3.0)
    }
}

/// `Petsc` dropped with its matrix (in MatDestroy): PETSc stays initialized and usable.
#[test]
fn petsc_dropped_by_matrix_destroy_stays_initialized() {
    let p = Petsc::init(&[]).unwrap();
    // Owned: the communicator outlives `p`.
    let owned = p.comm().duplicate().unwrap();
    let comm = owned.comm();
    let x = seq(&p, &[1.0, 2.0, 3.0]);
    let mut y = Vector::create_seq(&p, 3).unwrap();
    let a = Matrix::shell(&comm, 3, 3, Box::new(OwnsPetsc(RefCell::new(Some(p))))).unwrap();
    drop(a);
    let mut ksp = Ksp::create(&comm).unwrap();
    ksp.set_from_options().unwrap();
    let b = Matrix::shell(&comm, 3, 3, Box::new(Double)).unwrap();
    b.mult(&x, &mut y).unwrap();
    assert_eq!(values(&y), [2.0, 4.0, 6.0]);
}

/// `Petsc` dropped during `mult` (in MatMult): the call completes, PETSc stays usable.
#[test]
fn petsc_dropped_during_mult_stays_initialized() {
    let p = Petsc::init(&[]).unwrap();
    // Owned: the communicator outlives `p`.
    let owned = p.comm().duplicate().unwrap();
    let comm = owned.comm();
    let x = seq(&p, &[1.0, 2.0, 3.0]);
    let mut y = Vector::create_seq(&p, 3).unwrap();
    let a = Matrix::shell(&comm, 3, 3, Box::new(OwnsPetsc(RefCell::new(Some(p))))).unwrap();
    a.mult(&x, &mut y).unwrap();
    assert_eq!(values(&y), [3.0, 6.0, 9.0]);
    drop(a);
    assert_eq!(y.norm2().unwrap(), 126.0_f64.sqrt());
    Ksp::create(&comm).unwrap();
}

/// Set in the run of [`abort_ends_every_process`] under `mpiexec`.
const ABORT_RUN: &str = "SLIDESIM_TEST_ABORT_RUN";

#[test]
fn abort_ends_every_process() {
    if std::env::var_os(ABORT_RUN).is_some() {
        // Process 0 aborts, while the others wait for it in a collective call.
        let p = Petsc::init(&[]).unwrap();
        let comm = p.comm();
        if comm.rank() == 0 {
            // Output still in the buffers of Rust's stdout and of C's stdio (no newline): flushed
            // before the processes end.
            print!("from Rust before the abort; ");
            // SAFETY: a NUL terminated format without conversions.
            unsafe { printf(c"from C before the abort".as_ptr()) };
            p.abort_world(3);
        }
        comm.sum_real(1.0).unwrap();
        panic!("process {} left the collective call", comm.rank());
    }
    let Some(mpiexec) = test_support::Mpiexec::find() else {
        eprintln!("skipped: no mpiexec");
        return;
    };
    let mut cmd = mpiexec.command(2, std::env::current_exe().unwrap());
    // Without capture: the output of process 0 goes to its standard output.
    cmd.args([
        "--exact",
        "abort_ends_every_process",
        "--test-threads",
        "1",
        "--nocapture",
    ])
    .env(ABORT_RUN, "1")
    .env(test_support::MPI_RANKS, "2");
    let run = test_support::run_with_timeout(cmd, std::time::Duration::from_secs(60));
    let output = format!("{}\n{}", run.stdout, run.stderr);
    let status = run.status.unwrap_or_else(|| panic!("hangs:\n{output}"));
    assert!(!status.success(), "{output}");
    assert!(!output.contains("left the collective call"), "{output}");
    // In either order: C's stdout need not be buffered (MPICH makes it unbuffered).
    for text in ["from Rust before the abort", "from C before the abort"] {
        assert!(run.stdout.contains(text), "{text:?} is lost:\n{output}");
    }
}

/// Set in the run of [`panic_ends_every_process`] under `mpiexec`.
const PANIC_RUN: &str = "SLIDESIM_TEST_PANIC_RUN";

/// With [`Petsc::abort_world_on_panic`], a panic on one process ends every process, the others
/// waiting for it in a collective call: no hang in `PetscFinalize` while unwinding.
#[test]
fn panic_ends_every_process() {
    if std::env::var_os(PANIC_RUN).is_some() {
        let p = Petsc::init(&[]).unwrap();
        p.abort_world_on_panic(4);
        let comm = p.comm();
        if comm.rank() == 1 {
            panic!("a bug on one process");
        }
        comm.sum_real(1.0).unwrap();
        println!("process {} left the collective call", comm.rank());
        return;
    }
    let Some(mpiexec) = test_support::Mpiexec::find() else {
        eprintln!("skipped: no mpiexec");
        return;
    };
    let mut cmd = mpiexec.command(2, std::env::current_exe().unwrap());
    cmd.args([
        "--exact",
        "panic_ends_every_process",
        "--test-threads",
        "1",
        "--nocapture",
    ])
    .env(PANIC_RUN, "1")
    .env(test_support::MPI_RANKS, "2");
    let run = test_support::run_with_timeout(cmd, std::time::Duration::from_secs(60));
    let output = format!("{}\n{}", run.stdout, run.stderr);
    let status = run.status.unwrap_or_else(|| panic!("hangs:\n{output}"));
    assert!(!status.success(), "{output}");
    assert!(!output.contains("left the collective call"), "{output}");
    assert!(
        run.stderr
            .contains("[rank 1] panic: a bug on one process (at crates/petsc/tests/basics.rs:"),
        "{output}"
    );
}

unsafe extern "C" {
    fn printf(format: *const std::ffi::c_char, ...) -> std::ffi::c_int;
}

/// The bytes of every process on the root, in the order of the ranks, whatever their lengths
/// (none included); `None` on the other processes. A root which is not a rank is an error on
/// every process.
#[test]
fn gather_bytes_collects_in_rank_order() {
    let p = Petsc::init(&[]).unwrap();
    let comm = p.comm();
    let (rank, size) = (comm.rank(), comm.size());
    assert_eq!(size, test_support::ranks());
    // Process r sends r + 1 bytes of value r, process 1 none.
    let bytes = |r: usize| {
        if r == 1 {
            Vec::new()
        } else {
            vec![r as u8; r + 1]
        }
    };
    for root in [0, size - 1] {
        let gathered = comm.gather_bytes(root, &bytes(rank)).unwrap();
        if rank == root {
            let expected: Vec<Vec<u8>> = (0..size).map(bytes).collect();
            assert_eq!(gathered, Some(expected), "root {root}");
        } else {
            assert_eq!(gathered, None, "root {root}");
        }
    }
    // Nothing from any process.
    let gathered = comm.gather_bytes(0, &[]).unwrap();
    assert_eq!(gathered, (rank == 0).then(|| vec![Vec::new(); size]));
    let e = comm.gather_bytes(size, b"x").unwrap_err();
    assert!(e.message.contains(&format!("rank {size}")), "{e}");
}

/// Every process gets the bytes each process sent to it, in the order of the ranks, whatever
/// their lengths (none included); a list of other than one vector per process is an error on
/// every process.
#[test]
fn exchange_bytes_delivers_to_each_process() {
    let p = Petsc::init(&[]).unwrap();
    let comm = p.comm();
    let (rank, size) = (comm.rank(), comm.size());
    assert_eq!(size, test_support::ranks());
    // From p to q: p + 2 q bytes, of the value 10 p + q (none from 0 to 0).
    let bytes = |from: usize, to: usize| vec![(10 * from + to) as u8; from + 2 * to];
    let sends: Vec<Vec<u8>> = (0..size).map(|to| bytes(rank, to)).collect();
    let received = comm.exchange_bytes(&sends).unwrap();
    let expected: Vec<Vec<u8>> = (0..size).map(|from| bytes(from, rank)).collect();
    assert_eq!(received, expected);
    // Nothing at all.
    let received = comm.exchange_bytes(&vec![Vec::new(); size]).unwrap();
    assert_eq!(received, vec![Vec::<u8>::new(); size]);
    // One process gives one vector too many.
    let mut sends = vec![vec![1u8]; size];
    if rank == size - 1 {
        sends.push(Vec::new());
    }
    let e = comm.exchange_bytes(&sends).unwrap_err();
    assert!(
        e.message.contains("vectors of bytes") || e.message.contains("another process"),
        "{e}"
    );
}

/// The tests `parallel_on_*_processes` run on several processes.
const PARALLEL_TESTS: [&str; 2] = [
    "gather_bytes_collects_in_rank_order",
    "exchange_bytes_delivers_to_each_process",
];

#[test]
fn parallel_on_2_processes() {
    test_support::parallel(2, &PARALLEL_TESTS);
}

#[test]
fn parallel_on_3_processes() {
    test_support::parallel(3, &PARALLEL_TESTS);
}
