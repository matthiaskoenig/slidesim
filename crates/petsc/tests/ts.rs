//! Tests of time steppers with implicit equations from Rust, their solvers and their converged
//! reasons, and of options scoped to a prefix. nextest runs each test in its own process, so
//! every test initializes PETSc once.

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use petsc::sys;
use petsc::{
    Error, ExactFinalTime, Implicit, Matrix, OptionsScope, Petsc, ProblemType, Result,
    ShellOperator, Ts, VecMut, VecRef, Vector,
};

const N: usize = 4;

/// The operator `(shift + lambda_i) I`: the Jacobian of [`Decay`].
struct Diagonal {
    shift: Rc<Cell<f64>>,
}

/// The rates `lambda_i` of [`Decay`].
fn lambda(i: usize) -> f64 {
    0.5 + i as f64
}

impl ShellOperator for Diagonal {
    fn mult(&self, x: &VecRef<'_>, y: &mut VecMut<'_>) -> Result<()> {
        let x = x.with_array(<[f64]>::to_vec)?;
        y.with_array_mut(|y| {
            for (i, (y, x)) in y.iter_mut().zip(x).enumerate() {
                *y = (self.shift.get() + lambda(i)) * x;
            }
        })
    }

    fn diagonal(&self, d: &mut VecMut<'_>) -> Result<()> {
        d.with_array_mut(|d| {
            for (i, d) in d.iter_mut().enumerate() {
                *d = self.shift.get() + lambda(i);
            }
        })
    }
}

/// `u_t + lambda_i u_i = 0`, recording the shifts and times of its Jacobians (one per stage
/// solve, at the end of the step).
struct Decay {
    shift: Rc<Cell<f64>>,
    shifts: Rc<RefCell<Vec<f64>>>,
    times: Vec<f64>,
    residuals: usize,
    /// Fail the residual from this evaluation on.
    fail_at: Option<usize>,
}

impl Implicit for Decay {
    fn residual(
        &mut self,
        _t: f64,
        u: &VecRef<'_>,
        u_t: &VecRef<'_>,
        f: &mut VecMut<'_>,
    ) -> Result<()> {
        self.residuals += 1;
        if self.fail_at.is_some_and(|n| self.residuals >= n) {
            return Err(Error {
                code: sys::PETSC_ERR_USER,
                message: "residual failed on purpose".to_string(),
            });
        }
        let u = u.with_array(<[f64]>::to_vec)?;
        let u_t = u_t.with_array(<[f64]>::to_vec)?;
        f.with_array_mut(|f| {
            for (i, f) in f.iter_mut().enumerate() {
                *f = u_t[i] + lambda(i) * u[i];
            }
        })
    }

    fn jacobian(&mut self, t: f64, _u: &VecRef<'_>, _u_t: &VecRef<'_>, shift: f64) -> Result<bool> {
        self.shifts.borrow_mut().push(shift);
        self.times.push(t);
        let changed = self.shift.get() != shift;
        self.shift.set(shift);
        Ok(changed)
    }
}

/// A BDF2 stepper of [`Decay`] with time step `dt`, CG and Jacobi, with the options under
/// `decay_`; its solution, all ones; and its Jacobian.
fn decay(p: &Petsc, dt: f64, fail_at: Option<usize>) -> (Ts<Decay>, Vector, Matrix) {
    let shift = Rc::new(Cell::new(f64::NAN));
    let jacobian = Matrix::shell(
        &p.comm_self(),
        N,
        N,
        Box::new(Diagonal {
            shift: Rc::clone(&shift),
        }),
    )
    .unwrap();
    let equation = Decay {
        shift,
        shifts: Rc::default(),
        times: Vec::new(),
        residuals: 0,
        fail_at,
    };
    let mut ts = Ts::implicit(&p.comm_self(), equation, &jacobian, &jacobian).unwrap();
    ts.set_options_prefix("decay_").unwrap();
    ts.set_type("bdf").unwrap();
    ts.set_bdf_order(2).unwrap();
    ts.set_problem_type(ProblemType::Linear).unwrap();
    ts.set_adapt_type("none").unwrap();
    ts.set_exact_final_time(ExactFinalTime::MatchStep).unwrap();
    ts.set_time(0.0).unwrap();
    ts.set_time_step(dt).unwrap();
    let mut ksp = ts.snes().unwrap().ksp().unwrap();
    ksp.set_type("cg").unwrap();
    ksp.set_pc_type("jacobi").unwrap();
    ts.set_from_options().unwrap();
    let mut u = Vector::create_seq(p, N).unwrap();
    u.set(1.0).unwrap();
    (ts, u, jacobian)
}

/// The largest error of `u` against `exp(-lambda_i t)`.
fn error(u: &Vector, t: f64) -> f64 {
    u.with_array(|u| {
        u.iter()
            .enumerate()
            .map(|(i, u)| (u - (-lambda(i) * t).exp()).abs())
            .fold(0.0, f64::max)
    })
    .unwrap()
}

#[test]
fn bdf2_converges_with_order_two() {
    let p = Petsc::init(&["-decay_ksp_rtol", "1e-14"]).unwrap();
    let errors: Vec<f64> = [0.02, 0.01]
        .into_iter()
        .map(|dt| {
            let (mut ts, mut u, _) = decay(&p, dt, None);
            ts.set_max_time(1.0).unwrap();
            ts.solve(&mut u).unwrap();
            assert_eq!(ts.time().unwrap(), 1.0);
            assert_eq!(ts.step_number().unwrap(), (1.0 / dt).round() as usize);
            let reason = ts.converged_reason().unwrap();
            assert_eq!(reason.code, sys::TS_CONVERGED_TIME);
            assert_eq!(reason.name, "CONVERGED_TIME");
            assert!(!reason.diverged());
            let snes = ts.snes().unwrap();
            assert!(snes.converged_reason().unwrap().code > 0);
            let ksp = snes.ksp().unwrap();
            assert!(ksp.converged_reason().unwrap().code > 0);
            // Every step solved one system, in at least one iteration.
            let steps = ts.step_number().unwrap();
            let iterations = ts.linear_iterations().unwrap();
            assert!(iterations >= steps && iterations >= ksp.iterations().unwrap());
            error(&u, 1.0)
        })
        .collect();
    let order = (errors[0] / errors[1]).log2();
    assert!((1.9..=2.1).contains(&order), "{errors:?}: order {order}");
}

/// The state of `m` (`PetscObjectStateGet`), which increases with each change.
fn state(m: &Matrix) -> sys::PetscObjectState {
    let mut state = 0;
    assert_eq!(
        unsafe { sys::PetscObjectStateGet(m.as_raw() as sys::PetscObject, &mut state) },
        0
    );
    state
}

/// The number of changes of shift in `shifts`, from none.
fn changes(shifts: &[f64]) -> usize {
    1 + shifts.windows(2).filter(|w| w[0] != w[1]).count()
}

#[test]
fn jacobian_sees_the_shifts_and_marks_changes() {
    let p = Petsc::init(&[]).unwrap();
    let (mut ts, mut u, jacobian) = decay(&p, 0.1, None);
    let before = state(&jacobian);
    ts.set_max_time(0.5).unwrap();
    ts.solve(&mut u).unwrap();
    // The first step in two halves, of BDF1 and BDF2 (TSBDF's start), then steps of BDF2.
    let shifts = ts.context().shifts.borrow().clone();
    assert_eq!(shifts.len(), 6, "{shifts:?}");
    let expected = [20.0, 30.0, 15.0, 15.0, 15.0, 15.0];
    for (s, e) in shifts.iter().zip(expected) {
        assert!((s - e).abs() < 1e-12, "{shifts:?}");
    }
    // The matrix was marked changed with each change of shift (rounding included).
    assert_eq!((state(&jacobian) - before) as usize, changes(&shifts));
    // Restarted, the next step starts afresh; the context is reachable between solves.
    ts.context_mut().shifts.borrow_mut().clear();
    ts.restart_step().unwrap();
    ts.set_max_time(0.6).unwrap();
    ts.solve(&mut u).unwrap();
    let shifts = ts.context().shifts.borrow().clone();
    assert!((shifts[0] - 20.0).abs() < 1e-12, "{shifts:?}");
    assert!((ts.time().unwrap() - 0.6).abs() < 1e-15);
}

/// The sizes of the steps between the times `times` (each step's end), from `start`.
fn step_sizes(start: f64, times: &[f64]) -> Vec<f64> {
    std::iter::once(start)
        .chain(times.iter().copied())
        .collect::<Vec<_>>()
        .windows(2)
        .map(|w| w[1] - w[0])
        .collect()
}

fn assert_steps(sizes: &[f64], expected: &[f64]) {
    assert_eq!(sizes.len(), expected.len(), "{sizes:?}");
    for (h, e) in sizes.iter().zip(expected) {
        assert!((h - e).abs() < 1e-12, "{sizes:?} != {expected:?}");
    }
}

#[test]
fn solves_continue_from_the_time_reached() {
    let p = Petsc::init(&["-decay_ksp_rtol", "1e-14"]).unwrap();
    let (mut ts, mut u, _) = decay(&p, 0.1, None);
    // 0.1 (two halves: the start of BDF2), then the remaining 0.15 in two: TSAdapt matches the
    // final time by splitting a remainder under two steps rather than taking a short last one.
    ts.set_max_time(0.25).unwrap();
    ts.solve(&mut u).unwrap();
    assert_eq!(ts.time().unwrap(), 0.25);
    assert_eq!(ts.step_number().unwrap(), 3);
    let sizes = step_sizes(0.0, &ts.context().times);
    assert_steps(&sizes, &[0.05, 0.05, 0.075, 0.075]);
    // TSAdapt `none` keeps the shortened step for the next solve: the caller sets it back.
    assert!((ts.time_step().unwrap() - 0.075).abs() < 1e-12);
    ts.set_time_step(0.1).unwrap();
    ts.context_mut().times.clear();
    ts.set_max_time(1.0).unwrap();
    ts.solve(&mut u).unwrap();
    assert_eq!(ts.time().unwrap(), 1.0);
    assert_eq!(ts.step_number().unwrap(), 11);
    let sizes = step_sizes(0.25, &ts.context().times);
    assert_steps(&sizes, &[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.075, 0.075]);
    assert!(error(&u, 1.0) < 1e-2, "{}", error(&u, 1.0));
}

#[test]
fn adaptive_solves_continue_with_the_choice_of_the_adaptor() {
    let p = Petsc::init(&["-decay_ksp_rtol", "1e-14"]).unwrap();
    let (mut ts, mut u, _) = decay(&p, 0.01, None);
    ts.set_adapt_type("basic").unwrap();
    ts.set_max_time(0.5).unwrap();
    ts.solve(&mut u).unwrap();
    assert_eq!(ts.time().unwrap(), 0.5);
    // The last step was shortened to end at 0.5; the step chosen after it was not.
    let times = ts.context().times.clone();
    let last = times[times.len() - 1] - times[times.len() - 2];
    let next = ts.time_step().unwrap();
    assert!(
        next > 0.01 && next != last,
        "next {next}, last {last}: {times:?}"
    );
    // The next solve starts with it.
    ts.context_mut().times.clear();
    ts.set_max_time(2.0).unwrap();
    ts.solve(&mut u).unwrap();
    assert_eq!(ts.time().unwrap(), 2.0);
    let first = ts.context().times[0] - 0.5;
    assert!((first - next).abs() < 1e-12, "{first} != {next}");
}

#[test]
fn not_converged_linear_solve_is_error() {
    let p = Petsc::init(&[
        "-decay_ksp_max_it",
        "1",
        "-decay_ksp_type",
        "gmres",
        "-decay_pc_type",
        "none",
        "-decay_ksp_error_if_not_converged",
    ])
    .unwrap();
    let (mut ts, mut u, _) = decay(&p, 0.1, None);
    ts.set_max_time(1.0).unwrap();
    let e = ts.solve(&mut u).unwrap_err();
    assert_eq!(e.code, sys::PETSC_ERR_NOT_CONVERGED, "{e}");
    let reason = ts
        .snes()
        .unwrap()
        .ksp()
        .unwrap()
        .converged_reason()
        .unwrap();
    assert_eq!(reason.code, sys::KSP_DIVERGED_ITS);
    assert_eq!(reason.name, "DIVERGED_ITS");
    assert!(reason.diverged());
    assert_eq!(ts.time().unwrap(), 0.0);
}

#[test]
fn failed_step_sets_a_diverged_reason() {
    let p = Petsc::init(&[
        "-decay_ksp_max_it",
        "1",
        "-decay_ksp_type",
        "gmres",
        "-decay_pc_type",
        "none",
        "-decay_ts_error_if_step_fails",
        "false",
    ])
    .unwrap();
    let (mut ts, mut u, _) = decay(&p, 0.1, None);
    ts.set_max_time(1.0).unwrap();
    ts.solve(&mut u).unwrap();
    let reason = ts.converged_reason().unwrap();
    assert_eq!(reason.code, sys::TS_DIVERGED_NONLINEAR_SOLVE);
    assert_eq!(reason.name, "DIVERGED_NONLINEAR_SOLVE");
    let snes = ts.snes().unwrap().converged_reason().unwrap();
    assert_eq!(snes.name, "DIVERGED_LINEAR_SOLVE");
}

#[test]
fn residual_error_reaches_solve() {
    let p = Petsc::init(&[]).unwrap();
    let (mut ts, mut u, _) = decay(&p, 0.1, Some(3));
    ts.set_max_time(1.0).unwrap();
    let e = ts.solve(&mut u).unwrap_err();
    assert_eq!(e.code, sys::PETSC_ERR_USER, "{e}");
    assert!(e.message.contains("residual failed on purpose"), "{e}");
    assert!(e.message.contains("Implicit::residual"), "{e}");
}

#[test]
fn options_scope_sets_and_removes_prefixed_options() {
    let p = Petsc::init(&["-s_ksp_type", "gmres"]).unwrap();
    let scope = OptionsScope::new(
        "s_",
        &["-ksp_monitor -ksp_type cg -ksp_rtol 1e-3", "-ksp_rtol 1e-6"],
    )
    .unwrap();
    assert_eq!(p.option("-s_ksp_monitor").unwrap().as_deref(), Some(""));
    // The command line holds over the scope.
    assert_eq!(p.option("-s_ksp_type").unwrap().as_deref(), Some("gmres"));
    // A later text overrides an earlier one.
    assert_eq!(p.option("-s_ksp_rtol").unwrap().as_deref(), Some("1e-6"));
    assert_eq!(p.option("-ksp_type").unwrap(), None);
    // One scope per prefix (option names are case insensitive).
    let e = OptionsScope::new("S_", &[]).unwrap_err();
    assert_eq!(e.code, sys::PETSC_ERR_ARG_WRONGSTATE);
    assert!(e.message.contains("S_"), "{e}");
    let other = OptionsScope::new("t_", &["-ksp_type bcgs"]).unwrap();
    drop(scope);
    // As before: unset, or the value from the command line.
    assert_eq!(p.option("-s_ksp_monitor").unwrap(), None);
    assert_eq!(p.option("-s_ksp_rtol").unwrap(), None);
    assert_eq!(p.option("-s_ksp_type").unwrap().as_deref(), Some("gmres"));
    assert_eq!(p.option("-t_ksp_type").unwrap().as_deref(), Some("bcgs"));
    drop(other);
    assert_eq!(p.option("-t_ksp_type").unwrap(), None);
    OptionsScope::new("s_", &[]).unwrap();
}

#[test]
fn options_already_set_survive_the_scope_and_its_drop() {
    // From the command line, and set by the caller (in another case: names are case
    // insensitive).
    let p = Petsc::init(&["-s_ksp_rtol", "1e-2", "-s_ksp_monitor"]).unwrap();
    p.set_option("-S_KSP_MAX_IT", Some("5")).unwrap();
    let scope = OptionsScope::new(
        "s_",
        &["-ksp_rtol 1e-10 -ksp_max_it 100 -ksp_monitor_short -ksp_type cg"],
    )
    .unwrap();
    assert_eq!(p.option("-s_ksp_rtol").unwrap().as_deref(), Some("1e-2"));
    assert_eq!(p.option("-s_ksp_max_it").unwrap().as_deref(), Some("5"));
    assert_eq!(p.option("-s_ksp_monitor").unwrap().as_deref(), Some(""));
    // Options not set before are the scope's.
    assert_eq!(p.option("-s_ksp_type").unwrap().as_deref(), Some("cg"));
    assert_eq!(
        p.option("-s_ksp_monitor_short").unwrap().as_deref(),
        Some("")
    );
    drop(scope);
    // The options set before are untouched, the scope's are gone.
    assert_eq!(p.option("-s_ksp_rtol").unwrap().as_deref(), Some("1e-2"));
    assert_eq!(p.option("-s_ksp_max_it").unwrap().as_deref(), Some("5"));
    assert_eq!(p.option("-s_ksp_monitor").unwrap().as_deref(), Some(""));
    assert_eq!(p.option("-s_ksp_type").unwrap(), None);
    assert_eq!(p.option("-s_ksp_monitor_short").unwrap(), None);
}

#[test]
fn options_scope_reaches_the_solvers_of_its_prefix() {
    let p = Petsc::init(&[]).unwrap();
    let _scope = OptionsScope::new("decay_", &["-ksp_type gmres -ts_type beuler"]).unwrap();
    let (ts, _, _) = decay(&p, 0.1, None);
    let mut name: sys::TSType = std::ptr::null();
    assert_eq!(unsafe { sys::TSGetType(ts.as_raw(), &mut name) }, 0);
    assert_eq!(unsafe { std::ffi::CStr::from_ptr(name) }, c"beuler");
    let ksp = ts.snes().unwrap().ksp().unwrap();
    let mut name: sys::KSPType = std::ptr::null();
    assert_eq!(unsafe { sys::KSPGetType(ksp.as_raw(), &mut name) }, 0);
    assert_eq!(unsafe { std::ffi::CStr::from_ptr(name) }, c"gmres");
}

#[test]
fn invalid_options_scopes_are_errors() {
    let _p = Petsc::init(&[]).unwrap();
    for prefix in ["", "1a_", "a b_", "a-b_", "-a_"] {
        let e = OptionsScope::new(prefix, &[]).unwrap_err();
        assert_eq!(e.code, sys::PETSC_ERR_ARG_WRONG, "{prefix:?}: {e}");
    }
    let e = OptionsScope::new("a_", &["-options_file /nonexistent/options"]).unwrap_err();
    assert!(e.message.contains("/nonexistent/options"), "{e}");
    // The failed scope left nothing behind.
    OptionsScope::new("a_", &[]).unwrap();
}
