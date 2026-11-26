//! The integral over time of the solution of a [`crate::Problem`] by the weights of its BDF
//! steps: [`Integral`], for the mass balance over an interval
//! ([`crate::Problem::interval_balance`]).

use petsc::{VecRef, Vector};

use crate::{PdeError, Result};

/// The shift of a stage which [`Integral`] computes, relative to the shift PETSc's `TSBDF` took,
/// beyond which the integral does not follow the steps (rounding of the times aside).
const SHIFT_MATCH: f64 = 1e-8;

/// The integral over time of the solution `c` and the source `f` of a problem, by the weights of
/// its BDF steps, so that the mass balance of the steps adds up exactly.
///
/// A stage of `TSBDF` at time `t_0` solves its equations with the derivative of the polynomial
/// through its solution `c_0` and those of the history points `t_1` (and `t_2`):
/// `c_t = a_0 c_0 + a_1 c_1 + a_2 c_2` (`a_2 = 0` for the first order), the Lagrange weights of the
/// derivative at `t_0`, whose sum is zero. The mass balance holds at the stage for any linear
/// functional `T` of the terms (the storage, the sources, the inflows): `T(c_t) = sum` of the
/// other terms at `c_0`. Hence, with `X` an integral of the terms which satisfies
/// `a_0 (X_0 - X_1) - a_2 (X_2 - X_1) = x_0` at every stage, for `x_0` the terms at the stage,
/// the integral of the storage is exactly the change of the amount, and the integrals of the
/// terms add up as the terms do. `X` is affine in `x`, so this integrates the vectors the terms are
/// linear in: the solution (and the source, in case it changes between steps) at the stages,
/// `X_0 = X_1 + (x_0 - a_2 (X_2 - X_1)) / a_0`, from the integrals at the history points. Of a
/// function linear in time it is its exact integral.
///
/// The history follows `TSBDF` of order 2: a step from `t_s` to `t_0` takes the points `t_s` and
/// the start of the step before; the first step after a restart (the first step, and after every
/// change of the solution, [`Integral::restart`]) solves a half step of the first order from
/// `t_s` first, then the step with the points `t_s` and the half step. A stage which an adaptive
/// step control rejects is followed by a stage of the same step at an earlier time, which
/// replaces it and every stage after it. Each stage checks its weight `a_0` against the shift
/// the step took; if they differ, the integral is broken: [`Integral::check`] says why.
///
/// The source changes between steps only ([`Integral::source_changes`]). The integral of the
/// source at each point is `b + s f + r g`: `f` the source the steps apply, `b` and `g` vectors
/// fixed at its last change (the integrals at the history points then), `s` and `r` numbers of
/// the point. The recurrence keeps this form, so a stage integrates the source by updating two
/// numbers (a constant source adds `f` times the length of the step, the history aside), and the
/// vectors of the solution of points which are no longer needed are used again: a stage
/// allocates nothing.
#[derive(Debug)]
pub(crate) struct Integral {
    /// The history points of the next step, oldest first: the last is the current time.
    history: Vec<Node>,
    /// The stages of the current step solved so far, in order.
    stages: Vec<Node>,
    /// Whether the current step is the first after a restart.
    restart: bool,
    /// Why the integral does not follow the steps, if it does not.
    broken: Option<String>,
    /// `b` and `g` of the integrals of the source, at the quadrature points.
    base: Vec<f64>,
    rest: Vec<f64>,
    /// The vectors of points no longer needed, for the next stages.
    spare: Vec<Vector>,
}

/// The integral at a time.
#[derive(Debug)]
pub(crate) struct Node {
    /// s.
    pub(crate) t: f64,
    /// Of the solution: a global vector (without the essential dofs).
    pub(crate) c: Vector,
    /// Of the source: `s` and `r` of `b + s f + r g` (see [`Integral`]).
    source: [f64; 2],
}

impl Integral {
    /// The integral from the time `t`, zero there: `zero` is a global vector of zeros of the
    /// solution and `points` the number of quadrature points. The first step restarts.
    pub(crate) fn new(t: f64, zero: Vector, points: usize) -> Integral {
        Integral {
            history: vec![Node {
                t,
                c: zero,
                source: [0.0; 2],
            }],
            stages: Vec::new(),
            restart: true,
            broken: None,
            base: vec![0.0; points],
            rest: vec![0.0; points],
            spare: Vec::new(),
        }
    }

    /// The integral at the current time: the end of the last step.
    pub(crate) fn current(&self) -> &Node {
        self.history
            .last()
            .expect("the history has the current time")
    }

    /// The integral of the source at the current time, at the quadrature points, for the source
    /// `f` the steps apply.
    pub(crate) fn current_source(&self, f: &[f64]) -> Vec<f64> {
        self.source(self.current(), f)
    }

    fn source(&self, node: &Node, f: &[f64]) -> Vec<f64> {
        let [s, r] = node.source;
        (self.base.iter().zip(f).zip(&self.rest))
            .map(|((b, f), g)| b + s * f + r * g)
            .collect()
    }

    /// The source changes from `f`, the source of the steps so far, between two steps.
    pub(crate) fn source_changes(&mut self, f: &[f64]) {
        // Of the history points, the current one last: `b` and `g` from their integrals.
        let current = self.current_source(f);
        self.rest = match self.history.as_slice() {
            [before, _] => {
                let before = self.source(before, f);
                before.iter().zip(&current).map(|(x, b)| x - b).collect()
            }
            _ => vec![0.0; current.len()],
        };
        self.base = current;
        let n = self.history.len();
        for (k, node) in self.history.iter_mut().enumerate() {
            node.source = [0.0, if k + 1 == n { 0.0 } else { 1.0 }];
        }
        // None between steps; any other would be replaced by the next stage.
        self.recycle_stages(f64::NEG_INFINITY);
    }

    /// The next step starts the time integration afresh, from the current time.
    pub(crate) fn restart(&mut self) {
        let current = self
            .history
            .pop()
            .expect("the history has the current time");
        self.spare.extend(self.history.drain(..).map(|node| node.c));
        self.history.push(current);
        self.recycle_stages(f64::NEG_INFINITY);
        self.restart = true;
    }

    /// Drops the stages at `t` and after, keeping their vectors.
    fn recycle_stages(&mut self, t: f64) {
        let keep = self.stages.iter().take_while(|s| s.t < t).count();
        self.spare
            .extend(self.stages.drain(keep..).map(|node| node.c));
    }

    /// An error if the integral does not follow the steps.
    pub(crate) fn check(&self) -> Result<()> {
        match &self.broken {
            None => Ok(()),
            Some(why) => Err(PdeError::Unsupported(format!(
                "the mass balance over an interval does not follow the time steps: {why}"
            ))),
        }
    }

    /// A stage at the time `t` with the solution `y`, solved with the shift `shift` (the weight
    /// of its solution in its time derivative), and the source of the steps.
    pub(crate) fn stage(&mut self, t: f64, y: &VecRef<'_>, shift: f64) -> Result<()> {
        // A stage at the time of an earlier one, or before: the earlier ones were rejected.
        // The stages are in the order of their times.
        self.recycle_stages(t);
        let spare = self.spare.pop();
        let points = if self.restart {
            match self.stages.as_slice() {
                [] => Some((self.current(), None)),
                [half] => Some((self.current(), Some(half))),
                _ => None,
            }
        } else {
            match self.history.as_slice() {
                [before, current] => Some((current, Some(before))),
                _ => None,
            }
        };
        let Some((p1, p2)) = points else {
            // Not a method of at most two stages per step (another TS type, or options).
            self.broken
                .get_or_insert_with(|| format!("a stage at t = {t} s of an unknown history"));
            return Ok(());
        };
        let (a0, a2) = match p2 {
            None => (1.0 / (t - p1.t), 0.0),
            Some(p2) => {
                let (t1, t2) = (p1.t, p2.t);
                (
                    1.0 / (t - t1) + 1.0 / (t - t2),
                    (t - t1) / ((t2 - t) * (t2 - t1)),
                )
            }
        };
        let mismatch = (a0 - shift).abs() > SHIFT_MATCH * shift.abs() || shift.is_nan();
        // X_0 = X_1 + (x_0 - a_2 (X_2 - X_1)) / a_0, with x_0 the solution and the source.
        let mut c = match spare {
            Some(c) => c,
            None => p1.c.duplicate()?,
        };
        c.with_array_mut(|x0| {
            p1.c.with_array(|x1| {
                y.with_array(|y| match p2 {
                    None => {
                        for ((x0, x1), y) in x0.iter_mut().zip(x1).zip(y) {
                            *x0 = x1 + y / a0;
                        }
                        Ok::<(), petsc::Error>(())
                    }
                    Some(p2) => p2.c.with_array(|x2| {
                        for (((x0, x1), y), x2) in x0.iter_mut().zip(x1).zip(y).zip(x2) {
                            *x0 = x1 + (y - a2 * (x2 - x1)) / a0;
                        }
                    }),
                })
            })
        })????;
        // The source is 1 * f, and 0 * b and g.
        let [s1, r1] = p1.source;
        let source = match p2 {
            None => [s1 + 1.0 / a0, r1],
            Some(p2) => {
                let [s2, r2] = p2.source;
                [s1 + (1.0 - a2 * (s2 - s1)) / a0, r1 - a2 * (r2 - r1) / a0]
            }
        };
        if mismatch {
            self.broken.get_or_insert_with(|| {
                format!(
                    "a stage at t = {t} s with the shift {shift}, where the history points give \
                     {a0}"
                )
            });
        }
        self.stages.push(Node { t, c, source });
        Ok(())
    }

    /// The step which ended at the time `t` is accepted: its last stage.
    pub(crate) fn step(&mut self, t: f64) {
        let Some(end) = self.stages.pop() else {
            self.broken
                .get_or_insert_with(|| format!("a step to t = {t} s without a stage"));
            return;
        };
        if end.t != t {
            self.broken.get_or_insert_with(|| {
                format!(
                    "a step to t = {t} s whose last stage was at t = {} s",
                    end.t
                )
            });
        }
        let start = self
            .history
            .pop()
            .expect("the history has the current time");
        self.spare.extend(self.history.drain(..).map(|node| node.c));
        self.recycle_stages(f64::NEG_INFINITY);
        self.history.extend([start, end]);
        self.restart = false;
    }
}
