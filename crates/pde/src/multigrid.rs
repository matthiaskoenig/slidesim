//! Geometric multigrid on the refinement hierarchy of a mesh: `PCMG` with the transport
//! operator discretized again on every level.

use mesh::Mesh;
use petsc::{Ksp, Matrix};

use crate::{Discretization, PdeError, Result, SharedOperator};

/// The iterations of a smoother, before and after the coarse correction each: a Chebyshev
/// polynomial of degree 2.
const SMOOTHING_STEPS: usize = 2;

/// The interval a Chebyshev smoother targets, from the estimates `emin`, `emax` of the extreme
/// eigenvalues of its Jacobi preconditioned operator (`KSPChebyshevEstEigSet(a, b, c, d)`:
/// `[a emin + b emax, c emin + d emax]`): `[0.1 emax, 1.1 emax]`, the upper part of the
/// spectrum, which the coarser levels cannot represent (PETSc's default transform).
const CHEBYSHEV_TRANSFORM: [f64; 4] = [0.0, 0.1, 0.0, 1.1];

/// The levels of a geometric multigrid preconditioner from [`configure_multigrid`], for keeping
/// them in step with the shift of the operator ([`Multigrid::set_shift`]).
///
/// The preconditioner (PETSc's `PCMG`) holds references to what it needs; this handle shares
/// the operators of the levels and holds the matrices whose changes it marks.
#[derive(Debug)]
pub struct Multigrid {
    /// The operators of the levels, coarse first.
    levels: Vec<SharedOperator>,
    /// The shell matrices of the levels between the coarse and the finest level.
    intermediate: Vec<Matrix>,
    /// The assembled operator of the coarse level.
    coarse: Matrix,
}

/// Makes the preconditioner of `ksp` geometric multigrid (`PCMG`, V-cycles) over the levels of
/// `mesh`, with the operators `levels`: one per level of the mesh, coarse first, on the
/// discretizations of [`Discretization::hierarchy`] (each linked to the one before), each with
/// the coefficients of its level (the same coefficient functions at its quadrature points) and
/// the same terms and shift. The finest level is the operator of `ksp`, which should be the
/// operator `levels` ends with (as a shell matrix). Collective.
///
/// - Interpolation: `DMCreateInterpolation` from each level to the next, between the DMs of
///   the discretizations (nested P1 spaces; essential dofs are left out on both levels).
///   Restriction: its transpose.
/// - Coarser levels: the operator discretized again on the level (a shell matrix of its
///   [`SharedOperator`]), not the Galerkin product.
/// - Smoothers, on every level but the coarse one: 2 iterations of Chebyshev, preconditioned
///   by Jacobi (the diagonal of the shell operator), on `[0.1 emax, 1.1 emax]` for the largest
///   eigenvalue `emax` estimated by 10 iterations of CG (`KSPChebyshevEstEigSet`, with the
///   estimator set to CG: it only applies the shell operator and Jacobi). The estimates are
///   renewed when the operator of the level changes.
/// - Coarse level (level 0): one application of algebraic multigrid (`PCGAMG`, `preonly`) to
///   the assembled operator ([`crate::Transport::assemble`]), which runs on any
///   number of processes without external packages. A mesh of one level is solved this way
///   alone.
///
/// The options of the solver override these settings (`KSPSetFromOptions` after this call):
/// under its prefix `<p>`, `-<p>pc_type`, `-<p>mg_levels_ksp_max_it`, `-<p>mg_coarse_pc_type`
/// and the like (`-<p>mg_coarse_` is the coarse solver for a mesh of one level too).
///
/// When the shift changes, [`Multigrid::set_shift`] changes it on every level, and the operator
/// of `ksp` must be marked changed ([`Matrix::mark_changed`]; a `TS` does
/// when [`petsc::Implicit::jacobian`] returns true) for the preconditioner to be rebuilt.
///
/// Errors: [`PdeError::NotHierarchy`] unless `levels` has one operator per level of `mesh`,
/// each on the discretization linked to the one before.
pub fn configure_multigrid(
    ksp: &mut Ksp,
    mesh: &Mesh,
    levels: &[SharedOperator],
) -> Result<Multigrid> {
    let n = levels.len();
    if n == 0 || n != mesh.levels().len() {
        return Err(PdeError::NotHierarchy(format!(
            "{n} operators for the {} levels of the mesh",
            mesh.levels().len()
        )));
    }
    let discs: Vec<Discretization> = levels
        .iter()
        .map(|op| op.0.borrow().discretization())
        .collect();
    for (level, pair) in discs.windows(2).enumerate() {
        let coarse = pair[1].dm().coarse_dm()?;
        if coarse.is_none_or(|dm| dm.as_raw() != pair[0].dm().as_raw()) {
            return Err(PdeError::NotHierarchy(format!(
                "the discretization of level {level} is not the coarse level of level {} \
                 (Discretization::hierarchy links them)",
                level + 1
            )));
        }
    }

    let mut pc = ksp.pc()?;
    let prefix = pc.options_prefix()?;
    pc.set_type("mg")?;
    pc.mg_set_levels(n)?;
    let mut intermediate = Vec::with_capacity(n.saturating_sub(2));
    for level in 1..n {
        let interpolation = discs[level - 1]
            .dm()
            .create_interpolation(discs[level].dm())?;
        pc.mg_set_interpolation(level, &interpolation)?;
        let mut smoother = pc.mg_smoother(level)?;
        smoother.set_type("chebyshev")?;
        smoother.set_pc_type("jacobi")?;
        smoother.set_tolerances(None, None, Some(SMOOTHING_STEPS))?;
        let [a, b, c, d] = CHEBYSHEV_TRANSFORM;
        smoother.chebyshev_est_eig_set(a, b, c, d)?;
        if let Some(mut estimator) = smoother.chebyshev_est_eig_ksp()? {
            estimator.set_type("cg")?;
        }
        if level < n - 1 {
            // The DMs of the levels: PCMG passes the state of a solver with a DM (a TS) down
            // through them (DMRestrict); without them it would create DMs which cannot.
            smoother.set_dm(discs[level].dm(), false)?;
            let mut shell = discs[level].shell_matrix(levels[level].clone())?;
            shell.set_spd()?;
            smoother.set_operators(&shell, &shell)?;
            intermediate.push(shell);
        }
    }

    let mut coarse = levels[0].0.borrow().assemble()?;
    coarse.set_spd()?;
    let mut coarse_solve = pc.mg_smoother(0)?;
    if n > 1 {
        coarse_solve.set_dm(discs[0].dm(), false)?;
    }
    if n == 1 {
        // PCMG names a single level `mg_levels_0_`: the coarse solver all the same.
        coarse_solve.set_options_prefix(&format!("{prefix}mg_coarse_"))?;
    }
    coarse_solve.set_operators(&coarse, &coarse)?;
    coarse_solve.set_type("preonly")?;
    coarse_solve.set_pc_type("gamg")?;
    Ok(Multigrid {
        levels: levels.to_vec(),
        intermediate,
        coarse,
    })
}

impl Multigrid {
    /// Sets the shift of the operators of every level (see [`crate::Transport`]) and assembles
    /// the coarse operator again (its values; the pattern stays). The coefficients of the levels
    /// stay as they are. The coarser levels are marked changed, so their smoothers are set up
    /// again when the preconditioner is rebuilt next, which the caller triggers by marking the
    /// operator of the solver changed (see [`configure_multigrid`]). Collective.
    pub fn set_shift(&mut self, shift: f64) -> Result<()> {
        for op in &self.levels {
            op.0.borrow_mut().shift = shift;
        }
        for shell in &mut self.intermediate {
            shell.mark_changed()?;
        }
        self.levels[0].0.borrow().reassemble(&mut self.coarse)?;
        Ok(())
    }
}
