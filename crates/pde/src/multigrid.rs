//! Geometric multigrid on the refinement hierarchy of a mesh: `PCMG` with the transport
//! operator discretized again on every level.

use mesh::Mesh;
use petsc::{Ksp, Matrix};

use crate::discretization::Counts;
use crate::{Discretization, PdeError, Result, SharedOperator};

/// The iterations of a smoother, before and after the coarse correction each: a Chebyshev
/// polynomial of degree 2.
const SMOOTHING_STEPS: usize = 2;

/// The interval a Chebyshev smoother targets, from the estimates `emin`, `emax` of the extreme
/// eigenvalues of its Jacobi preconditioned operator (`KSPChebyshevEstEigSet(a, b, c, d)`:
/// `[a emin + b emax, c emin + d emax]`): `[0.1 emax, 1.1 emax]`, the upper part of the
/// spectrum, which the coarser levels cannot represent (PETSc's default transform).
const CHEBYSHEV_TRANSFORM: [f64; 4] = [0.0, 0.1, 0.0, 1.1];

/// The interval of [`CHEBYSHEV_TRANSFORM`] for operators which are not symmetric (advection):
/// `[0.5 emax, 1.2 emax]`. Their Jacobi preconditioned spectrum is complex, where a Chebyshev
/// polynomial of the wider interval grows: with `[0.1 emax, 1.1 emax]` the preconditioned
/// GMRES of a manufactured solution at the cell Peclet number 100 (on the unit square,
/// refinement 2) stagnated beyond 300 iterations. With this interval the solves of the tests
/// of advection took at most 24 iterations (Peclet 100), 8 (a front at Peclet 100) and 6
/// (Peclet below 1), fewer than with `[0.25 emax, 1.1 emax]` or `[0.25 emax, 1.5 emax]`, also
/// for large steps, where diffusion dominates.
const CHEBYSHEV_TRANSFORM_NONSYMMETRIC: [f64; 4] = [0.0, 0.5, 0.0, 1.2];

/// The levels of a geometric multigrid preconditioner from [`configure_multigrid`], for keeping
/// them in step with the shift of the operator ([`Multigrid::set_shift`]).
///
/// The preconditioner (PETSc's `PCMG`) holds references to what it needs; this handle shares
/// the operators of the levels and holds their assembled matrices, which it assembles again
/// when the shift changes.
#[derive(Debug)]
pub struct Multigrid {
    /// The operators of the levels, coarse first.
    levels: Vec<SharedOperator>,
    /// The operators of the levels assembled, coarse first.
    assembled: Vec<Matrix>,
}

/// Makes the preconditioner of `ksp` geometric multigrid (`PCMG`, V-cycles) over the levels of
/// `mesh`, with the operators `levels`: one per level of the mesh, coarse first, on the
/// discretizations of [`Discretization::hierarchy`] (each linked to the one before), each with
/// the coefficients of its level (the same coefficient functions at its quadrature points) and
/// the same terms and shift. The operator of `ksp` should be the operator `levels` ends with: as
/// a shell matrix, or assembled, the matrix [`Multigrid::finest`]. For a solve on a coarser
/// level `l`, `levels` holds the operators of the levels `0..=l` only. `symmetric` says whether the
/// operators are symmetric positive definite (diffusion and reaction), or not (with advection).
/// Collective.
///
/// - Interpolation: `DMCreateInterpolation` from each level to the next, between the DMs of
///   the discretizations (nested P1 spaces; essential dofs are left out on both levels).
///   Restriction: its transpose.
/// - Coarser levels: the operator discretized again on the level (its [`SharedOperator`]), not
///   the Galerkin product.
/// - Every level works on the operator of the level assembled ([`crate::Transport::assemble`]),
///   the finest included: the smoothers, the residuals of the levels and the coarse solver apply
///   the assembled matrices, which are several times cheaper to apply than the operators of
///   libCEED, and equal them up to rounding. The Krylov method of `ksp` applies the operator of
///   `ksp`.
/// - Smoothers, on every level but the coarse one: 2 iterations of Chebyshev, preconditioned
///   by Jacobi (the diagonal of the assembled operator), on `[0.1 emax, 1.1 emax]` for the
///   largest eigenvalue `emax` estimated by 10 iterations of CG if `symmetric`, else on
///   `[0.5 emax, 1.2 emax]` estimated by GMRES (`KSPChebyshevEstEigSet`, with the estimator
///   set explicitly). The estimates are renewed when the operator of the level changes. The
///   assembled operators are marked symmetric positive definite if `symmetric`.
/// - Coarse level (level 0): one application of algebraic multigrid (`PCGAMG`, `preonly`) to
///   the assembled operator, which runs on any number of processes without external packages.
///   A mesh of one level is solved this way alone.
///
/// The options of the solver override these settings (`KSPSetFromOptions` after this call):
/// under its prefix `<p>`, `-<p>pc_type`, `-<p>mg_levels_ksp_max_it`, `-<p>mg_coarse_pc_type`
/// and the like (`-<p>mg_coarse_` is the coarse solver for a mesh of one level too).
///
/// When the shift changes, [`Multigrid::set_shift`] changes it on every level and assembles the
/// levels again, and the operator of `ksp` must be marked changed ([`Matrix::mark_changed`]; a
/// `TS` does when [`petsc::Implicit::jacobian`] returns true) for the preconditioner to be
/// rebuilt.
///
/// Errors (on every process): [`PdeError::NotHierarchy`] unless `levels` has one operator per
/// level of `mesh` from level 0 up to a level of the mesh, each on the discretization linked to
/// the one before, and the last of the size of the operator of `ksp`, if it has one.
pub fn configure_multigrid(
    ksp: &mut Ksp,
    mesh: &Mesh,
    levels: &[SharedOperator],
    symmetric: bool,
) -> Result<Multigrid> {
    let n = levels.len();
    if n == 0 || n > mesh.levels().len() {
        return Err(PdeError::NotHierarchy(format!(
            "{n} operators for the {} levels of the mesh",
            mesh.levels().len()
        )));
    }
    let discs: Vec<Discretization> = levels
        .iter()
        .map(|op| op.0.borrow().discretization())
        .collect();
    // The coarsest operator on level 0: its DM a clone of the level, with the same cells.
    let cells = |dm: &petsc::Dm| dm.height_stratum(0).map(|cells| cells.end - cells.start);
    let other = cells(discs[0].dm())? != cells(&mesh.levels()[0])?;
    if discs[0].dm().comm().max_int(other.into())? > 0 {
        return Err(PdeError::NotHierarchy(format!(
            "the first of {n} operators is not on level 0 of the mesh"
        )));
    }
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
    // The finest operator of the levels is the operator of the solver, as far as the sizes tell.
    if let Some(sizes) = ksp.operator_sizes()? {
        let last = discs[n - 1].shared().global_sizes;
        let other = sizes != last;
        if discs[0].dm().comm().max_int(other.into())? > 0 {
            return Err(PdeError::NotHierarchy(format!(
                "the last of {n} operators (level {}, {} dofs) is not of the size of the operator \
                 of the solver ({} rows)",
                n - 1,
                last.1,
                sizes.1
            )));
        }
    }

    let counts = &discs[n - 1].shared().counts;
    Counts::count(&counts.multigrids);
    let assembled = levels
        .iter()
        .map(|op| {
            let mut matrix = op.0.borrow().assemble()?;
            if symmetric {
                matrix.set_spd()?;
            }
            Ok(matrix)
        })
        .collect::<Result<Vec<_>>>()?;
    let mut pc = ksp.pc()?;
    let prefix = pc.options_prefix()?;
    pc.set_type("mg")?;
    pc.mg_set_levels(n)?;
    for level in 1..n {
        let interpolation = discs[level - 1]
            .dm()
            .create_interpolation(discs[level].dm())?;
        pc.mg_set_interpolation(level, &interpolation)?;
        let mut smoother = pc.mg_smoother(level)?;
        smoother.set_type("chebyshev")?;
        smoother.set_pc_type("jacobi")?;
        smoother.set_tolerances(None, None, Some(SMOOTHING_STEPS))?;
        let [a, b, c, d] = if symmetric {
            CHEBYSHEV_TRANSFORM
        } else {
            CHEBYSHEV_TRANSFORM_NONSYMMETRIC
        };
        smoother.chebyshev_est_eig_set(a, b, c, d)?;
        if let Some(mut estimator) = smoother.chebyshev_est_eig_ksp()? {
            estimator.set_type(if symmetric { "cg" } else { "gmres" })?;
        }
        if level < n - 1 {
            // The DMs of the levels: PCMG passes the state of a solver with a DM (a TS) down
            // through them (DMRestrict); without them it would create DMs which cannot.
            smoother.set_dm(discs[level].dm(), false)?;
        }
        // On the finest level too, in place of the operators of `ksp` PCMG would take.
        smoother.set_operators(&assembled[level], &assembled[level])?;
    }

    let mut coarse_solve = pc.mg_smoother(0)?;
    if n > 1 {
        coarse_solve.set_dm(discs[0].dm(), false)?;
    }
    if n == 1 {
        // PCMG names a single level `mg_levels_0_`: the coarse solver all the same.
        coarse_solve.set_options_prefix(&format!("{prefix}mg_coarse_"))?;
    }
    coarse_solve.set_operators(&assembled[0], &assembled[0])?;
    coarse_solve.set_type("preonly")?;
    coarse_solve.set_pc_type("gamg")?;
    Ok(Multigrid {
        levels: levels.to_vec(),
        assembled,
    })
}

impl Multigrid {
    /// The operator of the finest level assembled, in step with the shift
    /// ([`Multigrid::set_shift`]): for the solver itself, whose Krylov method then applies the
    /// assembled matrix too.
    pub fn finest(&self) -> &Matrix {
        self.assembled.last().expect("a multigrid has a level")
    }

    /// Sets the shift of the operators of every level (see [`crate::Transport`]) and assembles
    /// them again (their values; the patterns stay). The coefficients of the levels stay as they
    /// are. The smoothers and the coarse solver are set up again when the preconditioner is
    /// rebuilt next, which the caller triggers by marking the operator of the solver changed
    /// (see [`configure_multigrid`]). Collective.
    pub fn set_shift(&mut self, shift: f64) -> Result<()> {
        for (op, matrix) in self.levels.iter().zip(&mut self.assembled) {
            let mut op = op.0.borrow_mut();
            op.shift = shift;
            op.reassemble(matrix)?;
        }
        Ok(())
    }
}
