//! `PetscFE`: finite element spaces with their quadrature.

use std::ptr::{null, null_mut};
use std::slice;

use crate::{Comm, Error, Result, chk, sys, to_int};

/// An owned reference to a `PetscFE`: a finite element space on a reference cell, with a
/// quadrature. A `DM` given the element keeps its own reference.
#[derive(Debug)]
pub struct Fe {
    raw: sys::PetscFE,
}

/// A quadrature on the reference cell of a [`Fe`].
#[derive(Clone, Debug, PartialEq)]
pub struct Quadrature {
    /// The dimension of the reference cell.
    pub dim: usize,
    /// The points, `points[q * dim + d]` (PETSc's reference cells span `[-1, 1]` in each
    /// direction).
    pub points: Vec<f64>,
    /// The weights, one per point.
    pub weights: Vec<f64>,
}

/// The basis functions of a [`Fe`] and their derivatives at the points of its quadrature
/// (`PetscFEGetCellTabulation`).
#[derive(Clone, Debug, PartialEq)]
pub struct Tabulation {
    pub num_points: usize,
    pub num_basis: usize,
    pub num_comp: usize,
    pub dim: usize,
    /// `values[(q * num_basis + b) * num_comp + c]`: component `c` of basis function `b` at
    /// point `q`.
    pub values: Vec<f64>,
    /// `derivatives[((q * num_basis + b) * num_comp + c) * dim + d]`: its derivative in
    /// reference direction `d`.
    pub derivatives: Vec<f64>,
}

/// `n` values at `ptr`, copied (none if `n == 0`).
///
/// # Safety
/// Unless `n == 0`, `ptr` points to `n` valid values.
unsafe fn copy(ptr: *const f64, n: usize) -> Vec<f64> {
    if n == 0 {
        Vec::new()
    } else {
        unsafe { slice::from_raw_parts(ptr, n) }.to_vec()
    }
}

impl Fe {
    /// The Lagrange element of `degree` with `num_comp` components on the reference simplex
    /// (`simplex`) or box of dimension `dim`, with a quadrature exact for polynomials of degree
    /// `quadrature_degree` (`PetscFECreateLagrange` on `PETSC_COMM_SELF`: no communication).
    pub fn lagrange(
        dim: usize,
        num_comp: usize,
        simplex: bool,
        degree: usize,
        quadrature_degree: usize,
    ) -> Result<Fe> {
        let mut raw = null_mut();
        chk!(unsafe {
            sys::PetscFECreateLagrange(
                Comm::self_().as_raw(),
                to_int(dim, "dimension")?,
                to_int(num_comp, "number of components")?,
                simplex,
                to_int(degree, "degree")?,
                to_int(quadrature_degree, "quadrature degree")?,
                &mut raw,
            )
        });
        Ok(Fe { raw })
    }

    /// The quadrature of the element (`PetscFEGetQuadrature`, `PetscQuadratureGetData`).
    pub fn quadrature(&self) -> Result<Quadrature> {
        let mut quad = null_mut();
        // SAFETY: the quadrature is borrowed from the element.
        chk!(unsafe { sys::PetscFEGetQuadrature(self.raw, &mut quad) });
        let (mut dim, mut nc, mut np) = (0, 0, 0);
        let (mut points, mut weights) = (null(), null());
        chk!(unsafe {
            sys::PetscQuadratureGetData(quad, &mut dim, &mut nc, &mut np, &mut points, &mut weights)
        });
        if nc != 1 {
            return Err(Error::new(
                sys::PETSC_ERR_SUP,
                format!("quadrature with {nc} weight components (need 1)"),
            ));
        }
        let (dim, np) = (dim as usize, np as usize);
        // SAFETY: the quadrature holds `np * dim` coordinates and `np` weights.
        Ok(Quadrature {
            dim,
            points: unsafe { copy(points, np * dim) },
            weights: unsafe { copy(weights, np) },
        })
    }

    /// The basis functions and their first derivatives at the quadrature points
    /// (`PetscFEGetCellTabulation`).
    pub fn tabulation(&self) -> Result<Tabulation> {
        let mut tab: sys::PetscTabulation = null_mut();
        // SAFETY: the tabulation is borrowed from the element, which keeps it.
        chk!(unsafe { sys::PetscFEGetCellTabulation(self.raw, 1, &mut tab) });
        // SAFETY: a valid tabulation of `K + 1 = 2` arrays for `Nr = 1` reference cell.
        let t = unsafe { &*tab };
        let (np, nb, nc, dim) = (t.Np as usize, t.Nb as usize, t.Nc as usize, t.cdim as usize);
        let n = np * nb * nc;
        // SAFETY: `T[0]` holds `Np * Nb * Nc` values and `T[1]` `Np * Nb * Nc * cdim`.
        let (values, derivatives) = unsafe { (copy(*t.T, n), copy(*t.T.add(1), n * dim)) };
        Ok(Tabulation {
            num_points: np,
            num_basis: nb,
            num_comp: nc,
            dim,
            values,
            derivatives,
        })
    }

    pub fn as_raw(&self) -> sys::PetscFE {
        self.raw
    }
}

impl Drop for Fe {
    fn drop(&mut self) {
        if crate::alive() {
            unsafe { sys::PetscFEDestroy(&mut self.raw) };
        }
    }
}
