//! `PetscSF`: star forests, the communication graphs of distributed meshes.

use std::ptr::null;

use crate::{Error, Result, alive, chk, sys};

/// An owned `PetscSF`: a graph whose leaves (points of this process) each take the value of one
/// root (a point of some process), for example the migration of the points of a mesh into its
/// distribution with overlap ([`crate::Dm::distribute_overlap`]).
#[derive(Debug)]
pub struct Sf {
    raw: sys::PetscSF,
}

/// The leaf `leaf` of this process takes the value of the root `root` of process `rank`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Leaf {
    pub leaf: sys::PetscInt,
    pub rank: usize,
    pub root: sys::PetscInt,
}

impl Sf {
    /// Takes ownership of one reference to `raw`.
    ///
    /// # Safety
    /// `raw` is a valid `PetscSF`, and the caller gives up its reference.
    pub unsafe fn from_raw(raw: sys::PetscSF) -> Sf {
        Sf { raw }
    }

    pub fn as_raw(&self) -> sys::PetscSF {
        self.raw
    }

    /// The number of roots on this process, and the leaves of this process with their roots, in
    /// the order of the graph (`PetscSFGetGraph`).
    pub fn graph(&self) -> Result<(usize, Vec<Leaf>)> {
        let (mut roots, mut leaves) = (0, 0);
        let mut local: *const sys::PetscInt = null();
        let mut remote: *const sys::PetscSFNode = null();
        // SAFETY: `local` (null for the leaves 0..leaves) and `remote` are borrowed from the SF.
        chk!(unsafe {
            sys::PetscSFGetGraph(self.raw, &mut roots, &mut leaves, &mut local, &mut remote)
        });
        if roots < 0 {
            return Err(Error::new(
                sys::PETSC_ERR_ARG_WRONGSTATE,
                "a star forest without a graph".to_string(),
            ));
        }
        let leaves = leaves.max(0) as usize;
        let mut graph = Vec::with_capacity(leaves);
        for i in 0..leaves {
            // SAFETY: `remote` holds `leaves` nodes, `local` (if not null) `leaves` indices.
            let node = unsafe { *remote.add(i) };
            let leaf = if local.is_null() {
                i as sys::PetscInt
            } else {
                unsafe { *local.add(i) }
            };
            graph.push(Leaf {
                leaf,
                rank: node.rank as usize,
                root: node.index,
            });
        }
        Ok((roots as usize, graph))
    }

    /// Sets each leaf of `leaves` to the value of its root in `roots` (indexed by root, on the
    /// process of the root); other entries of `leaves` stay (`PetscSFBcastBegin` and
    /// `PetscSFBcastEnd` with `MPI_REPLACE`). Collective.
    ///
    /// Errors if `roots` is shorter than the number of roots of this process, or `leaves` does
    /// not hold every leaf of this process.
    pub fn bcast(&self, roots: &[f64], leaves: &mut [f64]) -> Result<()> {
        let (num_roots, graph) = self.graph()?;
        let needed = graph.iter().map(|l| l.leaf as usize + 1).max().unwrap_or(0);
        if roots.len() < num_roots || leaves.len() < needed {
            return Err(Error::new(
                sys::PETSC_ERR_ARG_SIZ,
                format!(
                    "a broadcast of {} roots into {} leaves; the star forest has {num_roots} \
                     roots and leaves up to {needed}",
                    roots.len(),
                    leaves.len()
                ),
            ));
        }
        let (real, replace) = unsafe { (sys::petsc_sys_mpiu_real(), sys::petsc_sys_mpi_replace()) };
        // SAFETY: the arrays hold every root and leaf (checked above), and stay borrowed until
        // the broadcast ends.
        chk!(unsafe {
            sys::PetscSFBcastBegin(
                self.raw,
                real,
                roots.as_ptr().cast(),
                leaves.as_mut_ptr().cast(),
                replace,
            )
        });
        chk!(unsafe {
            sys::PetscSFBcastEnd(
                self.raw,
                real,
                roots.as_ptr().cast(),
                leaves.as_mut_ptr().cast(),
                replace,
            )
        });
        Ok(())
    }
}

impl Drop for Sf {
    fn drop(&mut self) {
        if alive() {
            crate::error::destroyed(
                unsafe { sys::PetscSFDestroy(&mut self.raw) },
                "PetscSFDestroy",
            );
        }
    }
}
