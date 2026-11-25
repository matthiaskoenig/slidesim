//! `DM`: meshes (DMPlex).

use std::ffi::{CStr, c_char};
use std::ops::Range;
use std::path::Path;
use std::ptr::{null, null_mut};

use crate::{Comm, Error, Result, alive, c_string, chk, sys, to_int};

/// A mesh point (cell, face, edge or vertex) of a DMPlex.
pub type Point = sys::PetscInt;

/// An owned reference to a PETSc `DM` (a DMPlex). Cloning shares the `DM` (PETSc reference
/// counting); the `DM` is destroyed with its last reference.
#[derive(Debug)]
pub struct Dm {
    raw: sys::DM,
}

/// The indices of `is`, copied; `ISDestroy`s `is` if `owned`. A null `is` has no indices.
fn take_is(mut is: sys::IS, owned: bool) -> Result<Vec<sys::PetscInt>> {
    if is.is_null() {
        return Ok(Vec::new());
    }
    let copy = (|| -> Result<Vec<sys::PetscInt>> {
        let mut n = 0;
        chk!(unsafe { sys::ISGetLocalSize(is, &mut n) });
        let mut indices: *const sys::PetscInt = null();
        chk!(unsafe { sys::ISGetIndices(is, &mut indices) });
        let v = if n == 0 {
            Vec::new()
        } else {
            // SAFETY: ISGetIndices gives `n` indices, valid until ISRestoreIndices.
            unsafe { std::slice::from_raw_parts(indices, n as usize) }.to_vec()
        };
        chk!(unsafe { sys::ISRestoreIndices(is, &mut indices) });
        Ok(v)
    })();
    if owned {
        chk!(unsafe { sys::ISDestroy(&mut is) });
    }
    copy
}

impl Dm {
    /// Takes ownership of one reference to `raw`.
    ///
    /// # Safety
    /// `raw` is a valid `DM`, and the caller gives up its reference.
    pub unsafe fn from_raw(raw: sys::DM) -> Dm {
        Dm { raw }
    }

    pub fn as_raw(&self) -> sys::DM {
        self.raw
    }

    /// A DMPlex on `comm` read from the mesh file `path`, named `name`
    /// (`DMPlexCreateFromFile`). The format follows from the extension, for example `.msh` for
    /// gmsh. With `interpolate`, the mesh has edges (and faces in 3D), not only cells and
    /// vertices. Process 0 reads the file and holds the whole mesh: see [`Dm::distribute`].
    /// Readers take their settings from the options database (for example
    /// `-dm_plex_gmsh_use_regions`).
    pub fn plex_from_file(comm: &Comm, path: &Path, name: &str, interpolate: bool) -> Result<Dm> {
        let path = path.to_str().ok_or_else(|| {
            Error::new(
                sys::PETSC_ERR_ARG_WRONG,
                format!("mesh file path {} is not UTF-8", path.display()),
            )
        })?;
        let path = c_string(path, "mesh file path")?;
        let name = c_string(name, "DM name")?;
        let mut raw = null_mut();
        // SAFETY: PETSc copies the strings; `raw` receives a new DM.
        chk!(unsafe {
            sys::DMPlexCreateFromFile(
                comm.as_raw(),
                path.as_ptr(),
                name.as_ptr(),
                interpolate,
                &mut raw,
            )
        });
        Ok(Dm { raw })
    }

    /// `DMSetFromOptions`.
    pub fn set_from_options(&mut self) -> Result<()> {
        chk!(unsafe { sys::DMSetFromOptions(self.raw) });
        Ok(())
    }

    /// The communicator of the DM (`PetscObjectGetComm`).
    pub fn comm(&self) -> Comm {
        let mut comm = unsafe { sys::petsc_sys_comm_self() };
        // Cannot fail on a valid object.
        unsafe { sys::PetscObjectGetComm(self.raw as sys::PetscObject, &mut comm) };
        Comm::new(comm)
    }

    /// The DM distributed over the processes of its communicator, with `overlap` layers of cells
    /// shared with neighbouring processes (`DMPlexDistribute`). Labels are distributed with it.
    /// `None` if there is nothing to distribute (one process). Collective.
    pub fn distribute(&self, overlap: usize) -> Result<Option<Dm>> {
        let overlap = to_int(overlap, "overlap")?;
        let mut raw = null_mut();
        // SAFETY: `raw` receives a new DM, or stays null.
        chk!(unsafe { sys::DMPlexDistribute(self.raw, overlap, null_mut(), &mut raw) });
        Ok((!raw.is_null()).then_some(Dm { raw }))
    }

    /// The DM refined once, on the same communicator (`DMRefine`). A DMPlex is refined
    /// uniformly (regular refinement unless the options select another transform), and its labels
    /// are carried over: a refined point carries the labels of its parent. Collective.
    pub fn refine(&self) -> Result<Dm> {
        let mut raw = null_mut();
        // SAFETY: `raw` receives a new DM, or stays null.
        chk!(unsafe { sys::DMRefine(self.raw, self.comm().as_raw(), &mut raw) });
        if raw.is_null() {
            return Err(Error::new(
                sys::PETSC_ERR_SUP,
                "DMRefine did not refine the DM",
            ));
        }
        Ok(Dm { raw })
    }

    /// Makes `coarse` the next coarser level of this DM (`DMSetCoarseDM`), for multigrid. The DM
    /// keeps a reference to `coarse`.
    pub fn set_coarse_dm(&mut self, coarse: &Dm) -> Result<()> {
        chk!(unsafe { sys::DMSetCoarseDM(self.raw, coarse.raw) });
        Ok(())
    }

    /// The next coarser level set with [`Dm::set_coarse_dm`] (`DMGetCoarseDM`), shared.
    pub fn coarse_dm(&self) -> Result<Option<Dm>> {
        let mut raw = null_mut();
        chk!(unsafe { sys::DMGetCoarseDM(self.raw, &mut raw) });
        if raw.is_null() {
            return Ok(None);
        }
        // SAFETY: a borrowed reference of the DM, which becomes ours.
        chk!(unsafe { sys::PetscObjectReference(raw as sys::PetscObject) });
        Ok(Some(Dm { raw }))
    }

    /// The topological dimension (`DMGetDimension`).
    pub fn dimension(&self) -> Result<usize> {
        let mut dim = 0;
        chk!(unsafe { sys::DMGetDimension(self.raw, &mut dim) });
        Ok(dim as usize)
    }

    /// The dimension of the coordinates (`DMGetCoordinateDim`).
    pub fn coordinate_dimension(&self) -> Result<usize> {
        let mut dim = 0;
        chk!(unsafe { sys::DMGetCoordinateDim(self.raw, &mut dim) });
        Ok(dim as usize)
    }

    /// The local points of depth `depth` (vertices 0, edges 1, ...) of a DMPlex
    /// (`DMPlexGetDepthStratum`).
    pub fn depth_stratum(&self, depth: usize) -> Result<Range<Point>> {
        let depth = to_int(depth, "depth")?;
        let (mut start, mut end) = (0, 0);
        chk!(unsafe { sys::DMPlexGetDepthStratum(self.raw, depth, &mut start, &mut end) });
        Ok(start..end)
    }

    /// The local points of height `height` (cells 0, faces 1, ...) of a DMPlex
    /// (`DMPlexGetHeightStratum`).
    pub fn height_stratum(&self, height: usize) -> Result<Range<Point>> {
        let height = to_int(height, "height")?;
        let (mut start, mut end) = (0, 0);
        chk!(unsafe { sys::DMPlexGetHeightStratum(self.raw, height, &mut start, &mut end) });
        Ok(start..end)
    }

    /// Makes sure the local coordinates (with those of ghost points) exist
    /// (`DMGetCoordinatesLocalSetUp`), which noncollective geometry queries such as
    /// [`Dm::cell_volume`] need: a refined or distributed DM may have global coordinates only.
    /// Collective.
    pub fn set_up_local_coordinates(&self) -> Result<()> {
        chk!(unsafe { sys::DMGetCoordinatesLocalSetUp(self.raw) });
        Ok(())
    }

    /// The volume (area in 2D) of the local cell `cell` (`DMPlexComputeCellGeometryFVM`).
    /// Needs the local coordinates: see [`Dm::set_up_local_coordinates`].
    pub fn cell_volume(&self, cell: Point) -> Result<sys::PetscReal> {
        let cells = self.height_stratum(0)?;
        if !cells.contains(&cell) {
            return Err(Error::new(
                sys::PETSC_ERR_ARG_OUTOFRANGE,
                format!("point {cell} is not a cell (cells are {cells:?})"),
            ));
        }
        let mut volume = 0.0;
        // SAFETY: centroid and normal are optional outputs.
        chk!(unsafe {
            sys::DMPlexComputeCellGeometryFVM(self.raw, cell, &mut volume, null_mut(), null_mut())
        });
        Ok(volume)
    }

    /// The global numbers of the local cells, in the order of the local cells; a cell owned by
    /// another process has the number `-(g + 1)` for its global number `g`
    /// (`DMPlexGetCellNumbering`). Collective on its first call.
    pub fn cell_numbering(&self) -> Result<Vec<sys::PetscInt>> {
        let mut is = null_mut();
        // SAFETY: the IS is borrowed from the DM.
        chk!(unsafe { sys::DMPlexGetCellNumbering(self.raw, &mut is) });
        take_is(is, false)
    }

    /// The names of the labels, in the order of the DM (`DMGetNumLabels`, `DMGetLabelName`).
    pub fn label_names(&self) -> Result<Vec<String>> {
        let mut n = 0;
        chk!(unsafe { sys::DMGetNumLabels(self.raw, &mut n) });
        (0..n)
            .map(|i| {
                let mut name: *const c_char = null();
                chk!(unsafe { sys::DMGetLabelName(self.raw, i, &mut name) });
                // SAFETY: the name of a label of the DM, a C string.
                Ok(unsafe { CStr::from_ptr(name) }
                    .to_string_lossy()
                    .into_owned())
            })
            .collect()
    }

    /// Whether the DM has the label `name` (`DMHasLabel`).
    pub fn has_label(&self, name: &str) -> Result<bool> {
        let name = c_string(name, "label name")?;
        let mut has = false;
        chk!(unsafe { sys::DMHasLabel(self.raw, name.as_ptr(), &mut has) });
        Ok(has)
    }

    /// The values the label `name` takes on local points, sorted; none if there is no such label
    /// (`DMGetLabelIdIS`).
    pub fn label_values(&self, name: &str) -> Result<Vec<sys::PetscInt>> {
        let name = c_string(name, "label name")?;
        let mut is = null_mut();
        // SAFETY: the IS is ours to destroy.
        chk!(unsafe { sys::DMGetLabelIdIS(self.raw, name.as_ptr(), &mut is) });
        let mut values = take_is(is, true)?;
        values.sort_unstable();
        Ok(values)
    }

    /// The local points where the label `name` has the value `value`; none if there is no such
    /// label or value (`DMGetStratumIS`).
    pub fn stratum_points(&self, name: &str, value: sys::PetscInt) -> Result<Vec<Point>> {
        let name = c_string(name, "label name")?;
        let mut is = null_mut();
        // SAFETY: the IS (or null) is ours to destroy.
        chk!(unsafe { sys::DMGetStratumIS(self.raw, name.as_ptr(), value, &mut is) });
        take_is(is, true)
    }
}

impl Clone for Dm {
    fn clone(&self) -> Dm {
        // Cannot fail on a valid object. After PetscFinalize nothing is destroyed any more, so
        // the reference count does not matter.
        if alive() {
            unsafe { sys::PetscObjectReference(self.raw as sys::PetscObject) };
        }
        Dm { raw: self.raw }
    }
}

impl Drop for Dm {
    fn drop(&mut self) {
        if alive() {
            unsafe { sys::DMDestroy(&mut self.raw) };
        }
    }
}
