//! `PetscViewer`: output of PETSc objects to files.

use std::path::Path;
use std::ptr::null_mut;

use crate::{Comm, Error, Result, alive, c_string, chk, sys};

/// An owned PETSc viewer (`PetscViewer`). Viewing a vector with it ([`Vector::view`]) queues the
/// vector; [`Viewer::flush`] writes what is queued. Dropping it writes what is still queued and
/// closes it, with errors lost: flush first.
///
/// [`Vector::view`]: crate::Vector::view
#[derive(Debug)]
pub struct Viewer {
    raw: sys::PetscViewer,
}

impl Viewer {
    /// A VTK viewer of the file `path` (`PetscViewerVTKOpen` for writing), on the processes of
    /// `comm`. The extension chooses the format: `.vtu` is the XML unstructured grid, which
    /// process 0 writes with the pieces of every process. A vector of a DMPlex with a finite
    /// element field viewed with it ([`Vector::view`]) is written as point data named after
    /// the vector ([`Vector::set_name`]): a local vector is written as it is, its constrained
    /// (essential) values included. An existing file is overwritten. Collective.
    ///
    /// [`Vector::view`]: crate::Vector::view
    /// [`Vector::set_name`]: crate::Vector::set_name
    pub fn vtk(comm: &Comm, path: &Path) -> Result<Viewer> {
        let name = path.to_str().ok_or_else(|| {
            Error::new(
                sys::PETSC_ERR_ARG_WRONG,
                format!("VTK file name {} is not UTF-8", path.display()),
            )
        })?;
        let name = c_string(name, "VTK file name")?;
        let mut raw = null_mut();
        // SAFETY: a valid communicator and a NUL terminated name, which PETSc copies; `raw`
        // receives a new viewer, which `Viewer` owns.
        chk!(unsafe {
            sys::PetscViewerVTKOpen(comm.as_raw(), name.as_ptr(), sys::FILE_MODE_WRITE, &mut raw)
        });
        Ok(Viewer { raw })
    }

    pub fn as_raw(&self) -> sys::PetscViewer {
        self.raw
    }

    /// Writes what is queued (`PetscViewerFlush`). Collective.
    pub fn flush(&mut self) -> Result<()> {
        // SAFETY: a valid viewer.
        chk!(unsafe { sys::PetscViewerFlush(self.raw) });
        Ok(())
    }
}

impl Drop for Viewer {
    fn drop(&mut self) {
        if alive() {
            // SAFETY: the viewer is owned; nothing to report an error to.
            unsafe { sys::PetscViewerDestroy(&mut self.raw) };
        }
    }
}
