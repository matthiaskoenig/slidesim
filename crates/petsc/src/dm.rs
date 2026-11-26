//! `DM`: meshes (DMPlex).

use std::ffi::{CStr, c_char};
use std::marker::PhantomData;
use std::ops::Range;
use std::path::Path;
use std::ptr::{null, null_mut};

use crate::{
    AsVec, AsVecMut, Comm, Error, Fe, Matrix, Result, Sf, alive, c_string, chk, string_from, sys,
    to_int,
};

/// A mesh point (cell, face, edge or vertex) of a DMPlex.
pub type Point = sys::PetscInt;

/// How values reach the target vector of a scatter between local and global vectors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InsertMode {
    /// Replace the target values (`INSERT_VALUES`).
    Insert,
    /// Add to the target values (`ADD_VALUES`): the contributions of every process to a shared
    /// entry add up.
    Add,
}

impl InsertMode {
    fn raw(self) -> sys::InsertMode {
        match self {
            InsertMode::Insert => sys::INSERT_VALUES,
            InsertMode::Add => sys::ADD_VALUES,
        }
    }
}

/// The cell restriction of a field of a DMPlex: for every cell, the local vector indices of its
/// nodes, in closure order (`DMPlexGetLocalOffsets`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LocalOffsets {
    pub num_cells: usize,
    /// The number of nodes per cell.
    pub cell_size: usize,
    /// The number of components per node: node `i` of cell `e` holds its components at
    /// `offsets[e * cell_size + i] + c`.
    pub num_comp: usize,
    /// The length of the local vector.
    pub local_size: usize,
    pub offsets: Vec<sys::PetscInt>,
}

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

    /// The DM (distributed) with `overlap` more layers of cells shared with neighbouring
    /// processes (`DMPlexDistributeOverlap`), with the star forest which migrated the points
    /// of this DM (its roots) to those of the new one (its leaves): each point of the new DM is a
    /// leaf of the point it is on its process. The new DM has its own numbering of the points;
    /// the cones keep their order. Labels are distributed with it. With an overlap of one, the
    /// cells of the new DM hold every cell which shares a vertex with a cell of this one (the
    /// adjacency of finite elements, the default of a DMPlex). `None` on one process. Collective.
    pub fn distribute_overlap(&self, overlap: usize) -> Result<Option<(Dm, Sf)>> {
        let overlap = to_int(overlap, "overlap")?;
        let (mut raw, mut sf) = (null_mut(), null_mut());
        // SAFETY: `raw` and `sf` receive new objects, or stay null.
        chk!(unsafe { sys::DMPlexDistributeOverlap(self.raw, overlap, &mut sf, &mut raw) });
        // SAFETY: new references, given up here.
        let sf = (!sf.is_null()).then(|| unsafe { Sf::from_raw(sf) });
        Ok(match (raw.is_null(), sf) {
            (false, Some(sf)) => Some((Dm { raw }, sf)),
            (false, None) => {
                return Err(Error::new(
                    sys::PETSC_ERR_PLIB,
                    "DMPlexDistributeOverlap gave a DM without its star forest".to_string(),
                ));
            }
            (true, _) => None,
        })
    }

    /// The graph partitioner [`Dm::distribute`] decides with which process owns which cell
    /// (`DMPlexGetPartitioner`), borrowed from the DM.
    pub fn partitioner(&self) -> Result<Partitioner<'_>> {
        let mut raw = null_mut();
        // SAFETY: `raw` receives the partitioner, borrowed from the DM.
        chk!(unsafe { sys::DMPlexGetPartitioner(self.raw, &mut raw) });
        Ok(Partitioner {
            raw,
            _dm: PhantomData,
        })
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

    /// The interpolation from this DM, a coarse level, to the DM `fine` (`DMCreateInterpolation`):
    /// the matrix from global vectors of this DM to global vectors of `fine`, for the fields and
    /// sections of both (constrained dofs are not in global vectors, so they are left out). For
    /// DMPlex, the nested algorithm needs `fine` refined regularly from this DM and linked to it
    /// ([`Dm::set_coarse_dm`]); otherwise PETSc locates the nodes of `fine` in the cells of this
    /// DM. Collective.
    pub fn create_interpolation(&self, fine: &Dm) -> Result<Matrix> {
        let mut raw = null_mut();
        // SAFETY: valid DMs; `raw` receives a new matrix; no scaling vector.
        chk!(unsafe { sys::DMCreateInterpolation(self.raw, fine.raw, &mut raw, null_mut()) });
        // SAFETY: a new matrix, whose reference becomes ours.
        Ok(unsafe { Matrix::from_raw(raw) })
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

    /// The local points owned by another process (the leaves of the point SF, `DMGetPointSF`),
    /// sorted: each point of a distributed mesh has one owner, and is a ghost on the other
    /// processes which have it. None on one process.
    pub fn ghost_points(&self) -> Result<Vec<Point>> {
        let mut sf = null_mut();
        // SAFETY: the SF is borrowed from the DM.
        chk!(unsafe { sys::DMGetPointSF(self.raw, &mut sf) });
        let (mut roots, mut leaves) = (0, 0);
        let mut local: *const sys::PetscInt = null();
        // SAFETY: `local` (the leaf points, or null for 0..leaves) is borrowed from the SF.
        chk!(unsafe { sys::PetscSFGetGraph(sf, &mut roots, &mut leaves, &mut local, null_mut()) });
        // Without a graph (roots < 0), no point is shared.
        if roots < 0 || leaves <= 0 {
            return Ok(Vec::new());
        }
        let mut points = if local.is_null() {
            (0..leaves).collect()
        } else {
            unsafe { std::slice::from_raw_parts(local, leaves as usize) }.to_vec()
        };
        points.sort_unstable();
        Ok(points)
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

    /// The points of the cone of `point`, one level down, in order (`DMPlexGetCone`): the edges
    /// of a cell of a 2D mesh, or the two vertices of an edge.
    pub fn cone(&self, point: Point) -> Result<Vec<Point>> {
        let mut n = 0;
        // SAFETY: as for `support`.
        chk!(unsafe { sys::DMPlexGetConeSize(self.raw, point, &mut n) });
        let mut cone: *const sys::PetscInt = null();
        chk!(unsafe { sys::DMPlexGetCone(self.raw, point, &mut cone) });
        if n == 0 || cone.is_null() {
            return Ok(Vec::new());
        }
        // SAFETY: PETSc points `cone` into the DM's own storage, `n > 0` entries long, copied
        // before the DM can change.
        Ok(unsafe { std::slice::from_raw_parts(cone, n as usize) }.to_vec())
    }

    /// The local points whose closure holds `point`, one level up (`DMPlexGetSupport`): the
    /// cells of an edge of a 2D mesh, for example.
    pub fn support(&self, point: Point) -> Result<Vec<Point>> {
        let mut n = 0;
        // SAFETY: `self.raw` is a live DMPlex; PETSc writes the support size into `n` (an error
        // for a point outside the chart is returned, not undefined).
        chk!(unsafe { sys::DMPlexGetSupportSize(self.raw, point, &mut n) });
        let mut support: *const sys::PetscInt = null();
        // SAFETY: PETSc points `support` into the DM's own support storage, `n` entries long,
        // valid as long as the DM is not changed; nothing here changes it.
        chk!(unsafe { sys::DMPlexGetSupport(self.raw, point, &mut support) });
        if n == 0 || support.is_null() {
            return Ok(Vec::new());
        }
        // SAFETY: `support` is non-null and holds `n > 0` initialized `PetscInt`s (above), and
        // the slice is copied before `self` can be borrowed mutably again.
        Ok(unsafe { std::slice::from_raw_parts(support, n as usize) }.to_vec())
    }

    /// The local faces (edges of a 2D mesh) on the boundary of the domain, sorted: those with
    /// one cell in all, not those on the boundary between the parts of processes
    /// (`DMPlexMarkBoundaryFaces` into a label of its own). Collective.
    pub fn boundary_faces(&self) -> Result<Vec<Point>> {
        let name = c_string("boundary faces", "label name")?;
        let mut label = null_mut();
        // SAFETY: `label` receives a new label, which is destroyed below.
        chk!(unsafe { sys::DMLabelCreate(sys::petsc_sys_comm_self(), name.as_ptr(), &mut label) });
        let faces = (|| -> Result<Vec<Point>> {
            chk!(unsafe { sys::DMPlexMarkBoundaryFaces(self.raw, 1, label) });
            let mut is = null_mut();
            // SAFETY: the IS (or null) is ours to destroy.
            chk!(unsafe { sys::DMLabelGetStratumIS(label, 1, &mut is) });
            let mut faces = take_is(is, true)?;
            faces.sort_unstable();
            Ok(faces)
        })();
        chk!(unsafe { sys::DMLabelDestroy(&mut label) });
        faces
    }
}

/// The graph partitioner of a DMPlex (`PetscPartitioner`), borrowed from it ([`Dm::partitioner`]).
#[derive(Debug)]
pub struct Partitioner<'a> {
    raw: sys::PetscPartitioner,
    _dm: PhantomData<&'a Dm>,
}

impl Partitioner<'_> {
    /// The partitioner type (`PetscPartitionerGetType`), for example `simple` or `ptscotch`.
    pub fn type_name(&self) -> Result<String> {
        let mut name: sys::PetscPartitionerType = null();
        // SAFETY: `name` receives a string of the partitioner, or stays null.
        chk!(unsafe { sys::PetscPartitionerGetType(self.raw, &mut name) });
        // SAFETY: null or a NUL terminated string of the partitioner, valid for this call.
        Ok(unsafe { string_from(name) }.unwrap_or_default())
    }

    /// Sets the partitioner type (`PetscPartitionerSetType`): `simple` (the cells split by their
    /// number), or a graph partitioner PETSc was built with, for example `ptscotch`. An error if
    /// PETSc has no such type. Collective.
    pub fn set_type(&self, name: &str) -> Result<()> {
        let name = c_string(name, "partitioner type")?;
        // SAFETY: PETSc copies the name.
        chk!(unsafe { sys::PetscPartitionerSetType(self.raw, name.as_ptr()) });
        Ok(())
    }

    /// Takes the settings from the options database, for example the type from
    /// `-petscpartitioner_type` (`PetscPartitionerSetFromOptions`). Collective.
    pub fn set_from_options(&self) -> Result<()> {
        // SAFETY: a valid partitioner.
        chk!(unsafe { sys::PetscPartitionerSetFromOptions(self.raw) });
        Ok(())
    }
}

/// The PETSc array `ptr` of `n` indices, copied, and freed (`PetscFree`).
///
/// # Safety
/// `ptr` is null or an array PETSc allocated for the caller, holding `n` indices.
unsafe fn take_array(ptr: *mut sys::PetscInt, n: usize) -> Result<Vec<sys::PetscInt>> {
    if ptr.is_null() {
        return Ok(Vec::new());
    }
    let v = if n == 0 {
        Vec::new()
    } else {
        unsafe { std::slice::from_raw_parts(ptr, n) }.to_vec()
    };
    chk!(unsafe { sys::petsc_sys_free(ptr.cast()) });
    Ok(v)
}

/// Discretizations: fields, boundary conditions, sections, and the scatters between local and
/// global vectors.
impl Dm {
    /// A new DM with the topology, coordinates and point distribution of this one, copies of its
    /// labels, and no fields (`DMClone`): fields and boundary conditions set on it leave this
    /// DM unchanged. Collective.
    pub fn clone_mesh(&self) -> Result<Dm> {
        let mut raw = null_mut();
        // SAFETY: `raw` receives a new DM.
        chk!(unsafe { sys::DMClone(self.raw, &mut raw) });
        Ok(Dm { raw })
    }

    /// Whether the cells are simplices (`DMPlexIsSimplex`).
    pub fn is_simplex(&self) -> Result<bool> {
        let mut simplex = false;
        chk!(unsafe { sys::DMPlexIsSimplex(self.raw, &mut simplex) });
        Ok(simplex)
    }

    /// Discretizes the coordinates with `fe`, projecting them onto it if `project`
    /// (`DMSetCoordinateDisc`, not localized). Without `fe`, it makes sure the coordinates have
    /// the affine (P1 or Q1) element, which [`Dm::local_offsets`] of [`Dm::coordinate_dm`]
    /// needs. Collective.
    pub fn set_coordinate_disc(&mut self, fe: Option<&Fe>, project: bool) -> Result<()> {
        let fe = fe.map_or(null_mut(), Fe::as_raw);
        chk!(unsafe { sys::DMSetCoordinateDisc(self.raw, fe, false, project) });
        Ok(())
    }

    /// The DM of the coordinates (`DMGetCoordinateDM`), shared.
    pub fn coordinate_dm(&self) -> Result<Dm> {
        let mut raw = null_mut();
        chk!(unsafe { sys::DMGetCoordinateDM(self.raw, &mut raw) });
        // SAFETY: a borrowed reference of the DM, which becomes ours.
        chk!(unsafe { sys::PetscObjectReference(raw as sys::PetscObject) });
        Ok(Dm { raw })
    }

    /// The local coordinates (with those of ghost points), copied (`DMGetCoordinatesLocal`):
    /// `coordinate_dimension` values per node of [`Dm::coordinate_dm`]. Collective, unless
    /// they are set up (see [`Dm::set_up_local_coordinates`]).
    pub fn local_coordinates(&self) -> Result<Vec<f64>> {
        let mut raw = null_mut();
        // SAFETY: the vector is borrowed from the DM.
        chk!(unsafe { sys::DMGetCoordinatesLocal(self.raw, &mut raw) });
        if raw.is_null() {
            return Err(Error::new(
                sys::PETSC_ERR_ARG_WRONGSTATE,
                "the DM has no coordinates",
            ));
        }
        let mut n = 0;
        chk!(unsafe { sys::VecGetLocalSize(raw, &mut n) });
        let mut array = null();
        chk!(unsafe { sys::VecGetArrayRead(raw, &mut array) });
        let values = if n == 0 {
            Vec::new()
        } else {
            // SAFETY: `n` values, valid until the restore.
            unsafe { std::slice::from_raw_parts(array, n as usize) }.to_vec()
        };
        chk!(unsafe { sys::VecRestoreArrayRead(raw, &mut array) });
        Ok(values)
    }

    /// Adds the field discretized by `fe` on the whole mesh (`DMAddField`). The DM keeps a
    /// reference to `fe`. [`Dm::create_ds`] completes the setup.
    pub fn add_field(&mut self, fe: &Fe) -> Result<()> {
        chk!(unsafe { sys::DMAddField(self.raw, null_mut(), fe.as_raw() as sys::PetscObject) });
        Ok(())
    }

    /// Creates the discrete system of the fields (`DMCreateDS`). Collective.
    pub fn create_ds(&mut self) -> Result<()> {
        chk!(unsafe { sys::DMCreateDS(self.raw) });
        Ok(())
    }

    /// Makes the field `field` essential (Dirichlet) on the points where the label `label` has
    /// one of `values`, and on their closure (`DMAddBoundary` with `DM_BC_ESSENTIAL`, named
    /// `name`, without boundary value functions): the section constrains their dofs, so global
    /// vectors leave them out. Completes the label. Must come before the section is created
    /// (by the first vector, for example).
    pub fn add_essential_boundary(
        &mut self,
        name: &str,
        label: &str,
        values: &[sys::PetscInt],
        field: usize,
    ) -> Result<()> {
        let c_name = c_string(name, "boundary name")?;
        let c_label = c_string(label, "label name")?;
        let mut raw_label = null_mut();
        // SAFETY: the label is borrowed from the DM.
        chk!(unsafe { sys::DMGetLabel(self.raw, c_label.as_ptr(), &mut raw_label) });
        if raw_label.is_null() {
            return Err(Error::new(
                sys::PETSC_ERR_ARG_WRONG,
                format!("DM has no label {label:?}"),
            ));
        }
        // SAFETY: PETSc copies the name and the values; no functions and no context.
        chk!(unsafe {
            sys::DMAddBoundary(
                self.raw,
                sys::DM_BC_ESSENTIAL,
                c_name.as_ptr(),
                raw_label,
                to_int(values.len(), "number of label values")?,
                values.as_ptr(),
                to_int(field, "field")?,
                0,
                null(),
                None,
                None,
                null_mut(),
                null_mut(),
            )
        });
        Ok(())
    }

    /// The values the label `name` takes on any process, sorted; none if there is no such label
    /// (`DMLabelGetValueISGlobal`, values of nonempty strata). Collective.
    pub fn label_values_global(&self, name: &str) -> Result<Vec<sys::PetscInt>> {
        let c_name = c_string(name, "label name")?;
        let mut label = null_mut();
        // SAFETY: the label is borrowed from the DM.
        chk!(unsafe { sys::DMGetLabel(self.raw, c_name.as_ptr(), &mut label) });
        // Every process takes part in the reduction, also one without the label.
        let mut has = sys::PetscInt::from(!label.is_null());
        has = self.comm().max_int(has)?;
        if has == 0 {
            return Ok(Vec::new());
        }
        if label.is_null() {
            return Err(Error::new(
                sys::PETSC_ERR_ARG_WRONG,
                format!("label {name:?} exists on some processes only"),
            ));
        }
        let mut is = null_mut();
        // SAFETY: the IS is ours to destroy.
        chk!(unsafe { sys::DMLabelGetValueISGlobal(self.comm().as_raw(), label, true, &mut is) });
        let mut values = take_is(is, true)?;
        values.sort_unstable();
        Ok(values)
    }

    /// The local points in the closure of `point`, starting with `point`
    /// (`DMPlexGetTransitiveClosure`).
    pub fn closure(&self, point: Point) -> Result<Vec<Point>> {
        let mut n = 0;
        let mut pairs: *mut sys::PetscInt = null_mut();
        // SAFETY: PETSc lends `2 n` values (point, orientation) until the restore.
        chk!(unsafe { sys::DMPlexGetTransitiveClosure(self.raw, point, true, &mut n, &mut pairs) });
        let points = if n == 0 {
            Vec::new()
        } else {
            unsafe { std::slice::from_raw_parts(pairs, 2 * n as usize) }
                .iter()
                .step_by(2)
                .copied()
                .collect()
        };
        chk!(unsafe {
            sys::DMPlexRestoreTransitiveClosure(self.raw, point, true, &mut n, &mut pairs)
        });
        Ok(points)
    }

    /// The indices in local vectors of the dofs of `point`, constrained ones included
    /// (`DMGetLocalSection`, which creates the section from the fields). Collective on the first
    /// call, which creates the section.
    pub fn local_dofs(&self, point: Point) -> Result<Range<usize>> {
        let mut section = null_mut();
        // SAFETY: the section is borrowed from the DM.
        chk!(unsafe { sys::DMGetLocalSection(self.raw, &mut section) });
        let (mut dof, mut offset) = (0, 0);
        chk!(unsafe { sys::PetscSectionGetDof(section, point, &mut dof) });
        chk!(unsafe { sys::PetscSectionGetOffset(section, point, &mut offset) });
        let (offset, dof) = (offset as usize, dof as usize);
        Ok(offset..offset + dof)
    }

    /// Names the field `field` of the local section `name`, and its components `components`, one
    /// name each (`PetscSectionSetFieldName`, `PetscSectionSetComponentName`): for viewers, which
    /// name data after them (a VTK viewer names point data `<vector><field>.<component>`).
    /// Collective on the first call, which creates the section. Errors for a field which does
    /// not exist, or a number of names other than the number of components.
    pub fn set_field_names(&mut self, field: usize, name: &str, components: &[&str]) -> Result<()> {
        let section = self.local_section()?;
        let field_int = to_int(field, "field")?;
        let c_name = c_string(name, "field name")?;
        // SAFETY: a valid section; PETSc checks the field number and copies the name.
        chk!(unsafe { sys::PetscSectionSetFieldName(section, field_int, c_name.as_ptr()) });
        let mut n = 0;
        // SAFETY: a valid section and field (checked above).
        chk!(unsafe { sys::PetscSectionGetFieldComponents(section, field_int, &mut n) });
        if components.len() != n as usize {
            return Err(Error::new(
                sys::PETSC_ERR_ARG_SIZ,
                format!(
                    "{} component names for field {field}, which has {n} components",
                    components.len()
                ),
            ));
        }
        for (c, component) in components.iter().enumerate() {
            let c_component = c_string(component, "component name")?;
            // SAFETY: a valid section, field and component; PETSc copies the name.
            chk!(unsafe {
                sys::PetscSectionSetComponentName(
                    section,
                    field_int,
                    to_int(c, "component")?,
                    c_component.as_ptr(),
                )
            });
        }
        Ok(())
    }

    /// The name of the field `field` of the local section and the names of its components (see
    /// [`Dm::set_field_names`]). Collective on the first call, which creates the section.
    pub fn field_names(&self, field: usize) -> Result<(String, Vec<String>)> {
        let section = self.local_section()?;
        let field = to_int(field, "field")?;
        let mut name: *const c_char = null();
        // SAFETY: a valid section; PETSc checks the field number; `name` receives a string of
        // the section.
        chk!(unsafe { sys::PetscSectionGetFieldName(section, field, &mut name) });
        // SAFETY: null or a NUL terminated string of the section, valid for this call.
        let name = unsafe { string_from(name) }.unwrap_or_default();
        let mut n = 0;
        // SAFETY: a valid section and field.
        chk!(unsafe { sys::PetscSectionGetFieldComponents(section, field, &mut n) });
        let components = (0..n)
            .map(|c| {
                let mut component: *const c_char = null();
                // SAFETY: a valid section, field and component.
                chk!(unsafe {
                    sys::PetscSectionGetComponentName(section, field, c, &mut component)
                });
                // SAFETY: null or a NUL terminated string of the section, valid for this call.
                Ok(unsafe { string_from(component) }.unwrap_or_default())
            })
            .collect::<Result<Vec<_>>>()?;
        Ok((name, components))
    }

    /// The local section (`DMGetLocalSection`), borrowed from the DM: valid while the DM lives
    /// and its fields do not change. Collective on the first call, which creates it.
    fn local_section(&self) -> Result<sys::PetscSection> {
        let mut section = null_mut();
        // SAFETY: the section is borrowed from the DM.
        chk!(unsafe { sys::DMGetLocalSection(self.raw, &mut section) });
        Ok(section)
    }

    /// The number of dofs of `point` constrained by essential boundary conditions
    /// (`PetscSectionGetConstraintDof` of the local section).
    pub fn constrained_dofs(&self, point: Point) -> Result<usize> {
        let mut section = null_mut();
        // SAFETY: the section is borrowed from the DM.
        chk!(unsafe { sys::DMGetLocalSection(self.raw, &mut section) });
        let mut n = 0;
        chk!(unsafe { sys::PetscSectionGetConstraintDof(section, point, &mut n) });
        Ok(n as usize)
    }

    /// The cell restriction of field `field` on the cells of the local mesh, in the order of the
    /// cells (`DMPlexGetLocalOffsets` on height 0). Essential dofs are included (their indices
    /// are decoded). Needs the discrete system ([`Dm::create_ds`]).
    pub fn local_offsets(&self, field: usize) -> Result<LocalOffsets> {
        let (mut num_cells, mut cell_size, mut num_comp, mut local_size) = (0, 0, 0, 0);
        let mut offsets: *mut sys::PetscInt = null_mut();
        // SAFETY: `offsets` receives an array of `num_cells * cell_size` indices which we free.
        chk!(unsafe {
            sys::DMPlexGetLocalOffsets(
                self.raw,
                null_mut(),
                0,
                0,
                to_int(field, "field")?,
                &mut num_cells,
                &mut cell_size,
                &mut num_comp,
                &mut local_size,
                &mut offsets,
            )
        });
        let (num_cells, cell_size) = (num_cells as usize, cell_size as usize);
        let offsets = unsafe { take_array(offsets, num_cells * cell_size) }?;
        Ok(LocalOffsets {
            num_cells,
            cell_size,
            num_comp: num_comp as usize,
            local_size: local_size as usize,
            offsets,
        })
    }

    /// For every entry of a local vector, the index of its entry in global vectors: owned by this
    /// or another process, or negative for a constrained (essential) dof, which global vectors
    /// leave out (`DMGetLocalToGlobalMapping`). Collective on the first call.
    pub fn local_to_global_indices(&self) -> Result<Vec<sys::PetscInt>> {
        let mut ltog = null_mut();
        // SAFETY: the mapping is borrowed from the DM.
        chk!(unsafe { sys::DMGetLocalToGlobalMapping(self.raw, &mut ltog) });
        let mut n = 0;
        chk!(unsafe { sys::ISLocalToGlobalMappingGetSize(ltog, &mut n) });
        let mut indices: *const sys::PetscInt = null();
        chk!(unsafe { sys::ISLocalToGlobalMappingGetIndices(ltog, &mut indices) });
        let v = if n == 0 {
            Vec::new()
        } else {
            // SAFETY: `n` indices, valid until the restore.
            unsafe { std::slice::from_raw_parts(indices, n as usize) }.to_vec()
        };
        chk!(unsafe { sys::ISLocalToGlobalMappingRestoreIndices(ltog, &mut indices) });
        Ok(v)
    }

    /// Scatters the global vector `global` to the local vector `local`, with the entries of
    /// ghost points (`DMGlobalToLocal`, `INSERT_VALUES`). Constrained entries of `local` are
    /// left as they are. Collective.
    pub fn global_to_local(&self, global: &impl AsVec, local: &mut impl AsVecMut) -> Result<()> {
        chk!(unsafe {
            sys::DMGlobalToLocal(
                self.raw,
                global.as_raw(),
                sys::INSERT_VALUES,
                local.as_raw(),
            )
        });
        Ok(())
    }

    /// Scatters the local vector `local` to the global vector `global` (`DMLocalToGlobal`):
    /// with [`InsertMode::Add`], the entries of a point shared by several processes add up
    /// into `global`, which is not zeroed first. Constrained entries are dropped. Collective.
    pub fn local_to_global(
        &self,
        local: &impl AsVec,
        mode: InsertMode,
        global: &mut impl AsVecMut,
    ) -> Result<()> {
        chk!(unsafe {
            sys::DMLocalToGlobal(self.raw, local.as_raw(), mode.raw(), global.as_raw())
        });
        Ok(())
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
