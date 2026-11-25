//! gmsh meshes as distributed DMPlex, with named tags and a uniform refinement hierarchy.
//!
//! [`Mesh::load`] reads the gmsh file of a [`Scene`] into a DMPlex whose labels carry the gmsh
//! physical names (`-dm_plex_gmsh_use_regions`), distributes it over the processes
//! (`DMPlexDistribute`, no overlap) and refines it uniformly (`DMRefine`). The levels, coarse
//! first, are linked for multigrid (`DMSetCoarseDM`). The labels of the physical names are the
//! mesh's [`Tag`]s; they survive distribution and refinement, so every level has them.
//!
//! A tag is data from the mesh file: this crate knows no tag names.

use std::cmp::max;
use std::ops::Range;
use std::path::{Path, PathBuf};

use contracts::{MeshFormat, Scene};
use petsc::{Comm, Dm, Petsc, Point};

/// Labels PETSc creates itself (gmsh's generic sets, topology, ghosts, output), which are not
/// gmsh physical names.
const PETSC_LABELS: [&str; 9] = [
    "depth",
    "celltype",
    "Cell Sets",
    "Face Sets",
    "Edge Sets",
    "Vertex Sets",
    "marker",
    "ghost",
    "vtk",
];

/// The PETSc option which names the labels of a gmsh mesh after its physical names.
const USE_REGIONS: &str = "-dm_plex_gmsh_use_regions";

/// A named part of the mesh: a gmsh physical group.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tag {
    /// The physical name, which is also the name of the DMPlex label.
    pub name: String,
    /// The topological dimension of the points it marks: 2 for areas and 1 for edges in 2D.
    pub dim: u8,
}

#[derive(Debug, thiserror::Error)]
pub enum MeshError {
    #[error("mesh file not found: {}", .path.display())]
    NotFound { path: PathBuf },
    #[error("cannot read mesh {}: {source}", .path.display())]
    Read { path: PathBuf, source: petsc::Error },
    #[error("mesh {} has dimension {mesh}, but the scene has dimension {scene}", .path.display())]
    Dimension {
        path: PathBuf,
        mesh: usize,
        scene: u8,
    },
    #[error("mesh has no tag '{name}'; available: {}", .available.join(", "))]
    UnknownTag {
        name: String,
        available: Vec<String>,
    },
    #[error(transparent)]
    Petsc(#[from] petsc::Error),
}

/// A distributed mesh and its uniform refinements.
#[derive(Debug)]
pub struct Mesh {
    /// Coarse first; never empty.
    levels: Vec<Dm>,
    /// Sorted by name.
    tags: Vec<Tag>,
}

impl Mesh {
    /// Reads the mesh of `scene`, distributes it over the processes of `petsc` and refines it
    /// uniformly `refinements` times. Collective.
    ///
    /// Errors (on every process): [`MeshError::NotFound`] if process 0 finds no file at
    /// `scene.mesh.path`, [`MeshError::Read`] if PETSc cannot read it, [`MeshError::Dimension`]
    /// if its dimension is not `scene.dimension`, and [`MeshError::Petsc`] if distributing or
    /// refining fails.
    pub fn load(petsc: &Petsc, scene: &Scene, refinements: u32) -> Result<Mesh, MeshError> {
        let path = &scene.mesh.path;
        match scene.mesh.format {
            MeshFormat::Gmsh => {}
        }
        check_readable(petsc, path)?;
        let comm = petsc.comm();
        let mut coarse = read(petsc, &comm, path).map_err(|source| MeshError::Read {
            path: path.clone(),
            source,
        })?;
        let dim = coarse.dimension()?;
        if dim != usize::from(scene.dimension) {
            return Err(MeshError::Dimension {
                path: path.clone(),
                mesh: dim,
                scene: scene.dimension,
            });
        }
        if let Some(distributed) = coarse.distribute(0)? {
            coarse = distributed;
        }
        coarse.set_up_local_coordinates()?;
        let tags = tags(&coarse, dim)?;
        let mut levels = vec![coarse];
        for _ in 0..refinements {
            let coarser = levels.last().expect("levels is never empty");
            let mut finer = coarser.refine()?;
            finer.set_coarse_dm(coarser)?;
            finer.set_up_local_coordinates()?;
            levels.push(finer);
        }
        Ok(Mesh { levels, tags })
    }

    /// The levels, coarse first: the distributed mesh as read, then each refinement of the one
    /// before. The finest is the level of the PDE.
    pub fn levels(&self) -> &[Dm] {
        &self.levels
    }

    /// The finest level.
    pub fn finest(&self) -> &Dm {
        self.levels.last().expect("levels is never empty")
    }

    /// The tags, sorted by name.
    pub fn tags(&self) -> &[Tag] {
        &self.tags
    }

    /// The tag named `name`.
    pub fn tag(&self, name: &str) -> Result<Tag, MeshError> {
        self.tags
            .iter()
            .find(|t| t.name == name)
            .cloned()
            .ok_or_else(|| MeshError::UnknownTag {
                name: name.to_string(),
                available: self.tags.iter().map(|t| t.name.clone()).collect(),
            })
    }
}

/// The mesh file `path` read on `comm`, with labels named after the gmsh physical names.
/// Process 0 reads the file and holds the whole mesh.
fn read(petsc: &Petsc, comm: &Comm, path: &Path) -> petsc::Result<Dm> {
    petsc.with_option(USE_REGIONS, Some("true"), || {
        Dm::plex_from_file(comm, path, "mesh", true)
    })
}

/// An error on every process unless process 0 finds the mesh file `path` and can read it.
/// Collective.
///
/// Process 0 alone reads the file within the collective read: if it failed there, the other
/// processes would wait for it forever. So on several processes, process 0 first reads the
/// file alone, and all processes learn the outcome. Only process 0 needs to see the file.
fn check_readable(petsc: &Petsc, path: &Path) -> Result<(), MeshError> {
    const NOT_FOUND: Point = -1;
    let comm = petsc.comm();
    let mut error = None;
    let mut status = 0;
    if comm.rank() == 0 {
        if !path.is_file() {
            status = NOT_FOUND;
        } else if comm.size() > 1
            && let Err(e) = read(petsc, &petsc.comm_self(), path)
        {
            status = e.code.max(1);
            error = Some(e);
        }
    }
    if comm.size() > 1 {
        status = comm.sum_int(status)?;
    }
    match status {
        0 => Ok(()),
        NOT_FOUND => Err(MeshError::NotFound {
            path: path.to_path_buf(),
        }),
        code => Err(MeshError::Read {
            path: path.to_path_buf(),
            source: error.unwrap_or_else(|| petsc::Error {
                code,
                message: "process 0 cannot read the mesh file (its error tells why)".to_string(),
            }),
        }),
    }
}

/// The tags of the distributed mesh `dm` of dimension `dim`: its labels except PETSc's own,
/// sorted by name, with the largest dimension of the points they mark on any process. Labels
/// marking no point are left out. Collective.
fn tags(dm: &Dm, dim: usize) -> Result<Vec<Tag>, MeshError> {
    let mut names: Vec<String> = dm
        .label_names()?
        .into_iter()
        .filter(|n| !PETSC_LABELS.contains(&n.as_str()))
        .collect();
    // The same order on every process: the reductions below match up.
    names.sort();
    let strata = (0..=dim)
        .map(|d| dm.depth_stratum(d))
        .collect::<Result<Vec<_>, _>>()?;
    let mut tags = Vec::with_capacity(names.len());
    for name in names {
        let local = local_dim(dm, &name, &strata)?;
        let global = dm.comm().max_int(local)?;
        if let Ok(dim) = u8::try_from(global) {
            tags.push(Tag { name, dim });
        }
    }
    Ok(tags)
}

/// The largest depth of the local points the label `name` marks, or -1 if it marks none;
/// `strata[d]` are the points of depth `d`.
fn local_dim(dm: &Dm, name: &str, strata: &[Range<Point>]) -> Result<Point, MeshError> {
    let mut dim = -1;
    for value in dm.label_values(name)? {
        for p in dm.stratum_points(name, value)? {
            if let Some(d) = strata.iter().position(|s| s.contains(&p)) {
                dim = max(dim, d as Point);
            }
        }
    }
    Ok(dim)
}
