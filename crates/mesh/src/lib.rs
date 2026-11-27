//! gmsh meshes as distributed DMPlex, with named tags and a uniform refinement hierarchy.
//!
//! [`Mesh::load`] reads the gmsh file of a [`Scene`] into a DMPlex whose labels carry the gmsh
//! physical names (`-dm_plex_gmsh_use_regions`), splits each vertex where the boundary of a 2D
//! mesh touches itself into one vertex per fan of its cells (see [`Mesh::load`]), distributes
//! it over the processes
//! (`DMPlexDistribute`, no overlap) and refines it uniformly (`DMRefine`). The graph partitioner
//! PT-Scotch decides which process owns which cell, if PETSc has it (see [`PARTITIONER`]). The
//! levels, coarse first, are linked for multigrid (`DMSetCoarseDM`). The labels of the physical
//! names are the mesh's [`Tag`]s; they survive distribution and refinement, so every level has
//! them.
//!
//! A tag is data from the mesh file: this crate knows no tag names.

use std::cmp::max;
use std::ops::Range;
use std::path::{Path, PathBuf};

use contracts::{MeshSource, Scene};
use petsc::{Comm, Dm, Petsc, Point};

mod split;

/// The name of the DM of a mesh.
const NAME: &str = "mesh";

/// The PETSc option which names the labels of a gmsh mesh after its physical names.
const USE_REGIONS: &str = "-dm_plex_gmsh_use_regions";

/// The partitioner which distributes the mesh, if PETSc has it: PT-Scotch keeps the cells of a
/// process together, so few faces are shared between processes (a small edge cut, little
/// communication). Without it, PETSc's `simple` partitioner splits the cells by their number in
/// the file, which scatters them. The option `-petscpartitioner_type` selects another.
///
/// A PT-Scotch partition is reproducible: the same mesh file on the same number of processes gives
/// the same partition, in every run and every time in a run. That takes PT-Scotch built without
/// threads (`THREADS=OFF`, whose routines may depend on the scheduling) and with its
/// deterministic algorithms with a fixed random seed (`SCOTCH_DETERMINISTIC=FULL`), as
/// `scripts/deps/build-native.sh` builds it, and its random generator reset before each
/// partitioning ([`Petsc::reset_partitioner_random`]). The environment variables
/// `SCOTCH_DETERMINISTIC=0` or `SCOTCH_RANDOM_FIXED_SEED=0` would undo the build options.
pub const PARTITIONER: &str = "ptscotch";

/// The partitioner without [`PARTITIONER`].
const FALLBACK_PARTITIONER: &str = "simple";

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
    #[error("mesh format '{format}' is not yet supported by the mesher")]
    Source { format: &'static str },
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
    /// See [`Mesh::local_bounds`].
    bounds: Option<[[f64; 2]; 2]>,
    /// The topological dimension.
    dim: usize,
}

impl Mesh {
    /// Reads the mesh of `scene`, distributes it over the processes of `petsc` and refines it
    /// uniformly `refinements` times. Collective.
    ///
    /// A vertex of a 2D mesh where the boundary touches itself (two holes, or a hole and the
    /// outer boundary, meeting at one point) is split first into one vertex per fan of its
    /// cells (cells joined by the edges at the vertex), at the same coordinates: a point contact
    /// is an aperture of zero width, through which no field passes, so fields are discontinuous
    /// there. Every cell and edge keeps its tags, the copies of a vertex the tags of the vertex.
    /// A mesh without such a vertex is used as read.
    ///
    /// Errors (on every process): [`MeshError::NotFound`] if process 0 finds no file at
    /// the gmsh path of the scene, [`MeshError::Read`] if PETSc cannot read it,
    /// [`MeshError::Dimension`] if its dimension is not `scene.dimension`,
    /// [`MeshError::Source`] for the `polygons` and `lobules` sources (not yet supported), and
    /// [`MeshError::Petsc`] if distributing or refining fails.
    pub fn load(petsc: &Petsc, scene: &Scene, refinements: u32) -> Result<Mesh, MeshError> {
        let path = match &scene.mesh {
            MeshSource::Gmsh { path } => path,
            MeshSource::Polygons { .. } => {
                return Err(MeshError::Source { format: "polygons" });
            }
            MeshSource::Lobules(_) => return Err(MeshError::Source { format: "lobules" }),
        };
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
        if dim == 2
            && let Some(split) = split::split_pinched_vertices(&coarse, &comm)?
        {
            coarse = split;
        }
        select_partitioner(petsc, &coarse)?;
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
        let bounds = local_bounds(&levels)?;
        Ok(Mesh {
            levels,
            tags,
            bounds,
            dim,
        })
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

    /// The bounding box `[[xmin, ymin], [xmax, ymax]]` of the coordinates of this process's cells
    /// on all levels (their vertices, ghosts included; the first two coordinates), or `None` if
    /// this process has no cells. Every point of a local cell is inside it, so it is the region
    /// of a raster this process needs (see `maps::Maps::read`).
    pub fn local_bounds(&self) -> Option<[[f64; 2]; 2]> {
        self.bounds
    }

    /// The local cell of the coarse level (level 0) which contains the local cell `cell` of the
    /// level `level` (cells numbered from 0 in the order of the cells of the level).
    ///
    /// The regular refinement (`DMRefine`) splits each cell into `2^dim` children (4 triangles
    /// in 2D), numbered consecutively in the order of their parents (the cells of a mesh of one
    /// cell type), and keeps them on the process of their parent: the parent of cell `c` is cell
    /// `c / 2^dim` of the level before. Not collective.
    ///
    /// Panics if the mesh has no level `level`.
    pub fn coarse_cell(&self, level: usize, cell: usize) -> usize {
        assert!(
            level < self.levels.len(),
            "mesh has no level {level} (it has {})",
            self.levels.len()
        );
        cell >> (self.dim * level)
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

/// The bounding box of the coordinates of the local cells of `levels` (see
/// [`Mesh::local_bounds`]), `None` without local cells. The levels have their local coordinates
/// set up, so this is not collective.
fn local_bounds(levels: &[Dm]) -> petsc::Result<Option<[[f64; 2]; 2]>> {
    let mut bounds: Option<[[f64; 2]; 2]> = None;
    for level in levels {
        if level.height_stratum(0)?.is_empty() {
            continue;
        }
        let dim = level.coordinate_dimension()?;
        for x in level.local_coordinates()?.chunks_exact(dim) {
            let [lo, hi] = bounds.get_or_insert([[x[0], x[1]], [x[0], x[1]]]);
            for axis in 0..2 {
                lo[axis] = lo[axis].min(x[axis]);
                hi[axis] = hi[axis].max(x[axis]);
            }
        }
    }
    Ok(bounds)
}

/// The mesh file `path` read on `comm`, with labels named after the gmsh physical names.
/// Process 0 reads the file and holds the whole mesh.
fn read(petsc: &Petsc, comm: &Comm, path: &Path) -> petsc::Result<Dm> {
    petsc.with_option(USE_REGIONS, Some("true"), || {
        Dm::plex_from_file(comm, path, NAME, true)
    })
}

/// Sets the partitioner of `dm`: [`PARTITIONER`] if PETSc has it, else [`FALLBACK_PARTITIONER`],
/// unless the option `-petscpartitioner_type` selects another. Resets PT-Scotch's random
/// generator, so that the partition does not depend on earlier partitionings in the process (see
/// [`PARTITIONER`] for what else a reproducible partition takes). Collective.
fn select_partitioner(petsc: &Petsc, dm: &Dm) -> petsc::Result<()> {
    petsc.reset_partitioner_random();
    let partitioner = dm.partitioner()?;
    let default = if petsc.has_external_package(PARTITIONER)? {
        PARTITIONER
    } else {
        FALLBACK_PARTITIONER
    };
    partitioner.set_type(default)?;
    partitioner.set_from_options()
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
        .filter(|n| !contracts::RESERVED_TAGS.contains(&n.as_str()))
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
