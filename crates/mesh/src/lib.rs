//! Meshes as distributed DMPlex, with named tags and a uniform refinement hierarchy.
//!
//! [`Mesh::load`] makes the mesh of a [`Scene`]: it reads a gmsh file into a DMPlex whose labels
//! carry the gmsh physical names (`-dm_plex_gmsh_use_regions`), or, for the `polygons` and
//! `lobules` sources, process 0 meshes the tagged polygons or the lobules with the [`mesher`]
//! and turns the [`TaggedMesh`] into a DMPlex with one label per tag ([`Mesh::from_tagged`]).
//! The rest is the same for every source ([`Mesh::from_dm`]): it splits each vertex where the
//! boundary of a 2D mesh touches itself into one vertex per fan of its cells (see
//! [`Mesh::load`]), distributes the mesh over the processes
//! (`DMPlexDistribute`, no overlap) and refines it uniformly (`DMRefine`). The graph partitioner
//! PT-Scotch decides which process owns which cell, if PETSc has it (see [`PARTITIONER`]). The
//! levels, coarse first, are linked for multigrid (`DMSetCoarseDM`). The labels of the physical
//! names (or of the tags of the mesher) are the mesh's [`Tag`]s; they survive distribution and
//! refinement, so every level has them.
//!
//! The numbering of the points of a distributed mesh depends on the partition. What does not:
//! the element ids ([`Mesh::element_ids`], the number of each coarse cell in the undistributed
//! mesh, carried through the distribution) and the fine element ids of the refined cells
//! ([`Mesh::fine_element_id`]). On them rests the canonical finest mesh on process 0
//! ([`Mesh::canonical`], the same for any number of processes) and the gathers of values into
//! its order ([`Mesh::gather_vertex_values`], [`Mesh::gather_element_values`]).
//!
//! A tag is data from the mesh file or the scene: this crate knows no tag names.

use std::cell::OnceCell;
use std::cmp::max;
use std::ops::Range;
use std::path::{Path, PathBuf};

use contracts::{ContractError, MeshSource, Polygons, Scene};
use mesher::{Domain, Lobules, Quality, Size, TaggedMesh};
use petsc::{Comm, Dm, Petsc, Point};

mod canonical;
mod split;

pub use canonical::{CanonicalMesh, CanonicalTag, ElementRows};

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
    /// The source of the scene cannot be meshed: the polygons file cannot be read, or the
    /// mesher rejects the polygons or the lobules (the message names the file, if there is one,
    /// and where in um), or the scene is not two dimensional.
    #[error("{message}")]
    Mesher { message: String },
    /// A gather onto process 0 ([`Mesh::canonical`], [`Mesh::gather_vertex_values`],
    /// [`Mesh::gather_element_values`]) failed: its arguments do not fit the mesh on some
    /// process, or PETSc failed on one (the message of the lowest such process). The same on
    /// every process.
    #[error("gather onto process 0: {0}")]
    Gather(String),
    /// The distribution of the mesh over the processes lost cells: a cell of the distributed
    /// mesh comes from no cell of the coarse one. The same on every process.
    #[error("cannot distribute the mesh: {0}")]
    Distribute(String),
    #[error(transparent)]
    Petsc(#[from] petsc::Error),
}

impl MeshError {
    /// Whether every process of [`Mesh::load`] has this error, with the same message, at the
    /// same point: [`MeshError::NotFound`] and [`MeshError::Mesher`] (process 0 finds them and
    /// tells the others), [`MeshError::Dimension`] (every process reads the dimension of the
    /// same mesh), [`MeshError::Gather`] and [`MeshError::Distribute`] (the processes agree on
    /// them). After such an error no process waits for another, so a caller may end each
    /// process alone. A PETSc error, or one of reading (the collective read may fail on one
    /// process only), may be on some processes only.
    pub fn is_collective(&self) -> bool {
        matches!(
            self,
            MeshError::NotFound { .. }
                | MeshError::Dimension { .. }
                | MeshError::Mesher { .. }
                | MeshError::Gather(_)
                | MeshError::Distribute(_)
        )
    }
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
    /// See [`Mesh::generated`].
    generated: Option<(TaggedMesh, Quality)>,
    /// See [`Mesh::element_ids`].
    element_ids: Vec<u64>,
    /// The order of [`Mesh::gather_vertex_values`], made with the first canonical mesh.
    vertex_order: OnceCell<canonical::VertexOrder>,
}

impl Mesh {
    /// Makes the mesh of `scene`, distributes it over the processes of `petsc` and refines it
    /// uniformly `refinements` times. Collective.
    ///
    /// A gmsh source is read by process 0. For a `polygons` or a `lobules` source, process 0
    /// alone meshes the polygons file (with the size field of the scene) or the lobules with the
    /// [`mesher`] ([`mesh_serial`]), sends the outcome to the others, and the [`TaggedMesh`]
    /// becomes the coarse mesh ([`Mesh::from_tagged`]); process 0 keeps it and its quality
    /// ([`Mesh::generated`]). Then [`Mesh::from_dm`].
    ///
    /// A vertex of a 2D mesh where the boundary touches itself (two holes, or a hole and the
    /// outer boundary, meeting at one point) is split first into one vertex per fan of its
    /// cells (cells joined by the edges at the vertex), at the same coordinates: a point contact
    /// is an aperture of zero width, through which no field passes, so fields are discontinuous
    /// there. Every cell and edge keeps its tags, the copies of a vertex the tags of the vertex.
    /// A mesh without such a vertex is used as is.
    ///
    /// Errors (on every process): [`MeshError::NotFound`] if process 0 finds no file at
    /// the gmsh path of the scene, [`MeshError::Read`] if PETSc cannot read it,
    /// [`MeshError::Dimension`] if its dimension is not `scene.dimension`,
    /// [`MeshError::Mesher`] if process 0 cannot read the polygons file or the mesher fails
    /// (the message names the file and the place, see [`mesh_serial`]) or a generated source
    /// is in a scene of dimension 3, and [`MeshError::Petsc`] if distributing or refining fails.
    pub fn load(petsc: &Petsc, scene: &Scene, refinements: u32) -> Result<Mesh, MeshError> {
        let path = match &scene.mesh {
            MeshSource::Gmsh { path } => path,
            MeshSource::Polygons { .. } | MeshSource::Lobules(_) => {
                return Mesh::generate(petsc, scene, refinements);
            }
        };
        check_readable(petsc, path)?;
        let comm = petsc.comm();
        let coarse = read(petsc, &comm, path).map_err(|source| MeshError::Read {
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
        Mesh::from_dm(petsc, coarse, refinements)
    }

    /// [`Mesh::load`] of a `polygons` or a `lobules` source: process 0 meshes, sends the
    /// outcome to the others, then [`Mesh::from_tagged`]. Collective.
    fn generate(petsc: &Petsc, scene: &Scene, refinements: u32) -> Result<Mesh, MeshError> {
        // Every process has the scene: no process meshes.
        two_dimensional(scene)?;
        let comm = petsc.comm();
        let generated = (comm.rank() == 0).then(|| mesh_serial(petsc, scene));
        // The first byte 0 for a mesh, or 1 and the error.
        let mut message = Vec::new();
        if let Some(generated) = &generated {
            match generated {
                Ok(_) => message.push(0),
                Err(e) => {
                    message.push(1);
                    message.extend(e.to_string().into_bytes());
                }
            }
        }
        comm.broadcast_bytes(0, &mut message)?;
        if let Some((1, text)) = message.split_first() {
            return Err(MeshError::Mesher {
                message: String::from_utf8_lossy(text).into_owned(),
            });
        }
        let generated = generated.transpose()?;
        let mut mesh = Mesh::from_tagged(petsc, generated.as_ref().map(|g| &g.0), refinements)?;
        mesh.generated = generated;
        Ok(mesh)
    }

    /// The mesh `mesh` (on process 0; `None` on the others, whose meshes would be ignored)
    /// as the coarse mesh ([`TaggedMesh::to_dm`]: one label per tag, on every process), then
    /// [`Mesh::from_dm`]. Collective.
    ///
    /// Errors: [`MeshError::Petsc`] if process 0's mesh is invalid ([`TaggedMesh::validate`],
    /// the same error on every process) or distributing or refining fails.
    ///
    /// Panics if process 0 has no mesh.
    pub fn from_tagged(
        petsc: &Petsc,
        mesh: Option<&TaggedMesh>,
        refinements: u32,
    ) -> Result<Mesh, MeshError> {
        let comm = petsc.comm();
        assert!(
            comm.rank() != 0 || mesh.is_some(),
            "process 0 holds the mesh of Mesh::from_tagged"
        );
        let empty = TaggedMesh {
            tags: Vec::new(),
            vertices: Vec::new(),
            triangles: Vec::new(),
            triangle_tags: Vec::new(),
            edges: Vec::new(),
            edge_tags: Vec::new(),
        };
        let coarse = mesh.unwrap_or(&empty).to_dm(&comm)?;
        Mesh::from_dm(petsc, coarse, refinements)
    }

    /// The mesh of the DMPlex `coarse` on the processes of `petsc`, held by process 0 (as
    /// [`Dm::plex_from_file`] reads a file or [`TaggedMesh::to_dm`] makes one), its labels the
    /// tags: the vertices where the boundary of a 2D mesh touches itself split (see
    /// [`Mesh::load`]), partitioned ([`PARTITIONER`]), distributed and refined uniformly
    /// `refinements` times. Collective.
    ///
    /// The cell numbering of the split coarse mesh gives the element ids
    /// ([`Mesh::element_ids`]): the migration star forest of `DMPlexDistribute` carries the
    /// number of each cell to the process and the local cell it becomes.
    ///
    /// Errors: [`MeshError::Petsc`] if splitting, distributing or refining fails;
    /// [`MeshError::Distribute`] (on every process) if the distribution loses cells.
    pub fn from_dm(petsc: &Petsc, coarse: Dm, refinements: u32) -> Result<Mesh, MeshError> {
        let comm = petsc.comm();
        let mut coarse = coarse;
        let dim = coarse.dimension()?;
        if dim == 2
            && let Some(split) = split::split_pinched_vertices(&coarse, &comm)?
        {
            coarse = split;
        }
        select_partitioner(petsc, &coarse)?;
        // The numbers of the cells of the undistributed mesh: the element ids.
        let numbers = coarse.cell_numbering()?;
        let element_ids = match coarse.distribute_with_migration(0)? {
            Some((distributed, migration)) => {
                let ids = migrated_numbers(&coarse, &numbers, &distributed, &migration)?;
                coarse = distributed;
                ids
            }
            None => numbers.iter().map(|&g| owned_number(g)).collect(),
        };
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
            generated: None,
            element_ids,
            vertex_order: OnceCell::new(),
        })
    }

    /// On process 0, the mesh the mesher made for a `polygons` or `lobules` source and its
    /// quality ([`Mesh::load`]); `None` on the other processes and for a gmsh source.
    pub fn generated(&self) -> Option<&(TaggedMesh, Quality)> {
        self.generated.as_ref()
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

/// The global number `g` of a cell or vertex of a numbering of PETSc (`-(g + 1)` for a point of
/// another process).
fn owned_number(g: petsc::sys::PetscInt) -> u64 {
    (if g >= 0 { g } else { -(g + 1) }) as u64
}

/// The numbers `numbers` of the cells of `coarse` (its cell numbering), carried to the cells of
/// `distributed` by the star forest `migration` of [`Dm::distribute_with_migration`]: the number
/// of each local cell of `distributed`, in the order of its cells. Collective.
///
/// Errors: [`MeshError::Distribute`] (on every process) if a cell of `distributed` comes from
/// no cell of `coarse`; [`MeshError::Petsc`] if PETSc fails.
fn migrated_numbers(
    coarse: &Dm,
    numbers: &[petsc::sys::PetscInt],
    distributed: &Dm,
    migration: &petsc::Sf,
) -> Result<Vec<u64>, MeshError> {
    let points = |dm: &Dm| -> petsc::Result<usize> {
        let chart = (0..=dm.dimension()?)
            .map(|d| dm.depth_stratum(d).map(|s| s.end))
            .collect::<petsc::Result<Vec<_>>>()?;
        Ok(chart.into_iter().max().unwrap_or(0).max(0) as usize)
    };
    let cells = coarse.height_stratum(0)?;
    let mut roots = vec![-1; points(coarse)?];
    for (cell, &g) in cells.zip(numbers) {
        roots[cell as usize] = owned_number(g) as petsc::sys::PetscInt;
    }
    let mut leaves = vec![-1; points(distributed)?];
    migration.bcast_int(&roots, &mut leaves)?;
    let cells = distributed.height_stratum(0)?;
    let ids: Vec<petsc::sys::PetscInt> = cells.map(|c| leaves[c as usize]).collect();
    let missing = ids.iter().filter(|&&g| g < 0).count();
    let missing = distributed
        .comm()
        .sum_int(petsc::sys::PetscInt::try_from(missing).unwrap_or(petsc::sys::PetscInt::MAX))?;
    if missing > 0 {
        return Err(MeshError::Distribute(format!(
            "{missing} cells of the distributed mesh come from no cell of the coarse mesh"
        )));
    }
    Ok(ids.into_iter().map(|g| g as u64).collect())
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

/// The 2D mesh of the source of `scene` as a [`TaggedMesh`], and its quality, made on this
/// process alone (no collective call; `PETSC_COMM_SELF`): for a gmsh source the file read and
/// converted ([`TaggedMesh::from_dm`], the quality without a size field); for a `polygons`
/// source the polygons file meshed with the size field of the scene, for a `lobules` source the
/// lobules ([`mesher::generate`], the quality against the size field).
///
/// Errors: [`MeshError::NotFound`] and [`MeshError::Read`] for a gmsh file which is not there or
/// which PETSc cannot read; [`MeshError::Mesher`] if the scene is not two dimensional, the
/// polygons file cannot be read, or the mesh cannot be made (a mesher error, with the path of
/// the polygons or gmsh file, and the place in um).
pub fn mesh_serial(petsc: &Petsc, scene: &Scene) -> Result<(TaggedMesh, Quality), MeshError> {
    let mesher = |message: String| MeshError::Mesher { message };
    two_dimensional(scene)?;
    match &scene.mesh {
        MeshSource::Gmsh { path } => {
            if !path.is_file() {
                return Err(MeshError::NotFound { path: path.clone() });
            }
            let dm = read(petsc, &petsc.comm_self(), path).map_err(|source| MeshError::Read {
                path: path.clone(),
                source,
            })?;
            let tagged = TaggedMesh::from_dm(&dm)
                .map_err(|e| mesher(format!("cannot convert the mesh {}: {e}", path.display())))?;
            let quality = tagged
                .quality(None)
                .map_err(|e| mesher(format!("the mesh {}: {e}", path.display())))?;
            Ok((tagged, quality))
        }
        MeshSource::Polygons { path, size } => {
            let polygons = Polygons::load(path).map_err(|e| match e {
                // These name the file.
                ContractError::Io { .. } | ContractError::Parse { .. } => {
                    mesher(format!("cannot read the polygons: {e}"))
                }
                e => mesher(format!("cannot read the polygons {}: {e}", path.display())),
            })?;
            let cannot = |e: mesher::MesherError| {
                mesher(format!("cannot mesh the polygons {}: {e}", path.display()))
            };
            let domain = Domain::from_polygons(&polygons).map_err(cannot)?;
            let size = Size::new(size, &domain).map_err(cannot)?;
            mesher::generate(petsc, &domain, &size).map_err(cannot)
        }
        MeshSource::Lobules(spec) => {
            let cannot = |e: mesher::MesherError| match e {
                // Its own message starts with "lobules: ".
                mesher::MesherError::InvalidLobules(m) => {
                    mesher(format!("cannot mesh the lobules: {m}"))
                }
                e => mesher(format!("cannot mesh the lobules: {e}")),
            };
            let lobules = Lobules::new(spec).map_err(cannot)?;
            mesher::generate(petsc, lobules.domain(), lobules.size()).map_err(cannot)
        }
    }
}

/// An error ([`MeshError::Mesher`]) unless `scene` is two dimensional, as the meshes of the
/// mesher are.
fn two_dimensional(scene: &Scene) -> Result<(), MeshError> {
    if scene.dimension == 2 {
        Ok(())
    } else {
        Err(MeshError::Mesher {
            message: format!(
                "the mesher makes 2D meshes, but the scene has dimension {}",
                scene.dimension
            ),
        })
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
