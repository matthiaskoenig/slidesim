//! The canonical finest mesh on process 0, and values gathered into its order.
//!
//! The finest level of a [`Mesh`] is distributed: each process holds the cells of its partition,
//! numbered in an order of its own. Its canonical form ([`Mesh::canonical`]) is the same for any
//! number of processes, bitwise:
//!
//! - the triangles are the cells of the finest level sorted by their fine element id
//!   ([`Mesh::fine_element_id`]), which is partition independent;
//! - each triangle starts at its vertex with the smallest coordinates (by `x`, then `y`) and is
//!   counterclockwise: a rule on the coordinates, which are the same on every process (those of
//!   the coarse mesh, and midpoints `(a + b) / 2` of edges, whose sum does not depend on the
//!   order), so it does not depend on the order of the vertices of a cell on its process;
//! - the vertices are numbered by their first appearance in the triangles in that order (two
//!   vertices at the same coordinates, the copies of a split pinched vertex, stay two);
//! - `element` holds the coarse element id of each triangle ([`Mesh::element_ids`]);
//! - an area tag (dimension 2) holds the indices of the triangles it marks, ascending; an edge
//!   tag (dimension 1) the edges it marks as pairs of vertex indices, the smaller first, the
//!   pairs ascending. A tag of another dimension holds nothing.
//!
//! The processes identify the vertices they share by their global numbers
//! ([`Dm::vertex_numbering`]); each process sends the coordinates of the vertices it owns.

use petsc::{Comm, Dm, Point};

use crate::{Mesh, MeshError};

/// The finest level of a mesh in a form independent of the partition (see the module).
#[derive(Debug, Clone, PartialEq)]
pub struct CanonicalMesh {
    /// The coordinates (um), numbered by first appearance in `triangles`.
    pub vertices: Vec<[f64; 2]>,
    /// The vertex indices of each triangle, counterclockwise, sorted by fine element id.
    pub triangles: Vec<[u32; 3]>,
    /// The coarse element id of each triangle.
    pub element: Vec<u64>,
    /// The tags of the mesh, in its order (sorted by name).
    pub tags: Vec<CanonicalTag>,
}

/// Rows of values per element from [`Mesh::gather_element_values`]: the element ids, ascending,
/// and their rows, one after the other.
pub type ElementRows = (Vec<u64>, Vec<f64>);

/// A tag of a [`CanonicalMesh`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CanonicalTag {
    pub name: String,
    /// 2 for an area, 1 for edges.
    pub dim: u8,
    /// An area tag: the indices of the triangles it marks, ascending; else empty.
    pub triangles: Vec<u32>,
    /// An edge tag: the edges it marks, the smaller vertex index first, ascending; else empty.
    pub edges: Vec<[u32; 2]>,
}

/// How values at the local vertices of the finest level go into the canonical order: made with
/// the canonical mesh, the same on every later gather.
#[derive(Debug)]
pub(crate) struct VertexOrder {
    /// The local vertices this process owns (indices in the order of the local vertices), in the
    /// order it sends their values.
    owned: Vec<usize>,
    /// On process 0: the number of canonical vertices, and for each process the canonical
    /// index of each vertex it sends.
    slots: Option<(usize, Vec<Vec<u32>>)>,
}

/// A process's part of the canonical mesh: its cells, the vertices it owns, the points of the
/// tags it holds.
struct Part {
    /// For each cell, its fine element id and the global numbers of its vertices.
    cells: Vec<(u64, [u64; 3])>,
    /// For each owned vertex, its global number and coordinates.
    vertices: Vec<(u64, [f64; 2])>,
    /// For each tag of the mesh: of an area tag the fine element ids of the cells it marks, of
    /// an edge tag the global numbers of the vertices of each edge it marks.
    tags: Vec<Vec<u64>>,
}

impl Part {
    fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        let mut put = |x: u64| bytes.extend(x.to_le_bytes());
        put(self.cells.len() as u64);
        for (id, vertices) in &self.cells {
            put(*id);
            vertices.iter().for_each(|&v| put(v));
        }
        put(self.vertices.len() as u64);
        for (g, [x, y]) in &self.vertices {
            put(*g);
            put(x.to_bits());
            put(y.to_bits());
        }
        for tag in &self.tags {
            put(tag.len() as u64);
            tag.iter().for_each(|&x| put(x));
        }
        bytes
    }

    /// The part of the bytes of [`Part::to_bytes`] with `tags` tags.
    fn from_bytes(bytes: &[u8], tags: usize) -> Part {
        let mut words = bytes
            .as_chunks::<8>()
            .0
            .iter()
            .map(|w| u64::from_le_bytes(*w));
        let mut next = || words.next().expect("a part is complete");
        let cells = (0..next())
            .map(|_| (next(), [next(), next(), next()]))
            .collect();
        let vertices = (0..next())
            .map(|_| (next(), [f64::from_bits(next()), f64::from_bits(next())]))
            .collect();
        let tags = (0..tags)
            .map(|_| (0..next()).map(|_| next()).collect())
            .collect();
        Part {
            cells,
            vertices,
            tags,
        }
    }
}

impl Mesh {
    /// The element id of each local coarse cell (level 0, numbered from 0 in the order of its
    /// cells): its number in the undistributed coarse mesh, after the pinched vertices are split
    /// (as [`Mesh::load`] makes it, the cells of process 0 in their order). The same element has
    /// the same id on any number of processes: [`Mesh::from_dm`] carries the numbers through
    /// `DMPlexDistribute` with its migration star forest.
    pub fn element_ids(&self) -> &[u64] {
        &self.element_ids
    }

    /// The fine element id of the local cell `cell` of level `level` (numbered from 0 in the
    /// order of the cells of the level): the element id of its coarse cell, shifted by
    /// `dim * level` bits, with the bits of its place in the refinement below,
    /// `(element_ids[cell >> (dim level)] << (dim level)) | (cell & ((1 << (dim level)) - 1))`.
    ///
    /// The regular refinement numbers the `2^dim` children of a cell consecutively, in the order
    /// of their parents (see [`Mesh::coarse_cell`]), and makes the same children of a cell on
    /// any process (the cones of the distributed cells keep their order): so the low bits are
    /// the places of the cell among its siblings on each level, coarse first, and the id is
    /// partition independent. On level 0 it is the element id. Not collective.
    ///
    /// Panics if the mesh has no level `level`.
    pub fn fine_element_id(&self, level: usize, cell: usize) -> u64 {
        let shift = self.dim * level;
        let coarse = self.coarse_cell(level, cell);
        (self.element_ids[coarse] << shift) | (cell as u64 & ((1 << shift) - 1))
    }

    /// The finest level as a [`CanonicalMesh`] on process 0, the same for any number of
    /// processes (see the module); `None` on the other processes. Collective.
    ///
    /// Errors (on every process): [`MeshError::Gather`] if PETSc fails on a process while it
    /// collects its part (the message of the lowest such process); [`MeshError::Petsc`] if a
    /// collective call fails.
    pub fn canonical(&self) -> Result<Option<CanonicalMesh>, MeshError> {
        let (canonical, order) = self.build_canonical()?;
        // The order is the same as of an earlier call: keep the first.
        let _ = self.vertex_order.set(order);
        Ok(canonical)
    }

    /// The values `values` at the local vertices of the finest level (one per vertex, in the
    /// order of the local vertices, depth 0), in the order of the vertices of the
    /// [`Mesh::canonical`] mesh on process 0; `None` on the other processes. Each vertex has
    /// the value of the process which owns it. Collective. The first gather (without an earlier
    /// [`Mesh::canonical`]) makes the canonical mesh to learn its order.
    ///
    /// Errors (on every process): [`MeshError::Gather`] if a process has not one value per
    /// local vertex, or as [`Mesh::canonical`].
    pub fn gather_vertex_values(&self, values: &[f64]) -> Result<Option<Vec<f64>>, MeshError> {
        let finest = self.finest();
        let comm = finest.comm();
        let vertices = finest.depth_stratum(0)?.len();
        let fits = if values.len() == vertices {
            Ok(())
        } else {
            Err(format!(
                "{} values for the {vertices} local vertices of the finest level on process {}",
                values.len(),
                comm.rank()
            ))
        };
        agree(&comm, fits)?;
        let order = self.vertex_order()?;
        let bytes: Vec<u8> = order
            .owned
            .iter()
            .flat_map(|&v| values[v].to_le_bytes())
            .collect();
        let Some(gathered) = comm.gather_bytes(0, &bytes)? else {
            return Ok(None);
        };
        let (n, slots) = order.slots.as_ref().expect("process 0 has the slots");
        let mut canonical = vec![f64::NAN; *n];
        for (bytes, slots) in gathered.iter().zip(slots) {
            for (value, &slot) in bytes.as_chunks::<8>().0.iter().zip(slots) {
                canonical[slot as usize] = f64::from_le_bytes(*value);
            }
        }
        Ok(Some(canonical))
    }

    /// The rows `values` of the elements `ids` (`width` values per element, row by row) of every
    /// process on process 0, sorted by element id: the ids and the rows; `None` on the other
    /// processes. An element held by several processes appears once per process. Collective.
    ///
    /// Errors (on every process): [`MeshError::Gather`] if a process has not `width` values per
    /// id, or `width` is 0.
    pub fn gather_element_values(
        &self,
        ids: &[u64],
        values: &[f64],
        width: usize,
    ) -> Result<Option<ElementRows>, MeshError> {
        let comm = self.finest().comm();
        let fits = if width > 0 && values.len() == ids.len() * width {
            Ok(())
        } else {
            Err(format!(
                "{} values for {} elements of width {width} on process {}",
                values.len(),
                ids.len(),
                comm.rank()
            ))
        };
        agree(&comm, fits)?;
        let mut bytes = Vec::with_capacity(8 * (ids.len() + values.len()));
        for (id, row) in ids.iter().zip(values.chunks_exact(width)) {
            bytes.extend(id.to_le_bytes());
            row.iter().for_each(|v| bytes.extend(v.to_le_bytes()));
        }
        let Some(gathered) = comm.gather_bytes(0, &bytes)? else {
            return Ok(None);
        };
        let mut rows: Vec<&[[u8; 8]]> = gathered
            .iter()
            .flat_map(|b| b.as_chunks::<8>().0.chunks_exact(1 + width))
            .collect();
        rows.sort_by_key(|row| u64::from_le_bytes(row[0]));
        let ids = rows.iter().map(|row| u64::from_le_bytes(row[0])).collect();
        let values = rows
            .iter()
            .flat_map(|row| row[1..].iter().map(|v| f64::from_le_bytes(*v)))
            .collect();
        Ok(Some((ids, values)))
    }

    /// The order of [`Mesh::gather_vertex_values`]: made with the canonical mesh on the first
    /// call. Collective on the first call (every process makes it in the same call).
    fn vertex_order(&self) -> Result<&VertexOrder, MeshError> {
        if let Some(order) = self.vertex_order.get() {
            return Ok(order);
        }
        let (_, order) = self.build_canonical()?;
        Ok(self.vertex_order.get_or_init(|| order))
    }

    /// The canonical mesh on process 0 and the order of the vertices of this process.
    /// Collective.
    fn build_canonical(&self) -> Result<(Option<CanonicalMesh>, VertexOrder), MeshError> {
        let finest = self.finest();
        let comm = finest.comm();
        // Collective on its first call: before anything one process may fail alone.
        let numbering = finest.vertex_numbering()?;
        let part = self.part(&numbering);
        agree(
            &comm,
            part.as_ref().map(|_| ()).map_err(ToString::to_string),
        )?;
        let (part, owned) = part?;
        let Some(gathered) = comm.gather_bytes(0, &part.to_bytes())? else {
            return Ok((None, VertexOrder { owned, slots: None }));
        };
        let parts: Vec<Part> = gathered
            .iter()
            .map(|b| Part::from_bytes(b, self.tags.len()))
            .collect();
        let (canonical, slots) = self.assemble(&parts);
        let n = canonical.vertices.len();
        Ok((
            Some(canonical),
            VertexOrder {
                owned,
                slots: Some((n, slots)),
            },
        ))
    }

    /// This process's [`Part`], and the indices of the local vertices it owns (in the order of
    /// the part's vertices), with the vertex numbering `numbering` of the finest level. Not
    /// collective; the caller agrees on errors.
    fn part(&self, numbering: &[petsc::sys::PetscInt]) -> Result<(Part, Vec<usize>), MeshError> {
        let level = self.levels.len() - 1;
        let dm = self.finest();
        let cells = dm.height_stratum(0)?;
        let edges = dm.depth_stratum(1)?;
        let vertices = dm.depth_stratum(0)?;
        let global = |v: Point| -> u64 {
            let g = numbering[(v - vertices.start) as usize];
            (if g >= 0 { g } else { -(g + 1) }) as u64
        };
        let coordinates = vertex_coordinates(dm, &vertices)?;
        let mut part = Part {
            cells: Vec::with_capacity(cells.len()),
            vertices: Vec::new(),
            tags: Vec::with_capacity(self.tags.len()),
        };
        let fine_id = |cell: Point| self.fine_element_id(level, (cell - cells.start) as usize);
        for cell in cells.clone() {
            let corners: Vec<u64> = dm
                .closure(cell)?
                .into_iter()
                .filter(|p| vertices.contains(p))
                .map(global)
                .collect();
            let corners: [u64; 3] = corners.try_into().map_err(|c: Vec<u64>| petsc::Error {
                code: petsc::sys::PETSC_ERR_SUP,
                message: format!(
                    "a cell of {} vertices; the canonical mesh is of triangles",
                    c.len()
                ),
            })?;
            part.cells.push((fine_id(cell), corners));
        }
        let mut owned = Vec::new();
        for (i, v) in vertices.clone().enumerate() {
            if numbering[i] >= 0 {
                owned.push(i);
                part.vertices.push((global(v), coordinates[i]));
            }
        }
        for tag in &self.tags {
            let mut points = Vec::new();
            for value in dm.label_values(&tag.name)? {
                points.extend(dm.stratum_points(&tag.name, value)?);
            }
            let mut marked = Vec::new();
            for p in points {
                match tag.dim {
                    2 if cells.contains(&p) => marked.push(fine_id(p)),
                    1 if edges.contains(&p) => {
                        marked.extend(dm.cone(p)?.into_iter().map(global));
                    }
                    _ => {}
                }
            }
            part.tags.push(marked);
        }
        Ok((part, owned))
    }

    /// The canonical mesh of the parts of the processes (in the order of the ranks), and for
    /// each process the canonical index of each vertex of its part.
    fn assemble(&self, parts: &[Part]) -> (CanonicalMesh, Vec<Vec<u32>>) {
        let level = self.levels.len() - 1;
        let n_vertices: usize = parts.iter().map(|p| p.vertices.len()).sum();
        let mut coordinates = vec![[f64::NAN; 2]; n_vertices];
        for (g, x) in parts.iter().flat_map(|p| &p.vertices) {
            coordinates[*g as usize] = *x;
        }
        let mut cells: Vec<(u64, [u64; 3])> =
            parts.iter().flat_map(|p| p.cells.iter().copied()).collect();
        cells.sort_unstable_by_key(|c| c.0);
        assert!(
            cells.windows(2).all(|w| w[0].0 < w[1].0),
            "the fine element ids of the cells are unique"
        );
        const NONE: u32 = u32::MAX;
        let mut canonical_of = vec![NONE; n_vertices];
        let mut vertices = Vec::with_capacity(n_vertices);
        let mut triangles = Vec::with_capacity(cells.len());
        for (_, corners) in &cells {
            let x = corners.map(|g| coordinates[g as usize]);
            let first = (0..3)
                .min_by(|&a, &b| {
                    x[a][0]
                        .total_cmp(&x[b][0])
                        .then(x[a][1].total_cmp(&x[b][1]))
                })
                .expect("three corners");
            let mut order = [first, (first + 1) % 3, (first + 2) % 3];
            let [a, b, c] = order.map(|i| x[i]);
            if (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) < 0.0 {
                order.swap(1, 2);
            }
            triangles.push(order.map(|i| {
                let g = corners[i] as usize;
                if canonical_of[g] == NONE {
                    canonical_of[g] = vertices.len() as u32;
                    vertices.push(coordinates[g]);
                }
                canonical_of[g]
            }));
        }
        assert_eq!(vertices.len(), n_vertices, "every vertex is in a cell");
        let ids: Vec<u64> = cells.iter().map(|c| c.0).collect();
        let element = ids.iter().map(|id| id >> (self.dim * level)).collect();
        let tags = self
            .tags
            .iter()
            .enumerate()
            .map(|(t, tag)| {
                let marked = || parts.iter().flat_map(|p| p.tags[t].iter().copied());
                let mut triangles = Vec::new();
                let mut edges = Vec::new();
                if tag.dim == 2 {
                    triangles = marked()
                        .map(|id| ids.binary_search(&id).expect("a marked cell") as u32)
                        .collect();
                    triangles.sort_unstable();
                    triangles.dedup();
                } else if tag.dim == 1 {
                    for p in parts {
                        for pair in p.tags[t].as_chunks::<2>().0 {
                            let [a, b] = pair.map(|g| canonical_of[g as usize]);
                            edges.push([a.min(b), a.max(b)]);
                        }
                    }
                    edges.sort_unstable();
                    edges.dedup();
                }
                CanonicalTag {
                    name: tag.name.clone(),
                    dim: tag.dim,
                    triangles,
                    edges,
                }
            })
            .collect();
        let slots = parts
            .iter()
            .map(|p| {
                p.vertices
                    .iter()
                    .map(|(g, _)| canonical_of[*g as usize])
                    .collect()
            })
            .collect();
        (
            CanonicalMesh {
                vertices,
                triangles,
                element,
                tags,
            },
            slots,
        )
    }
}

/// The coordinates of the local vertices `vertices` of `dm` (the first two), from its local
/// coordinates (set up).
fn vertex_coordinates(dm: &Dm, vertices: &std::ops::Range<Point>) -> petsc::Result<Vec<[f64; 2]>> {
    let coordinate_dm = dm.coordinate_dm()?;
    let coordinates = dm.local_coordinates()?;
    vertices
        .clone()
        .map(|v| {
            let i = coordinate_dm.local_dofs(v)?.start;
            Ok([coordinates[i], coordinates[i + 1]])
        })
        .collect()
}

/// An error ([`MeshError::Gather`]) on every process if `local` is one on any: the message of
/// the lowest process with one. Collective.
pub(crate) fn agree(comm: &Comm, local: Result<(), String>) -> Result<(), MeshError> {
    let size = comm.size();
    // The lowest failing process has the largest `size - rank`.
    let failing = comm.max_int(if local.is_err() {
        (size - comm.rank()) as petsc::sys::PetscInt
    } else {
        0
    })?;
    if failing == 0 {
        return Ok(());
    }
    let root = size - failing as usize;
    let mut message = local.err().unwrap_or_default().into_bytes();
    comm.broadcast_bytes(root, &mut message)?;
    Err(MeshError::Gather(
        String::from_utf8_lossy(&message).into_owned(),
    ))
}
