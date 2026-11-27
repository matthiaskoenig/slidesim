//! The tagged mesh, and its conversion from and to a DMPlex.

use std::collections::HashMap;
use std::ops::Range;

use petsc::{Comm, Dm, Point};
use serde::{Deserialize, Serialize};

use crate::{MeshTag, MesherError};

/// A triangle mesh with tags: `triangles` counterclockwise (rotated to start at their smallest
/// vertex), each with the index of its area tag (`triangle_tags`); `edges` `[a, b]` with
/// `a < b`, one entry per tagged edge and tag, with the index of the tag (`edge_tags`), sorted by
/// tag and vertices.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TaggedMesh {
    pub tags: Vec<MeshTag>,
    pub vertices: Vec<[f64; 2]>,
    pub triangles: Vec<[u32; 3]>,
    pub triangle_tags: Vec<u16>,
    pub edges: Vec<[u32; 2]>,
    pub edge_tags: Vec<u16>,
}

/// The labels PETSc creates itself (topology, gmsh's generic sets, ghosts, output), which are no
/// tags.
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

/// The name of the DM of a mesh.
const NAME: &str = "mesh";

/// `t` rotated to start at its smallest vertex, the orientation kept.
pub(crate) fn canonical_triangle(t: [u32; 3]) -> [u32; 3] {
    let k = (0..3).min_by_key(|&k| t[k]).unwrap();
    [t[k], t[(k + 1) % 3], t[(k + 2) % 3]]
}

/// Twice the signed area of the triangle `a`, `b`, `c`.
pub(crate) fn twice_area(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> f64 {
    (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
}

/// A tag value as a label value (`PetscInt` has 32 or 64 bits, depending on the PETSc build).
#[allow(clippy::useless_conversion)]
fn label_value(value: i32) -> Point {
    Point::from(value)
}

/// A label value as a tag value, if it fits.
#[allow(clippy::useless_conversion)]
fn tag_value(value: Point) -> Option<i32> {
    i32::try_from(value).ok()
}

/// A DMPlex of triangles read into arrays: the coordinates of the vertices, the triangles
/// (counterclockwise, canonical), the end points of the edges, and the point ranges.
pub(crate) struct Plex {
    pub vertices: Vec<[f64; 2]>,
    pub triangles: Vec<[u32; 3]>,
    pub cells: Range<Point>,
    pub edges: Range<Point>,
    pub edge_ends: Vec<[u32; 2]>,
}

impl Plex {
    /// The local points of `dm`, a 2D DMPlex of triangles with vertex coordinates.
    pub fn read(dm: &Dm) -> Result<Plex, MesherError> {
        let bad = |m: String| MesherError::Triangulation(m);
        if dm.dimension()? != 2 || dm.coordinate_dimension()? != 2 {
            return Err(bad("the DMPlex is not two dimensional".to_string()));
        }
        let (cells, edges, verts) = (
            dm.height_stratum(0)?,
            dm.depth_stratum(1)?,
            dm.depth_stratum(0)?,
        );
        let xy = dm.local_coordinates()?;
        if xy.len() != 2 * verts.len() {
            return Err(bad(format!(
                "{} coordinates for {} vertices",
                xy.len(),
                verts.len()
            )));
        }
        let vertices: Vec<[f64; 2]> = xy.chunks(2).map(|c| [c[0], c[1]]).collect();
        let vertex = |p: Point| (p - verts.start) as u32;
        let mut triangles = Vec::with_capacity(cells.len());
        for c in cells.clone() {
            let v: Vec<u32> = dm
                .closure(c)?
                .into_iter()
                .filter(|p| verts.contains(p))
                .map(vertex)
                .collect();
            let &[a, b, d] = v.as_slice() else {
                return Err(bad(format!("cell {c} has {} vertices, not 3", v.len())));
            };
            let t = if twice_area(
                vertices[a as usize],
                vertices[b as usize],
                vertices[d as usize],
            ) < 0.0
            {
                [a, d, b]
            } else {
                [a, b, d]
            };
            triangles.push(canonical_triangle(t));
        }
        let edge_ends = edges
            .clone()
            .map(|e| {
                let cone = dm.cone(e)?;
                let &[a, b] = cone.as_slice() else {
                    return Err(bad(format!("edge {e} has {} vertices, not 2", cone.len())));
                };
                let (a, b) = (vertex(a), vertex(b));
                Ok([a.min(b), a.max(b)])
            })
            .collect::<Result<_, MesherError>>()?;
        Ok(Plex {
            vertices,
            triangles,
            cells,
            edges,
            edge_ends,
        })
    }

    /// The index of the triangle of point `p`, if a cell.
    pub fn cell(&self, p: Point) -> Option<usize> {
        self.cells
            .contains(&p)
            .then(|| (p - self.cells.start) as usize)
    }

    /// The end points of point `p`, if an edge.
    pub fn edge(&self, p: Point) -> Option<[u32; 2]> {
        self.edges
            .contains(&p)
            .then(|| self.edge_ends[(p - self.edges.start) as usize])
    }
}

/// A DMPlex on `comm` of the triangles of `mesh` (on process 0; the other processes give none)
/// and the edge point of every vertex pair `[a, b]`, `a < b`.
pub(crate) fn plex(
    comm: &Comm,
    mesh: &TaggedMesh,
) -> petsc::Result<(Dm, HashMap<[u32; 2], Point>)> {
    let root = comm.rank() == 0;
    let (cells, coordinates): (Vec<Point>, Vec<f64>) = if root {
        (
            mesh.triangles
                .iter()
                .flat_map(|t| t.map(|v| v as Point))
                .collect(),
            mesh.vertices.iter().flatten().copied().collect(),
        )
    } else {
        (Vec::new(), Vec::new())
    };
    let dm = Dm::plex_from_cells(comm, NAME, 2, 3, &cells, 2, &coordinates)?;
    let verts = dm.depth_stratum(0)?;
    let mut edges = HashMap::new();
    for e in dm.depth_stratum(1)? {
        let cone = dm.cone(e)?;
        let (a, b) = (
            (cone[0] - verts.start) as u32,
            (cone[1] - verts.start) as u32,
        );
        edges.insert([a.min(b), a.max(b)], e);
    }
    Ok((dm, edges))
}

impl TaggedMesh {
    /// The mesh as a DMPlex on `comm`, held by process 0 (the meshes of the other processes are
    /// ignored): the cells and vertices in the order of the arrays, one label per tag, named
    /// after it, with its value on the triangles or edges of the tag. Collective.
    pub fn to_dm(&self, comm: &Comm) -> petsc::Result<Dm> {
        let (mut dm, edge_points) = plex(comm, self)?;
        for tag in &self.tags {
            dm.create_label(&tag.name)?;
        }
        if comm.rank() == 0 {
            let cells = dm.height_stratum(0)?;
            for (i, &t) in self.triangle_tags.iter().enumerate() {
                let tag = &self.tags[t as usize];
                dm.set_label_value(&tag.name, cells.start + i as Point, label_value(tag.value))?;
            }
            for (e, &t) in self.edges.iter().zip(&self.edge_tags) {
                let tag = &self.tags[t as usize];
                let point = *edge_points.get(e).ok_or_else(|| petsc::Error {
                    code: petsc::sys::PETSC_ERR_ARG_WRONG,
                    message: format!("the tagged edge {e:?} is no edge of a triangle"),
                })?;
                dm.set_label_value(&tag.name, point, label_value(tag.value))?;
            }
        }
        Ok(dm)
    }

    /// The mesh of the local points of a serial DMPlex of triangles: one tag per label (but
    /// PETSc's own) and value, of dimension 2 on cells and 1 on edges (both if the value marks
    /// both), in the order of the labels and their values.
    ///
    /// Errors: [`MesherError::AreaTags`] for a triangle without exactly one area tag.
    pub fn from_dm(dm: &Dm) -> Result<TaggedMesh, MesherError> {
        let plex = Plex::read(dm)?;
        let mut tags = Vec::new();
        let mut area_tags: Vec<Vec<u16>> = vec![Vec::new(); plex.triangles.len()];
        let mut edges: Vec<(u16, [u32; 2])> = Vec::new();
        for name in dm.label_names()? {
            if PETSC_LABELS.contains(&name.as_str()) {
                continue;
            }
            for value in dm.label_values(&name)? {
                let points = dm.stratum_points(&name, value)?;
                let int = tag_value(value).ok_or_else(|| {
                    MesherError::Triangulation(format!(
                        "the value {value} of label {name:?} is larger than {}",
                        i32::MAX
                    ))
                })?;
                let mut add = |dim: u8| -> Result<u16, MesherError> {
                    tags.push(MeshTag {
                        name: name.clone(),
                        value: int,
                        dim,
                    });
                    u16::try_from(tags.len() - 1)
                        .map_err(|_| MesherError::Triangulation("more than 65536 tags".to_string()))
                };
                let cells: Vec<usize> = points.iter().filter_map(|&p| plex.cell(p)).collect();
                if !cells.is_empty() {
                    let tag = add(2)?;
                    for c in cells {
                        area_tags[c].push(tag);
                    }
                }
                let ends: Vec<[u32; 2]> = points.iter().filter_map(|&p| plex.edge(p)).collect();
                if !ends.is_empty() {
                    let tag = add(1)?;
                    edges.extend(ends.into_iter().map(|e| (tag, e)));
                }
            }
        }
        let triangle_tags = area_tags
            .iter()
            .enumerate()
            .map(|(triangle, t)| match t.as_slice() {
                [tag] => Ok(*tag),
                _ => Err(MesherError::AreaTags {
                    triangle,
                    count: t.len(),
                }),
            })
            .collect::<Result<_, _>>()?;
        edges.sort_unstable();
        Ok(TaggedMesh {
            tags,
            vertices: plex.vertices,
            triangles: plex.triangles,
            triangle_tags,
            edge_tags: edges.iter().map(|e| e.0).collect(),
            edges: edges.into_iter().map(|e| e.1).collect(),
        })
    }
}
