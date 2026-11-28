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

/// The tags as bytes: per tag the length of the name (u32), the name, the value (i32) and the
/// dimension (u8), little endian.
fn encode(tags: &[MeshTag]) -> Vec<u8> {
    let mut out = Vec::new();
    for t in tags {
        out.extend((t.name.len() as u32).to_le_bytes());
        out.extend(t.name.as_bytes());
        out.extend(t.value.to_le_bytes());
        out.push(t.dim);
    }
    out
}

/// The tags of [`encode`].
fn decode(bytes: &[u8]) -> Option<Vec<MeshTag>> {
    let mut at = 0;
    let mut take = |n: usize| -> Option<Vec<u8>> {
        let part = bytes.get(at..at + n)?.to_vec();
        at += n;
        Some(part)
    };
    let mut tags = Vec::new();
    while let Some(length) = take(4) {
        let length = u32::from_le_bytes(length.try_into().ok()?) as usize;
        let name = String::from_utf8(take(length)?).ok()?;
        let value = i32::from_le_bytes(take(4)?.try_into().ok()?);
        let dim = take(1)?[0];
        tags.push(MeshTag { name, value, dim });
    }
    Some(tags)
}

impl TaggedMesh {
    /// Whether the arrays fit together: tags of valid names (not reserved) and dimensions 1 or
    /// 2, at most 65536; finite vertices; one area tag per triangle and one edge tag per edge;
    /// triangles of three distinct vertices; edges `[a, b]` with `a < b`, each an edge of a
    /// triangle. Errors: [`MesherError::InvalidTag`] or [`MesherError::InvalidMesh`].
    pub fn validate(&self) -> Result<(), MesherError> {
        let bad = |m: String| Err(MesherError::InvalidMesh(m));
        if self.tags.len() > usize::from(u16::MAX) + 1 {
            return bad(format!("{} tags, at most 65536", self.tags.len()));
        }
        for t in &self.tags {
            contracts::check_tag_name(&t.name).map_err(MesherError::InvalidTag)?;
            if t.dim != 1 && t.dim != 2 {
                return bad(format!("the tag {:?} has the dimension {}", t.name, t.dim));
            }
        }
        if let Some(v) = self
            .vertices
            .iter()
            .find(|v| !(v[0].is_finite() && v[1].is_finite()))
        {
            return bad(format!("the vertex {v:?} is not finite"));
        }
        if self.triangle_tags.len() != self.triangles.len() {
            return bad(format!(
                "{} triangle tags for {} triangles",
                self.triangle_tags.len(),
                self.triangles.len()
            ));
        }
        if self.edge_tags.len() != self.edges.len() {
            return bad(format!(
                "{} edge tags for {} edges",
                self.edge_tags.len(),
                self.edges.len()
            ));
        }
        let n = self.vertices.len() as u64;
        let dim = |t: u16| self.tags.get(t as usize).map(|t| t.dim);
        let mut sides = Vec::with_capacity(3 * self.triangles.len());
        for (i, (t, &tag)) in self.triangles.iter().zip(&self.triangle_tags).enumerate() {
            if t.iter().any(|&v| u64::from(v) >= n) {
                return bad(format!(
                    "triangle {i} {t:?} names a vertex which is not there"
                ));
            }
            if t[0] == t[1] || t[1] == t[2] || t[2] == t[0] {
                return bad(format!("triangle {i} {t:?} repeats a vertex"));
            }
            if dim(tag) != Some(2) {
                return bad(format!(
                    "triangle {i} has the tag {tag}, which is no area tag"
                ));
            }
            for k in 0..3 {
                let (a, b) = (t[k], t[(k + 1) % 3]);
                sides.push([a.min(b), a.max(b)]);
            }
        }
        sides.sort_unstable();
        for (e, &tag) in self.edges.iter().zip(&self.edge_tags) {
            if e[0] >= e[1] || u64::from(e[1]) >= n {
                return bad(format!(
                    "the edge {e:?} names a vertex which is not there, or is not ordered"
                ));
            }
            if dim(tag) != Some(1) {
                return bad(format!(
                    "the edge {e:?} has the tag {tag}, which is no edge tag"
                ));
            }
            if sides.binary_search(e).is_err() {
                return bad(format!("the tagged edge {e:?} is no edge of a triangle"));
            }
        }
        Ok(())
    }

    /// The mesh as a DMPlex on `comm`, held by process 0 (the meshes of the other processes are
    /// ignored, and may be empty): the cells and vertices in the order of the arrays, one label
    /// per tag on every process, named after it, with its value on the triangles or edges of the
    /// tag. Process 0 validates its mesh ([`TaggedMesh::validate`]) and sends the outcome and
    /// its tags to the others, so every process gets the same labels, or the same error.
    /// Collective.
    pub fn to_dm(&self, comm: &Comm) -> petsc::Result<Dm> {
        let root = comm.rank() == 0;
        // The first byte 0 and the tags, or 1 and the error.
        let mut message = Vec::new();
        if root {
            match self.validate() {
                Ok(()) => {
                    message.push(0);
                    message.extend(encode(&self.tags));
                }
                Err(e) => {
                    message.push(1);
                    message.extend(e.to_string().into_bytes());
                }
            }
        }
        comm.broadcast_bytes(0, &mut message)?;
        let error = |m: String| petsc::Error {
            code: petsc::sys::PETSC_ERR_ARG_WRONG,
            message: m,
        };
        let tags = match message.split_first() {
            Some((0, tags)) => decode(tags).ok_or_else(|| error("undecodable tags".to_string()))?,
            Some((_, m)) => return Err(error(String::from_utf8_lossy(m).into_owned())),
            None => return Err(error("no message from process 0".to_string())),
        };
        let (mut dm, edge_points) = plex(comm, self)?;
        for tag in &tags {
            dm.create_label(&tag.name)?;
        }
        if root {
            let cells = dm.height_stratum(0)?;
            for (i, &t) in self.triangle_tags.iter().enumerate() {
                let tag = &tags[t as usize];
                dm.set_label_value(&tag.name, cells.start + i as Point, label_value(tag.value))?;
            }
            for (e, &t) in self.edges.iter().zip(&self.edge_tags) {
                let tag = &tags[t as usize];
                // Validated: every tagged edge is an edge of a triangle.
                dm.set_label_value(&tag.name, edge_points[e], label_value(tag.value))?;
            }
        }
        Ok(dm)
    }

    /// The mesh of the local points of a serial DMPlex of triangles: one tag per label (but
    /// PETSc's own, [`contracts::RESERVED_TAGS`]), value and dimension (2 on cells, 1 on edges,
    /// both if the value marks both), sorted by name, value and dimension.
    ///
    /// Errors: [`MesherError::AreaTags`] for a triangle without exactly one area tag.
    pub fn from_dm(dm: &Dm) -> Result<TaggedMesh, MesherError> {
        let plex = Plex::read(dm)?;
        // Per tag: the triangles, or the edges.
        let mut found: Vec<(MeshTag, Vec<usize>, Vec<[u32; 2]>)> = Vec::new();
        for name in dm.label_names()? {
            if contracts::RESERVED_TAGS.contains(&name.as_str()) {
                continue;
            }
            for value in dm.label_values(&name)? {
                let points = dm.stratum_points(&name, value)?;
                let int = tag_value(value).ok_or_else(|| {
                    MesherError::InvalidTag(format!(
                        "the value {value} of label {name:?} is larger than {}",
                        i32::MAX
                    ))
                })?;
                let tag = |dim: u8| MeshTag {
                    name: name.clone(),
                    value: int,
                    dim,
                };
                let cells: Vec<usize> = points.iter().filter_map(|&p| plex.cell(p)).collect();
                if !cells.is_empty() {
                    found.push((tag(2), cells, Vec::new()));
                }
                let ends: Vec<[u32; 2]> = points.iter().filter_map(|&p| plex.edge(p)).collect();
                if !ends.is_empty() {
                    found.push((tag(1), Vec::new(), ends));
                }
            }
        }
        if found.len() > usize::from(u16::MAX) + 1 {
            return Err(MesherError::InvalidMesh(format!(
                "{} tags, at most 65536",
                found.len()
            )));
        }
        found.sort_by(|a, b| (&a.0.name, a.0.value, a.0.dim).cmp(&(&b.0.name, b.0.value, b.0.dim)));
        let mut area_tags: Vec<Vec<u16>> = vec![Vec::new(); plex.triangles.len()];
        let mut edges: Vec<(u16, [u32; 2])> = Vec::new();
        for (i, (_, cells, ends)) in found.iter().enumerate() {
            for &c in cells {
                area_tags[c].push(i as u16);
            }
            edges.extend(ends.iter().map(|&e| (i as u16, e)));
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
            tags: found.into_iter().map(|f| f.0).collect(),
            vertices: plex.vertices,
            triangles: plex.triangles,
            triangle_tags,
            edge_tags: edges.iter().map(|e| e.0).collect(),
            edges: edges.into_iter().map(|e| e.1).collect(),
        })
    }
}
