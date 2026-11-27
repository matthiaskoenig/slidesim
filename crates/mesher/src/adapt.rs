//! The adaptation of a tagged mesh to the size field: Mmg through PETSc's `DMAdaptMetric`.
//!
//! The mesh becomes a DMPlex on `PETSC_COMM_SELF` with two labels: one on the edges whose value
//! stands for the set of tags of the edge (or for an untagged interface between two area tags),
//! and one on the cells whose value is the region of the cell: the triangles connected across
//! untagged edges of one area tag. Mmg (with `no_surf`) keeps every edge on the boundary and
//! between cells of different values as it is, so a tagged chain between two parts of one area
//! survives too: the region ids differ on its two sides where the area tags do not. The labels
//! are split back into the tags afterwards.

use std::collections::HashMap;

use petsc::{MetricOptions, Petsc, Point};

use crate::tagged::{Plex, plex};
use crate::{MesherError, Size, TaggedMesh};

/// The label of the edges: the index of the set of tags of the edge, plus 1.
const EDGES: &str = "mesher_edges";
/// The label of the cells: the region of the cell, plus 1.
const REGIONS: &str = "mesher_regions";

/// The representative of the set of `i` (path halving).
fn find(parent: &mut [usize], mut i: usize) -> usize {
    while parent[i] != i {
        parent[i] = parent[parent[i]];
        i = parent[i];
    }
    i
}

/// `mesh` remeshed by Mmg (`DMAdaptMetric` on `PETSC_COMM_SELF`) to the isotropic metric
/// `1 / h^2` of `size` at its vertices, with PETSc's parameters of [`MetricOptions::default`]:
/// the boundary and the interfaces (tagged edges, and edges between area tags) keep their
/// vertices; every triangle keeps its area tag, every edge on a tagged edge its tags.
///
/// Errors: [`MesherError::UntaggedBoundary`] for a boundary edge without a tag,
/// [`MesherError::EdgeOffBoundary`] for a tagged edge inside a region (Mmg would drop it),
/// [`MesherError::InvalidMesh`] for an invalid mesh (see [`TaggedMesh::validate`]) or an edge of more
/// than two triangles, [`MesherError::Petsc`] if Mmg fails.
pub fn adapt(petsc: &Petsc, mesh: &TaggedMesh, size: &Size) -> Result<TaggedMesh, MesherError> {
    let bad = |m: String| MesherError::InvalidMesh(m);
    mesh.validate()?;
    if mesh.triangles.is_empty() {
        return Err(MesherError::InvalidMesh("no triangles".to_string()));
    }
    let midpoint = |e: [u32; 2]| {
        let (a, b) = (mesh.vertices[e[0] as usize], mesh.vertices[e[1] as usize]);
        [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5]
    };

    // Every edge with its triangles, and the tagged edges with their tags, both sorted by edge
    // (every tagged edge is an edge of a triangle: validated).
    let mut sides: Vec<([u32; 2], usize)> = Vec::with_capacity(3 * mesh.triangles.len());
    for (i, t) in mesh.triangles.iter().enumerate() {
        for k in 0..3 {
            let (a, b) = (t[k], t[(k + 1) % 3]);
            sides.push(([a.min(b), a.max(b)], i));
        }
    }
    sides.sort_unstable();
    let mut tagged: Vec<([u32; 2], u16)> = mesh
        .edges
        .iter()
        .copied()
        .zip(mesh.edge_tags.iter().copied())
        .collect();
    tagged.sort_unstable();
    tagged.dedup();
    // (edge, triangles, tags) per edge of a triangle.
    let mut edges: Vec<([u32; 2], Vec<usize>, Vec<u16>)> = Vec::new();
    let mut k = 0;
    for chunk in sides.chunk_by(|x, y| x.0 == y.0) {
        let e = chunk[0].0;
        let mut tags = Vec::new();
        while k < tagged.len() && tagged[k].0 == e {
            tags.push(tagged[k].1);
            k += 1;
        }
        if chunk.len() > 2 {
            return Err(bad(format!("the edge {e:?} has {} triangles", chunk.len())));
        }
        edges.push((e, chunk.iter().map(|s| s.1).collect(), tags));
    }
    // Regions: triangles connected across untagged edges of one area tag, numbered in the order
    // of their first triangle.
    let mut parent: Vec<usize> = (0..mesh.triangles.len()).collect();
    for (_, triangles, tags) in &edges {
        if let &[s, t] = triangles.as_slice()
            && tags.is_empty()
            && mesh.triangle_tags[s] == mesh.triangle_tags[t]
        {
            let (s, t) = (find(&mut parent, s), find(&mut parent, t));
            parent[s.max(t)] = s.min(t);
        }
    }
    let mut region_of_root: HashMap<usize, usize> = HashMap::new();
    let mut region_tags: Vec<u16> = Vec::new();
    let regions: Vec<usize> = (0..mesh.triangles.len())
        .map(|t| {
            let root = find(&mut parent, t);
            *region_of_root.entry(root).or_insert_with(|| {
                region_tags.push(mesh.triangle_tags[t]);
                region_tags.len() - 1
            })
        })
        .collect();

    // The sets of tags of the edges given to Mmg: tagged edges, and untagged interfaces between
    // area tags (the empty set).
    let mut sets: Vec<Vec<u16>> = Vec::new();
    let mut set_of: HashMap<Vec<u16>, usize> = HashMap::new();
    let mut marked: Vec<([u32; 2], usize)> = Vec::new();
    for (e, triangles, tags) in &edges {
        let set = match *triangles.as_slice() {
            [_] if tags.is_empty() => {
                return Err(MesherError::UntaggedBoundary { at: midpoint(*e) });
            }
            [s, t] if !tags.is_empty() && regions[s] == regions[t] => {
                return Err(MesherError::EdgeOffBoundary {
                    tag: mesh.tags[tags[0] as usize].name.clone(),
                    at: midpoint(*e),
                });
            }
            [s, t] if tags.is_empty() && mesh.triangle_tags[s] == mesh.triangle_tags[t] => {
                continue;
            }
            _ => tags,
        };
        let i = *set_of.entry(set.clone()).or_insert_with(|| {
            sets.push(set.clone());
            sets.len() - 1
        });
        marked.push((*e, i));
    }

    let comm = petsc.comm_self();
    let (mut dm, edge_points) = plex(&comm, mesh)?;
    let cells = dm.height_stratum(0)?;
    for (i, &r) in regions.iter().enumerate() {
        dm.set_label_value(REGIONS, cells.start + i as Point, (r + 1) as Point)?;
    }
    dm.create_label(EDGES)?;
    for (e, i) in marked {
        let point = *edge_points
            .get(&e)
            .ok_or_else(|| bad(format!("the edge {e:?} is not in the DMPlex")))?;
        dm.set_label_value(EDGES, point, (i + 1) as Point)?;
    }
    let sizes: Vec<f64> = mesh.vertices.iter().map(|&x| size.at(x)).collect();
    let metric = dm.isotropic_metric(&sizes, &MetricOptions::default())?;
    let adapted = dm.adapt_metric(&metric, Some(EDGES), Some(REGIONS))?;

    let out = Plex::read(&adapted)?;
    let mut triangle_tags: Vec<Option<u16>> = vec![None; out.triangles.len()];
    for (r, &tag) in region_tags.iter().enumerate() {
        for p in adapted.stratum_points(REGIONS, (r + 1) as Point)? {
            if let Some(c) = out.cell(p) {
                triangle_tags[c] = Some(tag);
            }
        }
    }
    let triangle_tags = triangle_tags
        .iter()
        .enumerate()
        .map(|(triangle, t)| t.ok_or(MesherError::AreaTags { triangle, count: 0 }))
        .collect::<Result<_, _>>()?;
    let mut tagged: Vec<(u16, [u32; 2])> = Vec::new();
    for (i, set) in sets.iter().enumerate() {
        for p in adapted.stratum_points(EDGES, (i + 1) as Point)? {
            if let Some(e) = out.edge(p) {
                tagged.extend(set.iter().map(|&tag| (tag, e)));
            }
        }
    }
    tagged.sort_unstable();
    Ok(TaggedMesh {
        tags: mesh.tags.clone(),
        vertices: out.vertices,
        triangles: out.triangles,
        triangle_tags,
        edge_tags: tagged.iter().map(|e| e.0).collect(),
        edges: tagged.into_iter().map(|e| e.1).collect(),
    })
}
