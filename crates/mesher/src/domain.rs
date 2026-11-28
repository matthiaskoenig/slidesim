//! The domain to mesh: tagged areas (polygons with holes) and tagged edge chains.

use contracts::Polygons;
use serde::{Deserialize, Serialize};

use crate::MesherError;

/// A tag of a mesh: an area tag (`dim` 2, on triangles) or an edge tag (`dim` 1, on edges), with
/// the name and value of its DMPlex label.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MeshTag {
    pub name: String,
    pub value: i32,
    pub dim: u8,
}

/// Tagged polygons in um: `areas` holds the tag index (into `tags`) and the rings of each polygon
/// (the outer ring first, then the holes; closed or not, either orientation), `edges` the tag
/// index and the points of each edge chain (closed if its last point is its first).
///
/// Every segment of the boundary of the areas needs an edge tag; chains may also separate areas
/// or parts of one area. Rings and chains share their vertices where they meet (exactly equal
/// coordinates, or a vertex exactly on a segment) and do not cross.
#[derive(Clone, Debug, PartialEq)]
pub struct Domain {
    pub tags: Vec<MeshTag>,
    pub areas: Vec<(usize, Vec<Vec<[f64; 2]>>)>,
    pub edges: Vec<(usize, Vec<[f64; 2]>)>,
}

impl Domain {
    /// The domain of a `polygons/1` file. A tag is a name with a value; the tags are sorted by
    /// name, value and dimension (the order in which [`crate::TaggedMesh::from_dm`] reads them
    /// back).
    ///
    /// Errors ([`MesherError::InvalidTag`]) for a reserved or empty tag name, and for a value
    /// which does not fit a DMPlex label value (`i32`).
    pub fn from_polygons(p: &Polygons) -> Result<Domain, MesherError> {
        let tag = |name: &str, value: u32, dim: u8| -> Result<MeshTag, MesherError> {
            contracts::check_tag_name(name).map_err(MesherError::InvalidTag)?;
            let value = i32::try_from(value).map_err(|_| {
                MesherError::InvalidTag(format!(
                    "the value {value} of the tag {name:?} is larger than {}",
                    i32::MAX
                ))
            })?;
            Ok(MeshTag {
                name: name.to_string(),
                value,
                dim,
            })
        };
        let areas: Vec<(MeshTag, Vec<Vec<[f64; 2]>>)> = p
            .areas
            .iter()
            .map(|a| Ok((tag(&a.tag, a.value, 2)?, a.rings.clone())))
            .collect::<Result<_, MesherError>>()?;
        let edges: Vec<(MeshTag, Vec<[f64; 2]>)> = p
            .edges
            .iter()
            .map(|e| Ok((tag(&e.tag, e.value, 1)?, e.points.clone())))
            .collect::<Result<_, MesherError>>()?;
        let mut tags: Vec<MeshTag> = areas
            .iter()
            .map(|a| a.0.clone())
            .chain(edges.iter().map(|e| e.0.clone()))
            .collect();
        tags.sort_by(|a, b| (&a.name, a.value, a.dim).cmp(&(&b.name, b.value, b.dim)));
        tags.dedup();
        let index = |t: &MeshTag| {
            tags.binary_search_by(|u| (&u.name, u.value, u.dim).cmp(&(&t.name, t.value, t.dim)))
        };
        let areas = areas
            .into_iter()
            .map(|(t, rings)| (index(&t).unwrap(), rings))
            .collect();
        let edges = edges
            .into_iter()
            .map(|(t, points)| (index(&t).unwrap(), points))
            .collect();
        Ok(Domain { tags, areas, edges })
    }
}
