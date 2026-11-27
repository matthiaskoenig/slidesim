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
    /// The domain of a `polygons/1` file: the area tags in the order of their first polygon, then
    /// the edge tags in the order of their first chain; a tag is a name with a value.
    ///
    /// Errors if a tag value does not fit a DMPlex label value (`i32`).
    pub fn from_polygons(p: &Polygons) -> Result<Domain, MesherError> {
        let mut tags: Vec<MeshTag> = Vec::new();
        let mut index = |name: &str, value: u32, dim: u8| -> Result<usize, MesherError> {
            let value = i32::try_from(value).map_err(|_| {
                MesherError::Triangulation(format!(
                    "the value {value} of tag {name:?} is larger than {}",
                    i32::MAX
                ))
            })?;
            let tag = MeshTag {
                name: name.to_string(),
                value,
                dim,
            };
            Ok(match tags.iter().position(|t| *t == tag) {
                Some(i) => i,
                None => {
                    tags.push(tag);
                    tags.len() - 1
                }
            })
        };
        let areas = p
            .areas
            .iter()
            .map(|a| Ok((index(&a.tag, a.value, 2)?, a.rings.clone())))
            .collect::<Result<_, MesherError>>()?;
        let edges = p
            .edges
            .iter()
            .map(|e| Ok((index(&e.tag, e.value, 1)?, e.points.clone())))
            .collect::<Result<_, MesherError>>()?;
        Ok(Domain { tags, areas, edges })
    }
}
