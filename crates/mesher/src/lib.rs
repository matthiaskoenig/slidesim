//! Tagged polygons to a graded, adapted triangle mesh with the tags.
//!
//! [`triangulate`] builds a constrained Delaunay triangulation (`spade`) of the rings of the areas
//! and the edge chains of a [`Domain`], with the boundary already split to the [`Size`], and
//! keeps the triangles inside the areas; [`adapt`] remeshes it with Mmg through PETSc's
//! `DMAdaptMetric` to the isotropic metric `1 / h^2` of the size field, the boundary and the
//! interfaces frozen; [`generate`] does both and measures the [`Quality`]. The result is a
//! [`TaggedMesh`]: vertices, triangles with one area tag each and the tagged edges, which
//! [`TaggedMesh::to_dm`] turns into a DMPlex with one label per tag.
//!
//! Everything but the adaptation is plain Rust with exact predicates, and no step iterates a hash
//! map, so the same input gives the same mesh, bit for bit, on one target. Tag names are data: the
//! mesher knows none.

mod adapt;
mod domain;
mod quality;
mod size;
mod tagged;
mod triangulate;

pub use adapt::adapt;
pub use domain::{Domain, MeshTag};
pub use quality::Quality;
pub use size::Size;
pub use tagged::TaggedMesh;
pub use triangulate::triangulate;

use petsc::Petsc;

/// Why a domain cannot be meshed; coordinates are in um.
#[derive(Debug, thiserror::Error)]
pub enum MesherError {
    /// A segment of the boundary of the domain (an area on one side only) without an edge tag;
    /// `at` is the midpoint of the segment between two input vertices.
    #[error("an untagged boundary segment at ({}, {}) um", at[0], at[1])]
    UntaggedBoundary { at: [f64; 2] },
    /// A part of the plane inside two areas (or inside one polygon twice); `at` lies inside it.
    #[error("areas overlap at ({}, {}) um", at[0], at[1])]
    Overlap { at: [f64; 2] },
    /// An edge chain of tag `tag` which is not on the boundary of an area, nor between two
    /// parts of the domain (it ends inside an area, or lies outside the domain); `at` is the
    /// midpoint of the segment between two input vertices.
    #[error("the edge {tag:?} at ({}, {}) um is not on a boundary of the areas", at[0], at[1])]
    EdgeOffBoundary { tag: String, at: [f64; 2] },
    /// Two holes overlap (the walls of tags `a` and `b`).
    #[error("the holes of {a:?} and {b:?} overlap")]
    HolesOverlap { a: String, b: String },
    /// A ring or chain crosses itself or another one at `at`.
    #[error("rings or edges cross at ({}, {}) um", at[0], at[1])]
    SelfIntersection { at: [f64; 2] },
    /// A triangle of a DMPlex with `count` area tags, not one ([`TaggedMesh::from_dm`]).
    #[error("triangle {triangle} has {count} area tags, not one")]
    AreaTags { triangle: usize, count: usize },
    /// The size field cannot be used: a wall which is no edge tag of the domain, or a size which
    /// is not finite and positive.
    #[error("size field: {0}")]
    SizeField(String),
    /// The input cannot be triangulated (degenerate rings, too many tags, ...).
    #[error("triangulation: {0}")]
    Triangulation(String),
    #[error(transparent)]
    Petsc(#[from] petsc::Error),
}

/// [`triangulate`], [`adapt`] (on `PETSC_COMM_SELF`) and the [`Quality`] of the result against
/// `size`. Calls no collective on a larger communicator: run it on one process.
pub fn generate(
    petsc: &Petsc,
    domain: &Domain,
    size: &Size,
) -> Result<(TaggedMesh, Quality), MesherError> {
    let input = triangulate(domain, size)?;
    let mesh = adapt(petsc, &input, size)?;
    let quality = mesh.quality(Some(size));
    Ok((mesh, quality))
}
