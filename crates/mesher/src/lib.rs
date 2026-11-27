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
//! Everything but the adaptation is plain Rust with exact predicates, and no step (of the mesher or
//! of the parts of `spade` and `rstar` it calls) iterates a hash map or set, so the same input
//! gives the same mesh, bit for bit, on one target. Tag names are data: the mesher knows none.
//!
//! [`Lobules`] builds the domain of parametric ideal lobules (a hexagonal lattice of veins, the
//! Voronoi cells around them, circular holes at their centres and corners) from a
//! [`contracts::LobuleSpec`], and the relative position of a point between the two kinds of holes.

mod adapt;
mod domain;
mod lobules;
mod quality;
mod size;
mod tagged;
mod triangulate;

pub use adapt::adapt;
pub use domain::{Domain, MeshTag};
pub use lobules::Lobules;
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
    #[error(
        "an untagged boundary segment at ({}, {}) um (a gap between areas? borders shared by \
         areas must match exactly)",
        at[0], at[1]
    )]
    UntaggedBoundary { at: [f64; 2] },
    /// A part of the plane inside two areas (or inside one polygon twice); `at` lies inside it.
    #[error(
        "areas overlap at ({}, {}) um (borders shared by areas must match exactly)",
        at[0], at[1]
    )]
    Overlap { at: [f64; 2] },
    /// An edge chain of tag `tag` which is not on the boundary of an area, nor between two
    /// parts of the domain (it ends inside an area, or lies outside the domain); `at` is the
    /// midpoint of the segment between two input vertices.
    #[error("the edge {tag:?} at ({}, {}) um is not on a boundary of the areas", at[0], at[1])]
    EdgeOffBoundary { tag: String, at: [f64; 2] },
    /// Two holes overlap or touch (the walls of tags `a` and `b`), or a hole reaches the outer
    /// border (`b` its tag).
    #[error("the holes of {a:?} and {b:?} overlap")]
    HolesOverlap { a: String, b: String },
    /// A [`LobuleSpec`](contracts::LobuleSpec) which gives no lobules: a radius which is not
    /// finite and positive, a jitter out of range, tags which are not distinct, ...
    #[error("lobules: {0}")]
    InvalidLobules(String),
    /// A ring or chain crosses itself or another one at `at`.
    #[error(
        "rings or edges cross at ({}, {}) um (borders shared by areas must match exactly)",
        at[0], at[1]
    )]
    SelfIntersection { at: [f64; 2] },
    /// A triangle of a DMPlex with `count` area tags, not one ([`TaggedMesh::from_dm`]).
    #[error("triangle {triangle} has {count} area tags, not one")]
    AreaTags { triangle: usize, count: usize },
    /// The size field cannot be used: a wall which is no edge tag of the domain, or a size which
    /// is not finite and positive.
    #[error("size field: {0}")]
    SizeField(String),
    /// A tag which cannot become a DMPlex label: a reserved name
    /// ([`contracts::RESERVED_TAGS`]), an empty one, or a value beyond `i32`.
    #[error("tag: {0}")]
    InvalidTag(String),
    /// A [`TaggedMesh`] whose arrays do not fit together (indices out of range, tags of the wrong
    /// dimension, a tagged edge which is no edge of a triangle, ...).
    #[error("invalid mesh: {0}")]
    InvalidMesh(String),
    /// The input cannot be triangulated (degenerate rings, too many tags, ...).
    #[error("triangulation: {0}")]
    Triangulation(String),
    #[error(transparent)]
    Petsc(#[from] petsc::Error),
}

/// [`triangulate`], [`adapt`] (on `PETSC_COMM_SELF`) and the [`Quality`] of the result against
/// `size`. Calls no collective on a larger communicator: run it on one process.
///
/// Guaranteed for any input it accepts: every triangle has the area tag of the polygon it lies
/// in, the triangles of an area cover its polygons exactly (the boundary and the interfaces keep
/// their input vertices), and every tagged chain keeps its tags on every sub-edge and its length.
/// The quality (`min_angle_deg`, `min_sicn`, `edge_ratio` against `size`, which includes the cap
/// by the gap widths) is measured and returned, not enforced: for a size field which the
/// boundary splitting resolves and gaps no narrower than `min / 1.6` (where the cap reaches its
/// floor) the tests see a smallest angle of 20 degrees or more, a SICN of 0.4 or more and edge
/// ratios within [0.4, 2.0]; inputs with sharper corners, narrower gaps or steeper size fields
/// may give less, which the caller reads from the [`Quality`].
pub fn generate(
    petsc: &Petsc,
    domain: &Domain,
    size: &Size,
) -> Result<(TaggedMesh, Quality), MesherError> {
    let input = triangulate(domain, size)?;
    let mesh = adapt(petsc, &input, size)?;
    let quality = mesh.quality(Some(size))?;
    Ok((mesh, quality))
}
