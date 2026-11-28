//! Tagged polygons to a graded, adapted triangle mesh with the tags.
//!
//! [`triangulate`] builds a constrained Delaunay triangulation (`spade`) of the rings of the areas
//! and the edge chains of a [`Domain`], with the boundary already split to the [`Size`], and
//! keeps the triangles inside the areas; [`adapt`] remeshes it with Mmg through PETSc's
//! `DMAdaptMetric` to the isotropic metric `1 / h^2` of the size field, the boundary and the
//! interfaces frozen; [`generate`] does both, improves the elements which Mmg leaves near the
//! quality bounds, and measures the [`Quality`]. The result is a
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
mod improve;
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
    /// Two holes overlap or touch: the walls of tags `a` and `b`, the holes centred at `a_at`
    /// and `b_at`.
    #[error(
        "the holes of {a:?} at ({}, {}) um and of {b:?} at ({}, {}) um overlap",
        a_at[0], a_at[1], b_at[0], b_at[1]
    )]
    HolesOverlap {
        a: String,
        a_at: [f64; 2],
        b: String,
        b_at: [f64; 2],
    },
    /// A hole (the wall of tag `hole`, centred at `at`) reaches the outer border (tag `border`).
    #[error(
        "the hole of {hole:?} at ({}, {}) um reaches the outer border {border:?}",
        at[0], at[1]
    )]
    HoleReachesBorder {
        hole: String,
        at: [f64; 2],
        border: String,
    },
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

/// [`triangulate`], [`adapt`] (on `PETSC_COMM_SELF`), a local improvement of the elements which
/// Mmg leaves near the quality bounds (smoothing, edge collapses and flips of free vertices and
/// edges, each applied only if it improves the worst of the elements it changes, so the worst
/// element never gets worse; the boundary, the interfaces and the tagged edges stay) and the
/// [`Quality`] of the result against `size`. Calls no collective on a larger communicator: run
/// it on one process.
///
/// Guaranteed for any input it accepts: every triangle has the area tag of the polygon it lies
/// in, the triangles of an area cover its polygons exactly (the boundary and the interfaces keep
/// their input vertices), and every tagged chain keeps its tags on every sub-edge and its length.
/// The quality (`min_angle_deg`, `min_sicn`, `edge_ratio` against `size`, which includes the caps
/// by the gap widths and the segment lengths) is measured and returned, not enforced. The tests
/// see a smallest angle of 20 degrees or more, a SICN of 0.4 or more and edge ratios within
/// [0.4, 2.0] when the input has
///
/// - a size field which the boundary splitting resolves (no jump over less than an element),
/// - gaps no narrower than `min / 1.6` (where the gap cap reaches its floor `min / 4`),
/// - input segments no shorter than `min / 4` (the cap by a segment's length, `1.5` times it,
///   stops at the floor `min / 4`), and
/// - no sharp corners of the boundary (the triangles at a corner of angle `t` have angles of at
///   most `t` there);
///
/// the lobules of any valid spec, polygonized circles and the pixel steps of masks with pixels of
/// `min / 4` or more are such inputs. These bounds rest on the local improvement lifting the
/// elements Mmg leaves below them to its margin (`GOOD` of `improve.rs`), which it does not
/// guarantee: the quality sweeps of the tests check it (`quality_sweep_heavy`, rings 2 to 4, in
/// a release step of `scripts/check.sh`).
/// Inputs with sharper corners, narrower gaps, shorter segments or steeper size fields may give
/// less, which the caller reads from the [`Quality`].
pub fn generate(
    petsc: &Petsc,
    domain: &Domain,
    size: &Size,
) -> Result<(TaggedMesh, Quality), MesherError> {
    let input = triangulate(domain, size)?;
    let mut mesh = adapt(petsc, &input, size)?;
    improve::improve(&mut mesh, size);
    let quality = mesh.quality(Some(size))?;
    Ok((mesh, quality))
}
