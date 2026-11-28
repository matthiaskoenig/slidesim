//! The constrained Delaunay triangulation of a domain, with its boundary split to the size field.
//!
//! 1. The rings and chains become segments between their vertices, matched by exact
//!    coordinates; a segment with an input vertex exactly on it is split there, so that shared
//!    borders become the same pieces (each inserted once) whatever vertices each side has.
//! 2. Pieces which cross are an error ([`MesherError::SelfIntersection`]).
//! 3. Every piece is split to the size field (equal steps of `integral ds / h`): the boundary is
//!    final, the adaptation keeps it.
//! 4. `spade` triangulates the points with every sub-edge as a constraint. No outer face
//!    exclusion: spade's parity rule is wrong for a chain between two areas.
//! 5. The triangles are flood filled into regions across edges which are no constraints. The
//!    winding number of every region around every polygon follows from the outer face across the
//!    pieces (the rings are oriented with the area on their left): a region inside exactly one
//!    polygon belongs to its area, inside none it is outside, inside two it is an
//!    [`MesherError::Overlap`]. Classifying regions (not triangles by their centroids) is immune
//!    to the slivers between nearly collinear sub-edges and the convex hull.
//! 6. Every piece between an area and the outside needs an edge tag, and every tagged piece lies
//!    between an area and something else (the outside, or another region).
//! 7. The triangles of the areas are refined until they resolve the size field (longest edge at
//!    most twice the size at the centroid), so that Mmg, which interpolates the metric on this
//!    mesh, starts from a mesh fine enough to carry it.

use std::collections::{HashMap, VecDeque};

use rstar::primitives::{GeomWithData, Rectangle};
use rstar::{AABB, RTree};
use spade::handles::{FixedFaceHandle, FixedVertexHandle, InnerTag};
use spade::{ConstrainedDelaunayTriangulation, Point2, Triangulation};

use crate::tagged::canonical_triangle;
use crate::{Domain, MesherError, Size, TaggedMesh};

type P = [f64; 2];

/// The orientation of `c` against the line from `a` to `b`, exact in sign: positive if `a`, `b`,
/// `c` turn counterclockwise, zero if collinear.
fn orient(a: P, b: P, c: P) -> f64 {
    let coord = |p: P| robust::Coord { x: p[0], y: p[1] };
    robust::orient2d(coord(a), coord(b), coord(c))
}

fn midpoint(a: P, b: P) -> P {
    [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5]
}

fn bbox(a: P, b: P) -> AABB<P> {
    AABB::from_corners(a, b)
}

/// Twice the signed area of a ring (positive if counterclockwise).
fn twice_area(ring: &[P]) -> f64 {
    let n = ring.len();
    (0..n)
        .map(|i| {
            let (a, b) = (ring[i], ring[(i + 1) % n]);
            a[0] * b[1] - a[1] * b[0]
        })
        .sum()
}

/// What a piece lies on: the polygons whose rings run along it (with whether the ring runs from
/// `a` to `b`, so that the polygon is on its left) and the edge tags of the chains along it.
struct Piece {
    a: u32,
    b: u32,
    rings: Vec<(u32, bool)>,
    tags: Vec<u16>,
}

/// An input segment: its ends (in the direction of its ring, which has its polygon on the left),
/// and the polygon of its ring or the tag of its chain.
type Segment = (P, P, Option<u32>, Option<u16>);

/// The planar straight line graph of the input: distinct vertices and the pieces between them.
struct Graph {
    vertices: Vec<P>,
    pieces: Vec<Piece>,
    /// The tag of the first ring without area, if any: an error unless it crosses itself.
    flat_ring: Option<String>,
}

impl Graph {
    fn new(domain: &Domain) -> Result<Graph, MesherError> {
        let bad = |m: String| MesherError::Triangulation(m);
        if domain.areas.is_empty() {
            return Err(bad("no area".to_string()));
        }
        if domain.tags.len() > usize::from(u16::MAX) + 1 {
            return Err(bad(format!("{} tags, at most 65536", domain.tags.len())));
        }
        let tag_of = |i: usize, dim: u8| -> Result<(), MesherError> {
            match domain.tags.get(i) {
                Some(t) if t.dim == dim => Ok(()),
                Some(t) => Err(bad(format!("tag {:?} has the dimension {}", t.name, t.dim))),
                None => Err(bad(format!("no tag {i}"))),
            }
        };

        // Rings without repeated points, oriented with the polygon on their left; chains without
        // repeated points.
        let dedup = |points: &[P]| -> Vec<P> {
            let mut out: Vec<P> = Vec::with_capacity(points.len());
            for &p in points {
                if out.last() != Some(&p) {
                    out.push(p);
                }
            }
            out
        };
        let mut segments: Vec<Segment> = Vec::new();
        let mut flat_ring = None;
        for (polygon, (tag, rings)) in domain.areas.iter().enumerate() {
            tag_of(*tag, 2)?;
            let name = &domain.tags[*tag].name;
            if rings.is_empty() {
                return Err(bad(format!("a polygon of {name:?} has no ring")));
            }
            for (r, ring) in rings.iter().enumerate() {
                let mut ring = dedup(ring);
                while ring.len() > 1 && ring.first() == ring.last() {
                    ring.pop();
                }
                let area = twice_area(&ring);
                if ring.len() < 3 {
                    return Err(bad(format!("a ring of {name:?} with fewer than 3 points")));
                }
                if area == 0.0 && flat_ring.is_none() {
                    flat_ring = Some(name.clone());
                }
                // Outer rings counterclockwise, holes clockwise.
                if area != 0.0 && (area > 0.0) != (r == 0) {
                    ring.reverse();
                }
                let n = ring.len();
                for i in 0..n {
                    segments.push((ring[i], ring[(i + 1) % n], Some(polygon as u32), None));
                }
            }
        }
        for (tag, points) in &domain.edges {
            tag_of(*tag, 1)?;
            let points = dedup(points);
            if points.len() < 2 {
                return Err(bad(format!(
                    "a chain of {:?} without a segment",
                    domain.tags[*tag].name
                )));
            }
            for w in points.windows(2) {
                segments.push((w[0], w[1], None, Some(*tag as u16)));
            }
        }

        // Distinct vertices by exact coordinates (-0 is 0).
        let key = |p: P| ((p[0] + 0.0).to_bits(), (p[1] + 0.0).to_bits());
        let mut ids: HashMap<(u64, u64), u32> = HashMap::new();
        let mut vertices: Vec<P> = Vec::new();
        let mut id = |p: P| -> Result<u32, MesherError> {
            if !(p[0].is_finite() && p[1].is_finite()) {
                return Err(bad(format!("the point ({}, {}) is not finite", p[0], p[1])));
            }
            Ok(*ids.entry(key(p)).or_insert_with(|| {
                vertices.push([p[0] + 0.0, p[1] + 0.0]);
                (vertices.len() - 1) as u32
            }))
        };
        let segments = segments
            .into_iter()
            .map(|(a, b, ring, tag)| Ok((id(a)?, id(b)?, ring, tag)))
            .collect::<Result<Vec<_>, MesherError>>()?;

        // Split every segment at the vertices exactly on it, into pieces; shared pieces merged.
        let tree = RTree::bulk_load(
            vertices
                .iter()
                .enumerate()
                .map(|(i, &p)| GeomWithData::new(p, i as u32))
                .collect(),
        );
        let mut pieces: Vec<Piece> = Vec::new();
        let mut piece_of: HashMap<(u32, u32), usize> = HashMap::new();
        for (va, vb, ring, tag) in segments {
            let (a, b) = (vertices[va as usize], vertices[vb as usize]);
            let mut on: Vec<u32> = tree
                .locate_in_envelope_intersecting(bbox(a, b))
                .map(|v| v.data)
                .filter(|&v| v != va && v != vb && orient(a, b, vertices[v as usize]) == 0.0)
                .collect();
            // Collinear points are ordered along the segment by their coordinates.
            let forward = (a[0], a[1]) < (b[0], b[1]);
            on.sort_by(|&u, &v| {
                let (p, q) = (vertices[u as usize], vertices[v as usize]);
                let o = (p[0], p[1]).partial_cmp(&(q[0], q[1])).unwrap();
                if forward { o } else { o.reverse() }
            });
            let mut chain = Vec::with_capacity(on.len() + 2);
            chain.push(va);
            chain.extend(on);
            chain.push(vb);
            for w in chain.windows(2) {
                let (u, v) = (w[0], w[1]);
                let i = *piece_of.entry((u.min(v), u.max(v))).or_insert_with(|| {
                    pieces.push(Piece {
                        a: u,
                        b: v,
                        rings: Vec::new(),
                        tags: Vec::new(),
                    });
                    pieces.len() - 1
                });
                let piece = &mut pieces[i];
                if let Some(polygon) = ring {
                    piece.rings.push((polygon, piece.a == u));
                }
                if let Some(tag) = tag
                    && !piece.tags.contains(&tag)
                {
                    piece.tags.push(tag);
                }
            }
        }
        for piece in &mut pieces {
            piece.tags.sort_unstable();
        }
        Ok(Graph {
            vertices,
            pieces,
            flat_ring,
        })
    }

    fn ends(&self, piece: &Piece) -> (P, P) {
        (
            self.vertices[piece.a as usize],
            self.vertices[piece.b as usize],
        )
    }

    /// The first crossing of two pieces (in the order of the pieces), if any. After the splitting
    /// at vertices, pieces meet only at their ends or cross properly.
    fn crossing(&self) -> Option<P> {
        let boxes = RTree::bulk_load(
            self.pieces
                .iter()
                .enumerate()
                .map(|(i, piece)| {
                    let (a, b) = self.ends(piece);
                    GeomWithData::new(Rectangle::from_corners(a, b), i)
                })
                .collect(),
        );
        for (i, p) in self.pieces.iter().enumerate() {
            let (a, b) = self.ends(p);
            let mut others: Vec<usize> = boxes
                .locate_in_envelope_intersecting(bbox(a, b))
                .map(|g| g.data)
                .filter(|&j| j > i)
                .collect();
            others.sort_unstable();
            for j in others {
                let q = &self.pieces[j];
                if p.a == q.a || p.a == q.b || p.b == q.a || p.b == q.b {
                    continue;
                }
                let (c, d) = self.ends(q);
                let (o1, o2) = (orient(a, b, c), orient(a, b, d));
                let (o3, o4) = (orient(c, d, a), orient(c, d, b));
                if o1 * o2 < 0.0 && o3 * o4 < 0.0 {
                    let cross = |u: P, v: P| u[0] * v[1] - u[1] * v[0];
                    let sub = |u: P, v: P| [u[0] - v[0], u[1] - v[1]];
                    let t = cross(sub(c, a), sub(d, c)) / cross(sub(b, a), sub(d, c));
                    return Some([a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]);
                }
            }
        }
        None
    }
}

/// The parameters (0 and 1 included) which split the segment from `a` to `b` to the size field:
/// `n = round(integral ds / h)` sub-edges (at least one) of equal `integral ds / h`, the integral
/// by the trapezoidal rule.
fn split_params(a: P, b: P, size: &Size) -> Vec<f64> {
    let length = (b[0] - a[0]).hypot(b[1] - a[1]);
    let m = ((4.0 * length / size.min()).ceil() as usize).clamp(4, 4096);
    let at = |t: f64| [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])];
    let inverse: Vec<f64> = (0..=m)
        .map(|i| 1.0 / size.at(at(i as f64 / m as f64)))
        .collect();
    let mut g = vec![0.0; m + 1];
    for i in 0..m {
        g[i + 1] = g[i] + 0.5 * (inverse[i] + inverse[i + 1]) * length / m as f64;
    }
    let n = (g[m].round() as usize).max(1);
    let mut ts = Vec::with_capacity(n + 1);
    ts.push(0.0);
    let mut i = 0;
    for k in 1..n {
        let target = g[m] * k as f64 / n as f64;
        while g[i + 1] < target {
            i += 1;
        }
        let f = (target - g[i]) / (g[i + 1] - g[i]);
        ts.push((i as f64 + f) / m as f64);
    }
    ts.push(1.0);
    ts
}

/// The longest edge of a background triangle in units of the size at its centroid.
const BACKGROUND: f64 = 2.0;

/// Inserts the centroid of every triangle of the areas (`in_area` of its region) whose longest
/// edge is longer than [`BACKGROUND`] times the size at the centroid, and of the new triangles
/// in turn, until none is: Mmg interpolates the metric on this mesh, so the mesh has to resolve
/// the size field. Constraints are final, so only a triangle whose longest free edge is longer
/// than its constraints (where an insertion helps) is split. Every triangle an insertion creates
/// holds the new vertex, and lies in the region of the triangle the vertex was inserted into
/// (flips never cross a constraint), so `region` (by face index) stays valid. Centroids which
/// are not strictly inside their triangle (slivers) are not inserted.
///
/// Errors if the triangulation would get more than `limit` vertices: a safeguard against a
/// refinement which does not end, never reached by the inputs seen (the caller's limit is the
/// count of equilateral triangles of `BACKGROUND` times the smallest size over the areas).
fn refine_background(
    cdt: &mut ConstrainedDelaunayTriangulation<Point2<f64>>,
    region: &mut Vec<usize>,
    in_area: &dyn Fn(usize) -> bool,
    size: &Size,
    limit: usize,
) -> Result<(), MesherError> {
    let position =
        |v: spade::handles::VertexHandle<'_, Point2<f64>, (), spade::CdtEdge<()>, ()>| {
            let p = v.position();
            [p.x, p.y]
        };
    // The centroid of a face if the face is to be split, and a vertex of the face.
    let split = |cdt: &ConstrainedDelaunayTriangulation<Point2<f64>>,
                 face: FixedFaceHandle<InnerTag>| {
        let vertices = cdt.face(face).vertices();
        let p = vertices.map(position);
        let c = [
            (p[0][0] + p[1][0] + p[2][0]) / 3.0,
            (p[0][1] + p[1][1] + p[2][1]) / 3.0,
        ];
        // The longest free edge and the longest constraint (which no insertion shortens).
        let (mut free, mut fixed) = (0.0f64, 0.0f64);
        for e in cdt.face(face).adjacent_edges() {
            let [a, b] = e.vertices().map(position);
            let length = (b[0] - a[0]).hypot(b[1] - a[1]);
            if e.is_constraint_edge() {
                fixed = fixed.max(length);
            } else {
                free = free.max(length);
            }
        }
        let inside = (0..3).all(|i| orient(p[i], p[(i + 1) % 3], c) > 0.0);
        (inside && free > fixed && free > BACKGROUND * size.at(c)).then_some((c, vertices[0].fix()))
    };
    let mut queue: VecDeque<FixedFaceHandle<InnerTag>> = cdt
        .inner_faces()
        .map(|f| f.fix())
        .filter(|f| in_area(region[f.index()]))
        .collect();
    while let Some(face) = queue.pop_front() {
        if cdt.num_vertices() >= limit {
            return Err(MesherError::Triangulation(format!(
                "the background refinement exceeds its limit of {limit} vertices"
            )));
        }
        let Some((c, hint)) = split(cdt, face) else {
            continue;
        };
        let r = region[face.index()];
        let v = cdt
            .insert_with_hint(Point2::new(c[0], c[1]), hint)
            .map_err(|e| MesherError::Triangulation(format!("{e:?}")))?;
        region.resize(cdt.num_all_faces(), usize::MAX);
        let faces: Vec<FixedFaceHandle<InnerTag>> = cdt
            .vertex(v)
            .out_edges()
            .filter_map(|e| e.face().as_inner().map(|f| f.fix()))
            .collect();
        for f in faces {
            region[f.index()] = r;
            queue.push_back(f);
        }
    }
    Ok(())
}

/// The region of a side of a constraint: a region of triangles, or the outer face.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Side {
    Region(usize),
    Outer,
}

/// The constrained Delaunay triangulation of `domain` with its boundary split to `size` and the
/// interior refined for [`crate::adapt`]: the triangles inside the areas, each with its area tag,
/// and the tagged sub-edges.
///
/// Errors: [`MesherError::SelfIntersection`] for crossing rings or chains,
/// [`MesherError::Overlap`] for areas which overlap, [`MesherError::UntaggedBoundary`] for a
/// boundary segment without an edge tag, [`MesherError::EdgeOffBoundary`] for a tagged chain off
/// the boundaries, [`MesherError::Triangulation`] for degenerate input.
pub fn triangulate(domain: &Domain, size: &Size) -> Result<TaggedMesh, MesherError> {
    triangulate_capped(domain, size, None)
}

/// [`triangulate`] with the limit of the background refinement `limit` (vertices in all) instead
/// of the one from the areas.
fn triangulate_capped(
    domain: &Domain,
    size: &Size,
    limit: Option<usize>,
) -> Result<TaggedMesh, MesherError> {
    let bad = |m: String| MesherError::Triangulation(m);
    let graph = Graph::new(domain)?;
    if let Some(at) = graph.crossing() {
        return Err(MesherError::SelfIntersection { at });
    }
    if let Some(name) = &graph.flat_ring {
        return Err(bad(format!("a ring of {name:?} without area")));
    }

    // The points: the input vertices, then the split points of every piece in turn.
    let mut points = graph.vertices.clone();
    let mut constraints: Vec<[usize; 2]> = Vec::new();
    // The sub-edges of every piece, in its direction.
    let mut sub_edges: Vec<Vec<[usize; 2]>> = Vec::with_capacity(graph.pieces.len());
    for piece in &graph.pieces {
        let (a, b) = graph.ends(piece);
        let ts = split_params(a, b, size);
        let mut ids = Vec::with_capacity(ts.len());
        ids.push(piece.a as usize);
        for &t in &ts[1..ts.len() - 1] {
            points.push([a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]);
            ids.push(points.len() - 1);
        }
        ids.push(piece.b as usize);
        let edges: Vec<[usize; 2]> = ids.windows(2).map(|w| [w[0], w[1]]).collect();
        constraints.extend(&edges);
        sub_edges.push(edges);
    }

    // Inserted one by one in a fixed order: spade's bulk loading iterates hash sets.
    let mut cdt = ConstrainedDelaunayTriangulation::<Point2<f64>>::new();
    let mut previous = None;
    for (i, p) in points.iter().enumerate() {
        let point = Point2::new(p[0], p[1]);
        let v = match previous {
            Some(hint) => cdt.insert_with_hint(point, hint),
            None => cdt.insert(point),
        }
        .map_err(|e| bad(format!("the point ({}, {}): {e:?}", p[0], p[1])))?;
        if v.index() != i {
            return Err(bad(format!(
                "a split point coincides with another point at ({}, {}) um",
                p[0], p[1]
            )));
        }
        previous = Some(v);
    }
    for &[u, v] in &constraints {
        let (a, b) = (
            FixedVertexHandle::from_index(u),
            FixedVertexHandle::from_index(v),
        );
        if !cdt.can_add_constraint(a, b) {
            let at = midpoint(points[u], points[v]);
            return Err(bad(format!(
                "sub-edges cross near ({}, {}) um (segments at too small an angle)",
                at[0], at[1]
            )));
        }
        cdt.add_constraint(a, b);
    }
    if cdt.num_constraints() != constraints.len() {
        return Err(bad(
            "split points lie on other segments (segments at too small an angle)".to_string(),
        ));
    }
    // Regions: faces connected across edges which are no constraints.
    let mut region = vec![usize::MAX; cdt.num_all_faces()];
    // Per region: whether it touches the outer face, and its largest face (area, centroid).
    let mut regions: Vec<(bool, f64, P)> = Vec::new();
    for f in cdt.inner_faces() {
        if region[f.fix().index()] != usize::MAX {
            continue;
        }
        let r = regions.len();
        regions.push((false, -1.0, [0.0; 2]));
        region[f.fix().index()] = r;
        let mut stack = vec![f];
        while let Some(face) = stack.pop() {
            let p = face.vertices().map(|v| [v.position().x, v.position().y]);
            let area = 0.5 * orient(p[0], p[1], p[2]);
            if area > regions[r].1 {
                regions[r].1 = area;
                regions[r].2 = [
                    (p[0][0] + p[1][0] + p[2][0]) / 3.0,
                    (p[0][1] + p[1][1] + p[2][1]) / 3.0,
                ];
            }
            for e in face.adjacent_edges() {
                if e.is_constraint_edge() {
                    continue;
                }
                match e.rev().face().as_inner() {
                    None => regions[r].0 = true,
                    Some(n) => {
                        if region[n.fix().index()] == usize::MAX {
                            region[n.fix().index()] = r;
                            stack.push(n);
                        }
                    }
                }
            }
        }
    }

    // The sides of every piece (left and right of its direction), from its first sub-edge.
    let mut sides: Vec<(Side, Side)> = Vec::with_capacity(sub_edges.len());
    for edges in &sub_edges {
        let [u, v] = edges[0];
        let (u, v) = (
            FixedVertexHandle::from_index(u),
            FixedVertexHandle::from_index(v),
        );
        let e = cdt
            .get_edge_from_neighbors(u, v)
            .ok_or_else(|| bad("a constraint is missing".to_string()))?;
        let [left, right] = [e.face(), e.rev().face()].map(|face| {
            face.as_inner()
                .map_or(Side::Outer, |f| Side::Region(region[f.fix().index()]))
        });
        sides.push((left, right));
    }

    // Winding numbers (polygon, number) of the regions, from the outer face across the pieces.
    let node = |s: Side| match s {
        Side::Region(r) => r,
        Side::Outer => regions.len(),
    };
    let mut adjacent: Vec<Vec<(usize, usize, i32)>> = vec![Vec::new(); regions.len() + 1];
    for (i, &(left, right)) in sides.iter().enumerate() {
        if left == right {
            continue;
        }
        // From left to right: out of the polygons on the left, into those on the right.
        adjacent[node(left)].push((node(right), i, -1));
        adjacent[node(right)].push((node(left), i, 1));
    }
    let mut winding: Vec<Option<Vec<(u32, i32)>>> = vec![None; regions.len() + 1];
    let mut queue = VecDeque::new();
    for (r, w) in winding.iter_mut().enumerate() {
        if r == regions.len() || regions[r].0 {
            *w = Some(Vec::new());
            queue.push_back(r);
        }
    }
    while let Some(r) = queue.pop_front() {
        for &(n, piece, sign) in &adjacent[r] {
            let mut w = winding[r].clone().unwrap_or_default();
            for &(polygon, forward) in &graph.pieces[piece].rings {
                // The polygon is on the left of the piece if `forward`.
                let delta = if forward { sign } else { -sign };
                match w.binary_search_by_key(&polygon, |&(p, _)| p) {
                    Ok(k) => {
                        w[k].1 += delta;
                        if w[k].1 == 0 {
                            w.remove(k);
                        }
                    }
                    Err(k) => w.insert(k, (polygon, delta)),
                }
            }
            match &winding[n] {
                None => {
                    winding[n] = Some(w);
                    queue.push_back(n);
                }
                Some(known) if *known != w => {
                    return Err(bad("inconsistent winding numbers".to_string()));
                }
                Some(_) => {}
            }
        }
    }

    // The polygon of every region, if any.
    let mut polygon_of: Vec<Option<usize>> = Vec::with_capacity(regions.len());
    for (r, &(_, _, at)) in regions.iter().enumerate() {
        let w = winding[r]
            .as_ref()
            .ok_or_else(|| bad("a region out of reach".to_string()))?;
        polygon_of.push(match w.as_slice() {
            [] => None,
            [(p, 1)] => Some(*p as usize),
            w if w.iter().all(|&(_, n)| n > 0) => return Err(MesherError::Overlap { at }),
            _ => {
                return Err(bad(format!(
                    "a hole outside its polygon at ({}, {}) um",
                    at[0], at[1]
                )));
            }
        });
    }
    let area_of = |s: Side| match s {
        Side::Region(r) => polygon_of[r],
        Side::Outer => None,
    };

    // Boundaries need a tag, tags need a boundary.
    for (piece, &(left, right)) in graph.pieces.iter().zip(&sides) {
        let (a, b) = graph.ends(piece);
        let (l, r) = (area_of(left), area_of(right));
        let dangling = left == right || (l.is_none() && r.is_none());
        if let (true, Some(&tag)) = (dangling, piece.tags.first()) {
            return Err(MesherError::EdgeOffBoundary {
                tag: domain.tags[tag as usize].name.clone(),
                at: midpoint(a, b),
            });
        }
        if !dangling && l.is_none() != r.is_none() && piece.tags.is_empty() {
            return Err(MesherError::UntaggedBoundary { at: midpoint(a, b) });
        }
    }

    // The background for the metric: triangles of the areas longer than `BACKGROUND` times the
    // size at their centroid get a vertex at it, until none is.
    let in_area = |region: usize| polygon_of[region].is_some();
    // At most the vertices of equilateral triangles of `BACKGROUND` times the smallest size over
    // the areas, which bounds the refinement whatever the size field.
    let area: f64 = cdt
        .inner_faces()
        .filter(|f| in_area(region[f.fix().index()]))
        .map(|f| {
            let p = f.vertices().map(|v| [v.position().x, v.position().y]);
            0.5 * orient(p[0], p[1], p[2])
        })
        .sum();
    let side = BACKGROUND * size.min();
    let limit = limit
        .unwrap_or(cdt.num_vertices() + (2.0 * area / (0.25 * 3f64.sqrt() * side * side)) as usize);
    refine_background(&mut cdt, &mut region, &in_area, size, limit)?;
    if cdt.num_constraints() != constraints.len() {
        return Err(bad("the refinement split a constraint".to_string()));
    }

    // The triangles of the areas, the vertices numbered in the order of first use.
    let mut number = vec![u32::MAX; cdt.num_vertices()];
    let mut vertices: Vec<P> = Vec::new();
    let mut triangles = Vec::new();
    let mut triangle_tags = Vec::new();
    for f in cdt.inner_faces() {
        let Some(polygon) = polygon_of[region[f.fix().index()]] else {
            continue;
        };
        let t = f.vertices().map(|v| {
            let i = v.fix().index();
            if number[i] == u32::MAX {
                number[i] = vertices.len() as u32;
                vertices.push([v.position().x, v.position().y]);
            }
            number[i]
        });
        triangles.push(canonical_triangle(t));
        triangle_tags.push(domain.areas[polygon].0 as u16);
    }
    let mut edges: Vec<(u16, [u32; 2])> = Vec::new();
    for (piece, subs) in graph.pieces.iter().zip(&sub_edges) {
        for &tag in &piece.tags {
            for &[u, v] in subs {
                let (u, v) = (number[u], number[v]);
                edges.push((tag, [u.min(v), u.max(v)]));
            }
        }
    }
    edges.sort_unstable();
    Ok(TaggedMesh {
        tags: domain.tags.clone(),
        vertices,
        triangles,
        triangle_tags,
        edge_tags: edges.iter().map(|e| e.0).collect(),
        edges: edges.into_iter().map(|e| e.1).collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MeshTag;
    use contracts::SizeField;

    #[test]
    fn background_refinement_limit_is_an_error() {
        let tag = |name: &str, dim| MeshTag {
            name: name.to_string(),
            value: 1,
            dim,
        };
        let square = vec![
            [0.0, 0.0],
            [100.0, 0.0],
            [100.0, 100.0],
            [0.0, 100.0],
            [0.0, 0.0],
        ];
        let domain = Domain {
            tags: vec![tag("a", 2), tag("cut", 1)],
            areas: vec![(0, vec![square.clone()])],
            edges: vec![(1, square)],
        };
        let spec = SizeField {
            min_um: 5.0,
            max_um: 5.0,
            dist_min_um: 0.0,
            dist_max_um: 1.0,
            walls: vec![],
        };
        let size = Size::new(&spec, &domain).unwrap();
        assert!(triangulate(&domain, &size).is_ok());
        match triangulate_capped(&domain, &size, Some(100)) {
            Err(MesherError::Triangulation(m)) => assert!(m.contains("limit of 100"), "{m}"),
            r => panic!("unexpected {r:?}"),
        }
    }
}
