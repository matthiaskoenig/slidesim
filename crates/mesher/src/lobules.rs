//! Parametric ideal lobules: a tagged [`Domain`] from a [`LobuleSpec`], and the relative position
//! of a point between the walls of its holes.
//!
//! The veins lie on a hexagonal lattice of spacing `sqrt(3) R` (`R` the circumradius of a cell),
//! ring by ring up to ring `rings + 1`, each displaced by a uniform point of the disc of radius
//! `jitter R` (SplitMix64 of the seed, two draws per vein in lattice order, so a vein keeps its
//! displacement whatever `rings`). The tissue is the union of the Voronoi cells of the veins
//! within ring `rings`; the outer ring only bounds them. The Voronoi cells come from `spade`'s
//! Delaunay triangulation of the veins: a corner is the circumcentre of a triangle, computed once
//! per triangle, and the outer border is made of the Voronoi edges between a vein of the tissue
//! and one of the outer ring.
//!
//! The holes are circles around the veins of the tissue (`centre`) and around every corner
//! (`corner`), polygonized with `max(12, ceil(2 pi r / wall_um))` vertices on the full circle. A
//! corner on the outer border has its hole cut off by the border: the border is trimmed by the
//! corner radius at both ends of each of its edges, and the arc between two trimmed ends runs
//! through the tissue with the same vertex spacing. Every point of the outer ring is computed
//! once and copied into the edge chains, so the rings and the chains share their vertices bit for
//! bit, and the domain is one area: the outer ring and the holes inside the tissue.

use std::f64::consts::TAU;

use contracts::{AreaPolygon, EdgeChain, LobuleSpec, Polygons, SizeField};
use rstar::RTree;
use rstar::primitives::{GeomWithData, Line};
use spade::{DelaunayTriangulation, Point2, Triangulation};

use crate::{Domain, MesherError, Size};

type P = [f64; 2];

/// The largest jitter in units of the circumradius (the contract's): the Delaunay triangulation
/// of the veins stays the one of the lattice, and adjacent corners stay at least `R / 4` apart.
const MAX_JITTER: f64 = 0.25;

/// SplitMix64, a seeded generator of uniform numbers in `[0, 1)`, as sbmlsim-c's tests have it.
struct Random(u64);

impl Random {
    fn uniform(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^= z >> 31;
        (z >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn distance(a: P, b: P) -> f64 {
    (b[0] - a[0]).hypot(b[1] - a[1])
}

fn invalid(message: String) -> MesherError {
    MesherError::InvalidLobules(message)
}

/// The axial coordinates of the hexagonal lattice and the ring of each, ring by ring up to
/// `rings`, each ring counterclockwise.
fn lattice(rings: i64) -> Vec<(i64, i64, i64)> {
    const DIRECTIONS: [(i64, i64); 6] = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)];
    let mut out = vec![(0, 0, 0)];
    for k in 1..=rings {
        let (mut q, mut r) = (DIRECTIONS[4].0 * k, DIRECTIONS[4].1 * k);
        for (dq, dr) in DIRECTIONS {
            for _ in 0..k {
                out.push((q, r, k));
                q += dq;
                r += dr;
            }
        }
    }
    out
}

/// The circumcentre of the triangle `a`, `b`, `c`, relative to `a`.
fn circumcentre(a: P, b: P, c: P) -> P {
    let (bx, by) = (b[0] - a[0], b[1] - a[1]);
    let (cx, cy) = (c[0] - a[0], c[1] - a[1]);
    let d = 2.0 * (bx * cy - by * cx);
    let (b2, c2) = (bx * bx + by * by, cx * cx + cy * cy);
    [
        a[0] + (cy * b2 - by * c2) / d,
        a[1] + (bx * c2 - cx * b2) / d,
    ]
}

/// The number of vertices of a full hole circle of radius `r`.
fn circle_vertices(r: f64, wall: f64) -> usize {
    ((TAU * r / wall).ceil() as usize).max(12)
}

/// The points of the circle of radius `r` around `c` at the angles `start - sweep k / m` for
/// `k` in `1..m`, `m` the vertex count of the full circle scaled to the sweep (at least 1).
fn arc(c: P, r: f64, start: f64, sweep: f64, n: usize) -> Vec<P> {
    let m = ((n as f64 * sweep / TAU).ceil() as usize).max(1);
    (1..m)
        .map(|k| {
            let a = start - sweep * k as f64 / m as f64;
            [c[0] + r * a.cos(), c[1] + r * a.sin()]
        })
        .collect()
}

/// Ideal lobules: the veins, the corners, the tagged domain and its size field.
#[derive(Clone, Debug)]
pub struct Lobules {
    veins: Vec<P>,
    corners: Vec<P>,
    domain: Domain,
    size_field: SizeField,
    bounds: [P; 2],
    centre_radius: f64,
    corner_radius: f64,
    vein_tree: RTree<P>,
    corner_tree: RTree<P>,
}

impl Lobules {
    /// The lobules of `spec`: geometry only (no mesh), deterministic, and the same on every
    /// process.
    ///
    /// Errors ([`MesherError::InvalidLobules`]) for a radius which is not finite and positive, a
    /// jitter outside `[0, 0.25]` or tags which are not distinct; [`MesherError::InvalidTag`] for
    /// a reserved or empty tag; [`MesherError::SizeField`] for sizes which are not finite with
    /// `0 < wall_um <= interior_um`; [`MesherError::HolesOverlap`] if two holes overlap or touch
    /// (the tags of both), or a hole reaches the outer border (the hole's tag and the outer tag),
    /// which the contract's bounds on the radii rule out.
    pub fn new(spec: &LobuleSpec) -> Result<Lobules, MesherError> {
        let radius = spec.lobule_radius_um;
        let (rc, rk) = (spec.centre.radius_um, spec.corner.radius_um);
        for (key, v) in [
            ("lobule_radius_um", radius),
            ("centre.radius_um", rc),
            ("corner.radius_um", rk),
        ] {
            if !(v.is_finite() && v > 0.0) {
                return Err(invalid(format!("{key} must be finite and > 0, got {v}")));
            }
        }
        if !(spec.jitter.is_finite() && (0.0..=MAX_JITTER).contains(&spec.jitter)) {
            return Err(invalid(format!(
                "jitter must be within 0 and {MAX_JITTER}, got {}",
                spec.jitter
            )));
        }
        let tags = [
            ("outer_tag", &spec.outer_tag),
            ("area_tag", &spec.area_tag),
            ("centre.tag", &spec.centre.tag),
            ("corner.tag", &spec.corner.tag),
        ];
        for (i, (key, tag)) in tags.iter().enumerate() {
            contracts::check_tag_name(tag)
                .map_err(|e| MesherError::InvalidTag(format!("{key}: {e}")))?;
            if let Some((other, _)) = tags[..i].iter().find(|(_, t)| t == tag) {
                return Err(invalid(format!(
                    "{key} {tag:?} equals {other}: the tags must differ"
                )));
            }
        }
        let (wall, interior) = (spec.size.wall_um, spec.size.interior_um);
        // What `Size::new` checks of the size field (`size` cannot fail).
        if !(wall > 0.0 && wall <= interior && (10.0 * interior).is_finite()) {
            return Err(MesherError::SizeField(format!(
                "wall_um {wall} and interior_um {interior}: need 0 < wall_um <= interior_um"
            )));
        }

        // The veins, displaced in lattice order.
        let rings = i64::from(spec.rings);
        let mut random = Random(spec.seed);
        let lattice = lattice(rings + 1);
        let positions: Vec<P> = lattice
            .iter()
            .map(|&(q, r, _)| {
                let rho = spec.jitter * radius * random.uniform().sqrt();
                let angle = TAU * random.uniform();
                [
                    radius * (3f64.sqrt() * (q as f64 + 0.5 * r as f64)) + rho * angle.cos(),
                    radius * (1.5 * r as f64) + rho * angle.sin(),
                ]
            })
            .collect();
        let tissue: Vec<bool> = lattice.iter().map(|&(_, _, k)| k <= rings).collect();

        // The Delaunay triangulation; the index of a vertex handle is the index of its vein.
        let mut delaunay = DelaunayTriangulation::<Point2<f64>>::new();
        let mut handles = Vec::with_capacity(positions.len());
        for (i, p) in positions.iter().enumerate() {
            let h = delaunay
                .insert(Point2::new(p[0], p[1]))
                .map_err(|e| invalid(format!("vein {i} at {p:?}: {e:?}")))?;
            if h.index() != handles.len() {
                return Err(invalid(format!("vein {i} at {p:?} coincides with another")));
            }
            handles.push(h);
        }

        // The corners: the circumcentres of the triangles around the veins of the tissue, each
        // computed once, with the number of veins of the tissue of its triangle.
        let mut corner_of_face: Vec<Option<usize>> = vec![None; delaunay.num_all_faces()];
        let mut corners: Vec<P> = Vec::new();
        let mut interior: Vec<bool> = Vec::new();
        for (i, &h) in handles.iter().enumerate().filter(|&(i, _)| tissue[i]) {
            for e in delaunay.vertex(h).out_edges() {
                let face = e.face().as_inner().ok_or_else(|| {
                    invalid(format!(
                        "the cell of vein {i} at {:?} is unbounded",
                        positions[i]
                    ))
                })?;
                let slot = &mut corner_of_face[face.fix().index()];
                if slot.is_none() {
                    let [a, b, c] = face.positions().map(|p| [p.x, p.y]);
                    *slot = Some(corners.len());
                    corners.push(circumcentre(a, b, c));
                    interior.push(face.vertices().iter().all(|v| tissue[v.fix().index()]));
                }
            }
        }

        // The outer border, counterclockwise: the Voronoi edge of `a -> b` (a in the tissue, b
        // not) runs from the corner of the triangle right of it to the one left of it.
        let mut next: Vec<Option<usize>> = vec![None; corners.len()];
        for (_, &h) in handles.iter().enumerate().filter(|&(i, _)| tissue[i]) {
            for e in delaunay.vertex(h).out_edges() {
                if tissue[e.to().fix().index()] {
                    continue;
                }
                let from =
                    (e.rev().face().as_inner()).and_then(|f| corner_of_face[f.fix().index()]);
                let to = (e.face().as_inner()).and_then(|f| corner_of_face[f.fix().index()]);
                match (from, to) {
                    (Some(from), Some(to)) if next[from].is_none() => next[from] = Some(to),
                    _ => return Err(invalid("the outer border is not a simple ring".into())),
                }
            }
        }
        let border_count = next.iter().flatten().count();
        let mut ring = Vec::with_capacity(border_count);
        if let Some(start) = next.iter().position(Option::is_some) {
            let mut c = start;
            loop {
                ring.push(c);
                c = next[c].ok_or_else(|| invalid("the outer border is open".into()))?;
                if c == start || ring.len() > border_count {
                    break;
                }
            }
        }
        let on_border = (0..corners.len()).filter(|&c| next[c].is_some()).count();
        if ring.len() != border_count
            || border_count < 3
            || on_border != interior.iter().filter(|&&i| !i).count()
        {
            return Err(invalid("the outer border is not a simple ring".into()));
        }

        let veins: Vec<P> = (0..positions.len())
            .filter(|&i| tissue[i])
            .map(|i| positions[i])
            .collect();

        // No two holes overlap or touch: the veins' first, then the corners'.
        let holes: Vec<(P, f64, &str)> = veins
            .iter()
            .map(|&v| (v, rc, spec.centre.tag.as_str()))
            .chain(corners.iter().map(|&c| (c, rk, spec.corner.tag.as_str())))
            .collect();
        let r_max = rc.max(rk);
        let hole_tree = RTree::bulk_load(
            holes
                .iter()
                .enumerate()
                .map(|(i, h)| GeomWithData::new(h.0, i))
                .collect(),
        );
        for (i, &(c, r, tag)) in holes.iter().enumerate() {
            let reach = r + r_max;
            let other = hole_tree
                .locate_within_distance(c, reach * reach)
                .map(|g| g.data)
                .filter(|&j| j > i && distance(c, holes[j].0) <= r + holes[j].1)
                .min();
            if let Some(j) = other {
                return Err(MesherError::HolesOverlap {
                    a: tag.to_string(),
                    b: holes[j].2.to_string(),
                });
            }
        }

        // The edges of the outer border, trimmed by the corner holes at both ends.
        let k = ring.len();
        let trimmed: Vec<[P; 2]> = (0..k)
            .map(|i| {
                let (p, q) = (corners[ring[i]], corners[ring[(i + 1) % k]]);
                let l = distance(p, q);
                let u = [(q[0] - p[0]) / l, (q[1] - p[1]) / l];
                [
                    [p[0] + rk * u[0], p[1] + rk * u[1]],
                    [q[0] - rk * u[0], q[1] - rk * u[1]],
                ]
            })
            .collect();
        // No hole reaches an edge of the border but at its own ends.
        let border_tree = RTree::bulk_load(
            trimmed
                .iter()
                .enumerate()
                .map(|(i, s)| GeomWithData::new(Line::new(s[0], s[1]), i))
                .collect(),
        );
        let corner_hole = |hole: usize| hole.checked_sub(veins.len());
        for (i, &(c, r, tag)) in holes.iter().enumerate() {
            let reaches = border_tree
                .locate_within_distance(c, r * r)
                .map(|g| g.data)
                .any(|s| corner_hole(i).is_none_or(|h| h != ring[s] && h != ring[(s + 1) % k]));
            if reaches {
                return Err(MesherError::HolesOverlap {
                    a: tag.to_string(),
                    b: spec.outer_tag.clone(),
                });
            }
        }

        // The outer ring and its chains: each trimmed edge, then the arc of the next corner hole
        // through the tissue (clockwise around the corner, from the direction of the previous
        // corner to the one of the next).
        let nk = circle_vertices(rk, wall);
        let chain = |tag: &str, points: Vec<P>| EdgeChain {
            tag: tag.to_string(),
            value: 1,
            points,
        };
        let mut outer: Vec<P> = Vec::new();
        let mut edges: Vec<EdgeChain> = Vec::new();
        for i in 0..k {
            let [a, b] = trimmed[i];
            let j = (i + 1) % k;
            let (p, q, r) = (
                corners[ring[i]],
                corners[ring[j]],
                corners[ring[(j + 1) % k]],
            );
            let from = (p[1] - q[1]).atan2(p[0] - q[0]);
            let to = (r[1] - q[1]).atan2(r[0] - q[0]);
            let points = arc(q, rk, from, (from - to).rem_euclid(TAU), nk);
            outer.extend([a, b]);
            outer.extend(&points);
            edges.push(chain(&spec.outer_tag, vec![a, b]));
            let mut wall_points = Vec::with_capacity(points.len() + 2);
            wall_points.push(b);
            wall_points.extend(points);
            wall_points.push(trimmed[j][0]);
            edges.push(chain(&spec.corner.tag, wall_points));
        }

        // The holes inside the tissue: the veins' and the interior corners'.
        let nc = circle_vertices(rc, wall);
        let circle = |c: P, r: f64, n: usize| -> Vec<P> {
            (0..n)
                .map(|k| {
                    let a = TAU * k as f64 / n as f64;
                    [c[0] + r * a.cos(), c[1] + r * a.sin()]
                })
                .collect()
        };
        let mut rings = vec![outer];
        let inside = veins
            .iter()
            .map(|&v| (circle(v, rc, nc), &spec.centre.tag))
            .chain(
                (0..corners.len())
                    .filter(|&c| interior[c])
                    .map(|c| (circle(corners[c], rk, nk), &spec.corner.tag)),
            );
        for (hole, tag) in inside {
            let mut closed = hole.clone();
            closed.push(hole[0]);
            edges.push(chain(tag, closed));
            rings.push(hole);
        }

        let bounds = rings[0].iter().fold(
            [[f64::INFINITY; 2], [f64::NEG_INFINITY; 2]],
            |[lo, hi], x| {
                [
                    [lo[0].min(x[0]), lo[1].min(x[1])],
                    [hi[0].max(x[0]), hi[1].max(x[1])],
                ]
            },
        );
        let domain = Domain::from_polygons(&Polygons {
            areas: vec![AreaPolygon {
                tag: spec.area_tag.clone(),
                value: 1,
                rings,
            }],
            edges,
        })?;
        Ok(Lobules {
            vein_tree: RTree::bulk_load(veins.clone()),
            corner_tree: RTree::bulk_load(corners.clone()),
            veins,
            corners,
            domain,
            size_field: spec.size_field(),
            bounds,
            centre_radius: rc,
            corner_radius: rk,
        })
    }

    /// The veins of the tissue (the centres of the cells), ring by ring.
    pub fn veins(&self) -> &[P] {
        &self.veins
    }

    /// The corners of the cells.
    pub fn corners(&self) -> &[P] {
        &self.corners
    }

    /// The tagged domain: the tissue with the area tag, the outer border with the outer tag, the
    /// hole walls with the centre and corner tags.
    pub fn domain(&self) -> &Domain {
        &self.domain
    }

    /// The size field of the domain ([`LobuleSpec::size_field`]), built on each call: it measures
    /// the gaps of the domain, which only the process which meshes needs.
    pub fn size(&self) -> Size {
        Size::new(&self.size_field, &self.domain)
            .expect("Lobules::new checks the sizes, and the domain has both walls")
    }

    /// The bounding box `[min, max]` of the domain in um.
    pub fn bounds(&self) -> [P; 2] {
        self.bounds
    }

    /// The relative position of `x` between the walls: `d_p / (d_p + d_c)` with `d_p` the
    /// distance to the nearest corner wall and `d_c` to the nearest centre wall (the circles,
    /// both clamped at 0 inside a hole): 0 on the corner walls, 1 on the centre walls.
    pub fn position(&self, x: P) -> f64 {
        let wall = |tree: &RTree<P>, r: f64| {
            tree.nearest_neighbor(x)
                .map_or(f64::INFINITY, |c| (distance(*c, x) - r).max(0.0))
        };
        let d_p = wall(&self.corner_tree, self.corner_radius);
        let d_c = wall(&self.vein_tree, self.centre_radius);
        if d_p + d_c > 0.0 {
            d_p / (d_p + d_c)
        } else {
            // On both walls at once, which overlapping holes only could give.
            0.5
        }
    }
}
