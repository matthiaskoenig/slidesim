//! The size field: the edge length as a function of the distance to the nearest wall, capped by
//! the width of narrow gaps and by the length of short boundary segments.

use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;

use contracts::SizeField;
use rstar::primitives::{GeomWithData, Line};
use rstar::{PointDistance, RTree};

use crate::{Domain, MesherError};

/// The cap of the size at a boundary point in units of the width of the gap there (the distance
/// to the boundary across): 0.4 puts at least two element layers across a gap (the size grows by
/// [`GRADING`] towards the middle of the gap, `0.4 w + 0.3 w / 2 = 0.55 w`), so that triangles
/// between two frozen boundaries are not stretched.
const FEATURE: f64 = 0.4;

/// The growth of the cap with the distance from a gap (dimensionless, size per length): about
/// Mmg's gradation `hgrad = 1.3`, so that Mmg needs not grade the metric further.
const GRADING: f64 = 0.3;

/// The smallest cap in units of the smallest size of the field: a gap narrower than
/// `FLOOR min / FEATURE` gets no smaller elements (its quality is then reported, not bounded),
/// which bounds the element count whatever the input.
const FLOOR: f64 = 0.25;

/// The cap of the size at the ends of a boundary segment in units of its length: the ends are
/// input vertices, which every step keeps, so a segment much shorter than the field (a pixel
/// step of a mask: 1.77 um under 5 um elements) would be an edge of metric length well below 1
/// with elements of the full size beside it. With the cap the elements next to it are at most
/// 1.5 times its length (an isosceles triangle on it has SICN 0.89; the edge ratio of the
/// segment is at least `1 / (1.5 + GRADING / 2) = 0.6`), growing by [`GRADING`]. Segments of
/// nearly the field's length (a polygonized circle of `ceil(2 pi r / h)` segments) need no cap
/// and get none. Not below [`FLOOR`] `min`: shorter segments are reported, not bounded.
const SEGMENT: f64 = 1.5;

/// A boundary point and the nearest point of another segment are no gap if the boundary joins
/// them within `PATH` times their distance. Two sides of a gap are joined the long way round
/// (around the end of a neck, or not at all), unless a chain crosses the gap: an interface across
/// a neck joins its two sides within its own length, so next to it the neck is no gap. A corner
/// of the boundary joins them directly.
/// At a corner of angle `t` between two polylines, a point and its foot on the other side are
/// joined by `cot(t / 2)` times their distance (1 for a right angle, the polygonized arc of a
/// hole meeting a straight border), so corners of 53 degrees or wider make no gap; for a
/// sharper corner between polylines the cap keeps two element layers across the wedge.
const PATH: f64 = 2.0;

/// One threshold field: `min` up to the distance `dist_min` from the nearest segment of a wall
/// chain, linear up to `max` at `dist_max`, `max` beyond (gmsh's Threshold field); without
/// walls, `max` everywhere. The distance comes from an R-tree of the wall segments.
#[derive(Clone, Debug)]
struct Threshold {
    min: f64,
    max: f64,
    dist_min: f64,
    dist_max: f64,
    walls: Option<RTree<Line<[f64; 2]>>>,
}

impl Threshold {
    fn new(spec: &SizeField, domain: &Domain) -> Result<Threshold, MesherError> {
        let finite = [spec.min_um, spec.max_um, spec.dist_min_um, spec.dist_max_um]
            .iter()
            .all(|v| v.is_finite());
        if !(finite
            && spec.min_um > 0.0
            && spec.min_um <= spec.max_um
            && spec.dist_min_um >= 0.0
            && spec.dist_min_um < spec.dist_max_um)
        {
            return Err(MesherError::SizeField(format!(
                "sizes {} to {} um at distances {} to {} um: need 0 < min <= max and \
                 0 <= dist_min < dist_max",
                spec.min_um, spec.max_um, spec.dist_min_um, spec.dist_max_um
            )));
        }
        let mut walls = Vec::new();
        for wall in &spec.walls {
            let tags: Vec<usize> = (0..domain.tags.len())
                .filter(|&i| domain.tags[i].dim == 1 && domain.tags[i].name == *wall)
                .collect();
            if tags.is_empty() {
                return Err(MesherError::SizeField(format!(
                    "the wall {wall:?} is no edge tag of the domain"
                )));
            }
            for (tag, points) in &domain.edges {
                if tags.contains(tag) {
                    walls.extend(
                        points
                            .windows(2)
                            .filter(|w| w[0] != w[1])
                            .map(|w| Line::new(w[0], w[1])),
                    );
                }
            }
        }
        Ok(Threshold {
            min: spec.min_um,
            max: spec.max_um,
            dist_min: spec.dist_min_um,
            dist_max: spec.dist_max_um,
            walls: (!walls.is_empty()).then(|| RTree::bulk_load(walls)),
        })
    }

    fn at(&self, x: [f64; 2]) -> f64 {
        let Some(walls) = &self.walls else {
            return self.max;
        };
        let d = walls
            .nearest_neighbor(x)
            .map_or(f64::INFINITY, |s| s.distance_2(&x).sqrt());
        if d <= self.dist_min {
            self.min
        } else if d >= self.dist_max {
            self.max
        } else {
            self.min + (self.max - self.min) * (d - self.dist_min) / (self.dist_max - self.dist_min)
        }
    }
}

/// The edge length in um at a point: the smallest of one or more threshold fields, each `min` up
/// to the distance `dist_min` from the nearest segment of its wall chains, linear up to `max` at
/// `dist_max`, `max` beyond (gmsh's Threshold fields under a Min field); without walls, `max`
/// everywhere.
///
/// The field is capped by the local width of the domain: at a point of a ring or chain whose
/// nearest other segment across (not touching it, at an end or a T-junction; not collinear with
/// it; not along it; not joined to the point along the boundary within twice the distance, see
/// [`PATH`]) is at the distance `w`, the size is at most `0.4 w` (but not below `min / 4`),
/// growing by 0.3 per um away from it. Narrow necks of tissue and chains close to each other so
/// get elements which fit between their frozen boundaries. It is capped by the boundary
/// segments too: at both ends of a segment of length `l` the size is at most `1.5 l` (but not
/// below `min / 4`), growing by 0.3 per um away from them, so that the elements beside a short
/// frozen segment (a pixel step of a mask) take its length ([`SEGMENT`]).
#[derive(Clone, Debug)]
pub struct Size {
    fields: Vec<Threshold>,
    /// The smallest `min` of the fields.
    min: f64,
    /// The largest `max` of the fields.
    max: f64,
    /// The boundary points where a cap (of a gap or of a short segment) is below the field, with
    /// the cap.
    gaps: Option<RTree<GeomWithData<[f64; 2], f64>>>,
    /// The smallest cap.
    gap_min: f64,
}

fn sub(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    [a[0] - b[0], a[1] - b[1]]
}

fn dot(a: [f64; 2], b: [f64; 2]) -> f64 {
    a[0] * b[0] + a[1] * b[1]
}

fn distance(a: [f64; 2], b: [f64; 2]) -> f64 {
    (b[0] - a[0]).hypot(b[1] - a[1])
}

/// Every segment of the rings (closed) and chains of `domain`, zero-length ones left out.
fn segments(domain: &Domain) -> Vec<[[f64; 2]; 2]> {
    let mut out = Vec::new();
    for (_, rings) in &domain.areas {
        for ring in rings {
            let n = ring.len();
            out.extend((0..n).map(|i| [ring[i], ring[(i + 1) % n]]));
        }
    }
    for (_, points) in &domain.edges {
        out.extend(points.windows(2).map(|w| [w[0], w[1]]));
    }
    out.retain(|s| s[0] != s[1]);
    out
}

/// A distance as a key of a min-heap, with the vertex as the tie break.
#[derive(PartialEq)]
struct Key(f64, usize);

impl Eq for Key {}

impl PartialOrd for Key {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Key {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0).then(self.1.cmp(&other.1))
    }
}

/// The segments as a graph: the vertices (shared by exactly equal coordinates), the end
/// vertices of each segment and the neighbours of each vertex with the segment lengths, and the
/// shortest paths along it from a vertex up to a length.
struct Boundary {
    ends: Vec<[usize; 2]>,
    neighbours: Vec<Vec<(usize, f64)>>,
    /// The distances of the last walk, infinite where it did not reach, and the vertices it
    /// reached (to reset them).
    distances: Vec<f64>,
    reached: Vec<usize>,
}

impl Boundary {
    fn new(segments: &[[[f64; 2]; 2]]) -> Boundary {
        let order = |a: &[f64; 2], b: &[f64; 2]| a[0].total_cmp(&b[0]).then(a[1].total_cmp(&b[1]));
        let mut points: Vec<[f64; 2]> = segments.iter().flatten().copied().collect();
        points.sort_by(order);
        points.dedup_by(|a, b| order(a, b) == Ordering::Equal);
        let vertex = |p: &[f64; 2]| points.binary_search_by(|q| order(q, p)).unwrap();
        let ends: Vec<[usize; 2]> = segments.iter().map(|s| s.each_ref().map(vertex)).collect();
        let mut neighbours = vec![Vec::new(); points.len()];
        for (s, &[a, b]) in segments.iter().zip(&ends) {
            let length = distance(s[0], s[1]);
            neighbours[a].push((b, length));
            neighbours[b].push((a, length));
        }
        Boundary {
            ends,
            neighbours,
            distances: vec![f64::INFINITY; points.len()],
            reached: Vec::new(),
        }
    }

    /// The lengths of the shortest paths along the boundary from `from`, up to `limit`
    /// (infinite beyond).
    fn walk(&mut self, from: usize, limit: f64) -> Vec<(usize, f64)> {
        for &v in &self.reached {
            self.distances[v] = f64::INFINITY;
        }
        self.reached.clear();
        let mut heap = BinaryHeap::new();
        self.distances[from] = 0.0;
        self.reached.push(from);
        heap.push(Reverse(Key(0.0, from)));
        let mut out = Vec::new();
        while let Some(Reverse(Key(d, v))) = heap.pop() {
            if d > self.distances[v] {
                continue;
            }
            out.push((v, d));
            for &(w, length) in &self.neighbours[v] {
                let e = d + length;
                if e <= limit && e < self.distances[w] {
                    if self.distances[w].is_infinite() {
                        self.reached.push(w);
                    }
                    self.distances[w] = e;
                    heap.push(Reverse(Key(e, w)));
                }
            }
        }
        out.sort_unstable_by_key(|&(v, _)| v);
        out
    }
}

/// The shortest path from the vertex of a walk to `v`, if the walk reached it.
fn reached(walk: &[(usize, f64)], v: usize) -> f64 {
    walk.binary_search_by_key(&v, |&(w, _)| w)
        .map_or(f64::INFINITY, |i| walk[i].1)
}

impl Size {
    /// The size field `spec` with the distance to the chains of the edge tags `spec.walls` of
    /// `domain`, capped by the widths of the gaps of `domain`.
    ///
    /// Errors if a wall is no edge tag of the domain, or the sizes and distances are not finite,
    /// positive and ordered (`min <= max`, `dist_min < dist_max`).
    pub fn new(spec: &SizeField, domain: &Domain) -> Result<Size, MesherError> {
        Size::min_of(std::slice::from_ref(spec), domain)
    }

    /// The smallest of the size fields `specs` at each point (each with the distance to its own
    /// walls), capped by the widths of the gaps of `domain`: walls of different sizes.
    ///
    /// Errors for no fields, and as [`Size::new`] for each field.
    pub fn min_of(specs: &[SizeField], domain: &Domain) -> Result<Size, MesherError> {
        if specs.is_empty() {
            return Err(MesherError::SizeField("no size field".to_string()));
        }
        let fields = specs
            .iter()
            .map(|spec| Threshold::new(spec, domain))
            .collect::<Result<Vec<_>, _>>()?;
        let mut size = Size {
            min: fields.iter().map(|f| f.min).fold(f64::INFINITY, f64::min),
            max: fields.iter().map(|f| f.max).fold(0.0, f64::max),
            fields,
            gaps: None,
            gap_min: f64::INFINITY,
        };
        size.gaps = size.gaps(domain);
        if let Some(gaps) = &size.gaps {
            size.gap_min = gaps.iter().map(|g| g.data).fold(f64::INFINITY, f64::min);
        }
        Ok(size)
    }

    /// The points of the rings and chains (every `min / 2` along each segment) where the cap by a
    /// gap is below the field, and the ends of the segments where the cap by their length is,
    /// with the cap.
    fn gaps(&self, domain: &Domain) -> Option<RTree<GeomWithData<[f64; 2], f64>>> {
        let segments = segments(domain);
        let tree = RTree::bulk_load(
            segments
                .iter()
                .enumerate()
                .map(|(i, s)| GeomWithData::new(Line::new(s[0], s[1]), i))
                .collect(),
        );
        let mut boundary = Boundary::new(&segments);
        // The farthest any gap reaches (where the cap meets the largest size), and the farthest a
        // path along the boundary needs to be followed.
        let limit = PATH * self.max / FEATURE;
        let orient = |a: [f64; 2], b: [f64; 2], c: [f64; 2]| {
            let coord = |p: [f64; 2]| robust::Coord { x: p[0], y: p[1] };
            robust::orient2d(coord(a), coord(b), coord(c))
        };
        let floor = FLOOR * self.min;
        let mut gaps = Vec::new();
        for (i, &[a, b]) in segments.iter().enumerate() {
            let length = distance(a, b);
            // A segment shorter than the field: its ends are frozen vertices, so the elements
            // around it take its length (see [`SEGMENT`]).
            let cap = (SEGMENT * length).max(floor);
            for x in [a, b] {
                if cap < self.field(x) {
                    gaps.push(GeomWithData::new(x, cap));
                }
            }
            let direction = sub(b, a);
            let n = (2.0 * length / self.min).ceil().max(1.0) as usize;
            // The paths along the boundary from the ends of this segment, walked when needed.
            let mut walks: Option<[Vec<(usize, f64)>; 2]> = None;
            for k in 0..=n {
                let t = k as f64 / n as f64;
                let x = [a[0] + t * direction[0], a[1] + t * direction[1]];
                let field = self.field(x);
                for (q, d2) in tree.nearest_neighbor_iter_with_distance_2(x) {
                    let d = d2.sqrt();
                    if FEATURE * d >= field {
                        break;
                    }
                    // Segments which share a point (an end, or an end inside the other: a
                    // T-junction) make no gap, however narrow the wedge between them; nor do
                    // collinear ones (shared borders), nor any at the distance 0 (rings and
                    // chains do not cross, so that is a junction too).
                    let [c, e] = [q.geom().from, q.geom().to];
                    // Whether `p` lies on the segment from `s` to `u`.
                    let inside = |p: [f64; 2], s: [f64; 2], u: [f64; 2]| {
                        orient(s, u, p) == 0.0
                            && p[0] >= s[0].min(u[0])
                            && p[0] <= s[0].max(u[0])
                            && p[1] >= s[1].min(u[1])
                            && p[1] <= s[1].max(u[1])
                    };
                    let touching = q.data == i
                        || [c, e].iter().any(|p| *p == a || *p == b)
                        || inside(a, c, e)
                        || inside(b, c, e)
                        || inside(c, a, b)
                        || inside(e, a, b);
                    let collinear = orient(a, b, c) == 0.0 && orient(a, b, e) == 0.0;
                    if touching || collinear || d == 0.0 {
                        continue;
                    }
                    // Across, not along: the way to the other segment makes more than 45 degrees
                    // with this one (the next segments of a polygonized curve lie along it).
                    let y = q.geom().nearest_point(&x);
                    let across = sub(y, x);
                    if dot(across, direction).abs() > std::f64::consts::FRAC_1_SQRT_2 * d * length {
                        continue;
                    }
                    let cap = (FEATURE * d).max(floor);
                    if cap >= field {
                        break;
                    }
                    // Not joined to `x` along the boundary within `PATH d` (a corner, no gap).
                    let [from_a, from_b] = walks
                        .get_or_insert_with(|| boundary.ends[i].map(|v| boundary.walk(v, limit)));
                    let path = boundary.ends[q.data]
                        .iter()
                        .zip([c, e])
                        .map(|(&v, p)| {
                            let along = (distance(x, a) + reached(from_a, v))
                                .min(distance(x, b) + reached(from_b, v));
                            along + distance(p, y)
                        })
                        .fold(f64::INFINITY, f64::min);
                    if path <= PATH * d {
                        continue;
                    }
                    gaps.push(GeomWithData::new(x, cap));
                    break;
                }
            }
        }
        (!gaps.is_empty()).then(|| RTree::bulk_load(gaps))
    }

    /// The size field without the cap.
    fn field(&self, x: [f64; 2]) -> f64 {
        self.fields
            .iter()
            .map(|f| f.at(x))
            .fold(f64::INFINITY, f64::min)
    }

    /// The edge length at `x`.
    pub fn at(&self, x: [f64; 2]) -> f64 {
        let field = self.field(x);
        let Some(gaps) = &self.gaps else {
            return field;
        };
        // The gap points by distance, until none can give less: every cap is at least
        // `gap_min` (the slack keeps a point whose distance rounds differently).
        let mut best = field;
        for (g, d2) in gaps.nearest_neighbor_iter_with_distance_2(x) {
            if self.gap_min + GRADING * d2.sqrt() > best * (1.0 + 1e-12) {
                break;
            }
            best = best.min(g.data + GRADING * distance(x, *g.geom()));
        }
        best
    }

    /// The smallest size anywhere.
    pub(crate) fn min(&self) -> f64 {
        self.min.min(self.gap_min)
    }
}
