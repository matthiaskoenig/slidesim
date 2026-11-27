//! The size field: the edge length as a function of the distance to the nearest wall, capped by
//! the width of narrow gaps.

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

/// The edge length in um at a point: `min` up to the distance `dist_min` from the nearest
/// segment of a wall chain, linear up to `max` at `dist_max`, `max` beyond (gmsh's Threshold
/// field); without walls, `max` everywhere. The distance comes from an R-tree of the wall
/// segments.
///
/// The field is capped by the local width of the domain: at a point of a ring or chain whose
/// nearest other segment across (not touching it, not collinear with it, not along it) is at the
/// distance `w`, the size is at most `0.4 w` (but not below `min / 4`), growing by 0.3 per um
/// away from it. Narrow necks of tissue and chains close to each other so get elements which fit
/// between their frozen boundaries.
#[derive(Clone, Debug)]
pub struct Size {
    min: f64,
    max: f64,
    dist_min: f64,
    dist_max: f64,
    walls: Option<RTree<Line<[f64; 2]>>>,
    /// The boundary points where the cap is below the field, with the cap.
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

impl Size {
    /// The size field `spec` with the distance to the chains of the edge tags `spec.walls` of
    /// `domain`, capped by the widths of the gaps of `domain`.
    ///
    /// Errors if a wall is no edge tag of the domain, or the sizes and distances are not finite,
    /// positive and ordered (`min <= max`, `dist_min < dist_max`).
    pub fn new(spec: &SizeField, domain: &Domain) -> Result<Size, MesherError> {
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
        let mut size = Size {
            min: spec.min_um,
            max: spec.max_um,
            dist_min: spec.dist_min_um,
            dist_max: spec.dist_max_um,
            walls: (!walls.is_empty()).then(|| RTree::bulk_load(walls)),
            gaps: None,
            gap_min: f64::INFINITY,
        };
        size.gaps = size.gaps(domain);
        if let Some(gaps) = &size.gaps {
            size.gap_min = gaps.iter().map(|g| g.data).fold(f64::INFINITY, f64::min);
        }
        Ok(size)
    }

    /// The points of the rings and chains (every `min / 2` along each segment) where the cap is
    /// below the field, with the cap.
    fn gaps(&self, domain: &Domain) -> Option<RTree<GeomWithData<[f64; 2], f64>>> {
        let segments = segments(domain);
        let tree = RTree::bulk_load(
            segments
                .iter()
                .enumerate()
                .map(|(i, s)| GeomWithData::new(Line::new(s[0], s[1]), i))
                .collect(),
        );
        let orient = |a: [f64; 2], b: [f64; 2], c: [f64; 2]| {
            let coord = |p: [f64; 2]| robust::Coord { x: p[0], y: p[1] };
            robust::orient2d(coord(a), coord(b), coord(c))
        };
        let floor = FLOOR * self.min;
        let mut gaps = Vec::new();
        for (i, &[a, b]) in segments.iter().enumerate() {
            let length = distance(a, b);
            let direction = sub(b, a);
            let n = (2.0 * length / self.min).ceil().max(1.0) as usize;
            for k in 0..=n {
                let t = k as f64 / n as f64;
                let x = [a[0] + t * direction[0], a[1] + t * direction[1]];
                let field = self.field(x);
                for (q, d2) in tree.nearest_neighbor_iter_with_distance_2(x) {
                    let d = d2.sqrt();
                    if FEATURE * d >= field {
                        break;
                    }
                    let [c, e] = [q.geom().from, q.geom().to];
                    let touching = q.data == i || [c, e].iter().any(|p| *p == a || *p == b);
                    let collinear = orient(a, b, c) == 0.0 && orient(a, b, e) == 0.0;
                    if touching || collinear {
                        continue;
                    }
                    // Across, not along: the way to the other segment makes more than 45 degrees
                    // with this one (the next segments of a polygonized curve lie along it).
                    let across = sub(q.geom().nearest_point(&x), x);
                    if dot(across, direction).abs() > std::f64::consts::FRAC_1_SQRT_2 * d * length {
                        continue;
                    }
                    let cap = (FEATURE * d).max(floor);
                    if cap < field {
                        gaps.push(GeomWithData::new(x, cap));
                    }
                    break;
                }
            }
        }
        (!gaps.is_empty()).then(|| RTree::bulk_load(gaps))
    }

    /// The size field without the cap.
    fn field(&self, x: [f64; 2]) -> f64 {
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

    /// The edge length at `x`.
    pub fn at(&self, x: [f64; 2]) -> f64 {
        let field = self.field(x);
        let Some(gaps) = &self.gaps else {
            return field;
        };
        let reach = (field - self.gap_min) / GRADING;
        if reach <= 0.0 {
            return field;
        }
        gaps.locate_within_distance(x, reach * reach)
            .map(|g| g.data + GRADING * distance(x, *g.geom()))
            .fold(field, f64::min)
    }

    /// The smallest size anywhere.
    pub(crate) fn min(&self) -> f64 {
        self.min.min(self.gap_min)
    }
}
