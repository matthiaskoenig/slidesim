//! The size field: the edge length as a function of the distance to the nearest wall.

use contracts::SizeField;
use rstar::primitives::Line;
use rstar::{PointDistance, RTree};

use crate::{Domain, MesherError};

/// The edge length in um at a point: `min` up to the distance `dist_min` from the nearest
/// segment of a wall chain, linear up to `max` at `dist_max`, `max` beyond (gmsh's Threshold
/// field). Without walls, `max` everywhere. The distance comes from an R-tree of the wall
/// segments.
#[derive(Clone, Debug)]
pub struct Size {
    min: f64,
    max: f64,
    dist_min: f64,
    dist_max: f64,
    walls: Option<RTree<Line<[f64; 2]>>>,
}

impl Size {
    /// The size field `spec` with the distance to the chains of the edge tags `spec.walls` of
    /// `domain`.
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
        let mut segments = Vec::new();
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
                    segments.extend(
                        points
                            .windows(2)
                            .filter(|w| w[0] != w[1])
                            .map(|w| Line::new(w[0], w[1])),
                    );
                }
            }
        }
        Ok(Size {
            min: spec.min_um,
            max: spec.max_um,
            dist_min: spec.dist_min_um,
            dist_max: spec.dist_max_um,
            walls: (!segments.is_empty()).then(|| RTree::bulk_load(segments)),
        })
    }

    /// The edge length at `x`.
    pub fn at(&self, x: [f64; 2]) -> f64 {
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

    /// The smallest size anywhere.
    pub(crate) fn min(&self) -> f64 {
        self.min
    }
}
