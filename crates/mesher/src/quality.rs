//! The quality of a triangle mesh: angles, SICN and the edge lengths against the size field.

use serde::Serialize;

use crate::tagged::twice_area;
use crate::{Size, TaggedMesh};

/// The quality of a mesh: the counts, the smallest angle in degrees, the smallest and mean SICN
/// (`4 sqrt(3) A / (l1^2 + l2^2 + l3^2)`, gmsh's for straight triangles: 1 for an equilateral
/// one, 0 for a degenerate one, negative if clockwise) and, against a size field, the smallest
/// and largest ratio `l / h(midpoint)` over the edges.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Quality {
    pub triangles: usize,
    pub vertices: usize,
    pub min_angle_deg: f64,
    pub min_sicn: f64,
    pub mean_sicn: f64,
    pub edge_ratio: Option<[f64; 2]>,
}

fn sub(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    [a[0] - b[0], a[1] - b[1]]
}

fn dot(a: [f64; 2], b: [f64; 2]) -> f64 {
    a[0] * b[0] + a[1] * b[1]
}

impl TaggedMesh {
    /// The quality of the mesh, with the edge ratios against `size` if given.
    pub fn quality(&self, size: Option<&Size>) -> Quality {
        let mut min_angle = f64::INFINITY;
        let mut min_sicn = f64::INFINITY;
        let mut sum_sicn = 0.0;
        let mut edges: Vec<[u32; 2]> = Vec::with_capacity(3 * self.triangles.len());
        for t in &self.triangles {
            let p = t.map(|v| self.vertices[v as usize]);
            let mut squares = 0.0;
            for i in 0..3 {
                let (a, b, c) = (p[i], p[(i + 1) % 3], p[(i + 2) % 3]);
                let (u, v) = (sub(b, a), sub(c, a));
                let angle = twice_area(a, b, c).abs().atan2(dot(u, v)).to_degrees();
                min_angle = min_angle.min(angle);
                squares += dot(u, u);
                let (x, y) = (t[i], t[(i + 1) % 3]);
                edges.push([x.min(y), x.max(y)]);
            }
            let sicn = 2.0 * 3f64.sqrt() * twice_area(p[0], p[1], p[2]) / squares;
            min_sicn = min_sicn.min(sicn);
            sum_sicn += sicn;
        }
        let edge_ratio = size.map(|size| {
            edges.sort_unstable();
            edges.dedup();
            edges.iter().fold([f64::INFINITY, 0.0], |[lo, hi], e| {
                let (a, b) = (self.vertices[e[0] as usize], self.vertices[e[1] as usize]);
                let mid = [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5];
                let ratio = (b[0] - a[0]).hypot(b[1] - a[1]) / size.at(mid);
                [lo.min(ratio), hi.max(ratio)]
            })
        });
        Quality {
            triangles: self.triangles.len(),
            vertices: self.vertices.len(),
            min_angle_deg: min_angle,
            min_sicn,
            mean_sicn: sum_sicn / self.triangles.len() as f64,
            edge_ratio,
        }
    }
}
