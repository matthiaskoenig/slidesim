//! The raster pyramid of a result (`raster/<field>/<level>` of `result/1`): the P1 fields of
//! the canonical mesh sampled at pixel centres, for a viewer which overlays results on a slide
//! image.
//!
//! - Level 0 covers the bounding box of the mesh from the origin (default: the lower corner of
//!   the box): `cols = ceil((x_max - x0) / h)` and `rows = ceil((y_max - y0) / h)`, at least 1
//!   each, with `h` the pixel size. Row 0 is at the smallest y. Pixel `(r, c)` has its centre at
//!   `(x0 + (c + 0.5) h, y0 + (r + 0.5) h)` and the value of the field there, NaN outside the
//!   mesh. With an origin right of or above the lower corner of the mesh the pixels cover only
//!   from the origin on.
//! - Level k+1 has `ceil(rows / 2)` x `ceil(cols / 2)` pixels, each the mean of the valid
//!   pixels of its block of 2 x 2 of level k (the block at an edge has the pixels it has), NaN
//!   if it has none. Levels past 1 x 1 stay 1 x 1.
//! - A pixel centre on a shared edge or vertex takes the first containing triangle in the order
//!   of the mesh; the field is continuous, so any gives the same value. A centre counts as
//!   inside a triangle when its barycentric coordinates are at least `-1e-9`.
//!
//! The pixels of the triangles are found by scanning the pixels of the bounding box of each
//! triangle (the cost is the number of pixels plus the overlaps of the boxes), once, when the
//! [`Raster`] is made; it keeps the triangle of each pixel (4 bytes per pixel), so each output
//! time costs a barycentric evaluation per pixel.

use anyhow::{Result, ensure};
use contracts::RasterMeta;
use mesh::CanonicalMesh;

/// The most pixels of level 0 a raster may have.
pub const MAX_PIXELS: u64 = 1 << 28;

/// Marks a pixel outside the mesh in [`Raster::triangle`].
const OUTSIDE: u32 = u32::MAX;

/// Slack of the inside test, in barycentric coordinates (relative to the triangle's size).
const TOLERANCE: f64 = 1e-9;

/// The pixels of a raster pyramid of a mesh and the triangle at each pixel centre of level 0.
#[derive(Debug, Clone)]
pub struct Raster {
    um_per_px: f64,
    origin_um: [f64; 2],
    shapes: Vec<[u64; 2]>,
    /// The triangle at the centre of each pixel of level 0, row by row; [`OUTSIDE`] if none.
    triangle: Vec<u32>,
    /// The triangles' vertex indices and the vertices of the mesh.
    triangles: Vec<[u32; 3]>,
    vertices: Vec<[f64; 2]>,
}

impl Raster {
    /// The pyramid of `levels` levels (at least 1) of pixels of `um_per_px` um over `mesh`, its
    /// corner of pixel (0, 0) at `origin_um` (default: the lower corner of the bounding box of
    /// the mesh).
    ///
    /// Errors: `um_per_px` not finite and > 0, `levels` 0, `origin_um` not finite, a mesh
    /// without triangles, or more than [`MAX_PIXELS`] pixels in level 0 (the message names
    /// `result.raster.um_per_px`).
    pub fn new(
        mesh: &CanonicalMesh,
        um_per_px: f64,
        origin_um: Option<[f64; 2]>,
        levels: u32,
    ) -> Result<Raster> {
        ensure!(
            um_per_px.is_finite() && um_per_px > 0.0,
            "result.raster.um_per_px must be finite and > 0, got {um_per_px}"
        );
        ensure!(levels >= 1, "result.raster.levels must be >= 1");
        ensure!(
            origin_um.is_none_or(|o| o.iter().all(|x| x.is_finite())),
            "result.raster.origin_um must be finite"
        );
        ensure!(
            !mesh.triangles.is_empty(),
            "the raster needs a mesh with triangles"
        );
        let mut low = [f64::INFINITY; 2];
        let mut high = [f64::NEG_INFINITY; 2];
        for v in &mesh.vertices {
            for d in 0..2 {
                low[d] = low[d].min(v[d]);
                high[d] = high[d].max(v[d]);
            }
        }
        let origin_um = origin_um.unwrap_or(low);
        let [rows, cols] = Self::level0_shape([low, high], um_per_px, origin_um)?;
        let mut shapes = vec![[rows, cols]];
        for _ in 1..levels {
            let [r, c] = shapes[shapes.len() - 1];
            shapes.push([r.div_ceil(2), c.div_ceil(2)]);
        }
        let mut raster = Raster {
            um_per_px,
            origin_um,
            shapes,
            triangle: vec![OUTSIDE; (rows * cols) as usize],
            triangles: mesh.triangles.clone(),
            vertices: mesh.vertices.clone(),
        };
        raster.locate();
        Ok(raster)
    }

    /// The `[rows, cols]` of level 0 over the bounding box `bounds` (`[[xmin, ymin], [xmax,
    /// ymax]]`) of a mesh, or the refusal of a raster of more than [`MAX_PIXELS`] pixels (the
    /// message names `result.raster.um_per_px`). `origin_um` is the corner of pixel (0, 0). A
    /// cheap check which needs no mesh: a run makes it before it writes anything.
    pub fn level0_shape(
        bounds: [[f64; 2]; 2],
        um_per_px: f64,
        origin_um: [f64; 2],
    ) -> Result<[u64; 2]> {
        let count = |d: usize| ((bounds[1][d] - origin_um[d]) / um_per_px).ceil().max(1.0);
        let (cols, rows) = (count(0), count(1));
        ensure!(
            rows * cols <= MAX_PIXELS as f64,
            "result.raster.um_per_px: the raster of level 0 would have {rows} x {cols} pixels, \
             more than {MAX_PIXELS} (2^28); use a larger um_per_px"
        );
        Ok([rows as u64, cols as u64])
    }

    /// The raster attributes of `result/1`.
    pub fn meta(&self) -> RasterMeta {
        RasterMeta {
            um_per_px: self.um_per_px,
            origin_um: self.origin_um,
            levels: self.shapes.len() as u32,
            shapes: self.shapes.clone(),
        }
    }

    /// Finds the first containing triangle of each pixel centre of level 0.
    fn locate(&mut self) {
        let [rows, cols] = self.shapes[0];
        let (rows, cols) = (rows as usize, cols as usize);
        let h = self.um_per_px;
        // The pixel indices whose centre lies in [low, high] along one axis, clipped to `n`.
        let span = |low: f64, high: f64, origin: f64, n: usize| {
            let first = ((low - origin) / h - 0.5).ceil().max(0.0);
            let last = ((high - origin) / h - 0.5).floor().min(n as f64 - 1.0);
            (first <= last).then_some((first as usize, last as usize))
        };
        for (t, tri) in self.triangles.iter().enumerate() {
            let p = tri.map(|i| self.vertices[i as usize]);
            let slack = TOLERANCE * h.max(1e-300);
            let bound = |d: usize, f: fn(f64, f64) -> f64, init: f64, s: f64| {
                p.iter().fold(init, |m, q| f(m, q[d])) + s
            };
            let (x_low, x_high) = (
                bound(0, f64::min, f64::INFINITY, -slack),
                bound(0, f64::max, f64::NEG_INFINITY, slack),
            );
            let (y_low, y_high) = (
                bound(1, f64::min, f64::INFINITY, -slack),
                bound(1, f64::max, f64::NEG_INFINITY, slack),
            );
            let (Some((c0, c1)), Some((r0, r1))) = (
                span(x_low, x_high, self.origin_um[0], cols),
                span(y_low, y_high, self.origin_um[1], rows),
            ) else {
                continue;
            };
            for r in r0..=r1 {
                let y = self.origin_um[1] + (r as f64 + 0.5) * h;
                for c in c0..=c1 {
                    let pixel = &mut self.triangle[r * cols + c];
                    if *pixel != OUTSIDE {
                        continue;
                    }
                    let x = self.origin_um[0] + (c as f64 + 0.5) * h;
                    if let Some(w) = barycentric(&p, [x, y])
                        && w.iter().all(|&w| w >= -TOLERANCE)
                    {
                        *pixel = t as u32;
                    }
                }
            }
        }
    }

    /// The pyramid of the P1 field of the nodal values `nodal` (one per mesh vertex): the pixels
    /// of each level, row by row, level 0 first.
    ///
    /// Panics if `nodal` is not one value per vertex.
    pub fn pyramid(&self, nodal: &[f64]) -> Vec<Vec<f32>> {
        assert_eq!(nodal.len(), self.vertices.len(), "one value per vertex");
        let mut level: Vec<f32> = self
            .triangle
            .iter()
            .enumerate()
            .map(|(k, &t)| {
                if t == OUTSIDE {
                    return f32::NAN;
                }
                let cols = self.shapes[0][1] as usize;
                let (r, c) = (k / cols, k % cols);
                let x = [
                    self.origin_um[0] + (c as f64 + 0.5) * self.um_per_px,
                    self.origin_um[1] + (r as f64 + 0.5) * self.um_per_px,
                ];
                let tri = self.triangles[t as usize];
                let p = tri.map(|i| self.vertices[i as usize]);
                let w = barycentric(&p, x).expect("a located triangle has an area");
                let v: f64 = (0..3).map(|i| w[i] * nodal[tri[i] as usize]).sum();
                v as f32
            })
            .collect();
        let mut levels = Vec::with_capacity(self.shapes.len());
        for k in 1..self.shapes.len() {
            let next = coarsen(&level, self.shapes[k - 1], self.shapes[k]);
            levels.push(std::mem::replace(&mut level, next));
        }
        levels.push(level);
        levels
    }
}

/// The barycentric coordinates of `x` in the triangle `p`, `None` if it has no area.
fn barycentric(p: &[[f64; 2]; 3], x: [f64; 2]) -> Option<[f64; 3]> {
    let area =
        (p[1][0] - p[0][0]) * (p[2][1] - p[0][1]) - (p[2][0] - p[0][0]) * (p[1][1] - p[0][1]);
    if area == 0.0 {
        return None;
    }
    let cross =
        |a: [f64; 2], b: [f64; 2]| (b[0] - a[0]) * (x[1] - a[1]) - (x[0] - a[0]) * (b[1] - a[1]);
    let w0 = cross(p[1], p[2]) / area;
    let w1 = cross(p[2], p[0]) / area;
    Some([w0, w1, 1.0 - w0 - w1])
}

/// The level of `to` pixels from the level `from` (row by row): the mean of the valid pixels of
/// each block of 2 x 2, NaN without one.
fn coarsen(level: &[f32], from: [u64; 2], to: [u64; 2]) -> Vec<f32> {
    let [rows, cols] = from.map(|n| n as usize);
    let [next_rows, next_cols] = to.map(|n| n as usize);
    let mut next = vec![f32::NAN; next_rows * next_cols];
    for r in 0..next_rows {
        for c in 0..next_cols {
            let (mut sum, mut n) = (0.0_f64, 0_u32);
            for rr in 2 * r..(2 * r + 2).min(rows) {
                for cc in 2 * c..(2 * c + 2).min(cols) {
                    let v = level[rr * cols + cc];
                    if !v.is_nan() {
                        sum += f64::from(v);
                        n += 1;
                    }
                }
            }
            if n > 0 {
                next[r * next_cols + c] = (sum / f64::from(n)) as f32;
            }
        }
    }
    next
}
