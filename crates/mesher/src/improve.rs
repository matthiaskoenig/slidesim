//! A local improvement of an adapted mesh against the quality bounds: smoothing, edge collapses
//! and edge flips around the elements below a margin above the bounds.
//!
//! Mmg aims at edge lengths, not at angles, and leaves a few elements below the bounds next to
//! frozen boundaries: an interior vertex a fraction of the size away from a wall, which Mmg can
//! neither collapse onto the frozen wall nor move. [`improve`] visits the vertices of the elements
//! which score below [`GOOD`] (by their smallest angle, SICN and edge ratios against the size
//! field, measured as [`Quality`](crate::Quality) measures them and scaled by the bounds: an angle
//! of 20 degrees, a SICN of 0.4, edge ratios within [0.4, 2.0] score 1) and tries, in this order,
//! to move a free vertex towards the centroid of its neighbours, to collapse a free vertex whose
//! elements still score below [`GOOD`] onto a neighbour, and to flip the free edges of its
//! triangles. An operation is applied if the worst of the elements it creates scores better than
//! the worst of the elements it replaces, so the worst element of the mesh never gets worse.
//! Nothing guarantees that every element ends above the bounds (a vertex may have no improving
//! operation, and [`BUDGET`] bounds the work): the quality bounds of `generate` rest on this
//! step reaching them, which the quality sweeps of the tests check.
//!
//! The vertices of the boundary, of the tagged edges and of the interfaces between area tags
//! (frozen) never move nor go, and their edges are never flipped, so every area keeps its
//! polygons exactly (an operation retriangulates the star of a vertex or the quadrilateral of an
//! edge, which stays inside one area tag) and every tagged edge stays. Vertices are visited in
//! the order of their indices, so the result is deterministic.

use std::collections::BTreeSet;

use crate::tagged::{canonical_triangle, twice_area};
use crate::{Size, TaggedMesh};

/// The score below which the elements of a vertex are worked on: an angle of 23 degrees, a SICN
/// of 0.46, edge ratios within [0.46, 1.74]; a margin above the bounds, so that the mesh changes
/// near its worst elements only.
const GOOD: f64 = 1.15;

/// The steps tried towards the centroid of the neighbours, as fractions of the way.
const STEPS: [f64; 3] = [1.0, 0.5, 0.25];

/// The most operations per vertex visited at the start: a bound on the work which the strict
/// improvements make unreachable in practice.
const BUDGET: usize = 20;

fn sub(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    [a[0] - b[0], a[1] - b[1]]
}

fn dot(a: [f64; 2], b: [f64; 2]) -> f64 {
    a[0] * b[0] + a[1] * b[1]
}

/// The score of a triangle by its smallest angle and its SICN against the bounds; `None` unless
/// it is counterclockwise.
fn triangle_score(p: [[f64; 2]; 3]) -> Option<f64> {
    let area = twice_area(p[0], p[1], p[2]);
    if area.is_nan() || area <= 0.0 {
        return None;
    }
    let mut min_angle = f64::INFINITY;
    let mut squares = 0.0;
    for i in 0..3 {
        let (a, b, c) = (p[i], p[(i + 1) % 3], p[(i + 2) % 3]);
        let (u, v) = (sub(b, a), sub(c, a));
        min_angle = min_angle.min(area.atan2(dot(u, v)).to_degrees());
        squares += dot(u, u);
    }
    let sicn = 2.0 * 3f64.sqrt() * area / squares;
    Some((min_angle / 20.0).min(sicn / 0.4))
}

/// The mesh being improved: triangles which may die, vertices which may go, the triangles of
/// every vertex.
struct Work<'a> {
    size: &'a Size,
    vertices: Vec<[f64; 2]>,
    triangles: Vec<[u32; 3]>,
    alive: Vec<bool>,
    gone: Vec<bool>,
    /// The live triangles of every vertex, sorted.
    of: Vec<Vec<usize>>,
    frozen: Vec<bool>,
    /// The boundary edges, the tagged edges and the interfaces, sorted.
    frozen_edges: Vec<[u32; 2]>,
}

impl Work<'_> {
    fn point(&self, v: u32) -> [f64; 2] {
        self.vertices[v as usize]
    }

    fn edge_score(&self, a: [f64; 2], b: [f64; 2]) -> f64 {
        let mid = [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5];
        let ratio = (b[0] - a[0]).hypot(b[1] - a[1]) / self.size.at(mid);
        (ratio / 0.4).min(2.0 / ratio)
    }

    fn score_of(&self, t: [u32; 3]) -> f64 {
        triangle_score(t.map(|v| self.point(v))).unwrap_or(f64::NEG_INFINITY)
    }

    fn edge_frozen(&self, a: u32, b: u32) -> bool {
        self.frozen_edges
            .binary_search(&[a.min(b), a.max(b)])
            .is_ok()
    }

    /// The neighbours of `v`, sorted.
    fn neighbours(&self, v: u32) -> Vec<u32> {
        let mut w: Vec<u32> = self.of[v as usize]
            .iter()
            .flat_map(|&t| self.triangles[t])
            .filter(|&w| w != v)
            .collect();
        w.sort_unstable();
        w.dedup();
        w
    }

    /// The worst score of the triangles and edges of `v`.
    fn vertex_score(&self, v: u32) -> f64 {
        let x = self.point(v);
        let triangles = self.of[v as usize]
            .iter()
            .map(|&t| self.score_of(self.triangles[t]));
        let edges = self
            .neighbours(v)
            .into_iter()
            .map(|w| self.edge_score(x, self.point(w)));
        triangles.chain(edges).fold(f64::INFINITY, f64::min)
    }

    /// Moves the free vertex `v` towards the centroid of its neighbours if that improves it.
    fn smooth(&mut self, v: u32) -> bool {
        let neighbours = self.neighbours(v);
        let x = self.point(v);
        let before = self.vertex_score(v);
        let k = neighbours.len() as f64;
        let centroid = neighbours.iter().fold([0.0, 0.0], |c, &w| {
            let p = self.point(w);
            [c[0] + p[0] / k, c[1] + p[1] / k]
        });
        for step in STEPS {
            let y = [
                x[0] + step * (centroid[0] - x[0]),
                x[1] + step * (centroid[1] - x[1]),
            ];
            let at = |w: u32| if w == v { y } else { self.point(w) };
            let triangles = self.of[v as usize]
                .iter()
                .map(|&t| triangle_score(self.triangles[t].map(at)).unwrap_or(f64::NEG_INFINITY));
            let edges = neighbours
                .iter()
                .map(|&w| self.edge_score(y, self.point(w)));
            let after = triangles.chain(edges).fold(f64::INFINITY, f64::min);
            if after > before {
                self.vertices[v as usize] = y;
                return true;
            }
        }
        false
    }

    /// Collapses the free vertex `v` onto the neighbour which gives the best elements, if they
    /// are better than the elements of `v`. Returns the neighbour.
    fn collapse(&mut self, v: u32) -> Option<u32> {
        let neighbours = self.neighbours(v);
        let x = self.point(v);
        let before = self.of[v as usize]
            .iter()
            .map(|&t| self.score_of(self.triangles[t]))
            .chain(
                neighbours
                    .iter()
                    .map(|&w| self.edge_score(x, self.point(w))),
            )
            .fold(f64::INFINITY, f64::min);
        let mut best: Option<(f64, u32)> = None;
        for &w in &neighbours {
            let shared: Vec<usize> = self.of[v as usize]
                .iter()
                .copied()
                .filter(|&t| self.triangles[t].contains(&w))
                .collect();
            let around_w = self.neighbours(w);
            let common = neighbours
                .iter()
                .filter(|u| around_w.binary_search(u).is_ok())
                .count();
            // An interior edge with two triangles, whose ends share their two opposite vertices
            // only (else the collapse folds the mesh).
            if shared.len() != 2 || common != 2 {
                continue;
            }
            let replaced = |t: [u32; 3]| t.map(|u| if u == v { w } else { u });
            let triangles = self.of[v as usize]
                .iter()
                .filter(|t| !shared.contains(t))
                .map(|&t| self.score_of(replaced(self.triangles[t])));
            let p = self.point(w);
            let edges = neighbours
                .iter()
                .filter(|&&u| u != w && around_w.binary_search(&u).is_err())
                .map(|&u| self.edge_score(p, self.point(u)));
            let after = triangles.chain(edges).fold(f64::INFINITY, f64::min);
            if after > before && best.is_none_or(|(b, _)| after > b) {
                best = Some((after, w));
            }
        }
        let (_, w) = best?;
        for t in std::mem::take(&mut self.of[v as usize]) {
            if self.triangles[t].contains(&w) {
                self.alive[t] = false;
                for u in self.triangles[t] {
                    self.of[u as usize].retain(|&s| s != t);
                }
            } else {
                self.triangles[t] = self.triangles[t].map(|u| if u == v { w } else { u });
                let of_w = &mut self.of[w as usize];
                if let Err(i) = of_w.binary_search(&t) {
                    of_w.insert(i, t);
                }
            }
        }
        self.gone[v as usize] = true;
        Some(w)
    }

    /// Flips the free edge `a`, `b` if both its triangles and the edge improve. Returns the
    /// opposite vertices.
    fn flip(&mut self, a: u32, b: u32) -> Option<[u32; 2]> {
        if self.edge_frozen(a, b) {
            return None;
        }
        let shared: Vec<usize> = self.of[a as usize]
            .iter()
            .copied()
            .filter(|&t| self.triangles[t].contains(&b))
            .collect();
        let &[s, t] = shared.as_slice() else {
            return None;
        };
        // `s` with `a -> b` counterclockwise, `c` left of it; `d` the third vertex of `t`.
        let (s, t) = {
            let k = (0..3).find(|&k| self.triangles[s][k] == a).unwrap();
            if self.triangles[s][(k + 1) % 3] == b {
                (s, t)
            } else {
                (t, s)
            }
        };
        let third = |t: usize| {
            *self.triangles[t]
                .iter()
                .find(|&&u| u != a && u != b)
                .unwrap()
        };
        let (c, d) = (third(s), third(t));
        if self.neighbours(c).binary_search(&d).is_ok() {
            return None;
        }
        let (n1, n2) = ([c, a, d], [c, d, b]);
        let before = self
            .score_of(self.triangles[s])
            .min(self.score_of(self.triangles[t]))
            .min(self.edge_score(self.point(a), self.point(b)));
        let after = self
            .score_of(n1)
            .min(self.score_of(n2))
            .min(self.edge_score(self.point(c), self.point(d)));
        if after <= before {
            return None;
        }
        self.triangles[s] = n1;
        self.triangles[t] = n2;
        let insert = |of: &mut Vec<usize>, x: usize| {
            if let Err(i) = of.binary_search(&x) {
                of.insert(i, x);
            }
        };
        self.of[a as usize].retain(|&x| x != t);
        self.of[b as usize].retain(|&x| x != s);
        insert(&mut self.of[c as usize], t);
        insert(&mut self.of[d as usize], s);
        Some([c, d])
    }
}

/// Improves the elements of `mesh` below a margin above the quality bounds (see the module).
pub(crate) fn improve(mesh: &mut TaggedMesh, size: &Size) {
    let n = mesh.vertices.len();
    // The edges with their triangles, sorted: boundary edges, interfaces and tagged edges are
    // frozen, and so are their ends.
    let mut sides: Vec<([u32; 2], usize)> = Vec::with_capacity(3 * mesh.triangles.len());
    for (i, t) in mesh.triangles.iter().enumerate() {
        for k in 0..3 {
            let (a, b) = (t[k], t[(k + 1) % 3]);
            sides.push(([a.min(b), a.max(b)], i));
        }
    }
    sides.sort_unstable();
    let mut frozen_edges: Vec<[u32; 2]> = mesh.edges.clone();
    let mut i = 0;
    while i < sides.len() {
        let e = sides[i].0;
        let j = (i..sides.len())
            .find(|&j| sides[j].0 != e)
            .unwrap_or(sides.len());
        if j - i != 2 || mesh.triangle_tags[sides[i].1] != mesh.triangle_tags[sides[i + 1].1] {
            frozen_edges.push(e);
        }
        i = j;
    }
    frozen_edges.sort_unstable();
    frozen_edges.dedup();
    let mut frozen = vec![false; n];
    for e in &frozen_edges {
        e.iter().for_each(|&v| frozen[v as usize] = true);
    }
    let mut of: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (i, t) in mesh.triangles.iter().enumerate() {
        t.iter().for_each(|&v| of[v as usize].push(i));
    }
    let mut work = Work {
        size,
        vertices: mesh.vertices.clone(),
        triangles: mesh.triangles.clone(),
        alive: vec![true; mesh.triangles.len()],
        gone: vec![false; n],
        of,
        frozen,
        frozen_edges,
    };

    // The vertices of the elements below `GOOD`.
    let mut queue: BTreeSet<u32> = BTreeSet::new();
    for t in &work.triangles {
        if work.score_of(*t) < GOOD {
            queue.extend(t);
        }
    }
    let mut edges: Vec<[u32; 2]> = sides.iter().map(|s| s.0).collect();
    edges.dedup();
    for [a, b] in edges {
        if work.edge_score(work.point(a), work.point(b)) < GOOD {
            queue.extend([a, b]);
        }
    }

    let mut budget = BUDGET * queue.len();
    while let Some(v) = queue.pop_first() {
        if budget == 0 {
            break;
        }
        if work.gone[v as usize] || work.vertex_score(v) >= GOOD {
            continue;
        }
        let free = !work.frozen[v as usize];
        let mut touched: Vec<u32> = Vec::new();
        if free && work.smooth(v) {
            touched = work.neighbours(v);
            touched.push(v);
        } else if free
            && work.vertex_score(v) < GOOD
            && let Some(w) = work.collapse(v)
        {
            touched = work.neighbours(w);
            touched.push(w);
        }
        if touched.is_empty() {
            // The edges of the triangles of `v`.
            let triangles = work.of[v as usize].clone();
            'flips: for t in triangles {
                if !work.alive[t] {
                    continue;
                }
                for k in 0..3 {
                    let (a, b) = (work.triangles[t][k], work.triangles[t][(k + 1) % 3]);
                    if let Some([c, d]) = work.flip(a, b) {
                        touched = vec![a, b, c, d];
                        break 'flips;
                    }
                }
            }
        }
        if !touched.is_empty() {
            budget -= 1;
            queue.extend(touched);
        }
    }

    // The live triangles and the remaining vertices, renumbered in their order.
    let mut index = vec![u32::MAX; n];
    let mut vertices = Vec::with_capacity(n);
    for v in 0..n {
        if !work.gone[v] {
            index[v] = vertices.len() as u32;
            vertices.push(work.vertices[v]);
        }
    }
    let (triangles, triangle_tags) = (0..work.triangles.len())
        .filter(|&t| work.alive[t])
        .map(|t| {
            (
                canonical_triangle(work.triangles[t].map(|v| index[v as usize])),
                mesh.triangle_tags[t],
            )
        })
        .unzip();
    mesh.vertices = vertices;
    mesh.triangles = triangles;
    mesh.triangle_tags = triangle_tags;
    for e in &mut mesh.edges {
        *e = e.map(|v| index[v as usize]);
    }
}
