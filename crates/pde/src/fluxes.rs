//! Equilibrated edge fluxes of a P1 Darcy pressure, and their lowest-order Raviart-Thomas
//! velocity.
//!
//! The P1 Darcy velocity `u_h = -D grad p_h` (`D = K / mu`) is not conservative: its normal
//! component jumps between the cells. The fluxes here are, as in the equilibration of Ladeveze
//! and of Ainsworth and Oden (and the conservative flux of Larson and Niklasson 2004), consistent
//! with the residual of the discrete equations, vertex patch by vertex patch.
//!
//! For a cell `K` and a vertex `z` of it, with `v_z` the P1 function of `z`, the residual of the
//! equation `div u = f` tested with `v_z` on `K` is
//!
//! `r_K(z) = int_K f v_z - int_K D grad p_h . grad v_z`
//!
//! (by the quadrature of the discretization, as the operator of the pressure takes it). The flux
//! through an edge `E` is the sum of two moments, one per end: `F_E = mu_(E,a) + mu_(E,b)`, and
//! the moments of each vertex `z` satisfy, for every cell `K` around `z`,
//!
//! `sum over the two edges E of K at z of mu_(E,z) (out of K) = r_K(z)`.
//!
//! Summed over the three vertices of `K` (`sum v_z = 1`, `sum grad v_z = 0`), the fluxes out of
//! `K` add up to `int_K f`: zero for the Darcy problem, which has no sources. The equations of a
//! vertex involve only the edges at it, a chain of cells from one edge of the boundary to another
//! (a vertex of the boundary) or a cycle (an inner vertex), which a sweep around the vertex
//! solves: the moment through each edge is that of the edge before plus the residual of the cell
//! between them. What is left free is chosen closest to the target moments
//! `(u_K . n) |E| / 2` of the cell mean `u_K` of the primal velocity (the mean of the two cells
//! of an inner edge), in the least squares sense:
//!
//! - an inner vertex which no pressure condition fixes (a free dof): the equations of the cycle
//!   hold when the residual of the vertex `sum_K r_K(z)` is zero, which the discrete equation
//!   says; the moments are determined up to a constant around the cycle, chosen closest to the
//!   targets;
//! - a free vertex of the boundary, both of whose edges of the boundary are walls (no pressure
//!   condition): zero moments there, and the chain determines the rest;
//! - a vertex with a pressure condition, of the tag `T` (the last listed which marks it, the
//!   edges at it or the cells at it, as its Dirichlet value): its residual is the flow through
//!   the tag at it ([`crate::Flow::tag_flows`]), which leaves through the edges of the boundary at
//!   it that count for `T` (the last listed tag which marks them); moments through the other
//!   edges of the boundary at it are zero. With one such edge the chain determines the moments,
//!   with two they are determined up to a constant, chosen closest to the targets.
//!
//! Where the boundary touches itself at a vertex (two holes, or a hole and the outer boundary,
//! meeting at a point), the cells around the vertex would be several fans which share no edge;
//! [`mesh::Mesh::load`] splits such a vertex into one per fan, so every vertex here has one
//! chain or cycle of cells.
//!
//! Hence the flux through an edge of a wall is zero, exactly, and the fluxes through the edges of
//! a pressure tag add up to its flow from the residual. The equations of a free vertex hold up to
//! its residual, which the iterative refinement of the pressure solve takes to rounding
//! ([`crate::Flow`]); a sweep starts at the cell of the smallest global number, so that every
//! process which solves a patch gets the same moments, bit for bit.
//!
//! Processes: the patch of a vertex at the boundary of the part of a process has cells of other
//! processes. Each process solves the patches of the vertices of its cells on the level
//! distributed with an overlap of one cell ([`petsc::Dm::distribute_overlap`]), which holds every
//! cell around them, with the data of the cells of other processes (their residuals, targets and
//! global numbers) broadcast from their processes. A process keeps the moments of its own cells.
//!
//! The velocity of the fluxes is the lowest-order Raviart-Thomas field of each cell,
//! `u(x) = sum_E F_E (x - x_E) / (2 |K|)`, with `F_E` the flux out of `K` through its edge `E` and
//! `x_E` the vertex opposite `E`: its normal component is `F_E / |E|` on `E`, the same from both
//! cells, and its divergence the sum of the fluxes over `|K|`.

use std::collections::BTreeSet;

use petsc::{Dm, Point, Sf, Vector};

use crate::{Discretization, PdeError, Result};

/// The equilibrated fluxes of a level, and the geometry of its local cells for their
/// Raviart-Thomas velocity.
#[derive(Debug)]
pub(crate) struct Fluxes {
    /// Per local cell, the fluxes out of it through the edges of its cone, in order.
    cells: Vec<[f64; 3]>,
    /// Per local cell, the vertex opposite each edge of its cone, and twice its area.
    geometry: Vec<([[f64; 2]; 3], f64)>,
    /// Per local edge (in the order of the edges of the DM), the flux in the direction of its
    /// reference normal: the vector from its first vertex to its second (in the order of its
    /// cone) turned by -90 degrees.
    edges: Vec<f64>,
}

/// The number of values of a [`Record`].
const RECORD: usize = 10;

/// What a patch needs of a cell, as broadcast to other processes: its global number; for each
/// edge `i` of its cone and end `j` of the edge (in the order of the cone of the edge), the
/// residual `r_K(z)` of that vertex at `1 + 2 i + j`; for each edge `i`, the target of the flux
/// moments out of the cell through it at `7 + i`.
type Record = [f64; RECORD];

/// A local cell of a level: its edges (its cone), the vertices of each edge (its cone), the
/// vertex opposite each edge and its coordinates, and the outward normal of each edge (of the
/// length of the edge).
struct Cell {
    edges: [Point; 3],
    ends: [[Point; 2]; 3],
    opposite: [Point; 3],
    corners: [[f64; 2]; 3],
    normals: [[f64; 2]; 3],
}

impl Fluxes {
    /// The equilibrated fluxes on the level of `disc` (with the pressure tags `tags`, in the
    /// order listed, essential) of the pressure `pressure` (a local vector of `disc`, with its
    /// Dirichlet values), for the diffusion tensors `diffusion` (`[D00, D01, D11]`) and the
    /// source `source` at the quadrature points of `disc`. Collective.
    ///
    /// Errors (on every process): [`PdeError::Unsupported`] for a vertex with a pressure
    /// condition whose flow cannot leave through an edge of the boundary of its tag (a pressure
    /// tag on edges inside the domain, on cells, or on vertices alone), naming the tag, or for a
    /// vertex whose cells are not fans around it (an edge of more than two cells).
    pub(crate) fn equilibrate(
        disc: &Discretization,
        diffusion: &[[f64; 3]],
        source: &[f64],
        pressure: &Vector,
        tags: &[&str],
    ) -> Result<Fluxes> {
        let dm = disc.dm();
        let cells = dm.height_stratum(0)?;
        let q = disc.points_per_cell();
        let nodes = disc.node_coordinates();
        let points = disc.quadrature_points();
        let weights = disc.quadrature_weights();
        let p = pressure.with_array(<[f64]>::to_vec)?;
        let numbering = dm.cell_numbering()?;

        let mut local = Vec::with_capacity(cells.len());
        let mut geometry = Vec::with_capacity(cells.len());
        let mut records = Vec::with_capacity(cells.len());
        for (c, cell) in cells.clone().enumerate() {
            let shape = Cell::new(dm, cell, nodes)?;
            let x = shape.corners;
            let mut pz = [0.0; 3];
            for (value, &v) in pz.iter_mut().zip(&shape.opposite) {
                *value = p[dm.local_dofs(v)?.start];
            }
            // The gradient of the P1 function of the vertex opposite each edge: `-N / (2 |K|)`
            // for the outward normal `N` of the edge.
            let area2 = ((x[1][0] - x[0][0]) * (x[2][1] - x[0][1])
                - (x[1][1] - x[0][1]) * (x[2][0] - x[0][0]))
                .abs();
            let grad_v = shape.normals.map(|n| [-n[0] / area2, -n[1] / area2]);
            let grad_p = (0..3).fold([0.0; 2], |g, i| {
                [g[0] + pz[i] * grad_v[i][0], g[1] + pz[i] * grad_v[i][1]]
            });
            // The residuals of the vertices (opposite edge i: vertex i) and the cell mean of the
            // primal velocity.
            let mut residual = [0.0; 3];
            let mut mean = [0.0; 2];
            let mut area = 0.0;
            for k in c * q..(c + 1) * q {
                let [d00, d01, d11] = diffusion[k];
                let flux = [
                    d00 * grad_p[0] + d01 * grad_p[1],
                    d01 * grad_p[0] + d11 * grad_p[1],
                ];
                let w = weights[k];
                for (i, r) in residual.iter_mut().enumerate() {
                    let g = grad_v[i];
                    let value =
                        1.0 + g[0] * (points[k][0] - x[i][0]) + g[1] * (points[k][1] - x[i][1]);
                    *r += w * (source[k] * value - (flux[0] * g[0] + flux[1] * g[1]));
                }
                mean = [mean[0] - w * flux[0], mean[1] - w * flux[1]];
                area += w;
            }
            let mean = mean.map(|m| m / area);
            let mut record = [0.0; RECORD];
            record[0] = numbering[c] as f64;
            for i in 0..3 {
                for j in 0..2 {
                    let vertex = shape
                        .opposite
                        .iter()
                        .position(|&v| v == shape.ends[i][j])
                        .expect("the ends of an edge are vertices of its cell");
                    record[1 + 2 * i + j] = residual[vertex];
                }
                let n = shape.normals[i];
                record[7 + i] = 0.5 * (mean[0] * n[0] + mean[1] * n[1]);
            }
            records.push(record);
            geometry.push((x, area2));
            local.push(shape);
        }

        let patches = Patches::new(dm, &records, tags)?;
        let moments = patches.solve()?;
        let fluxes: Vec<[f64; 3]> = moments
            .iter()
            .map(|m| [0, 1, 2].map(|i| m[i][0] + m[i][1]))
            .collect();

        // Each local edge from a local cell of it (with both on this process, from the first).
        let edge_points = dm.depth_stratum(1)?;
        let mut edges = vec![f64::NAN; edge_points.len()];
        for (c, shape) in local.iter().enumerate() {
            for i in 0..3 {
                let e = (shape.edges[i] - edge_points.start) as usize;
                if !edges[e].is_nan() {
                    continue;
                }
                let a = nodes[dm.local_dofs(shape.ends[i][0])?.start];
                let b = nodes[dm.local_dofs(shape.ends[i][1])?.start];
                let reference = [b[1] - a[1], a[0] - b[0]];
                let n = shape.normals[i];
                let sign = if reference[0] * n[0] + reference[1] * n[1] > 0.0 {
                    1.0
                } else {
                    -1.0
                };
                edges[e] = sign * fluxes[c][i];
            }
        }
        Ok(Fluxes {
            cells: fluxes,
            geometry,
            edges,
        })
    }

    /// The flux through each local edge, in the direction of its reference normal.
    pub(crate) fn edges(&self) -> &[f64] {
        &self.edges
    }

    /// The Raviart-Thomas velocity of local cell `cell` at `x`.
    pub(crate) fn velocity_at(&self, cell: usize, x: [f64; 2]) -> [f64; 2] {
        let (opposite, area2) = &self.geometry[cell];
        let mut u = [0.0; 2];
        for (f, o) in self.cells[cell].iter().zip(opposite) {
            u[0] += f * (x[0] - o[0]) / area2;
            u[1] += f * (x[1] - o[1]) / area2;
        }
        u
    }

    /// The Raviart-Thomas velocity at the quadrature points of `disc` (of the level of the
    /// fluxes), in the layout of [`Discretization::quadrature_points`].
    pub(crate) fn velocity(&self, disc: &Discretization) -> Vec<[f64; 2]> {
        let q = disc.points_per_cell();
        disc.quadrature_points()
            .iter()
            .enumerate()
            .map(|(k, &x)| self.velocity_at(k / q, x))
            .collect()
    }

    /// The Raviart-Thomas velocity at the Gauss points of each edge of the boundary of the
    /// domain of `disc` (of the level of the fluxes), sorted by edge (see
    /// [`crate::Flow::edge_velocity`]). Collective.
    pub(crate) fn edge_velocity(
        &self,
        disc: &Discretization,
    ) -> Result<Vec<(Point, [[f64; 2]; 2])>> {
        let mut velocity: Vec<(Point, [[f64; 2]; 2])> = disc
            .boundary_edges()?
            .iter()
            .map(|edge| {
                let at = |x| self.velocity_at(edge.cell, x);
                (edge.point, edge.points.map(at))
            })
            .collect();
        velocity.sort_unstable_by_key(|(point, _)| *point);
        Ok(velocity)
    }
}

impl Cell {
    /// The cell `cell` of `dm`, with the coordinates `nodes` of the entries of local vectors.
    fn new(dm: &Dm, cell: Point, nodes: &[[f64; 2]]) -> Result<Cell> {
        let cone = dm.cone(cell)?;
        let &[e0, e1, e2] = cone.as_slice() else {
            return Err(PdeError::Unsupported(format!(
                "a cell of {} edges",
                cone.len()
            )));
        };
        let edges = [e0, e1, e2];
        let mut ends = [[0; 2]; 3];
        for (end, &edge) in ends.iter_mut().zip(&edges) {
            let cone = dm.cone(edge)?;
            *end = cone.as_slice().try_into().map_err(|_| {
                PdeError::Unsupported(format!("an edge of {} vertices", cone.len()))
            })?;
        }
        let at = |v: Point| -> Result<[f64; 2]> { Ok(nodes[dm.local_dofs(v)?.start]) };
        // The vertex of the next edge which is not on this one.
        let mut opposite = [0; 3];
        for (i, o) in opposite.iter_mut().enumerate() {
            let next = ends[(i + 1) % 3];
            *o = if ends[i].contains(&next[0]) {
                next[1]
            } else {
                next[0]
            };
        }
        let mut corners = [[0.0; 2]; 3];
        let mut normals = [[0.0; 2]; 3];
        for i in 0..3 {
            let [a, b] = [at(ends[i][0])?, at(ends[i][1])?];
            corners[i] = at(opposite[i])?;
            let mut n = [b[1] - a[1], a[0] - b[0]];
            if n[0] * (a[0] - corners[i][0]) + n[1] * (a[1] - corners[i][1]) < 0.0 {
                n = [-n[0], -n[1]];
            }
            normals[i] = n;
        }
        Ok(Cell {
            edges,
            ends,
            opposite,
            corners,
            normals,
        })
    }
}

/// The patches of the vertices of the local cells of a level: the level with the cells around
/// them (with an overlap of one cell on several processes), the records of its cells, and the
/// pressure tags.
struct Patches {
    /// The level, or its distribution with an overlap of one cell.
    dm: Dm,
    /// For each cell of `dm`, the local cell of the level it is, if it is one of this process.
    local: Vec<Option<usize>>,
    /// For each cell of `dm`, its record.
    records: Vec<Record>,
    /// The points of `dm` each pressure tag marks, in the order listed.
    tags: Vec<(String, BTreeSet<Point>)>,
    /// The number of local cells of the level.
    num_local: usize,
    /// The first cell of `dm`.
    first_cell: Point,
}

/// A fan of cells around a vertex: the cells `K_1..K_n` of the walk, each with the indices
/// among its two edges at the vertex of the one it is entered through and the one it is left
/// by, and the edges `E_0..E_n` (`E_n = E_0` for a cycle).
struct Fan {
    walk: Vec<(Corner, usize, usize)>,
    edges: Vec<Point>,
    cycle: bool,
}

/// A cell of a patch and the two edges of it at the vertex of the patch: their indices in its
/// cone, and the end of each edge which is the vertex.
#[derive(Clone, Copy)]
struct Corner {
    cell: Point,
    edges: [(usize, usize); 2],
}

impl Patches {
    /// The patches of the level `level` (a DM of a level, with its labels) with the records
    /// `records` of its local cells and the pressure tags `tags`. Collective.
    fn new(level: &Dm, records: &[Record], tags: &[&str]) -> Result<Patches> {
        let cells = level.height_stratum(0)?;
        let base = level.clone_mesh()?;
        let (dm, local, records) = match base.distribute_overlap(1)? {
            None => (base, (0..cells.len()).map(Some).collect(), records.to_vec()),
            Some((dm, sf)) => {
                let (local, overlap_records) = migrate(&dm, &sf, &cells, records)?;
                (dm, local, overlap_records)
            }
        };
        let mut marked = Vec::with_capacity(tags.len());
        for &tag in tags {
            let mut points = BTreeSet::new();
            for value in dm.label_values(tag)? {
                points.extend(dm.stratum_points(tag, value)?);
            }
            marked.push((tag.to_string(), points));
        }
        Ok(Patches {
            first_cell: dm.height_stratum(0)?.start,
            dm,
            local,
            records,
            tags: marked,
            num_local: cells.len(),
        })
    }

    /// The moments of the local cells of the level: for each, for each edge of its cone and each
    /// end of it, the flux moment out of the cell. Collective.
    ///
    /// Errors (on every process): see [`Fluxes::equilibrate`].
    fn solve(&self) -> Result<Vec<[[f64; 2]; 3]>> {
        let dm = &self.dm;
        let cells = dm.height_stratum(0)?;
        let vertices = dm.depth_stratum(0)?;
        let mut moments = vec![[[f64::NAN; 2]; 3]; self.num_local];
        let mut done = vec![false; vertices.len()];
        let mut failure = None;
        // The first pressure tag found inside the domain, the same error on every process.
        let mut inside: Option<usize> = None;
        for (oc, local) in self.local.iter().enumerate() {
            if local.is_none() {
                continue;
            }
            let cell = cells.start + oc as Point;
            for edge in dm.cone(cell)? {
                for z in dm.cone(edge)? {
                    let seen = &mut done[(z - vertices.start) as usize];
                    if *seen {
                        continue;
                    }
                    *seen = true;
                    match self.solve_patch(z, &mut moments) {
                        Ok(None) => {}
                        Ok(Some(tag)) => inside = Some(inside.map_or(tag, |t| t.min(tag))),
                        Err(e) => {
                            failure.get_or_insert(e);
                        }
                    }
                }
            }
        }
        // The smallest index of such a tag on any process, as the largest of its complement.
        let count = self.tags.len() as petsc::sys::PetscInt;
        let first = count
            - dm.comm()
                .max_int(inside.map_or(0, |t| count - t as petsc::sys::PetscInt))?;
        if first < count {
            return Err(PdeError::Unsupported(format!(
                "pressure tag '{}' marks a point inside the domain (an edge, a cell or a vertex \
                 not on an edge of the boundary marked by it): the equilibrated fluxes of the \
                 conservative velocity need the pressure tags on edges of the boundary",
                self.tags[first as usize].0
            )));
        }
        if dm.comm().max_int(failure.is_some().into())? > 0 {
            return Err(failure.unwrap_or_else(|| {
                PdeError::Unsupported(
                    "the patch of a vertex of another process for the conservative flux".into(),
                )
            }));
        }
        Ok(moments)
    }

    /// The index of `cell` among the cells of `dm`.
    fn index(&self, cell: Point) -> usize {
        (cell - self.first_cell) as usize
    }

    /// The pressure tag an edge of the boundary counts for, or of the closure of which a vertex
    /// is (by the points `star` around it): the last listed which marks one of the points.
    fn tag_of(&self, star: &[Point]) -> Option<usize> {
        self.tags
            .iter()
            .rposition(|(_, points)| star.iter().any(|p| points.contains(p)))
    }

    /// Solves the patch of the vertex `z` (see the module documentation) and writes the moments
    /// of its local cells into `moments`.
    ///
    /// `Some` with the index of the pressure tag of `z` if its flow cannot leave through an edge
    /// of the boundary of the tag (the tag marks a point inside the domain); then nothing is
    /// written.
    fn solve_patch(&self, z: Point, moments: &mut [[[f64; 2]; 3]]) -> Result<Option<usize>> {
        let dm = &self.dm;
        let edges_at = dm.support(z)?;
        // The cells around z, each with its two edges at z, and the edges of the boundary at z.
        let mut corners: Vec<Corner> = Vec::new();
        let mut boundary = Vec::new();
        for &edge in &edges_at {
            let support = dm.support(edge)?;
            match support.len() {
                1 => boundary.push(edge),
                2 => {}
                n => {
                    return Err(PdeError::Unsupported(format!(
                        "an edge of {n} cells for the conservative flux"
                    )));
                }
            }
            for cell in support {
                if corners.iter().any(|c| c.cell == cell) {
                    continue;
                }
                let mut at_z = Vec::with_capacity(2);
                for (i, e) in dm.cone(cell)?.into_iter().enumerate() {
                    if let Some(j) = dm.cone(e)?.iter().position(|&v| v == z) {
                        at_z.push((i, j));
                    }
                }
                let &[first, second] = at_z.as_slice() else {
                    return Err(PdeError::Unsupported(format!(
                        "a cell with {} edges at a vertex",
                        at_z.len()
                    )));
                };
                corners.push(Corner {
                    cell,
                    edges: [first, second],
                });
            }
        }
        let mut star: Vec<Point> = vec![z];
        star.extend(&edges_at);
        star.extend(corners.iter().map(|c| c.cell));
        let tag = self.tag_of(&star);

        // One chain from an edge of the boundary to another, or one cycle.
        let fan = self.fan(z, &corners, boundary.first().copied())?;
        if boundary.len() > 2 || fan.walk.len() != corners.len() {
            return Err(PdeError::Unsupported(
                "a vertex whose cells are not one chain or cycle around it (where the boundary \
                 touches itself, which Mesh::load splits) for the conservative flux"
                    .into(),
            ));
        }

        // Which ends are free: edges of the boundary of the tag of z.
        let free = |edge: Point| tag.is_some() && self.tag_of(&[edge]) == tag;
        let (free_0, free_n) = if fan.cycle {
            (false, false)
        } else {
            (free(fan.edges[0]), free(fan.edges[fan.walk.len()]))
        };
        if let Some(t) = tag
            && !free_0
            && !free_n
        {
            return Ok(Some(t));
        }
        self.solve_fan(&fan, free_0, free_n, moments);
        Ok(None)
    }

    /// The fan of the cells `corners` around `z` from the edge of the boundary `start`, or the
    /// cycle of all of them without one; the walk starts at the end of the fan whose cell (then
    /// edge in its cone) comes first, or for a cycle at the cell of the smallest global number
    /// towards its neighbour of the smaller one.
    fn fan(&self, z: Point, corners: &[Corner], start: Option<Point>) -> Result<Fan> {
        let dm = &self.dm;
        let number = |cell: Point| self.records[self.index(cell)][0];
        let corner_of = |cell: Point| *corners.iter().find(|c| c.cell == cell).expect("a corner");
        let edge_of = |corner: &Corner, k: usize| -> Result<Point> {
            Ok(dm.cone(corner.cell)?[corner.edges[k].0])
        };
        let other_cell = |edge: Point, cell: Point| -> Result<Option<Point>> {
            Ok(dm.support(edge)?.into_iter().find(|&c| c != cell))
        };
        // The walk from `first`, entered through `entry`.
        let walk_from = |first: Corner, entry: Point| -> Result<Fan> {
            let mut walk: Vec<(Corner, usize, usize)> = Vec::new();
            let mut edges = vec![entry];
            let (mut corner, mut entered) = (first, entry);
            loop {
                let k_in = (0..2)
                    .find(|&k| edge_of(&corner, k).ok() == Some(entered))
                    .expect("entered through an edge at z");
                let k_out = 1 - k_in;
                let leave = edge_of(&corner, k_out)?;
                walk.push((corner, k_in, k_out));
                edges.push(leave);
                match other_cell(leave, corner.cell)? {
                    Some(next) if next == first.cell => {
                        return Ok(Fan {
                            walk,
                            edges,
                            cycle: true,
                        });
                    }
                    Some(next) => {
                        corner = corner_of(next);
                        entered = leave;
                    }
                    None => {
                        return Ok(Fan {
                            walk,
                            edges,
                            cycle: false,
                        });
                    }
                }
                if walk.len() > corners.len() {
                    return Err(PdeError::Unsupported(format!(
                        "vertex {z}: cells which are not chains or one cycle around it"
                    )));
                }
            }
        };
        match start {
            None => {
                let first = *corners
                    .iter()
                    .min_by(|a, b| number(a.cell).total_cmp(&number(b.cell)))
                    .expect("a vertex has a cell");
                let neighbours =
                    [0, 1].map(|k| edge_of(&first, k).and_then(|e| other_cell(e, first.cell)));
                let [Ok(Some(n0)), Ok(Some(n1))] = neighbours else {
                    return Err(PdeError::Unsupported(
                        "an inner vertex with an edge of one cell".into(),
                    ));
                };
                let toward = if number(n0) <= number(n1) { 0 } else { 1 };
                walk_from(first, edge_of(&first, 1 - toward)?)
            }
            Some(edge) => {
                let corner = corner_of(dm.support(edge)?[0]);
                let fan = walk_from(corner, edge)?;
                if fan.cycle {
                    return Err(PdeError::Unsupported(format!(
                        "vertex {z}: an edge of the boundary in a cycle of cells"
                    )));
                }
                // The key of an end: the number of its cell, then the index of its edge in the
                // cone of the cell.
                let n = fan.walk.len();
                let key = |(corner, k): (&Corner, usize)| (number(corner.cell), corner.edges[k].0);
                let first = key((&fan.walk[0].0, fan.walk[0].1));
                let last = key((&fan.walk[n - 1].0, fan.walk[n - 1].2));
                if first <= last {
                    Ok(fan)
                } else {
                    walk_from(fan.walk[n - 1].0, fan.edges[n])
                }
            }
        }
    }

    /// Solves the fan `fan` (see the module documentation), with its first and last edges free
    /// (`free_0`, `free_n`; neither for a cycle), and writes the moments of its local cells into
    /// `moments`. Without a free edge, the last cell takes the residual of the vertex, which the
    /// refined pressure solve leaves at rounding.
    fn solve_fan(&self, fan: &Fan, free_0: bool, free_n: bool, moments: &mut [[[f64; 2]; 3]]) {
        let walk = &fan.walk;
        let n = walk.len();
        // The residuals and the targets of the moments mu_i from K_i into K_(i+1) through E_i.
        let record = |corner: &Corner| &self.records[self.index(corner.cell)];
        let residual = |(corner, k_in, _): &(Corner, usize, usize)| {
            let (i, j) = corner.edges[*k_in];
            record(corner)[1 + 2 * i + j]
        };
        let target_out = |corner: &Corner, k: usize| record(corner)[7 + corner.edges[k].0];
        let mut targets = vec![0.0; n + 1];
        targets[0] = -target_out(&walk[0].0, walk[0].1);
        for i in 1..n {
            let (before, after) = (&walk[i - 1], &walk[i]);
            targets[i] = 0.5 * (target_out(&before.0, before.2) - target_out(&after.0, after.1));
        }
        targets[n] = target_out(&walk[n - 1].0, walk[n - 1].2);
        if fan.cycle {
            // E_n is E_0: one target, from both cells.
            let t = 0.5 * (targets[n] + targets[0]);
            targets[0] = t;
            targets[n] = t;
        }
        let mut prefix = vec![0.0; n + 1];
        for i in 1..=n {
            prefix[i] = prefix[i - 1] + residual(&walk[i - 1]);
        }

        let mut mu = vec![0.0; n + 1];
        match (fan.cycle, free_0, free_n) {
            (true, _, _) => {
                // mu_i = c + P_i, c closest to the targets; the last cell takes the residual of
                // the vertex (rounding).
                let c = (0..n).map(|i| targets[i] - prefix[i]).sum::<f64>() / n as f64;
                for i in 0..n {
                    mu[i] = c + prefix[i];
                }
                mu[n] = mu[0];
            }
            (false, false, false) => {
                mu[..n].copy_from_slice(&prefix[..n]);
                mu[n] = 0.0;
            }
            (false, true, false) => {
                for i in 0..n {
                    mu[i] = prefix[i] - prefix[n];
                }
                mu[n] = 0.0;
            }
            (false, false, true) => mu.copy_from_slice(&prefix),
            (false, true, true) => {
                let c = (0..=n).map(|i| targets[i] - prefix[i]).sum::<f64>() / (n + 1) as f64;
                for i in 0..=n {
                    mu[i] = c + prefix[i];
                }
            }
        }

        // K_i: in through E_(i-1) (index i-1), out through E_i.
        for (i, (corner, k_in, k_out)) in walk.iter().enumerate() {
            let Some(c) = self.local[self.index(corner.cell)] else {
                continue;
            };
            let (e_in, j_in) = corner.edges[*k_in];
            let (e_out, j_out) = corner.edges[*k_out];
            moments[c][e_in][j_in] = -mu[i];
            moments[c][e_out][j_out] = mu[i + 1];
        }
    }
}

/// The local cells of `dm` (the level of `cells`, distributed with an overlap of one cell by
/// `sf`) which are local cells of the level, and the records of the cells of `dm`, broadcast
/// from the records `records` of the local cells of the level of every process. Collective.
fn migrate(
    dm: &Dm,
    sf: &Sf,
    cells: &std::ops::Range<Point>,
    records: &[Record],
) -> Result<(Vec<Option<usize>>, Vec<Record>)> {
    let overlap_cells = dm.height_stratum(0)?;
    let (roots, graph) = sf.graph()?;
    let chart = graph
        .iter()
        .map(|l| l.leaf as usize + 1)
        .chain([overlap_cells.end as usize])
        .max()
        .unwrap_or(0);
    let mut root = vec![0.0; roots];
    let mut leaf = vec![0.0; chart];
    let mut migrated = vec![[0.0; RECORD]; overlap_cells.len()];
    for k in 0..RECORD {
        for (c, record) in records.iter().enumerate() {
            root[cells.start as usize + c] = record[k];
        }
        sf.bcast(&root, &mut leaf)?;
        for (c, record) in migrated.iter_mut().enumerate() {
            record[k] = leaf[overlap_cells.start as usize + c];
        }
    }
    let rank = dm.comm().rank();
    let mut local = vec![None; overlap_cells.len()];
    for l in &graph {
        if l.rank == rank && overlap_cells.contains(&l.leaf) && cells.contains(&l.root) {
            local[(l.leaf - overlap_cells.start) as usize] = Some((l.root - cells.start) as usize);
        }
    }
    Ok((local, migrated))
}
