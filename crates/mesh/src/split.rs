//! Vertices where the boundary of a 2D mesh touches itself, split into one vertex per fan.
//!
//! Where two holes, or a hole and the outer boundary, meet at a single point, the cells around
//! the vertex there form several fans: chains of cells joined by the edges at the vertex, which
//! share no edge with each other. With one P1 dof there, a field would pass through the point
//! from one fan to the other, as if the point contact were an opening; physically it is an
//! aperture of zero width, through which nothing flows. Splitting the vertex into one per fan
//! (at the same coordinates) makes the fans meet at the point only geometrically: fields are
//! discontinuous there, and the fans are joined through the rest of the mesh only.

use std::collections::{BTreeMap, BTreeSet};

use petsc::{Comm, Dm, Point};

/// Labels which a DMPlex creates itself, and which are not copied.
const TOPOLOGY_LABELS: [&str; 2] = ["depth", "celltype"];

/// The 2D mesh `dm` as read (process 0 holds all of it) with every vertex where the boundary
/// touches itself split into one vertex per fan of its cells, the first fan (of the cell first
/// in the mesh) keeping the vertex; `None` if there is no such vertex. The new mesh has the cells
/// in the same order with the same vertices but for the split ones, the vertices in the same
/// order followed by the new ones, and every label of `dm` on the corresponding points (a label
/// of a split vertex on each of its vertices). Collective.
pub(crate) fn split_pinched_vertices(dm: &Dm, comm: &Comm) -> petsc::Result<Option<Dm>> {
    let cells = dm.height_stratum(0)?;
    let vertices = dm.depth_stratum(0)?;
    let edges = dm.depth_stratum(1)?;
    // The vertices of each cell (in the order of its closure) and their indices in the new mesh.
    let mut corners: Vec<Vec<Point>> = Vec::with_capacity(cells.len());
    for cell in cells.clone() {
        let closure = dm.closure(cell)?;
        corners.push(
            closure
                .into_iter()
                .filter(|p| vertices.contains(p))
                .collect(),
        );
    }
    let mut index: Vec<Vec<usize>> = corners
        .iter()
        .map(|c| c.iter().map(|v| (v - vertices.start) as usize).collect())
        .collect();
    // For each split vertex, the vertex it copies, in the order of the new vertices.
    let mut copies: Vec<Point> = Vec::new();
    for vertex in vertices.clone() {
        let fans = fans(dm, vertex, &edges, &cells)?;
        for fan in fans.iter().skip(1) {
            let new = vertices.len() + copies.len();
            copies.push(vertex);
            for &cell in fan {
                let c = (cell - cells.start) as usize;
                let k = corners[c]
                    .iter()
                    .position(|&v| v == vertex)
                    .expect("a cell of the vertex");
                index[c][k] = new;
            }
        }
    }
    if comm.max_int(Point::from(!copies.is_empty()))? == 0 {
        return Ok(None);
    }

    let dim = dm.coordinate_dimension()?;
    let mut coordinates = dm.local_coordinates()?;
    for &vertex in &copies {
        let at = (vertex - vertices.start) as usize * dim;
        let x: Vec<f64> = coordinates[at..at + dim].to_vec();
        coordinates.extend(x);
    }
    let corners_per_cell = corners.first().map_or(3, Vec::len);
    let list: Vec<Point> = index.iter().flatten().map(|&i| i as Point).collect();
    let mut split = Dm::plex_from_cells(
        comm,
        crate::NAME,
        dm.dimension()?,
        corners_per_cell,
        &list,
        dim,
        &coordinates,
    )?;

    // The points of the new mesh for each point of `dm`.
    let new_cells = split.height_stratum(0)?;
    let new_vertices = split.depth_stratum(0)?;
    let mut new_edges: BTreeMap<(usize, usize), Point> = BTreeMap::new();
    for edge in split.depth_stratum(1)? {
        let ends: Vec<usize> = split
            .cone(edge)?
            .iter()
            .map(|v| (v - new_vertices.start) as usize)
            .collect();
        new_edges.insert((ends[0].min(ends[1]), ends[0].max(ends[1])), edge);
    }
    let mut copies_of: BTreeMap<Point, Vec<usize>> = BTreeMap::new();
    for (k, &vertex) in copies.iter().enumerate() {
        copies_of
            .entry(vertex)
            .or_default()
            .push(vertices.len() + k);
    }
    let image = |point: Point| -> petsc::Result<Vec<Point>> {
        if cells.contains(&point) {
            return Ok(vec![new_cells.start + (point - cells.start)]);
        }
        if vertices.contains(&point) {
            let mut points = vec![(point - vertices.start) as usize];
            points.extend(copies_of.get(&point).into_iter().flatten());
            return Ok(points
                .into_iter()
                .map(|i| new_vertices.start + i as Point)
                .collect());
        }
        // An edge: its ends as the vertices of one of its cells.
        let cell = dm.support(point)?[0];
        let c = (cell - cells.start) as usize;
        let ends: Vec<usize> = dm
            .cone(point)?
            .iter()
            .map(|v| {
                let k = corners[c]
                    .iter()
                    .position(|u| u == v)
                    .expect("a vertex of the cell");
                index[c][k]
            })
            .collect();
        Ok(vec![
            new_edges[&(ends[0].min(ends[1]), ends[0].max(ends[1]))],
        ])
    };
    for name in dm.label_names()? {
        if TOPOLOGY_LABELS.contains(&name.as_str()) {
            continue;
        }
        split.create_label(&name)?;
        for value in dm.label_values(&name)? {
            for point in dm.stratum_points(&name, value)? {
                for new in image(point)? {
                    split.set_label_value(&name, new, value)?;
                }
            }
        }
    }
    Ok(Some(split))
}

/// The fans of the cells around `vertex` of `dm`: the cells joined by edges at the vertex, each
/// fan sorted, the fans in the order of their first cells.
fn fans(
    dm: &Dm,
    vertex: Point,
    edges: &std::ops::Range<Point>,
    cells: &std::ops::Range<Point>,
) -> petsc::Result<Vec<Vec<Point>>> {
    let at: Vec<Point> = dm
        .support(vertex)?
        .into_iter()
        .filter(|e| edges.contains(e))
        .collect();
    let mut around = BTreeSet::new();
    let mut joins = Vec::new();
    for &edge in &at {
        let support: Vec<Point> = dm
            .support(edge)?
            .into_iter()
            .filter(|c| cells.contains(c))
            .collect();
        around.extend(&support);
        if let &[a, b] = support.as_slice() {
            joins.push((a, b));
        }
    }
    // Connected components of the cells around the vertex.
    let mut fans: Vec<Vec<Point>> = Vec::new();
    let mut left: BTreeSet<Point> = around;
    while let Some(&first) = left.iter().next() {
        left.remove(&first);
        let mut fan = vec![first];
        let mut k = 0;
        while k < fan.len() {
            let cell = fan[k];
            for &(a, b) in &joins {
                let other = if a == cell {
                    b
                } else if b == cell {
                    a
                } else {
                    continue;
                };
                if left.remove(&other) {
                    fan.push(other);
                }
            }
            k += 1;
        }
        fan.sort_unstable();
        fans.push(fan);
    }
    Ok(fans)
}
