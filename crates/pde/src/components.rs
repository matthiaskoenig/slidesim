//! The connected components of the cells of a level of a mesh (cells joined through their edges,
//! across processes), and the level with an overlap of one cell, which both they and the patches
//! of the equilibrated fluxes ([`crate::fluxes`]) see the cells of other processes through.

use std::ops::Range;

use petsc::{Dm, Point, Sf};

use crate::Result;

/// A level of a mesh (distributed without overlap) distributed again with an overlap of one cell
/// ([`Dm::distribute_overlap`]): every cell which shares a point with a cell of this process is
/// here too. On one process, the level itself.
pub(crate) struct Overlap {
    dm: Dm,
    /// From the points of the level (roots) to those of `dm` (leaves); `None` on one process.
    sf: Option<Sf>,
    /// The cells of the level.
    cells: Range<Point>,
    /// For each cell of `dm`, its index among the cells of the level if it is a cell of this
    /// process.
    local: Vec<Option<usize>>,
}

impl Overlap {
    /// The level `level` (a DM of a level of a mesh) with an overlap of one cell. Collective.
    pub(crate) fn new(level: &Dm) -> Result<Overlap> {
        let cells = level.height_stratum(0)?;
        let base = level.clone_mesh()?;
        let Some((dm, sf)) = base.distribute_overlap(1)? else {
            return Ok(Overlap {
                dm: base,
                sf: None,
                local: (0..cells.len()).map(Some).collect(),
                cells,
            });
        };
        let overlap_cells = dm.height_stratum(0)?;
        let (_, graph) = sf.graph()?;
        let rank = dm.comm().rank();
        let mut local = vec![None; overlap_cells.len()];
        for l in &graph {
            if l.rank == rank && overlap_cells.contains(&l.leaf) && cells.contains(&l.root) {
                local[(l.leaf - overlap_cells.start) as usize] =
                    Some((l.root - cells.start) as usize);
            }
        }
        Ok(Overlap {
            dm,
            sf: Some(sf),
            cells,
            local,
        })
    }

    /// The level with the overlap.
    pub(crate) fn dm(&self) -> &Dm {
        &self.dm
    }

    /// For each cell of [`Overlap::dm`], its index among the cells of the level if it is a cell
    /// of this process.
    pub(crate) fn local(&self) -> &[Option<usize>] {
        &self.local
    }

    /// The level with the overlap, and for each of its cells its index among the cells of the
    /// level if it is a cell of this process.
    pub(crate) fn into_parts(self) -> (Dm, Vec<Option<usize>>) {
        (self.dm, self.local)
    }

    /// The values `values` of the cells of the level (one per cell, on every process) at the
    /// cells of [`Overlap::dm`], each from the process of the cell. Collective.
    pub(crate) fn values(&self, values: &[f64]) -> Result<Vec<f64>> {
        let Some(sf) = &self.sf else {
            return Ok(values.to_vec());
        };
        let overlap_cells = self.dm.height_stratum(0)?;
        let (roots, graph) = sf.graph()?;
        let chart = graph
            .iter()
            .map(|l| l.leaf as usize + 1)
            .chain([overlap_cells.end as usize])
            .max()
            .unwrap_or(0);
        let mut root = vec![0.0; roots];
        root[self.cells.start as usize..self.cells.end as usize].copy_from_slice(values);
        let mut leaf = vec![0.0; chart];
        sf.bcast(&root, &mut leaf)?;
        Ok(leaf[overlap_cells.start as usize..overlap_cells.end as usize].to_vec())
    }
}

/// For each cell of this process of the level `level` (a DM of a level of a mesh, distributed
/// without overlap): the connected component of cells joined through edges which it belongs to,
/// named by the smallest global number of its cells, and whether a cell of the component is
/// marked (`marked`, one per cell of this process). The same on every process for the cells of a
/// component, and for any number of processes. Collective.
///
/// Each process joins the cells of the level with an overlap of one cell ([`Overlap`]) which
/// share an edge, and takes for its cells the smallest name and any mark of the cells joined to
/// them, the cells of other processes with the names and marks their processes have; until no
/// process changes a name or a mark (one round more than the longest chain of processes a
/// component passes through).
pub(crate) fn components(level: &Dm, marked: &[bool]) -> Result<Vec<(i64, bool)>> {
    let numbering = level.cell_numbering()?;
    // Global cell numbers are below 2^53: exact as f64, which the star forest broadcasts.
    let mut names: Vec<f64> = numbering.iter().map(|&n| n as f64).collect();
    let mut marks: Vec<f64> = marked.iter().map(|&m| f64::from(u8::from(m))).collect();
    let overlap = Overlap::new(level)?;
    let joined = joined_cells(overlap.dm())?;
    let groups = joined.iter().max().map_or(0, |&g| g + 1);
    loop {
        let overlap_names = overlap.values(&names)?;
        let overlap_marks = overlap.values(&marks)?;
        let mut group_name = vec![f64::INFINITY; groups];
        let mut group_mark = vec![0.0_f64; groups];
        for (c, &g) in joined.iter().enumerate() {
            group_name[g] = group_name[g].min(overlap_names[c]);
            group_mark[g] = group_mark[g].max(overlap_marks[c]);
        }
        let mut changed = false;
        for (c, &g) in joined.iter().enumerate() {
            if let Some(local) = overlap.local()[c] {
                changed |= names[local] != group_name[g] || marks[local] != group_mark[g];
                names[local] = group_name[g];
                marks[local] = group_mark[g];
            }
        }
        if level.comm().max_int(changed.into())? == 0 {
            break;
        }
    }
    Ok(names
        .iter()
        .zip(&marks)
        .map(|(&name, &mark)| (name as i64, mark > 0.0))
        .collect())
}

/// For each cell of `dm`, the group of the cells of `dm` joined to it through edges, numbered from
/// zero.
fn joined_cells(dm: &Dm) -> Result<Vec<usize>> {
    let cells = dm.height_stratum(0)?;
    let mut parent: Vec<usize> = (0..cells.len()).collect();
    fn find(parent: &mut [usize], mut c: usize) -> usize {
        while parent[c] != c {
            parent[c] = parent[parent[c]];
            c = parent[c];
        }
        c
    }
    for edge in dm.depth_stratum(1)? {
        if let [a, b] = dm.support(edge)?[..] {
            let a = find(&mut parent, (a - cells.start) as usize);
            let b = find(&mut parent, (b - cells.start) as usize);
            parent[a.max(b)] = a.min(b);
        }
    }
    let mut group = vec![usize::MAX; cells.len()];
    let mut groups = 0;
    let mut joined = Vec::with_capacity(cells.len());
    for c in 0..cells.len() {
        let root = find(&mut parent, c);
        if group[root] == usize::MAX {
            group[root] = groups;
            groups += 1;
        }
        joined.push(group[root]);
    }
    Ok(joined)
}
