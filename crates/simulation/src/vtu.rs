//! VTU files for ParaView: the fields on the finest level (point data), the flow, and values per
//! coarse element as cell data ([`CellWriter`]). Every process writes its piece of one file.

use std::path::Path;

use anyhow::{Context, Result};
use pde::{Flow, Problem};
use petsc::{Comm, Dm, Fe, Vector, Viewer};

/// Writes the pressure and the velocity of `flow` as `<out>/pressure.vtu` and
/// `<out>/velocity.vtu`. Collective.
pub(crate) fn write_flow(flow: &Flow, out: &Path) -> Result<()> {
    let comm = flow.discretization().dm().comm();
    for (name, vector) in [
        ("pressure.vtu", flow.local_pressure()?),
        ("velocity.vtu", flow.nodal_velocity()?),
    ] {
        let path = out.join(name);
        write_vector(&comm, &vector, &path)
            .with_context(|| format!("cannot write {}", path.display()))?;
    }
    Ok(())
}

/// Writes the solution of `problem` as the VTU file `path`. Collective.
pub(crate) fn write_problem(problem: &Problem, path: &Path) -> Result<()> {
    let solution = problem.local_solution()?;
    write_vector(&problem.discretization().dm().comm(), &solution, path)
}

/// Writes the local vector `vector` as the VTU file `path`. Collective.
fn write_vector(comm: &Comm, vector: &Vector, path: &Path) -> Result<()> {
    let mut viewer = Viewer::vtk(comm, path)?;
    vector.view(&mut viewer)?;
    viewer.flush()?;
    Ok(())
}

/// Writes values per coarse element as cell data of a VTU file: a field of one P0 component
/// per value on a copy of the coarse level, named after the value, so that the data arrays are
/// named after the values.
pub struct CellWriter {
    dm: Dm,
    names: Vec<String>,
    /// For each value, the entry of each local cell in a local vector.
    offsets: Vec<Vec<usize>>,
}

impl CellWriter {
    /// The writer of the values `names` on the coarse level `coarse`. Collective.
    pub fn new(coarse: &Dm, names: Vec<String>) -> Result<CellWriter> {
        let mut dm = coarse.clone_mesh()?;
        let dim = dm.dimension()?;
        let fe = Fe::lagrange(dim, 1, dm.is_simplex()?, 0, 1)?;
        for _ in &names {
            dm.add_field(&fe)?;
        }
        dm.create_ds()?;
        for (k, name) in names.iter().enumerate() {
            dm.set_field_names(k, name, &[name])?;
        }
        let offsets = (0..names.len())
            .map(|k| {
                let offsets = dm.local_offsets(k)?;
                Ok(offsets
                    .offsets
                    .iter()
                    .map(|&o| usize::try_from(o).expect("an entry of a local vector"))
                    .collect())
            })
            .collect::<Result<Vec<Vec<usize>>>>()?;
        Ok(CellWriter { dm, names, offsets })
    }

    /// The names of the values, in the order of a row of a table of [`CellWriter::write`].
    pub fn names(&self) -> &[String] {
        &self.names
    }

    /// Writes `table` (per cell of `cells`, local coarse cells, one value per name) as the VTU
    /// file `path`; NaN for the other cells. Collective.
    pub fn write(&self, cells: &[usize], table: &[f64], path: &Path) -> Result<()> {
        let mut local = Vector::create_local(&self.dm)?;
        local.set(f64::NAN)?;
        let n = self.names.len();
        local.with_array_mut(|values| {
            for (i, &cell) in cells.iter().enumerate() {
                for (k, offsets) in self.offsets.iter().enumerate() {
                    values[offsets[cell]] = table[i * n + k];
                }
            }
        })?;
        local.set_name("")?;
        let mut viewer = Viewer::vtk(&self.dm.comm(), path)?;
        local.view(&mut viewer)?;
        viewer.flush()?;
        Ok(())
    }
}
