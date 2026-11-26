//! The cells of a process and their parameters: [`CellSetup`].

use contracts::{CellsSpec, Coefficient};
use maps::Maps;
use mesh::Mesh;
use pde::Discretization;
use petsc::Comm;

use crate::{CellsError, Result, agree};

/// The cells of this process and the parameters of each: the coarse elements (level 0 of the
/// mesh) with tissue of the `areas` of a [`CellsSpec`], one model instance each.
///
/// The tissue of an element is its finest children (on the finest level of the mesh, the level
/// of the fields, [`Mesh::coarse_cell`]) whose area tag is one of the areas (default: every area
/// tag of the mesh). Its parameters are means over its tissue, by the quadrature of the
/// discretization of the fields (the same as the transport's and `pde::Exchange`'s):
/// `sum_q w_q c(x_q) / sum_q w_q` over the quadrature points `x_q` of the tissue, with the
/// quadrature weights `w_q`:
///
/// - every [`contracts::CellParameter`]: the mean of its value (a constant is taken as it is); with
///   a replacement rule, the mean of the replacement value instead where the mean of the rule's
///   map is above its threshold;
/// - the volumes ([`CellsSpec::volumes`]): the blood compartment the mean porosity `phi`, the
///   cell compartment `1 - phi`.
///
/// A map without value at a quadrature point of a cell's tissue is an error
/// ([`CellsError::Element`], naming the map and the element), wherever else it has none.
#[derive(Debug, Clone)]
pub struct CellSetup {
    /// The local coarse cells with a model (cells of level 0 of the mesh, numbered from 0 in
    /// their order), ascending.
    pub cells: Vec<usize>,
    /// For each cell, its parameters as changes of the model's initial values: the parameters
    /// of the spec in their order, then the blood and the cell volume.
    pub parameters: Vec<Vec<(String, f64)>>,
    /// For each cell, the global number of its element on the coarse level (the same for any
    /// number of processes only up to the partition).
    pub global: Vec<i64>,
    /// For each cell, the centroid of its element (um).
    pub centroids: Vec<[f64; 2]>,
    /// The communicator of the mesh, for errors on every process.
    comm: Comm,
}

impl CellSetup {
    /// The cells of `spec` on `mesh` and their parameters, from the maps `maps` (read for the
    /// cells of this process) and the porosity `porosity` of the blood space, by the
    /// quadrature of `disc`: the discretization of the fields on the finest level of `mesh`
    /// (`pde::Problem::discretization`, as for `pde::Exchange::new`). Collective.
    ///
    /// Errors: [`CellsError::UnknownArea`] for an area which is not an area tag of the mesh;
    /// on every process, [`CellsError::Invalid`] if `disc` is not on the finest level of
    /// `mesh`, and for a map the maps lack; [`CellsError::Element`] for a map without value in
    /// the tissue of an element, and for an element whose mean porosity is not in (0, 1) (no
    /// blood or no cell volume).
    pub fn new(
        disc: &Discretization,
        mesh: &Mesh,
        spec: &CellsSpec,
        maps: &Maps,
        porosity: &Coefficient,
    ) -> Result<CellSetup> {
        let comm = mesh.finest().comm();
        let areas = areas(mesh, spec)?;
        let finest = mesh.levels().len() - 1;
        let n_fine = mesh.finest().height_stratum(0)?.len();
        let n_disc = disc.dm().height_stratum(0)?.len();
        let points = disc.num_quadrature_points();
        let on_finest = if n_disc == n_fine && points.checked_rem(n_fine).unwrap_or(0) == 0 {
            Ok(())
        } else {
            Err(CellsError::Invalid(format!(
                "a discretization of {n_disc} local cells and {points} quadrature points on a \
                 finest level of {n_fine} cells"
            )))
        };
        agree(&comm, on_finest)?;
        // The quadrature points of each cell (none without cells).
        let q = points.checked_div(n_fine).unwrap_or(0);
        let n_coarse = mesh.levels()[0].height_stratum(0)?.len();

        // The tissue of each coarse element (its finest cells in the areas), and the centroids.
        let mut tissue: Vec<Vec<usize>> = vec![Vec::new(); n_coarse];
        let mut sums = vec![[0.0; 3]; n_coarse];
        let (points, weights) = (disc.quadrature_points(), disc.quadrature_weights());
        for e in 0..n_fine {
            let c = mesh.coarse_cell(finest, e);
            for k in e * q..(e + 1) * q {
                sums[c][0] += weights[k] * points[k][0];
                sums[c][1] += weights[k] * points[k][1];
                sums[c][2] += weights[k];
            }
            if disc
                .cell_area_tag(e)
                .is_some_and(|tag| areas.contains(&tag))
            {
                tissue[c].push(e);
            }
        }
        let cells: Vec<usize> = (0..n_coarse).filter(|&c| !tissue[c].is_empty()).collect();
        let numbering = mesh.levels()[0].cell_numbering()?;
        let global: Vec<i64> = cells
            .iter()
            .map(|&c| {
                let g = i64::from(numbering[c]);
                // Without overlap every local cell is owned; a cell of another process would
                // be `-(g + 1)`.
                if g >= 0 { g } else { -(g + 1) }
            })
            .collect();
        let centroids: Vec<[f64; 2]> = cells
            .iter()
            .map(|&c| [sums[c][0] / sums[c][2], sums[c][1] / sums[c][2]])
            .collect();
        let means = Means {
            disc,
            q,
            maps,
            tissue: cells.iter().map(|&c| tissue[c].as_slice()).collect(),
            global: &global,
            centroids: &centroids,
        };

        let computed = (|| {
            let mut parameters: Vec<Vec<(String, f64)>> = vec![Vec::new(); cells.len()];
            for p in &spec.parameters {
                let what = format!("cells.parameters '{}'", p.id);
                let mut values = means.coefficient(&p.value, &what)?;
                if let Some(rule) = &p.replace {
                    let map = means.map(&rule.map, &format!("{what}, replace.map"))?;
                    let replaced = means.coefficient(&rule.value, &format!("{what}, replace"))?;
                    for ((v, m), r) in values.iter_mut().zip(map).zip(replaced) {
                        if m > rule.above {
                            *v = r;
                        }
                    }
                }
                for (cell, v) in parameters.iter_mut().zip(values) {
                    cell.push((p.id.clone(), v));
                }
            }
            let porosity = means.coefficient(porosity, "the porosity of the blood")?;
            for (i, (cell, phi)) in parameters.iter_mut().zip(porosity).enumerate() {
                if !(phi > 0.0 && phi < 1.0) {
                    return Err(means.element(
                        i,
                        format!(
                            "the mean porosity {phi} leaves no {} volume (it must be in (0, 1))",
                            if phi <= 0.0 { "blood" } else { "cell" }
                        ),
                    ));
                }
                cell.push((spec.volumes.blood.clone(), phi));
                cell.push((spec.volumes.cells.clone(), 1.0 - phi));
            }
            Ok(parameters)
        })();
        let parameters = agree(&comm, computed)?;
        Ok(CellSetup {
            cells,
            parameters,
            global,
            centroids,
            comm,
        })
    }

    /// The communicator of the cells.
    pub fn comm(&self) -> Comm {
        self.comm
    }

    /// [`CellsError::Invalid`] unless the lists of the setup are one entry per cell.
    pub(crate) fn check(&self) -> Result<()> {
        let n = self.cells.len();
        let lengths = [
            self.parameters.len(),
            self.global.len(),
            self.centroids.len(),
        ];
        if lengths.iter().any(|&l| l != n) {
            return Err(CellsError::Invalid(format!(
                "a setup of {n} cells with {} parameter lists, {} global numbers and {} \
                 centroids",
                lengths[0], lengths[1], lengths[2]
            )));
        }
        Ok(())
    }
}

/// The areas of `spec` (default: every area tag of `mesh`), each an area tag of `mesh`.
fn areas(mesh: &Mesh, spec: &CellsSpec) -> Result<Vec<String>> {
    let dim = mesh.finest().dimension()?;
    let available: Vec<String> = mesh
        .tags()
        .iter()
        .filter(|tag| usize::from(tag.dim) == dim)
        .map(|tag| tag.name.clone())
        .collect();
    let Some(areas) = &spec.areas else {
        return Ok(available);
    };
    match areas.iter().find(|a| !available.contains(a)) {
        Some(area) => Err(CellsError::UnknownArea {
            area: area.clone(),
            available,
        }),
        None => Ok(areas.clone()),
    }
}

/// Means over the tissue of the cells.
struct Means<'a> {
    disc: &'a Discretization,
    /// The quadrature points of each cell.
    q: usize,
    maps: &'a Maps,
    /// The finest local cells of each cell's tissue.
    tissue: Vec<&'a [usize]>,
    global: &'a [i64],
    centroids: &'a [[f64; 2]],
}

impl Means<'_> {
    /// The mean of the coefficient `c` over each cell's tissue (a constant as it is); `what`
    /// names it in errors.
    fn coefficient(&self, c: &Coefficient, what: &str) -> Result<Vec<f64>> {
        match c {
            Coefficient::Constant(value) => Ok(vec![*value; self.tissue.len()]),
            Coefficient::Table(table) => {
                let map = self.get(&table.map)?;
                self.mean(what, |x| map.sample(x).map(|v| c.eval(v)))
            }
        }
    }

    /// The mean of the map `id` over each cell's tissue; `what` names it in errors.
    fn map(&self, id: &str, what: &str) -> Result<Vec<f64>> {
        let map = self.get(id)?;
        self.mean(what, |x| map.sample(x))
    }

    fn get(&self, id: &str) -> Result<&maps::Map> {
        self.maps
            .get(id)
            .map_err(|e| CellsError::Invalid(e.to_string()))
    }

    /// The mean of `f` over each cell's tissue: the first point where it fails is an error of
    /// its element, naming `what`.
    fn mean(
        &self,
        what: &str,
        f: impl Fn([f64; 2]) -> std::result::Result<f64, maps::MapError>,
    ) -> Result<Vec<f64>> {
        let q = self.q;
        let (points, weights) = (
            self.disc.quadrature_points(),
            self.disc.quadrature_weights(),
        );
        let mut means = Vec::with_capacity(self.tissue.len());
        for (i, tissue) in self.tissue.iter().enumerate() {
            let (mut sum, mut total) = (0.0, 0.0);
            for &e in *tissue {
                for k in e * q..(e + 1) * q {
                    let value = f(points[k]).map_err(|e| {
                        self.element(i, format!("{e} in the element's tissue ({what})"))
                    })?;
                    sum += weights[k] * value;
                    total += weights[k];
                }
            }
            means.push(sum / total);
        }
        Ok(means)
    }

    /// The error `message` of cell `i`.
    fn element(&self, i: usize, message: String) -> CellsError {
        CellsError::Element {
            cell: self.global[i],
            centroid: self.centroids[i],
            message,
        }
    }
}
