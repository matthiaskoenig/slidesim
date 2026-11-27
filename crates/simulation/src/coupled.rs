//! Coupled runs: cell models on the coarse elements of the mesh and the fields of a study, by
//! Strang splitting ([`Coupled`]).

use std::path::Path;

use anyhow::{Context, Result, bail};
use ceed::Ceed;
use cells::{CellModel, CellSetup, CellSolver, Cells};
use contracts::{CellsSpec, Study};
use maps::Maps;
use mesh::Mesh;
use pde::{Exchange, Problem};
use petsc::Comm;

use crate::phases::Phases;
use crate::summary::Seconds;
use crate::vtu::CellWriter;

/// The steps of the cells of every process ([`Coupled::stats`]).
pub(crate) struct CellStats {
    pub(crate) steps: u64,
    pub(crate) rejected: u64,
    pub(crate) seconds: Seconds,
}

/// The cells of a study coupled to its fields by Strang splitting.
///
/// The time from the current time to the next output time (or the end) is split into `n` equal
/// split steps of at most `cells.split_step`, `[a_k, b_k]`. Each is: the cells over the first
/// half, `[a_k, m_k]`; every field (linked or not) over the step, [`Problem::step_to`] `b_k`;
/// the cells over the second half, `[m_k, b_k]`. Two half steps of the cells with no transport
/// between them, the second half of a split step and the first half of the next, are one step
/// of the cells from `m_k` to `m_(k+1)`: the same scheme with half the cell steps (and half the
/// restarts of the time integration of the fields). At an output time the half steps are apart:
/// the cells reach it before the outputs are written.
///
/// A cell step from `t0` to `t1`: the means of the linked fields over each cell
/// ([`Exchange::means`]), the concentrations of the linked species in its blood at `t0`; the
/// cells advanced with them ([`CellSolver::advance`]); the change of each concentration over
/// the step given back to its field ([`Exchange::give_back`]), which acts on the field the
/// means were read from (no step of the fields between them), books the exchange and the
/// resupply of the Dirichlet values in the mass balance over an interval
/// ([`Problem::interval_balance`]), and starts the time integration of the field afresh.
///
/// With adaptive time steps, each field steps to the end of a split step
/// ([`Problem::step_to`]), whose last step is shortened to end there: its steps never pass the
/// end of a split step, whatever their size.
pub(crate) struct Coupled {
    solver: Cells,
    /// For each linked species (in the order of `cells.species`): the index of its field in the
    /// study and its exchange.
    links: Vec<(usize, Exchange)>,
    split_step: f64,
    /// The time the cells reached (s).
    t: f64,
    /// Per cell and linked species: the concentrations in the blood.
    blood: Vec<f64>,
    before: Vec<f64>,
    work: Vec<f64>,
    /// The steps and rejected steps of the cells of this process, and its time in them (s).
    steps: u64,
    rejected: u64,
    seconds: f64,
    /// For each linked species, the amount the cells gave to its field since t = 0 (mM*um^2),
    /// over every process.
    exchanged: Vec<f64>,
    /// The cells (local coarse cells) and their parameters, for the outputs.
    setup: CellSetup,
    /// Per cell: `V_i / Vb_i`, the blood volume of its element (um^2) over the blood volume of
    /// its model (in the model's units): the factor from amounts of the model to amounts of
    /// the fields.
    scale: Vec<f64>,
    /// Per cell: the volume of the cell compartment of the model.
    cell_volume: Vec<f64>,
    cells_compartment: String,
    writer: CellWriter,
    /// The number of cells of every process.
    count: usize,
}

impl Coupled {
    /// The cells of `spec`, the `cells` of `study`, with the model `model` (compiled and checked
    /// at the start of the study), on `mesh` with the maps `maps`, coupled to the fields of
    /// `problems` (those of `study.fields`, in order), on the path of `spec` (the batched one
    /// on `ceed`). Collective.
    ///
    /// Errors: those of [`linked_fields`], [`CellSetup::new`], [`Cells::new`] and
    /// [`Exchange::new`].
    pub(crate) fn new(
        study: &Study,
        spec: &CellsSpec,
        model: &CellModel,
        mesh: &Mesh,
        maps: &Maps,
        problems: &[Problem],
        ceed: &Ceed,
    ) -> Result<Coupled> {
        let fields = linked_fields(study, spec)?;
        let first = fields[0];
        let porosity = &study.fields[first].porosity;
        let disc = problems[first].discretization();
        let setup = CellSetup::new(disc, mesh, spec, maps, porosity).context("cells")?;
        let solver = Cells::new(&setup, spec, model, ceed).context("cells")?;
        let links = fields
            .iter()
            .map(|&f| {
                let exchange = Exchange::new(
                    problems[f].discretization(),
                    mesh,
                    &setup.cells,
                    porosity,
                    maps,
                )
                .with_context(|| {
                    format!(
                        "the exchange of the cells with field '{}'",
                        study.fields[f].id
                    )
                })?;
                Ok((f, exchange))
            })
            .collect::<Result<Vec<_>>>()?;
        let volume = |cell: &[(String, f64)], id: &str| {
            cell.iter()
                .find(|(p, _)| p == id)
                .expect("the setup sets both volumes")
                .1
        };
        let blood_volumes = links[0].1.volumes();
        let scale = setup
            .parameters
            .iter()
            .zip(blood_volumes)
            .map(|(cell, v)| v / volume(cell, &spec.volumes.blood))
            .collect();
        let cell_volume = setup
            .parameters
            .iter()
            .map(|cell| volume(cell, &spec.volumes.cells))
            .collect();
        let names: Vec<String> = solver
            .output_values()
            .iter()
            .map(|v| v.id.clone())
            .chain(parameter_ids(spec))
            .collect();
        let writer = CellWriter::new(&mesh.levels()[0], names)?;
        let comm = mesh.finest().comm();
        let count =
            usize::try_from(comm.sum_int(petsc::sys::PetscInt::try_from(setup.cells.len())?)?)?;
        let n = solver.n_cells() * spec.species.len();
        Ok(Coupled {
            solver,
            links,
            split_step: spec.split_step,
            t: problems[first].time(),
            blood: vec![0.0; n],
            before: vec![0.0; n],
            work: Vec::new(),
            steps: 0,
            rejected: 0,
            seconds: 0.0,
            exchanged: vec![0.0; spec.species.len()],
            setup,
            scale,
            cell_volume,
            cells_compartment: spec.volumes.cells.clone(),
            writer,
            count,
        })
    }

    /// Advances the cells and the fields of `problems` from their current time to `t_end`, by
    /// split steps (see [`Coupled`]), the cells ending at `t_end` too. Collective.
    pub(crate) fn advance(
        &mut self,
        study: &Study,
        problems: &mut [Problem],
        t_end: f64,
        phases: &mut Phases<'_>,
    ) -> Result<()> {
        let t0 = self.t;
        if t_end <= t0 {
            return Ok(());
        }
        // Equal split steps of at most the split step; a split step a rounding of a whole one
        // longer is one.
        let n = ((t_end - t0) / self.split_step * (1.0 - 1e-12))
            .ceil()
            .max(1.0);
        let h = (t_end - t0) / n;
        let n = n as usize;
        for k in 0..n {
            let a = t0 + k as f64 * h;
            let b = if k + 1 == n {
                t_end
            } else {
                t0 + (k + 1) as f64 * h
            };
            self.cell_step(problems, 0.5 * (a + b), phases)?;
            for (field, problem) in study.fields.iter().zip(problems.iter_mut()) {
                problem
                    .step_to(b)
                    .with_context(|| format!("field '{}'", field.id))?;
                // Adaptive steps too: the last step is shortened to end at the split step.
                if problem.time() != b {
                    bail!(
                        "field '{}' stepped to t = {} s instead of the end of the split step, \
                         t = {b} s",
                        field.id,
                        problem.time()
                    );
                }
            }
            phases.end("transport")?;
        }
        self.cell_step(problems, t_end, phases)
    }

    /// One step of the cells from where they are to `t1`, with the exchange before and after
    /// it. Collective.
    fn cell_step(
        &mut self,
        problems: &mut [Problem],
        t1: f64,
        phases: &mut Phases<'_>,
    ) -> Result<()> {
        let n = self.solver.n_cells();
        let ns = self.links.len();
        for (s, (field, exchange)) in self.links.iter().enumerate() {
            self.work.resize(n, 0.0);
            exchange.means(&problems[*field], &mut self.work)?;
            for (i, mean) in self.work.iter().enumerate() {
                self.blood[i * ns + s] = *mean;
            }
        }
        self.before.copy_from_slice(&self.blood);
        phases.end("exchange")?;
        let stats = self.solver.advance(self.t, t1, &mut self.blood)?;
        self.steps += stats.steps;
        self.rejected += stats.rejected;
        self.seconds += stats.seconds;
        self.t = t1;
        phases.end("cells")?;
        for (s, (field, exchange)) in self.links.iter().enumerate() {
            self.work.clear();
            self.work
                .extend((0..n).map(|i| self.blood[i * ns + s] - self.before[i * ns + s]));
            self.exchanged[s] += exchange
                .give_back(&mut problems[*field], &self.work)?
                .exchange;
        }
        phases.end("exchange")?;
        Ok(())
    }

    /// The number of cells of every process.
    pub(crate) fn count(&self) -> usize {
        self.count
    }

    /// The steps and the rejected steps of the cells of every process, and the mean and the
    /// largest time of a process in its cells (s, without the wait for the others).
    /// Collective.
    pub(crate) fn stats(&self, comm: &Comm) -> Result<CellStats> {
        let sum = |x: u64| -> Result<u64> {
            Ok(u64::try_from(
                comm.sum_int(petsc::sys::PetscInt::try_from(x)?)?,
            )?)
        };
        Ok(CellStats {
            steps: sum(self.steps)?,
            rejected: sum(self.rejected)?,
            seconds: Seconds {
                mean: comm.sum_real(self.seconds)? / comm.size() as f64,
                max: comm.max_real(self.seconds)?,
            },
        })
    }

    /// For each linked species, the index of its field and the amount the cells gave to it
    /// since t = 0 (mM*um^2).
    pub(crate) fn exchanged(&self) -> impl Iterator<Item = (usize, f64)> + '_ {
        self.links
            .iter()
            .map(|(f, _)| *f)
            .zip(self.exchanged.iter().copied())
    }

    /// The amount of every species of the cell compartment in every cell, in the units of the
    /// fields (mM*um^2): its concentration times the cell volume of the model, times the factor
    /// of the cell from the model's volumes to um^2. Collective.
    pub(crate) fn amounts(&self, comm: &Comm) -> Result<Vec<(String, f64)>> {
        let mut outputs = Vec::new();
        self.solver.outputs(&mut outputs);
        let values = self.solver.output_values();
        let no = values.len();
        let mut amounts: Vec<(String, f64)> = Vec::new();
        for (k, value) in values.iter().enumerate() {
            if value.compartment.as_deref() != Some(self.cells_compartment.as_str()) {
                continue;
            }
            let local: f64 = (0..self.solver.n_cells())
                .map(|i| outputs[i * no + k] * self.cell_volume[i] * self.scale[i])
                .sum();
            amounts.push((value.id.clone(), comm.sum_real(local)?));
        }
        Ok(amounts)
    }

    /// The names of the values of a cell ([`Coupled::table`]): the output values of the model
    /// (its states, a species of a concentration as its concentration, and `cells.outputs`),
    /// then the parameters of the element ([`CellSetup`]: `cells.parameters` and the two
    /// volumes).
    pub(crate) fn names(&self) -> &[String] {
        self.writer.names()
    }

    /// The element ids of the local cells ([`CellSetup::elements`]), in the order of the rows of
    /// [`Coupled::table`].
    pub(crate) fn elements(&self) -> &[u64] {
        &self.setup.elements
    }

    /// The values of the local cells now: one row per cell, one value per name of
    /// [`Coupled::names`].
    pub(crate) fn table(&self) -> Vec<f64> {
        let mut outputs = Vec::new();
        self.solver.outputs(&mut outputs);
        let no = self.solver.output_values().len();
        let parameters = self.setup.parameters.first().map_or(0, Vec::len);
        let columns = no + parameters;
        let mut table = Vec::with_capacity(self.solver.n_cells() * columns);
        for (i, cell) in self.setup.parameters.iter().enumerate() {
            table.extend_from_slice(&outputs[i * no..(i + 1) * no]);
            table.extend(cell.iter().map(|(_, v)| *v));
        }
        table
    }

    /// Writes the cells as the VTU file `path`: cell data of the coarse elements, one array per
    /// name of [`Coupled::names`], NaN where there is no cell. Collective.
    pub(crate) fn write(&self, path: &Path) -> Result<()> {
        self.writer
            .write(&self.setup.cells, &self.table(), path)
            .with_context(|| format!("cannot write {}", path.display()))
    }
}

/// The fields of `study` linked to the species of `spec`, in the order of the species: checked at
/// the start of a study, before anything is solved.
///
/// Errors: no linked species, a field linked to two species, linked fields of different porosity
/// (the blood volume of a cell is their porosity), and a field named `cells`, whose outputs
/// would be those of the cells.
pub(crate) fn linked_fields(study: &Study, spec: &CellsSpec) -> Result<Vec<usize>> {
    if study.fields.iter().any(|f| f.id == "cells") {
        bail!("a study with cells has no field 'cells': its outputs would be those of the cells");
    }
    let mut fields: Vec<usize> = Vec::new();
    for link in &spec.species {
        let index = study
            .fields
            .iter()
            .position(|f| f.id == link.field)
            .expect("the contracts check the linked fields");
        if fields.contains(&index) {
            bail!(
                "cells.species: the field '{}' is linked to two species of the model",
                link.field
            );
        }
        fields.push(index);
    }
    let Some(&first) = fields.first() else {
        bail!("cells.species: no species of the model is linked to a field");
    };
    let porosity = &study.fields[first].porosity;
    if let Some(&other) = fields
        .iter()
        .find(|&&f| study.fields[f].porosity != *porosity)
    {
        bail!(
            "cells.species: the linked fields '{}' and '{}' have different porosities; the \
             blood volume of the cells is the porosity of the fields they are linked to",
            study.fields[first].id,
            study.fields[other].id
        );
    }
    Ok(fields)
}

/// The ids of the parameters [`CellSetup`] sets per cell, in its order.
fn parameter_ids(spec: &CellsSpec) -> Vec<String> {
    spec.parameters
        .iter()
        .map(|p| p.id.clone())
        .chain([spec.volumes.blood.clone(), spec.volumes.cells.clone()])
        .collect()
}
