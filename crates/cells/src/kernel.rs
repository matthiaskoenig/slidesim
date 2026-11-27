//! The batched kernel of sbmlsim-core (`kernel/1`): its layout ([`KernelLayout`], from
//! `kernel.json`) and, without the feature `native`, the cells of a process on it
//! (`KernelCells`).
//!
//! The kernel is a WebAssembly module of the cell model with the Rosenbrock integrator RODAS4 of
//! the QFunction of the batched path (sbmlsim-core's `kernel/1`, see the documentation of
//! `sbmlsim_abi` at the pinned revision `SBMLSIM_CORE_KERNEL_REV` of `deps/versions.env`), which
//! the host compiles from the SBML of the study (`compiler.compileKernel` of sbmlsim-core's
//! JavaScript, `sbmlsim-core compile --kernel`). Its functions take byte offsets into the
//! memory of their caller: the browser build of slidesim imports them as
//! `slidesim_kernel.init`, `slidesim_kernel.advance` and `slidesim_kernel.assign`, and the host instantiates the kernel
//! with slidesim's memory and binds them (`wasm/host`, Task 17's JavaScript).

use sbmlsim_abi::ModelDescription;
use serde::Deserialize;

use crate::{CellsError, Result};

/// The layout of the points of a batched kernel and the description of its model: the keys of
/// `kernel.json` which slidesim reads (the others, such as `version`, `control` and `counts`,
/// are ignored: `kernel/1` fixes the four values of the control and the two of the counts).
/// `assigned_ids` is optional: a kernel of sbmlsim-core before `sbmlsim_kernel_assign` has
/// none, and then a target of an assignment rule cannot be an output ([`crate::CellModel`]).
#[derive(Debug, Clone, Deserialize)]
pub struct KernelLayout {
    /// The ids of the states of a point, in their order.
    pub state_ids: Vec<String>,
    /// The ids of the parameters of a point, in their order.
    pub parameter_ids: Vec<String>,
    /// The `f64` of the scratch of a point, the work of the integrator.
    pub scratch_per_point: usize,
    /// The parameters which `sbmlsim_kernel_assign` writes, the targets of the assignment rules
    /// in the order of the rules; `None` for a kernel without `sbmlsim_kernel_assign`.
    #[serde(default)]
    pub assigned_ids: Option<Vec<String>>,
    /// The description of the model (of sbmlsim-core's C target), which the checks of a study
    /// read as natively.
    pub model: ModelDescription,
}

impl KernelLayout {
    /// Checks the kernel: a model it integrates, of its states and parameters, and assigning
    /// parameters of it.
    ///
    /// Errors: [`CellsError::Unsupported`] for a model with events, delays or algebraic rules
    /// (which no kernel integrates), [`CellsError::Model`] for states or parameters other than
    /// those of the model, or an assigned id which is not a parameter.
    pub fn check(&self) -> Result<()> {
        let model = &self.model;
        let refused = [
            (!model.events.is_empty(), "events"),
            (model.n_delays > 0, "delays"),
            (model.n_algebraic > 0, "algebraic rules"),
        ];
        if let Some((_, feature)) = refused.iter().find(|(has, _)| *has) {
            return Err(CellsError::Unsupported {
                feature: format!("a model with {feature}"),
            });
        }
        let ids = |entities: &[sbmlsim_abi::Entity]| -> Vec<String> {
            entities.iter().map(|e| e.id.clone()).collect()
        };
        if self.state_ids != ids(&model.states) || self.parameter_ids != ids(&model.parameters) {
            return Err(CellsError::Model {
                message: format!(
                    "the kernel of the model {} has other states or parameters than its \
                     description",
                    model.id
                ),
            });
        }
        if let Some(id) = self
            .assigned_ids
            .iter()
            .flatten()
            .find(|id| !self.parameter_ids.contains(id))
        {
            return Err(CellsError::Model {
                message: format!(
                    "the kernel of the model {} assigns '{id}', which is not one of its \
                     parameters",
                    model.id
                ),
            });
        }
        Ok(())
    }
}

#[cfg(not(feature = "native"))]
pub use solver::KernelCells;

#[cfg(not(feature = "native"))]
mod solver {
    use std::time::Instant;

    use contracts::CellsSpec;
    use petsc::OwnedComm;
    use sbmlsim_abi::EntityKind;

    use crate::model::{Link, Outputs, Volume, Work};
    use crate::{
        CellModel, CellSetup, CellSolver, CellStats, CellsError, H_MIN, MAX_ATTEMPTS, OutputValue,
        Poison, Result, agree, failed_message, unfinished_message,
    };

    /// The values of the control of a point (`kernel/1`): `t`, `h`, done, failed.
    const CONTROL: usize = 4;
    const CONTROL_T: usize = 0;
    const CONTROL_DONE: usize = 2;
    const CONTROL_FAILED: usize = 3;
    /// The values of the counts of a point: the accepted steps and the rejected step attempts.
    const COUNTS: usize = 2;

    // The functions of the kernel, which the host binds (see the module documentation). The
    // pointers are byte offsets into the memory of slidesim on wasm32.
    #[link(wasm_import_module = "slidesim_kernel")]
    unsafe extern "C" {
        /// `sbmlsim_kernel_init(n, parameters, state) -> i32`.
        #[link_name = "init"]
        fn kernel_init(n: i32, parameters: *mut f64, state: *mut f64) -> i32;
        /// `sbmlsim_kernel_advance(n, state, parameters, control, counts, context, scratch) ->
        /// i32`.
        #[link_name = "advance"]
        fn kernel_advance(
            n: i32,
            state: *mut f64,
            parameters: *const f64,
            control: *mut f64,
            counts: *mut f64,
            context: *const f64,
            scratch: *mut f64,
        ) -> i32;
        /// `sbmlsim_kernel_assign(n, state, parameters, control) -> i32`.
        #[link_name = "assign"]
        fn kernel_assign(
            n: i32,
            state: *const f64,
            parameters: *mut f64,
            control: *const f64,
        ) -> i32;
    }

    /// The cells of a process on the batched kernel of their model: the batched path without
    /// native code (the browser build), one point of the kernel per cell. The kernel's
    /// integrator, step control and failures are those of the QFunction of [`crate::BatchedCells`]
    /// (`H_MIN` 1e-10, at most `MAX_ATTEMPTS` 10^6 step attempts of a cell in one advance), so a
    /// cell takes the steps it takes there, its math that of sbmlsim-core's mathlib (the same
    /// within the tolerances, not bitwise).
    ///
    /// The cells are initialized by the kernel (`sbmlsim_kernel_init`): every parameter NaN (not
    /// given: its SBML value and initial assignment) but those of the cell's parameters
    /// ([`CellSetup::parameters`]), which are fixed. A parameter of the setup which is a state
    /// of the model (the initial amount of a species, as the engine reads a change `S`) cannot
    /// go through the initialization: it is written into the state after it (divided by the
    /// volume of its compartment where the model stores the concentration), and the initial
    /// assignments which read that state are not evaluated again (the native paths evaluate
    /// them with it).
    ///
    /// An advance is one call of `sbmlsim_kernel_advance` for all cells, with the attempts of
    /// the context `MAX_ATTEMPTS`: each cell steps until it is done or failed. The scratch of
    /// the integrator is slidesim's (`scratch_per_point` values per cell). The outputs which are
    /// targets of assignment rules are evaluated at the time and the states of the cells by one
    /// call of `sbmlsim_kernel_assign` for all cells whenever they are read. One process: the
    /// browser build has one.
    pub struct KernelCells {
        species: Vec<String>,
        state_ids: Vec<String>,
        /// Of every cell, after the last advance.
        states: Vec<f64>,
        /// The states during an advance.
        next: Vec<f64>,
        /// Of every cell, in the order of the parameters of the kernel, as the initialization
        /// left them.
        parameters: Vec<f64>,
        n_parameters: usize,
        /// The control of every cell (`t`, `h`, done, failed), after the last advance.
        control: Vec<f64>,
        /// The accepted and rejected steps of every cell in the last advance.
        counts: Vec<f64>,
        scratch: Vec<f64>,
        links: Vec<Link>,
        outputs: Outputs,
        time_unit_s: f64,
        rtol: f64,
        atol: f64,
        elements: Vec<u64>,
        centroids: Vec<[f64; 2]>,
        comm: OwnedComm,
        poison: Poison,
        /// The time the cells reached (s).
        t: f64,
    }

    impl std::fmt::Debug for KernelCells {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("KernelCells")
                .field("cells", &self.elements.len())
                .field("species", &self.species)
                .field("state_ids", &self.state_ids)
                .field("time_unit_s", &self.time_unit_s)
                .finish_non_exhaustive()
        }
    }

    /// A count of points as the `i32` of the kernel.
    fn points(n: usize) -> Result<i32> {
        i32::try_from(n).map_err(|_| CellsError::Invalid(format!("{n} cells for the kernel")))
    }

    impl KernelCells {
        /// The cells of `setup` with the model `model` ([`CellModel::from_kernel`]), checked
        /// for `spec`, initialized by the kernel (see [`KernelCells`]). Collective.
        ///
        /// Errors: [`CellsError::Invalid`] for a setup whose lists are not one entry per cell;
        /// [`CellsError::Model`] if the kernel refuses its arguments; on every process,
        /// [`CellsError::Element`] for a cell whose initial state or parameters are not
        /// finite.
        pub fn with_model(
            setup: &CellSetup,
            spec: &CellsSpec,
            model: &CellModel,
        ) -> Result<KernelCells> {
            let comm = setup.owned_comm();
            agree(&comm.comm(), setup.check())?;
            let layout = &model.kernel;
            let (ns, np) = (layout.state_ids.len(), layout.parameter_ids.len());
            let n = setup.parameters.len();
            // The parameters: NaN but the cell's; its states the kernel's, then the cell's.
            let mut parameters = vec![f64::NAN; n * np];
            let mut changed_states = Vec::new();
            for (i, changes) in setup.parameters.iter().enumerate() {
                for (id, value) in changes {
                    if let Some(k) = layout.parameter_ids.iter().position(|p| p == id) {
                        parameters[i * np + k] = *value;
                    } else if let Some(k) = layout.state_ids.iter().position(|s| s == id) {
                        changed_states.push((i, k, *value));
                    } else {
                        return Err(CellsError::Invalid(format!(
                            "cells.parameters '{id}' is neither a parameter nor a state of the \
                             kernel"
                        )));
                    }
                }
            }
            let mut states = vec![0.0; n * ns];
            if n > 0 {
                // SAFETY: `parameters` holds `n * np` values and `states` `n * ns`, the arrays
                // of `n` points of the kernel's layout, in slidesim's memory.
                let code = unsafe {
                    kernel_init(points(n)?, parameters.as_mut_ptr(), states.as_mut_ptr())
                };
                if code != 0 {
                    return Err(CellsError::Model {
                        message: format!("the kernel refused its initialization ({code})"),
                    });
                }
            }
            let model_states = &model.description.states;
            for &(i, k, amount) in &changed_states {
                let p = &parameters[i * np..(i + 1) * np];
                let y = &states[i * ns..(i + 1) * ns];
                let value = match concentration_volume(model, k) {
                    Some(volume) => amount / volume.of(p, y),
                    None => amount,
                };
                states[i * ns + k] = value;
            }
            let finite = (0..n).try_for_each(|i| {
                let y = &states[i * ns..(i + 1) * ns];
                let p = &parameters[i * np..(i + 1) * np];
                let bad = y
                    .iter()
                    .position(|v| !v.is_finite())
                    .map(|k| &model_states[k].id)
                    .or_else(|| {
                        p.iter()
                            .position(|v| !v.is_finite())
                            .map(|k| &layout.parameter_ids[k])
                    });
                match bad {
                    Some(id) => Err(CellsError::Element {
                        element: setup.elements[i],
                        centroid: setup.centroids[i],
                        message: format!(
                            "the cell model does not initialize: '{id}' is not finite"
                        ),
                    }),
                    None => Ok(()),
                }
            });
            agree(&comm.comm(), finite)?;
            Ok(KernelCells {
                species: spec.species.iter().map(|l| l.species.clone()).collect(),
                state_ids: layout.state_ids.clone(),
                next: vec![0.0; states.len()],
                states,
                parameters,
                n_parameters: np,
                // At t = 0, a step to estimate, neither done nor failed.
                control: vec![0.0; n * CONTROL],
                counts: vec![0.0; n * COUNTS],
                scratch: vec![0.0; n * layout.scratch_per_point],
                links: model.links.clone(),
                outputs: Outputs::new(model, spec),
                time_unit_s: spec.time_unit_s,
                rtol: spec.rtol,
                atol: spec.atol,
                elements: setup.elements.clone(),
                centroids: setup.centroids.clone(),
                comm,
                poison: Poison::default(),
                t: 0.0,
            })
        }

        /// The advance of every cell; a failure an error of the first element which failed.
        fn advance_local(&mut self, t0: f64, t1: f64, blood: &mut [f64]) -> Result<CellStats> {
            let n = self.elements.len();
            let (ns, nb, np) = (self.state_ids.len(), self.links.len(), self.n_parameters);
            if blood.len() != n * nb {
                return Err(CellsError::Invalid(format!(
                    "{} blood values for {n} cells of {nb} species",
                    blood.len()
                )));
            }
            if !(t0.is_finite() && t1.is_finite() && t0 <= t1) {
                return Err(CellsError::Invalid(format!(
                    "an advance of the cells from t = {t0} s to t = {t1} s"
                )));
            }
            let (m0, m1) = (t0 / self.time_unit_s, t1 / self.time_unit_s);
            // Every cell which has not failed starts at t0, with the step it ended the last
            // advance with.
            for control in self.control.chunks_mut(CONTROL) {
                if control[CONTROL_FAILED] == 0.0 {
                    control[CONTROL_T] = m0;
                    control[CONTROL_DONE] = 0.0;
                }
            }
            // The state of each cell with the concentrations of the blood.
            self.next.copy_from_slice(&self.states);
            for i in 0..n {
                let y = &mut self.next[i * ns..(i + 1) * ns];
                let p = &self.parameters[i * np..(i + 1) * np];
                for (link, c) in self.links.iter().zip(&blood[i * nb..(i + 1) * nb]) {
                    let volume = link.volume.of(p, y);
                    y[link.state] = c * volume;
                }
            }
            self.counts.fill(0.0);
            let context = [m1, self.rtol, self.atol, H_MIN, f64::from(MAX_ATTEMPTS)];
            let start = Instant::now();
            if n > 0 {
                // SAFETY: the arrays of `n` points of the kernel's layout (the states, parameters,
                // controls, counts and scratch of `n` points, the five values of the context), in
                // slidesim's memory.
                let failed = unsafe {
                    kernel_advance(
                        points(n)?,
                        self.next.as_mut_ptr(),
                        self.parameters.as_ptr(),
                        self.control.as_mut_ptr(),
                        self.counts.as_mut_ptr(),
                        context.as_ptr(),
                        self.scratch.as_mut_ptr(),
                    )
                };
                if failed < 0 {
                    return Err(CellsError::Model {
                        message: format!(
                            "the kernel refused an advance from t = {t0} s to t = {t1} s"
                        ),
                    });
                }
            }
            let mut stats = CellStats {
                seconds: start.elapsed().as_secs_f64(),
                ..CellStats::default()
            };
            for c in self.counts.chunks(COUNTS) {
                stats.steps += c[0] as u64;
                stats.rejected += c[1] as u64;
            }
            let element = |i: usize, message: String| CellsError::Element {
                element: self.elements[i],
                centroid: self.centroids[i],
                message,
            };
            let reached = |i: usize| self.control[i * CONTROL + CONTROL_T] * self.time_unit_s;
            let control = |i: usize| &self.control[i * CONTROL..(i + 1) * CONTROL];
            if let Some(i) = (0..n)
                .find(|&i| control(i)[CONTROL_DONE] == 0.0 && control(i)[CONTROL_FAILED] == 0.0)
            {
                return Err(element(i, unfinished_message(t0, t1, reached(i))));
            }
            if let Some(i) = (0..n).find(|&i| control(i)[CONTROL_FAILED] != 0.0) {
                return Err(element(i, failed_message(t0, t1, reached(i))));
            }
            std::mem::swap(&mut self.states, &mut self.next);
            for i in 0..n {
                let y = &self.states[i * ns..(i + 1) * ns];
                let p = &self.parameters[i * np..(i + 1) * np];
                for (link, c) in self.links.iter().zip(&mut blood[i * nb..(i + 1) * nb]) {
                    *c = y[link.state] / link.volume.of(p, y);
                }
            }
            self.t = t1;
            Ok(stats)
        }
    }

    /// The volume which divides the amount of the state `k` of `model` into the value the model
    /// stores: that of its compartment for a species whose concentration the model stores, else
    /// none.
    fn concentration_volume(model: &CellModel, k: usize) -> Option<Volume> {
        let description = &model.description;
        let EntityKind::Species {
            compartment,
            stores_concentration: true,
            ..
        } = &description.states[k].kind
        else {
            return None;
        };
        if let Some(i) = description
            .parameters
            .iter()
            .position(|p| &p.id == compartment)
        {
            Some(Volume::Parameter(i))
        } else {
            description
                .states
                .iter()
                .position(|s| &s.id == compartment)
                .map(Volume::State)
        }
    }

    impl CellSolver for KernelCells {
        fn n_cells(&self) -> usize {
            self.elements.len()
        }

        fn species(&self) -> &[String] {
            &self.species
        }

        fn state_ids(&self) -> &[String] {
            &self.state_ids
        }

        fn advance(&mut self, t0: f64, t1: f64, blood: &mut [f64]) -> Result<CellStats> {
            self.poison.check()?;
            let local = self.advance_local(t0, t1, blood);
            let agreed = agree(&self.comm.comm(), local);
            self.poison.keep(agreed)
        }

        fn states(&self) -> &[f64] {
            &self.states
        }

        fn output_values(&self) -> &[OutputValue] {
            self.outputs.values()
        }

        fn outputs(&self, out: &mut Vec<f64>) {
            out.clear();
            let n = self.elements.len();
            let (ns, np) = (self.state_ids.len(), self.n_parameters);
            let t = self.t / self.time_unit_s;
            // The targets of the assignment rules at the time the cells reached and their states
            // (`sbmlsim_kernel_assign`), as the native paths evaluate them (`model_assign`), into
            // a copy of the parameters: the advances keep the values of the initialization.
            let assigned;
            let parameters = if self.outputs.has_rules() && n > 0 {
                let mut p = self.parameters.clone();
                let mut control = vec![0.0; n * CONTROL];
                for c in control.chunks_mut(CONTROL) {
                    c[CONTROL_T] = t;
                }
                // SAFETY: the states, parameters and controls of `n` points of the kernel's
                // layout, in slidesim's memory.
                let code = unsafe {
                    kernel_assign(
                        points(n).expect("the cells were initialized"),
                        self.states.as_ptr(),
                        p.as_mut_ptr(),
                        control.as_ptr(),
                    )
                };
                // Arguments the kernel refuses are a bug of slidesim: its own arrays.
                assert_eq!(
                    code, 0,
                    "sbmlsim_kernel_assign refused the arrays of {n} cells"
                );
                assigned = p;
                &assigned
            } else {
                &self.parameters
            };
            let mut work = Work::default();
            for i in 0..n {
                self.outputs.push(
                    t,
                    &self.states[i * ns..(i + 1) * ns],
                    &parameters[i * np..(i + 1) * np],
                    &mut work,
                    out,
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A layout of `kernel.json`: the keys slidesim reads, the others ignored.
    #[test]
    fn layout_reads_kernel_json() {
        let text = r#"{
            "version": "kernel/1",
            "state_ids": ["x"],
            "parameter_ids": ["k", "V"],
            "control": 4,
            "counts": 2,
            "scratch_per_point": 13,
            "model": {
                "contract": "model/2",
                "compiler": "sbmlsim-compiler 0.38.0",
                "id": "decay",
                "states": [{"id": "x", "kind": {"species": {"compartment": "V",
                    "has_only_substance_units": false, "stores_concentration": false}},
                    "initial": 1.0, "settable": true}],
                "parameters": [
                    {"id": "k", "kind": "parameter", "initial": 0.5, "settable": true},
                    {"id": "V", "kind": "compartment", "initial": 1.0, "settable": true}
                ],
                "selections": [],
                "jacobian": {"analytic": {"n": 1, "col_ptr": [0, 1], "row_idx": [0]}},
                "events": [],
                "n_roots": 0,
                "n_discontinuity_roots": 0,
                "n_constants": 0,
                "units": {}
            }
        }"#;
        let layout: KernelLayout = serde_json::from_str(text).unwrap();
        assert_eq!(
            layout.assigned_ids, None,
            "a kernel before sbmlsim_kernel_assign"
        );
        assert_eq!(layout.state_ids, ["x"]);
        assert_eq!(layout.parameter_ids, ["k", "V"]);
        assert_eq!(layout.scratch_per_point, 13);
        layout.check().unwrap();

        let mut assigning = layout.clone();
        assigning.assigned_ids = Some(vec!["k".into()]);
        assigning.check().unwrap();
        assigning.assigned_ids = Some(vec!["x".into()]);
        let e = assigning.check().unwrap_err();
        assert!(e.to_string().contains("assigns 'x'"), "{e}");
        let with_key = text.replace(
            r#""scratch_per_point": 13,"#,
            r#""scratch_per_point": 13, "assigned_ids": ["k"],"#,
        );
        let read: KernelLayout = serde_json::from_str(&with_key).unwrap();
        assert_eq!(read.assigned_ids, Some(vec!["k".to_string()]));

        let mut other = layout.clone();
        other.parameter_ids.reverse();
        assert!(matches!(other.check(), Err(CellsError::Model { .. })));
        let mut delays = layout;
        delays.model.n_delays = 1;
        let e = delays.check().unwrap_err();
        assert!(e.to_string().contains("a model with delays"), "{e}");
    }
}
