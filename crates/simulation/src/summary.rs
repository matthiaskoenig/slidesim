//! The summary of a simulation ([`Summary`]), `summary.json` of `slidesim run`:
//!
//! ```text
//! {"fields": {"<field id>": {
//!     "times": [...], "l2_norm": [...], "total_mass": [...],
//!     "mass_balance": [null | {
//!         "storage_rate": ..., "source": ..., "advection_defect": ..., "exchange": ...,
//!         "boundary": {"<tag>": {"advective": ..., "diffusive": ..., "resupply": ...}}}]}},
//!  "perfusion": {"tag_flows": {"<tag>": <um^2/s>},
//!                "pressure_islands": {"count": <n>, "area_um2": <um^2>}},
//!  "cells": {"count": <n>, "steps": <n>, "rejected": <n>, "seconds": {"mean": ..., "max": ...},
//!            "exchanged": {"<field id>": [...]}, "amounts": {"<species id>": [...]}},
//!  "phases": {"setup": ..., "pressure": ..., "transport": ..., "exchange": ...,
//!             "cells": ..., "output": ...}}
//! ```
//!
//! with, per output time, the time reached (s), the L2 norm of the field (mM*um), its amount of
//! substance in the domain, the integral of the field weighed by the porosity (mM*um^2,
//! [`Problem::total_mass`]), and its mass balance over the interval since the output time
//! before (since t = 0 for the first), each term its mean rate there
//! ([`Problem::interval_balance`], every term in mM*um^2/s; `null` for an interval of no
//! length, as at time 0):
//! `storage_rate = source + sum over the tags of (advective + diffusive + resupply) + exchange
//! - advection_defect`, the inflow through each tag of the boundary positive into the domain
//! ([`pde::BoundaryInflow`]), `exchange` what the cells gave to the field, and `resupply` what
//! the Dirichlet values of a tag resupplied when the cells changed the field next to them.
//! `storage_rate` is the change of `total_mass` over the interval divided by its length, and
//! the terms add up to it exactly, up to the residuals of the linear solves.
//! `cells` is there for a study with cells: their number, the steps and rejected steps of
//! their integrators summed over the cells and the run, the time of a process in its cells (s,
//! the mean over the processes and the largest; the phase `cells` is the largest and the wait
//! of the others for it), and per output time the amount each
//! linked field received from the cells since t = 0 (mM*um^2) and the amount of each species
//! of the cell compartment in all cells, in the units of the fields (mM*um^2: its
//! concentration times the cell volume of the model, times the blood volume of the element in
//! um^2 over the blood volume of the model). `perfusion` is there for a study with perfusion:
//! the flow through each pressure tag ([`Flow::tag_flows`], positive into the domain), and the
//! pressure islands, the connected components of the mesh which no pressure tag reaches, whose
//! pressure is 0 and which have no flow ([`Flow::islands`]): their number and area (0 and 0.0
//! without). `phases` is the wall time
//! of process 0 in each phase of the run (s), the phases separated by barriers: `setup` (from the
//! start of the process: PETSc, the study, the cell model, the mesh, the maps, the problems of
//! the fields and the cells), `pressure` (the flow, zero without perfusion), `transport` (the
//! steps of the fields), `exchange` (the means of the fields over the cells and the changes
//! given back), `cells` (the steps of the cells) and `output` (the values of the summary and
//! what the caller does between the output times, the files of `slidesim run`); together the
//! time of the run but the writing of `summary.json` itself.
//!
//! [`Problem::total_mass`]: pde::Problem::total_mass
//! [`Problem::interval_balance`]: pde::Problem::interval_balance
//! [`Flow::tag_flows`]: pde::Flow::tag_flows
//! [`Flow::islands`]: pde::Flow::islands

use std::collections::BTreeMap;

use pde::MassBalance;
use serde::Serialize;

/// `summary.json` (see the module documentation).
#[derive(Debug, Serialize)]
pub struct Summary {
    pub fields: BTreeMap<String, FieldSummary>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub perfusion: Option<PerfusionSummary>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cells: Option<CellsSummary>,
    /// The wall time of process 0 in each phase: s.
    pub phases: BTreeMap<&'static str, f64>,
}

/// A time over the processes: its mean and its largest value (s).
#[derive(Debug, Default, Clone, Copy, Serialize)]
pub struct Seconds {
    pub mean: f64,
    pub max: f64,
}

/// The cells of the study.
#[derive(Debug, Default, Serialize)]
pub struct CellsSummary {
    /// The number of cells.
    pub count: usize,
    /// The steps of the integrators of the cells and the rejected step attempts, summed.
    pub steps: u64,
    pub rejected: u64,
    /// The time of each process in its cells (without the wait for the others) over the
    /// processes: the mean and the largest; their ratio is the imbalance of the work.
    pub seconds: Seconds,
    /// Per linked field and output time, the amount the cells gave to the field since t = 0:
    /// mM*um^2.
    pub exchanged: BTreeMap<String, Vec<f64>>,
    /// Per species of the cell compartment and output time, its amount in all cells: mM*um^2.
    pub amounts: BTreeMap<String, Vec<f64>>,
}

/// The flow of the study.
#[derive(Debug, Serialize)]
pub struct PerfusionSummary {
    /// The flow through each pressure tag, positive into the domain: um^2/s.
    pub tag_flows: BTreeMap<String, f64>,
    pub pressure_islands: IslandsSummary,
}

/// [`pde::Floating`] of [`pde::Flow::islands`] in `summary.json`.
#[derive(Debug, Serialize)]
pub struct IslandsSummary {
    pub count: usize,
    pub area_um2: f64,
}

/// The values of one field at the output times.
#[derive(Debug, Default, Serialize)]
pub struct FieldSummary {
    /// s.
    pub times: Vec<f64>,
    /// mM*um.
    pub l2_norm: Vec<f64>,
    /// The amount of substance, `int phi c` (porosity `phi`): mM*um^2.
    pub total_mass: Vec<f64>,
    /// The mass balance over the interval since the output time before, `null` for an interval
    /// of no length.
    pub mass_balance: Vec<Option<BalanceSummary>>,
}

/// [`pde::MassBalance`] in `summary.json`: mM*um^2/s.
#[derive(Debug, Serialize)]
pub struct BalanceSummary {
    pub storage_rate: f64,
    pub source: f64,
    pub advection_defect: f64,
    pub exchange: f64,
    /// The inflow through each tag of the boundary, positive into the domain.
    pub boundary: BTreeMap<String, InflowSummary>,
}

/// [`pde::BoundaryInflow`] in `summary.json`: mM*um^2/s.
#[derive(Debug, Serialize)]
pub struct InflowSummary {
    pub advective: f64,
    pub diffusive: f64,
    pub resupply: f64,
}

impl From<MassBalance> for BalanceSummary {
    fn from(balance: MassBalance) -> BalanceSummary {
        BalanceSummary {
            storage_rate: balance.storage_rate,
            source: balance.source,
            advection_defect: balance.advection_defect,
            exchange: balance.exchange,
            boundary: balance
                .boundary
                .into_iter()
                .map(|b| {
                    let inflow = InflowSummary {
                        advective: b.advective,
                        diffusive: b.diffusive,
                        resupply: b.resupply,
                    };
                    (b.tag, inflow)
                })
                .collect(),
        }
    }
}
