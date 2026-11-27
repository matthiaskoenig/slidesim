//! The wall time of process 0 in each phase of a simulation ([`Phases`]).

use std::collections::BTreeMap;
use std::time::Instant;

use anyhow::Result;
use petsc::Comm;

/// The wall time of process 0 in each phase of a run, the phases separated by barriers, so that
/// a phase ends when every process ended it.
#[derive(Debug)]
pub(crate) struct Phases<'a> {
    comm: Comm<'a>,
    /// The end of the last phase.
    last: Instant,
    seconds: BTreeMap<&'static str, f64>,
}

impl<'a> Phases<'a> {
    /// The phases of a run started at `started`, each at zero seconds.
    pub(crate) fn new(comm: Comm<'a>, started: Instant) -> Phases<'a> {
        let seconds = [
            "setup",
            "pressure",
            "transport",
            "exchange",
            "cells",
            "output",
        ]
        .into_iter()
        .map(|name| (name, 0.0))
        .collect();
        Phases {
            comm,
            last: started,
            seconds,
        }
    }

    /// Ends a time spent in the phase `name`, from the end of the last phase, when every process
    /// ended it. Collective.
    pub(crate) fn end(&mut self, name: &'static str) -> Result<()> {
        self.comm.barrier()?;
        let now = Instant::now();
        *self.seconds.entry(name).or_default() += (now - self.last).as_secs_f64();
        self.last = now;
        Ok(())
    }

    /// The seconds of each phase so far.
    pub(crate) fn seconds(&self) -> &BTreeMap<&'static str, f64> {
        &self.seconds
    }
}
