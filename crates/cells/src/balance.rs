//! The balance of the work of the cells over the processes ([`Balance`]).
//!
//! The mesh partition gives each process the cells of its coarse elements ("its cells"), and it
//! stays: the outputs, the exchange with the fields and the steppers of the engine path stay
//! where they are. Before each step the work moves instead: a process whose cells made more step
//! attempts in the step before than the mean of the processes sends the cells beyond the mean
//! (the last ones, in their order) to processes below it, with everything a cell's step needs,
//! and gets their results back. A cell's step depends on nothing but what it is sent, so the
//! results are the same as without the balance, bit for bit.

use std::ops::Range;

use petsc::OwnedComm;

use crate::Result;

/// Which cells of a process its own steps advance and which other processes advance, by the
/// work of its cells in the step before; see the module documentation.
#[derive(Debug)]
pub(crate) struct Balance {
    comm: OwnedComm,
    /// Whether cells move: more than one process and the balance on.
    on: bool,
    /// The step attempts of each cell of this process in the step before, 1 before the first.
    weights: Vec<u64>,
    /// The cells of this process it advances itself: the first `kept`.
    kept: usize,
    /// Per process: the cells of this process which that process advances (in their order,
    /// after `kept`; empty for this process).
    sent: Vec<Range<usize>>,
    /// Per process: the number of its cells which this process advances.
    received: Vec<usize>,
}

impl Balance {
    /// The balance of the `n` cells of this process on `comm`, moving cells if `on` and there is
    /// more than one process.
    pub(crate) fn new(comm: OwnedComm, n: usize, on: bool) -> Balance {
        let size = comm.comm().size();
        Balance {
            on: on && size > 1,
            comm,
            weights: vec![1; n],
            kept: n,
            sent: vec![0..0; size],
            received: vec![0; size],
        }
    }

    /// The step attempts of each cell of this process in the step before, to set after a step.
    pub(crate) fn weights_mut(&mut self) -> &mut [u64] {
        &mut self.weights
    }

    /// The number of cells of this process it advances itself, the first ones.
    pub(crate) fn kept(&self) -> usize {
        self.kept
    }

    /// Per process: the number of its cells this process advances (after [`Balance::scatter`]),
    /// in the order of the cells [`Balance::scatter`] gives.
    pub(crate) fn received(&self) -> &[usize] {
        &self.received
    }

    /// The number of cells of other processes this process advances (after [`Balance::scatter`]).
    pub(crate) fn guests(&self) -> usize {
        self.received.iter().sum()
    }

    /// Plans the step: which cells of this process stay and where the others go, by the weights
    /// of every process. `valid` is whether this process can take the step; if a process
    /// cannot, nothing moves and every process is told so (`false`), so that none waits for
    /// another in [`Balance::scatter`]. Collective, if the balance is on.
    pub(crate) fn plan(&mut self, valid: bool) -> Result<bool> {
        let n = self.weights.len();
        self.kept = n;
        self.sent.iter_mut().for_each(|r| *r = 0..0);
        self.received.iter_mut().for_each(|r| *r = 0);
        if !self.on {
            return Ok(valid);
        }
        let comm = self.comm.comm();
        let total: u64 = self.weights.iter().sum();
        let mut mine = total.to_le_bytes().to_vec();
        mine.push(valid.into());
        let all = comm.exchange_bytes(&vec![mine; comm.size()])?;
        if all.iter().any(|b| b[8] == 0) {
            return Ok(false);
        }
        let totals: Vec<u64> = all
            .iter()
            .map(|b| u64::from_le_bytes(b[..8].try_into().expect("8 bytes")))
            .collect();
        let (kept, sent) = assign(&totals, comm.rank(), &self.weights);
        self.kept = kept;
        self.sent = sent;
        Ok(true)
    }

    /// Sends the cells of this process which others advance, `width` values each, which
    /// `record(i, out)` appends for cell `i`, and returns the cells of other processes this one
    /// advances, `width` values each, process by process in the order of the ranks and the cells
    /// of each in their order. Collective, if the balance is on.
    pub(crate) fn scatter(
        &mut self,
        width: usize,
        mut record: impl FnMut(usize, &mut Vec<f64>),
    ) -> Result<Vec<f64>> {
        if !self.on {
            return Ok(Vec::new());
        }
        let mut values = Vec::new();
        let sends: Vec<Vec<u8>> = self
            .sent
            .iter()
            .map(|cells| {
                values.clear();
                for i in cells.clone() {
                    record(i, &mut values);
                }
                debug_assert_eq!(values.len(), cells.len() * width);
                bytes(&values)
            })
            .collect();
        let received = self.comm.comm().exchange_bytes(&sends)?;
        let mut guests = Vec::with_capacity(received.iter().map(Vec::len).sum::<usize>() / 8);
        for (count, bytes) in self.received.iter_mut().zip(&received) {
            *count = bytes.len() / (8 * width.max(1));
            guests.extend(floats(bytes));
        }
        Ok(guests)
    }

    /// Sends the results of the cells of other processes this one advanced back to their
    /// processes, `width` values each in `results` (in the order of [`Balance::scatter`]), and
    /// gives the results of the cells of this process which others advanced to `store(i,
    /// values)` for cell `i`. Collective, if the balance is on.
    pub(crate) fn gather(
        &self,
        width: usize,
        results: &[f64],
        mut store: impl FnMut(usize, &[f64]),
    ) -> Result<()> {
        if !self.on {
            return Ok(());
        }
        let mut offset = 0;
        let sends: Vec<Vec<u8>> = self
            .received
            .iter()
            .map(|&count| {
                let values = &results[offset * width..(offset + count) * width];
                offset += count;
                bytes(values)
            })
            .collect();
        let received = self.comm.comm().exchange_bytes(&sends)?;
        for (cells, bytes) in self.sent.iter().zip(&received) {
            let values = floats(bytes);
            for (i, record) in cells.clone().zip(values.chunks_exact(width.max(1))) {
                store(i, record);
            }
        }
        Ok(())
    }
}

/// The bytes of `values`, little endian.
fn bytes(values: &[f64]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// The values of the bytes of [`bytes`].
fn floats(bytes: &[u8]) -> Vec<f64> {
    bytes
        .as_chunks::<8>()
        .0
        .iter()
        .map(|b| f64::from_le_bytes(*b))
        .collect()
}

/// The plan of the process `rank` with the cells of the weights `weights`, of the processes of
/// the total weights `totals`: the number of its cells it keeps, the first ones, and per process
/// the cells it sends there.
///
/// With `W` the sum of the totals over the `P` processes and `M = W / P` their mean, a process
/// above the mean keeps its cells up to the mean (each cell whose middle is at or below it, from
/// the first) and sends the others; a process at or below it keeps every cell. The cells sent
/// are laid out one after the other, process by process, the excess `T_p - M` of each process
/// above the mean taking its share of the line, and so are the deficits `M - T_q` of the
/// processes below the mean, which sum to the same: a cell goes to the process below the mean
/// whose deficit covers the position of its middle. Each process below the mean receives its
/// deficit within the largest weight of a cell, and every process computes the same from the
/// totals alone. Exact, in integers (times `2 P`).
fn assign(totals: &[u64], rank: usize, weights: &[u64]) -> (usize, Vec<Range<usize>>) {
    let size = totals.len();
    let n = weights.len();
    let mut sent = vec![0..0; size];
    let p = size as u128;
    let w: u128 = totals.iter().map(|&t| u128::from(t)).sum();
    let excess = |t: u64| (p * u128::from(t)).saturating_sub(w);
    let deficit = |t: u64| w.saturating_sub(p * u128::from(t));
    if excess(totals[rank]) == 0 {
        return (n, sent);
    }
    // Keep each cell whose middle is at or below the mean: 2 P (c + w_i / 2) <= 2 W.
    let mut c: u128 = 0;
    let mut kept = n;
    for (i, &wi) in weights.iter().enumerate() {
        if 2 * p * c + p * u128::from(wi) > 2 * w {
            kept = i;
            break;
        }
        c += u128::from(wi);
    }
    // The start of this process's part of the line of the excesses, and the ends of the parts
    // of the deficits (doubled).
    let start: u128 = 2 * totals[..rank].iter().map(|&t| excess(t)).sum::<u128>();
    let receivers: Vec<(usize, u128)> = totals
        .iter()
        .enumerate()
        .filter(|(_, t)| deficit(**t) > 0)
        .scan(0u128, |end, (q, &t)| {
            *end += 2 * deficit(t);
            Some((q, *end))
        })
        .collect();
    let mut x: u128 = 0;
    let mut first = kept;
    let mut current = None;
    for (i, &wi) in weights.iter().enumerate().skip(kept) {
        let middle = start + p * (2 * x + u128::from(wi));
        let q = receivers
            .iter()
            .find(|(_, end)| middle < *end)
            .or(receivers.last())
            .expect("a process above the mean means one below it")
            .0;
        if current != Some(q) {
            if let Some(previous) = current {
                sent[previous] = first..i;
            }
            current = Some(q);
            first = i;
        }
        x += u128::from(wi);
    }
    if let Some(q) = current {
        sent[q] = first..n;
    }
    (kept, sent)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The weight each process advances under the plans of every process.
    fn loads(weights: &[Vec<u64>]) -> Vec<u64> {
        let totals: Vec<u64> = weights.iter().map(|w| w.iter().sum()).collect();
        let mut loads = vec![0; weights.len()];
        for (rank, w) in weights.iter().enumerate() {
            let (kept, sent) = assign(&totals, rank, w);
            loads[rank] += w[..kept].iter().sum::<u64>();
            let mut next = kept;
            for (q, cells) in sent.iter().enumerate() {
                if cells.is_empty() {
                    continue;
                }
                assert_ne!(q, rank);
                assert_eq!(cells.start, next, "the cells sent follow each other");
                next = cells.end;
                loads[q] += w[cells.clone()].iter().sum::<u64>();
            }
            assert_eq!(next, w.len(), "every cell is advanced");
        }
        assert_eq!(loads.iter().sum::<u64>(), totals.iter().sum::<u64>());
        loads
    }

    #[test]
    fn heavy_process_spreads_its_cells() {
        // 10 cells of 10 on process 0, 3 cells of 10/3 on the others: 40 each.
        let weights = vec![vec![10; 10], vec![3, 3, 4], vec![4, 3, 3]];
        assert_eq!(loads(&weights), [40, 40, 40]);
        let totals = [100, 10, 10];
        let (kept, sent) = assign(&totals, 0, &weights[0]);
        assert_eq!(kept, 4);
        assert_eq!(sent, [0..0, 4..7, 7..10]);
    }

    #[test]
    fn balanced_processes_keep_their_cells() {
        let weights = vec![vec![5; 4], vec![4; 5], vec![]];
        let totals = [20, 20, 0];
        for (rank, w) in weights.iter().enumerate().take(2) {
            let (kept, sent) = assign(&totals, rank, w);
            // Above the mean of 40 / 3 they send, only to the process without cells.
            assert!(kept < w.len());
            assert!(sent[..2].iter().all(Range::is_empty));
        }
        let spread = loads(&weights);
        assert!(spread.iter().all(|&l| l.abs_diff(13) <= 5), "{spread:?}");
        assert_eq!(loads(&[vec![7; 3], vec![7; 3]]), [21, 21]);
        assert_eq!(loads(&[vec![0; 3], vec![]]), [0, 0]);
    }

    #[test]
    fn loads_are_within_a_cell_of_the_mean() {
        // Uneven weights and counts over 4 processes.
        let mut seed = 1u64;
        let mut next = |m: u64| {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 33) % m
        };
        for _ in 0..50 {
            let weights: Vec<Vec<u64>> = (0..4)
                .map(|r| {
                    let n = next(200) as usize;
                    (0..n)
                        .map(|_| 1 + next(100) * (1 + 9 * u64::from(r == 0)))
                        .collect()
                })
                .collect();
            let largest = weights.iter().flatten().copied().max().unwrap_or(0);
            let total: u64 = weights.iter().flatten().sum();
            let mean = total as f64 / 4.0;
            for l in loads(&weights) {
                assert!(
                    (l as f64 - mean).abs() <= largest as f64,
                    "{l} against {mean}"
                );
            }
        }
    }
}
