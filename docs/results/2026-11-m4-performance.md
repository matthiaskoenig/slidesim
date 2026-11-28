# Milestone 4: transport solves

Status: 2026-11-27, slidesim branch `feat/m4-lobules` (tasks 11 and 12 of the milestone 4 plan;
task 12 in "Cell work, the exchange and the compilation"). Verdict of task 11:
the linear solves of the transport take 5.8 times fewer flops on the generated lobule (one
process, 2 h of the batched SPT study), from four changes: a larger direct solve at the bottom of
the coarse solve (algebraic multigrid down to 5000 equations), LU as the coarse solve on one
process, the solution of the step before as the initial guess instead of its extrapolation, and
`solver.linear_rtol` 1e-6 in the SPT example studies (the default stays 1e-10, the setting of
the parity tests). The cells are now the larger part of a lobule run.

## Setup

- Machine: Intel Core i7-12700 (20 threads), Open MPI 4.1.6, PETSc 3.26.1
  (`deps/install/native-opt`), `cargo build --release`.
- The machine was shared with other sessions running CPU loads throughout (load average 4 to
  68 during the runs, recorded before and after each run). The runs were confined to CPUs 0-9
  at nice 10. Wall times are therefore noisy (the same run varied by a factor of 2); the
  decisions rest on the deterministic counts, the Krylov iterations per solve and the flops of
  the solves (`KSPSolve` in `-log_view`), with wall times of runs next to each other in time as a
  check. No quiet window occurred for the baseline the plan asked for.
- Benchmark: `scripts/bench/coupled.sh`. It runs the batched SPT studies, shortened to `END`
  seconds, with `-log_view` and `-<field>_ksp_converged_reason`, and prints the phases of
  `summary.json`, the solves of the transport, their iterations per solve, the preconditioner
  setups, the wall time and the load average before and after. Cases:
  - `quick` (the default): `testdata/studies/lobules/spt_quick.study.json`, the generated
    lobule on the scene's mesh unrefined (5,902 triangles, one mesh level), 1 h, 1 process, about
    20 s;
  - `lobule`: `testdata/studies/lobules/spt_batched.study.json` (one refinement: 23,608
    triangles, two levels, 3,093 coarse vertices), 1 process, `END=7200` here;
  - `lobules7`: the ring of 7 lobules (41,094 coarse triangles, 21,166 coarse vertices), 1
    process;
  - `slide`: the 281 mg/kg slide of milestone 3 (41,252 cells, 165,008 triangles), 8 processes
    in the runs below (before the 4-process cap of the later runs).
- Each simulation runs through `run_sim` of `scripts/lib/proc.sh`: a wall-clock limit, the CPU
  confinement, and no process left behind when the script ends (checked with TERM and KILL to
  the script and to a wrapper `bash -c "coupled.sh; ..."`, and Ctrl-C to the script as a job of
  a shell and to a wrapper as one: no `slidesim`, `mpiexec` or script process 3.1 s later, and
  neither the run nor the wrapper went on).

## Baseline

24 h, the code of task 10 (algebraic multigrid with PETSc's defaults on the coarse level, linear
solves to 1e-10, the extrapolated initial guess), on the loaded machine (load 11 to 15):

| Study | Processes | Setup | Pressure | Transport | Exchange | Cells | Output | Iterations / solve | Wall (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| slide | 8 | 1.8 | 1.3 | 714.8 | 46.2 | 114.5 | 7.3 | 19.60 | 886 |
| lobule | 1 | 0.9 | 0.4 | 566.1 | 47.5 | 154.4 | 6.2 | 28.08 | 776 |

Milestone 3 measured the slide at 573 s on a quiet machine, about 400 s of it in the transport.

## Candidates

### The coarse solve on several processes

The slide, 2 h, 8 processes, linear solves to 1e-10, PETSc options of the study. Transport phase
of runs at a load below 12 where available (base: 53.8 to 97.4 s over five runs at loads of 6 to
46):

| Variant | Options | Iterations / solve | Transport (s) |
| --- | --- | ---: | ---: |
| base | | 19.60 | 53.8, 56.6 |
| coarse limit 1000 | `-mg_coarse_pc_gamg_coarse_eq_limit 1000` | 16.59 | 43.5, 47.0 |
| coarse limit 2000 | `... 2000` | 16.59 | (load 42) |
| **coarse limit 5000** | `... 5000` | 15.12 | 42.3, 43.4, 44.0 |
| coarse limit 10,000 | `... 10000` | 15.12 | 42.7 |
| limit 5000, no aggressive coarsening | `... -mg_coarse_pc_gamg_aggressive_coarsening 0` | 16.02 | 45.4 |
| no aggressive coarsening | `-mg_coarse_pc_gamg_aggressive_coarsening 0` | 23.67 | 74.4 |
| unsmoothed aggregation | `-mg_coarse_pc_gamg_agg_nsmooths 0` | 36.97 | 116.4 |
| redundant LU | `-mg_coarse_pc_type redundant -mg_coarse_redundant_pc_type lu` | 10.42 | 172.8 |
| redundant GAMG | `-mg_coarse_pc_type redundant -mg_coarse_redundant_pc_type gamg` | 22.71 | 402.6 |
| block Jacobi ILU | `-mg_coarse_pc_type bjacobi -mg_coarse_sub_pc_type ilu` | 79.48 | 311.0 |
| one smoothing step in the coarse AMG | `-mg_coarse_mg_levels_ksp_max_it 1` | 24.90 | 145.6 |
| 3 smoothing steps | `-mg_levels_ksp_max_it 3` | 19.28 | 92.3 |
| BiCGStab | `-ksp_type bcgs` | 11.27 | 1033.3 |

Kept: the coarse limit of 5000 equations (PETSc's default is 50), 23 % fewer iterations and 20 %
less transport time, 61 % fewer messages and 15 % fewer flops in the solves: fewer levels in the
algebraic multigrid, with a larger direct solve at its bottom. A redundant LU of the whole coarse
level (21,000 equations on every process) halves the iterations but makes each one far more
expensive.

### The coarse solve on one process

On one process, LU of the coarse level is an exact coarse solve without the replication which
made it slow on 8 processes. Generated lobule, 2 h, 1 process, linear solves to 1e-6:

| Coarse solve | Iterations / solve | `KSPSolve` flops | Transport (s), pairs next to each other |
| --- | ---: | ---: | ---: |
| algebraic multigrid, limit 5000 | 7.00 | 2.02e10 | 17.0, 15.3, 10.0 |
| LU | 4.64 | 1.53e10 | 13.4, 30.4 (load rose from 27 to 51), 6.5 |

On the 7 lobules (21,166 coarse vertices), 15 min, 1 process, 1e-6: 4.13 against 7.00
iterations, 1.32e10 against 1.81e10 flops, transport 10.8 and 17.4 s against 26.0 s; the
factorization takes 0.11 s a setup. Kept: LU on one process (24 to 27 % fewer flops, 21 to 35 %
less transport time), algebraic multigrid on several.

### The initial guess

PETSc's BDF extrapolates the initial guess of each step from the steps before
(`-ts_bdf_initial_guess_extrapolate`, on by default); the plan took it for off. Turned off, the
guess is the solution of the step before:

| Case | Linear tolerance | Iterations / solve, extrapolated | Previous solution |
| --- | ---: | ---: | ---: |
| quick | 1e-10 | 13.53 | 13.07 |
| quick | 1e-6 | 8.01 | 7.96 |
| lobule, 2 h | 1e-10 | 12.02 | 12.00 |
| lobule, 2 h | 1e-6 | 7.00 | 7.00 |
| slide, 2 h, 8 processes | 1e-10 | 19.60 | 18.29 |

The iterations change by 0 to 7 %, but the solution is several times closer to that of a tight
solve (next section): on the lobule at 1e-6 the fields differ from those at 1e-12 by 6e-8
without extrapolation and by 4e-7 with it. The steps (10 to 15 s) are far longer than the time
in which a field relaxes after the cells changed it (about h^2 / D = 0.6 s for 20 um elements),
and a linear extrapolation carries the relaxation on past its end; the linear solve reduces the
residual of the guess by the tolerance, so a worse guess leaves a worse solution. Kept: the
previous solution as the guess (`-ts_bdf_initial_guess_extrapolate 0` among the defaults of a
problem), for the accuracy at no cost, and the 7 % on the slide.

### The linear tolerance

`solver.linear_rtol` sets the relative tolerance of the transport's linear solves (default
1e-10). Generated lobule, 2 h, 1 process, against the same run at 1e-12: the largest difference
of a field at 2 h relative to its largest value, the same for the cell states, and the worst
mass balance residual relative to the largest term of its balance (algebraic multigrid with
limit 5000 on the coarse level, extrapolated guess, except the last two rows):

| Linear tolerance | Iterations / solve | `KSPSolve` flops | `P` | `S` | cells `P` | cells `S` | balance `P` | balance `S` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1e-10 | 12.02 | 3.26e10 | 2.2e-11 | 1.1e-12 | 2.1e-11 | 9.9e-13 | 1.1e-10 | 1.2e-12 |
| 1e-8 | 9.27 | 2.57e10 | 8.7e-9 | 4.1e-10 | 8.4e-9 | 4.0e-10 | 5.2e-9 | 6.9e-11 |
| 1e-6 | 7.00 | 2.02e10 | 4.1e-7 | 1.9e-8 | 4.0e-7 | 1.9e-8 | 6.2e-7 | 1.5e-8 |
| 1e-6, previous solution as guess | 7.00 | 2.02e10 | 5.7e-8 | 2.7e-9 | 5.4e-8 | 2.6e-9 | 4.7e-7 | 1.1e-9 |
| 1e-6, all kept changes | 4.20 | 1.42e10 | 5.9e-8 | 2.8e-9 | 3.2e-8 | 1.5e-9 | 3.2e-7 | 4.7e-9 |

The cells take the same steps at every tolerance (24,834,388, of them 3,513,208 rejected) but
one or two at 1e-6. For scale, halving the time step of the transport (`quick`, 7.5 s against
15 s) changes `P` by 1.0e-3 and `S` by 5.0e-5 of their largest values: the error of the linear
solves at 1e-6 is a few ten-thousandths of that of the time steps. With fixed steps the time
integration has no error estimate to tie the tolerance to, so the study sets it.

Kept: 1e-6 for the SPT example studies (`testdata/studies/lobules/*spt*`,
`testdata/studies/slide_fields/*/spt*`), the loosest tolerance measured, 42 % fewer iterations
than 1e-10 at a field error far below the time-step error and the cells' own tolerance (rtol
1e-6). The default stays 1e-10, the setting at which runs of different process counts agree
within about 1e-10 (the parity tests pin 1e-12). On the slide, 2 h, 8 processes: 8.95
iterations and 25.9 s transport at 1e-6 with the coarse limit 5000, against 15.12 and 42.3 s at
1e-10.

The balance of `P` closes less well than that of `S` relative to its largest term at every
tolerance: its residual is of the size of that of `S` relative to the storage of the field
(mass / dt), which the linear solves set, but its terms are 400 times smaller (the cells make
`P` slowly), so the same absolute residual is a larger share of them. The residual scales with
the tolerance (`P`: 3.6e-12 at 1e-12, 9e-10 at 1e-10, 5e-8 at 1e-8, 4e-6 at 1e-6 on `quick`):
no defect of the balance itself.

## Result

| Case | Iterations / solve | `KSPSolve` flops | Transport (s) |
| --- | ---: | ---: | ---: |
| lobule, 2 h, task 10 | 27.82 | 8.19e10 | 22.9 (load 7 to 4) |
| lobule, 2 h, now, 1e-10 | 7.63 | 2.22e10 | 13.7 (load 34 to 23) |
| lobule, 2 h, now, 1e-6 | 4.20 | 1.42e10 | 11.0 (load 36 to 34) |
| quick, task 10 code with the coarse limit, 1e-10 | 13.53 | 3.98e09 | 2.9 |
| quick, now, 1e-6 | 2.20 | 7.36e08 | 0.8 |

At the flop rate of the task 10 run on the less loaded machine (3.7 Gflop/s), the transport of
the lobule would take about 4 s of the 2 h, against 22.9 s; the cells take about 20 s. The 24 h
runs and the slide at the final settings were not repeated (the runs were limited to small
problems); the slide's targets of the plan (573 s to at most 300 s) remain to be measured on a
quiet machine.

## Cell work, the exchange and the compilation

Task 12 of the plan. Three changes, each measured on its own:

- **Cell work balance** (`cells.balance`, default true). Before each step of the cells, a
  process whose cells made more step attempts in the step before than the mean of the
  processes sends the cells beyond the mean (its last ones) to the processes below it, with
  their states, parameters and step control, and gets their results back
  (`Comm::exchange_bytes`, `MPI_Alltoallv`); the mesh partition, the outputs and the exchange
  with the fields stay. A cell takes the same steps wherever it is advanced, so the results are
  those without the balance, bit for bit (tests on 2 and 3 processes, both paths). The engine
  path moves the cells of models without events, discontinuities and algebraic rules; a process
  keeps a stepper per cell of another process while the cell comes.
- **The exchange** on precomputed weights. A profile of the exchange (timers in its parts,
  `quick`, 1 process; `perf` is not permitted on the machine) gave 0.95 s, of it 0.50 s in the
  two integrals of the give-back against the test functions and 0.31 s in the split of the
  field into its free and Dirichlet nodes, each a libCEED operator on every quadrature point,
  and 0.07 s in the means; the allocations of the vectors were small. The weights
  `w_ij = int_(e_i) phi v_j` of the nodes of each cell are now computed once, the means, the
  split and the integrals are sparse sums over the nodes of the cells, and the vectors are kept
  from one call to the next. The results change by rounding (1.9e-13 of the largest term of a
  balance, `quick`, 1 process).
- **One compilation on a cold cache.** Process 0 compiles the cell model (and its QFunction)
  first; the others load the libraries from the cache. Before, every process compiled them at
  once (3 compilations of the model on 3 processes, `compile_once_on_cold_cache`).

Back-to-back runs, the machine shared (load average in the table), `scripts/bench/coupled.sh`,
`native-opt`; `base` is 45cdd22, `off` the new binary with `cells.balance` false, `on` with
it true. Times in s; `attempts` is the imbalance of the step attempts (the sum over the cell
steps of the largest attempts of a process over their mean, `cells.attempts` of
`summary.json`), `seconds` the largest time of a process in its cells over the mean.

| Case | Variant | Exchange | Cells | Transport | Attempts | Seconds | Wall | Load |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| quick, 1 process | base | 0.249, 0.251 | 10.2, 10.2 | 0.33, 0.33 | | 1.000 | 12, 12 | 2.7 to 2.2 |
| quick, 1 process | new | 0.026, 0.026 | 10.2, 10.3 | 0.33, 0.33 | 1.000 | 1.000 | 11, 11 | 2.5 to 2.2 |
| quick, 2 processes | base | 0.1, 0.1 | 5.2, 5.3 | 0.5, 0.5 | | 1.004, 1.003 | 7, 7 | 2.1 to 2.5 |
| quick, 2 processes | off | 0.0, 0.0 | 5.3, 5.3 | 0.5, 0.5 | 1.001, 1.001 | 1.005, 1.003 | 6, 6 | 2.3 to 2.5 |
| quick, 2 processes | on | 0.0, 0.0 | 5.3, 5.2 | 0.5, 0.5 | 1.000, 1.000 | 1.004, 1.000 | 6, 6 | 2.4 to 2.5 |
| quick, 4 processes | base | 0.1, 0.1 | 2.7, 2.7 | 0.4, 0.3 | | 1.016, 1.020 | 4, 4 | 2.5 to 3.3 |
| quick, 4 processes | off | 0.0, 0.0 | 2.7, 2.7 | 0.4, 0.3 | 1.016, 1.016 | 1.018, 1.017 | 4, 4 | 2.7 to 3.7 |
| quick, 4 processes | on | 0.0, 0.0 | 2.6, 2.6 | 0.4, 0.3 | 1.004, 1.004 | 1.007, 1.005 | 4, 4 | 2.7 to 3.7 |
| 7 lobules, 4 h, 4 processes | base | 8.4, 8.2 | 66.7, 66.5 | 34.2, 33.1 | | 1.029, 1.030 | 114, 113 | 6.6 to 4.4 |
| 7 lobules, 4 h, 4 processes | off | 0.9, 0.8 | 67.2, 66.8 | 33.7, 32.8 | 1.027, 1.027 | 1.021, 1.021 | 107, 105 | 4.7 to 4.3 |
| 7 lobules, 4 h, 4 processes | on | 0.8, 0.8 | 65.5, 65.7 | 32.8, 32.6 | 1.001, 1.001 | 1.009, 1.016 | 104, 104 | 4.4 to 4.5 |

(`quick` is 1 h of the generated lobule unrefined; the 7 lobules are
`ring1_spt_batched.study.json` shortened to 4 h, 350 million cell step attempts in all,
`END=14400 NP=4 STUDIES=lobules7`.)

- The exchange takes 9 to 10 times less time: 0.25 to 0.026 s on `quick`, 8.3 to 0.85 s on
  the 7 lobules on 4 processes, where it was 7 % of the run.
- The balance takes the imbalance of the step attempts from 1.027 to 1.001 on the 7 lobules and from
  1.016 to 1.004 on `quick` on 4 processes, and the phase `cells` 2 % down (67.0 to 65.6 s), as much
  as the imbalance it removes: these cases are nearly balanced by the partition in their first
  hours. Milestone 3 measured 1.36 (the time in the cells) on the 281 mg/kg slide at 24 h, where the
  necrosis concentrates the work, which the balance now removes at the cost of the sends: the phase
  `cells` less the largest integration time of a process (the sends and the waits) is 0.9 and 1.0 s
  without the balance and 0.6 and 0.4 s with it. The test `balance_evens_the_work` (process 0 with
  10 times the attempts) goes from 1.82 to 1.002 on 2 processes and from 2.50 to 1.0004 on 3.
- The compilation on a cold cache: the setup of `quick` on 4 processes 0.94 and 0.96 s
  (`base`) against 0.92 and 0.94 s; the wall time hardly changes on an idle machine, three
  compilations of the four are gone.
- Parity: the balance is bit for bit neutral in the tests of the cells. Whole runs on 2 processes
  are too (`quick`, `off` against `on`: equal); on 4 processes, runs of the same binary and settings
  are not reproducible bit for bit (`base` against `base`: up to 4e-13 of the largest term of a
  balance; on the 7 lobules, `on` against `on`, one rejected step in the 350 million step attempts
  of a run), and `off` against `on` differ by as much (up to 6e-13), so the balance adds nothing
  beyond that noise. The parity of 1, 2 and 3 processes at the tight settings (1e-10, the same step
  counts of the cells) holds (`cells_parallel_matches_one`, `two_ranks_match_one`). The mass
  balances close as before.

The slide target (573 s to at most 300 s, 8 processes, 24 h) was not re-measured, by ruling:
the runs stayed within 10 CPUs and 4 processes on the shared machine.

## Reproduce

```bash
scripts/bench/coupled.sh                                   # quick
STUDIES=lobule END=7200 TAG=r6 scripts/bench/coupled.sh
STUDIES=lobule END=7200 TAG=r12 EDIT='.solver.linear_rtol = 1e-12' scripts/bench/coupled.sh
EDIT='.solver.petsc_options = "-ts_bdf_initial_guess_extrapolate 1"' scripts/bench/coupled.sh
NP=4 TAG=on scripts/bench/coupled.sh                       # task 12
NP=4 TAG=off EDIT='.cells.balance = false' scripts/bench/coupled.sh
NP=4 END=14400 STUDIES=lobules7 TAG=on scripts/bench/coupled.sh
SLIDESIM=<binary of 45cdd22> NP=4 END=14400 STUDIES=lobules7 TAG=base scripts/bench/coupled.sh
```

The run folders under `target/bench/coupled` keep `summary.json`, `log_view.txt` and the outputs
of each run.
