# Milestone 4: split step of the coupled SPT runs

Status: 2026-11-27, slidesim branch `feat/m4-lobules` (task 11b of the milestone 4 plan). Verdict:
at the split step of 60 s of the example SPT studies, the Strang splitting of cells and transport
costs 1.7 % of the substrate uptake and delays the necrosis front by a few cells. A split step of
30 s is the choice for the substrate and the necrosis: it is the largest measured split step
whose substrate (S) uptake stays within 1 % of the reference and whose necrotic share and Dice
coefficient stay within 0.01, at 1.6 to 1.7 times the wall time of 60 s. It does not meet the
1 % for everything: the product P given back is off by 4.2 % at 30 s (lobule, 1 h; on the coarse
mesh 0.69 % at 4 h but 4.2 % at 1 h), and on the 7 lobules the S uptake in the first 15 min is
off by 1.7 % at the end and 13 % at 5 min. The necrosis numbers come from a coarser mesh over
4 h and are not measured on the production mesh at 24 h (see Setup). The splitting is of order 1.3 to 1.7 over the measured split steps, because a cell
step of seconds to minutes is long against the few seconds in which the perfusion refills a
cell's blood. The lobule and slide field example SPT studies use `split_step` 30 s since the commit
"feat(examples): split step 30 s in the SPT examples" (the user's decision of 2026-11-27); a
supply-aware coupling is a separate design task after milestone 4.

## Setup

- Machine: Intel Core i7-12700 (20 threads), Open MPI 4.1.6, PETSc 3.26.1
  (`deps/install/native-opt`), `cargo build --release` of 556c29e. Every run on 1 process,
  confined to CPUs 0-9 at nice 10, one run at a time; the machine was shared with other sessions,
  load average 1.5 to 8.4 before and after the runs (recorded per run below).
- Model and path: the SPT liver cell model on the batched path, the linear tolerance
  `solver.linear_rtol` 1e-6 of the example studies.
- Split steps 120, 60, 30, 15 and 10 s, the transport step `time.dt` = min(10 s, split step)
  (so 10 s at 60 s, where the example studies have 15 s), and a reference at split step 5 s with
  transport step 2 s.
- The budget (all runs together about 20 min, each run under about 2 min) did not allow 24 h.
  The reference costs 100 s per simulated hour on the lobule's own mesh, so the study is split in
  two, and the necrosis, which starts after 2 h, is measured on a coarser mesh:
  - **lobule**: `testdata/studies/lobules/spt_batched.study.json` as it is (its mesh and one
    refinement, 5,902 cells), **1 h**, outputs every 30 min. No cell is necrotic yet at 1 h.
  - **coarse**: the same study on a coarser mesh of the same lobule (the scene's
    `size.wall_um` 15 and `size.interior_um` 60 instead of 5 and 20, no refinement: 1,665 cells,
    mean cell area 137 um^2), **4 h**, outputs hourly. The necrotic share of the reference is
    0.158 at 3 h and 0.338 at 4 h, the pericentral front. Caveat: its cells are about 3.5 times
    larger than those of the production mesh (about 40 um^2), so the flow refills their blood
    more slowly, and the splitting error of the necrosis on the production mesh at 24 h may
    differ from the one measured here. The only link between the two meshes is that the S
    uptake errors at 1 h are similar (1.73 % and 1.73 % at 60 s, 0.62 % and 0.62 % at 30 s, the
    coarse values the largest over its outputs); that says nothing about the necrosis.
  - **lobules7**: the 7 lobules of `testdata/studies/lobules/ring1_spt_batched.study.json`
    (41,094 cells), **15 min**, outputs every 5 min, at 120, 60, 30 and 10 s, compared with the
    10 s run.
- Measures, against the reference (`scripts/bench/split_step.py`):
  - uptake: `cells.exchanged` of summary.json per field (the amount the field received from the
    cells since t = 0), relative error at the end and the largest over the output times; S is
    the substrate taken up, P the product the cells give back;
  - cell states at the output times: per state the difference scaled by the reference's largest
    magnitude over the cells at that time, the largest and the mean over cells and times;
  - necrosis: the share of the area of the cells with `necrosis > 0.5`, and the Dice coefficient
    of these cells against the reference (area weighted), at the end and the smallest over the
    output times;
  - observed order: ln(e1 / e2) / ln(h1 / h2) of successive split steps from the errors against
    the reference, and from the differences of successive runs (split step ratio 2);
  - cost: the phases of summary.json and the wall time.

## Lobule, 1 h

| split (s) | dt (s) | S uptake err end | S err max | P uptake err end | P err max | T max / mean | wall (s) | wall / 60 s |
|---|---|---|---|---|---|---|---|---|
| 120 | 10 | 4.63e-2 | 1.17e-1 | 2.19e-1 | 3.96e-1 | 1.0e-1 / 3.4e-2 | 11.1 | 0.69 |
| 60 | 10 | 1.73e-2 | 2.66e-2 | 1.03e-1 | 2.06e-1 | 4.5e-2 / 1.5e-2 | 16.2 | 1.00 |
| 30 | 10 | 6.17e-3 | 7.05e-3 | 4.21e-2 | 8.50e-2 | 1.5e-2 / 5.4e-3 | 25.7 | 1.59 |
| 15 | 10 | 1.97e-3 | 2.19e-3 | 1.51e-2 | 3.06e-2 | 4.3e-3 / 1.6e-3 | 44.0 | 2.72 |
| 10 | 10 | 1.10e-3 | 1.23e-3 | 8.22e-3 | 1.66e-2 | 2.5e-3 / 9.0e-4 | 57.1 | 3.53 |
| 5 (ref) | 2 | | | | | | 102.3 | 6.33 |

The reference took up 1.2095e6 mM*um^2 of S and gave 2.523e5 of P back. Cell states (scaled
difference, largest / mean):

| split (s) | S_ext | P_ext | S | P | T |
|---|---|---|---|---|---|
| 120 | 1.6e-1 / 8.5e-2 | 1.9e+1 / 8.4 | 1.5e-1 / 8.0e-2 | 1.7e+1 / 7.6 | 1.0e-1 / 3.4e-2 |
| 60 | 4.3e-2 / 2.1e-2 | 9.2 / 4.3 | 3.9e-2 / 1.9e-2 | 8.3 / 3.9 | 4.5e-2 / 1.5e-2 |
| 30 | 1.3e-2 / 6.2e-3 | 3.9 / 1.9 | 1.1e-2 / 5.5e-3 | 3.5 / 1.7 | 1.5e-2 / 5.4e-3 |
| 15 | 4.0e-3 / 2.0e-3 | 1.5 / 7.2e-1 | 3.6e-3 / 1.8e-3 | 1.3 / 6.2e-1 | 4.3e-3 / 1.6e-3 |
| 10 | 2.2e-3 / 1.1e-3 | 8.3e-1 / 3.9e-1 | 2.0e-3 / 9.8e-4 | 7.1e-1 / 3.3e-1 | 2.5e-3 / 9.0e-4 |

Observed orders (S uptake / P uptake, from the errors against the reference): 1.42 / 1.08
(120 to 60 s), 1.49 / 1.29 (60 to 30), 1.65 / 1.48 (30 to 15), 1.43 / 1.50 (15 to 10); from the
differences of successive runs 1.39 / 0.92 (120, 60, 30) and 1.40 / 1.18 (60, 30, 15).

## Coarse lobule, 4 h

| split (s) | dt (s) | S err end | S err max | P err end | P err max | necrotic share | share - ref | Dice end | Dice min | wall (s) | wall / 60 s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 120 | 10 | 2.16e-2 | 4.63e-2 | 4.67e-2 | 2.19e-1 | 0.3236 | -0.0145 | 0.9780 | 0.9462 | 7.1 | 0.60 |
| 60 | 10 | 8.75e-3 | 1.73e-2 | 1.94e-2 | 1.03e-1 | 0.3330 | -0.0051 | 0.9924 | 0.9842 | 11.7 | 1.00 |
| 30 | 10 | 3.03e-3 | 6.18e-3 | 6.87e-3 | 4.21e-2 | 0.3362 | -0.0019 | 0.9972 | 0.9972 | 20.4 | 1.74 |
| 15 | 10 | 9.39e-4 | 1.97e-3 | 2.18e-3 | 1.51e-2 | 0.3362 | -0.0019 | 0.9972 | 0.9972 | 36.0 | 3.08 |
| 10 | 10 | 5.35e-4 | 1.10e-3 | 1.23e-3 | 8.23e-3 | 0.3362 | -0.0019 | 0.9972 | 0.9972 | 48.6 | 4.15 |
| 10 | 2 | 3.74e-4 | 8.02e-4 | 8.82e-4 | 6.77e-3 | 0.3362 | -0.0019 | 0.9972 | 0.9972 | 56.7 | |
| 5 (ref) | 2 | | | | | 0.3381 | | | | 73.2 | 6.25 |

- The necrosis front lags at long split steps: the cells take up less S, make less T, and cross
  the threshold later. At 60 s, 6 cells (1,116 um^2) at 3 h and 5 at 4 h are below 0.5 which
  are above it in the reference (necrosis 0.23 to 0.49 against 0.51 to 0.75); at 30, 15 and
  10 s two cells (428 um^2) at 4 h, whose necrosis is 0.41 (30 s) and 0.49 (10 s) against 0.51,
  mid-ramp at the output time. One cell is 0.0006 of the area, so 0.0019 is the resolution of
  the share at the front, not an error floor.
- The run at 10 s with transport step 2 s separates the two steps: the transport step of 10 s
  is 30 % of the remaining uptake error at split step 10 s, the splitting the rest.
- The largest uptake errors are those of the first hour, while the cells fill with S (the
  1 h output); in hours 2 to 4 the S uptake per hour at 30 s is within 0.1 to 0.7 % of the
  reference, higher in hours 3 and 4 because fewer cells are necrotic.
- Observed orders (S / P / necrotic share): 1.31 / 1.27 / 1.52 (120 to 60 s), 1.53 / 1.50 /
  1.43 (60 to 30), 1.69 / 1.66 (30 to 15), 1.39 / 1.41 (15 to 10); from the differences of
  successive runs 1.17 / 1.12 (120, 60, 30) and 1.45 / 1.42 (60, 30, 15).

## Seven lobules, 15 min, against 10 s

| split (s) | S err end | S err max | P err end | P err max | wall (s) | wall / 60 s |
|---|---|---|---|---|---|---|
| 120 | 2.43e-1 | 5.69e-1 | 5.23e-1 | 7.57e-1 | 35.4 | 0.82 |
| 60 | 1.01e-1 | 3.86e-1 | 3.54e-1 | 6.21e-1 | 43.2 | 1.00 |
| 30 | 1.70e-2 | 1.30e-1 | 1.43e-1 | 3.56e-1 | 60.4 | 1.40 |
| 10 | | | | | 114.1 | 2.64 |

The first 15 min, while the substrate front enters, are the worst case of the splitting: at
30 s the S uptake differs from the 10 s run by 13 % at 5 min and 1.7 % at 15 min, against 0.7 %
at 30 min and 0.6 % at 1 h on the single lobule. The ordering of the split steps is the one of
the lobule; the 7 lobules confirm it but, at 15 min, not the 1 % of the criterion, which holds
after the first half hour. The setup and output of the larger mesh (10 s together) make the
relative cost of short split steps smaller here.

## Cost

The cells are the cost of a run, and their cost is set by the number of cell steps (the
advances from one midpoint of a split step to the next): the steps of the cells per cell and
advance on the lobule are 41 (120 s), 35 (60), 31 (30), 28 (15), 25 (10) and 19 (5 s). Each
advance starts with a transient, the blood the transport changed relaxing into the cell, which
costs the integrator 19 to 41 steps however short the advance is; so the cells phase
grows almost in proportion to the number of advances: 6.2, 10.3, 17.9, 31.6 and 42.7 s at 120,
60, 30, 15 and 10 s on the lobule for 1 h (1.74 times at 30 s against 60 s), and the transport
and exchange phases grow with it (dt 10 s, and a restart of BDF2 per split step).

## Recommendation

1. **Split step 30 s, the choice for the substrate and the necrosis.** It is the largest
   measured split step whose substrate (S) uptake error is below 1 % and whose necrotic share and
   Dice stay within 0.01 on both lobules: S uptake error 0.62 % (lobule, 1 h) and 0.30 % (coarse, 4 h) at the end, at most
   0.71 % over the outputs; necrotic share within 0.0019 and Dice 0.9972 at every output with
   necrosis. 60 s fails the uptake (1.73 % at 1 h) and the Dice at 3 h (0.984). The cost against
   60 s is 1.59 times the wall time on the lobule for 1 h, 1.74 times on the coarse lobule for
   4 h, and 1.40 times on the 7 lobules for 15 min; for the 24 h runs, where the cells dominate,
   an estimate of about 1.7 times (not measured). The transport step of the SPT studies (15 s)
   could stay 15 s, which `split_step` 30 s allows (at least `time.dt`), but split 30 s with
   dt 15 s was not measured: all runs here used dt 10 s. Limits of this choice: the product P
   misses 1 % at 30 s (item 2: 4.2 % on the lobule at 1 h, on the coarse mesh 0.69 % at 4 h but
   4.2 % at 1 h); on the 7 lobules the S uptake misses it in the first 15 min (1.7 % at 15 min,
   13 % at 5 min); and the necrosis criteria rest on the coarse mesh over 4 h, whose larger cells
   refill more slowly, so they may not carry over to the production mesh at 24 h (Setup).
2. **The product P and the blood species at the output times do not meet 1 % at any measured
   split step.** The P given back lags by up to 8.5 % (30 s) and 1.7 % (10 s) in the first hour,
   and the states P_ext and P of the cells at an output time differ from the reference by 1.9
   (30 s) and 0.39 (10 s) times the reference's largest value, on average over the cells. At an
   output time a cell has just made the half step of the splitting without transport, in which
   its export piles up in its blood instead of being washed out within seconds; the pile is
   proportional to the half step and dominates the small P_ext of the reference itself (largest
   0.023 mM). So these numbers measure the splitting, also in the reference; shorter split steps
   reduce them only in proportion. S_ext, S, T, necrosis and the uptake are not affected in this
   way.
3. **A supply-aware coupling is worth designing, for quality more than for speed.** One in
   which the cell step sees the supply and the washout of its blood by the flow (for example a
   relaxation of the blood towards the transported concentration at the refill rate of the
   element) would remove the first order error and the pile-up of item 2, so that the output
   states of fast exchanging species are meaningful. The speed it can buy is bounded by the
   measurements: if it made 60 s as accurate as 30 s is now, the runs would be 1.6 to 1.7 times
   faster, at 120 s 2.3 times (lobule, 1 h: 11.1 s against 25.7 s), and the transient at the
   start of every advance (19 to 41 cell steps) would shrink with the jumps of the blood. It is a
   change of the coupling scheme (conservation, positivity safeguard and order to be shown
   again), so the plan is: 30 s as the default now, the supply-aware coupling as a design task
   of its own, judged against these runs.

## Reproduce

```bash
scripts/bench/split_step.sh                         # all three studies, about 12 min
STUDIES=coarse SPLITS=10 DT_MAX=2 REF_RUN=0 OUT=<copy of the coarse reference> scripts/bench/split_step.sh
STUDIES=lobules7 SPLITS7=30 scripts/bench/split_step.sh
STUDIES=lobules7 SPLITS7="120 60 30 10" COMPARE=1 scripts/bench/split_step.sh
```

The run folders under `target/bench/split_step` keep the study, `summary.json`, the
`cells_<index>.vtu`, `run.log` and `wall.txt` (wall time and the load averages before and after)
of each run; `COMPARE=1` prints the tables again from them.
