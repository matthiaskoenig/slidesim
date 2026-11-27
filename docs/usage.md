# Running a study

`slidesim run` runs one `study/1` end to end: it loads the study and its scene, checks them,
compiles the cell model if the study has one, solves the flow, steps the fields (and the cells)
to every output time and writes the results.

```text
slidesim run <STUDY> --out <DIR> [-- <PETSC OPTIONS>...]
```

```bash
target/debug/slidesim run testdata/studies/tissue_oxygen/study.json --out out
mpiexec -n 2 target/debug/slidesim run testdata/studies/tissue_oxygen/study.json \
  --out out
```

The same command runs on one process or on many under `mpiexec`: the mesh is partitioned over
the processes (PT-Scotch), and every result agrees across process counts within 1e-10 relative
(with fixed time steps). `slidesim run --help` describes the outputs.

## What a run does

1. Loads the study and the scene, checks that every map the study names is in the scene, and
   checks the `cells` section against the study and the scene. With cells, it translates the
   model to C, compiles it and checks it against the study, all before anything is solved.
2. Loads the mesh, distributed over the processes and refined `mesh.refinements` times, and
   reads the maps over the cells of each process. Tagged polygons and lobules are meshed by
   process 0 first ([Meshing](methods/meshing.md)), and the run prints the quality of the mesh
   in one line; the zonation map of lobules is computed over the cells of each process.
3. With `perfusion`: solves the steady Darcy flow once ([Perfusion](methods/perfusion.md)).
4. Steps the fields to each output time of `time.outputs` in turn, then to `time.end`: each field
   alone, or with `cells` all fields and the cells together by Strang splitting
   ([Cell models](methods/cells.md)).
5. At each output time (time 0 before any step) writes every field, and with cells the cells;
   at the end, process 0 writes `summary.json`.

## Outputs

`DIR` is created if missing. The files `<field id>_<index>.vtu`, `cells_<index>.vtu`,
`pressure.vtu` and `velocity.vtu` of an earlier run in `DIR` are removed first; other files stay.

| File | Content |
| --- | --- |
| `<field id>_<index>.vtu` | the field at output time number `<index>` (of `time.outputs`, from 0), point data `<field id>.mM` on the finest mesh level, for ParaView |
| `cells_<index>.vtu` | with cells: cell data of the coarse elements, one array per value: the states of the model (species as concentrations), `cells.outputs` and the parameters of the element (NaN where an element has no cell) |
| `pressure.vtu` | with perfusion: the pressure, point data `pressure.Pa` |
| `velocity.vtu` | with perfusion: the point vectors `velocity` (the conservative velocity the fields are carried by) and `primal_velocity` (`-K grad p / mu`), um/s |
| `summary.json` | per field and output time: norms, amounts and the mass balance; flows, cells and the time per phase |

## summary.json

```text
{"fields": {"<field id>": {
    "times": [...], "l2_norm": [...], "total_mass": [...],
    "mass_balance": [null | {
        "storage_rate": ..., "source": ..., "advection_defect": ...,
        "exchange": ...,
        "boundary": {"<tag>": {
            "advective": ..., "diffusive": ..., "resupply": ...}}}]}},
 "perfusion": {"tag_flows": {"<tag>": <um^2/s>},
               "pressure_islands": {"count": <n>, "area_um2": <um^2>}},
 "cells": {"count": <n>, "steps": <n>, "rejected": <n>,
           "seconds": {"mean": <s>, "max": <s>},
           "exchanged": {"<field id>": [...]},
           "amounts": {"<species id>": [...]}},
 "phases": {"setup": ..., "pressure": ..., "transport": ..., "exchange": ...,
            "cells": ..., "output": ...}}
```

`fields`
:   Per output time: the time reached (s), the L2 norm of the field (mM*um) and its amount of
    substance, the integral of the field weighed by the porosity (`total_mass`, mM*um^2).
    `mass_balance` is the balance over the interval since the output time before, each term its
    mean rate there (mM*um^2/s; `null` at time 0, and at every time if PETSc options chose a
    time integration other than BDF2, such as `-oxygen_ts_type cn`, which the run says in one
    line):

    \[
    \begin{aligned}
    \text{storage_rate} = {} & \text{source} + \text{exchange} - \text{advection_defect} \\
    & + \sum_{\text{tags}} \left(\text{advective} + \text{diffusive} + \text{resupply}\right)
    \end{aligned}
    \]

    The inflow through each tag of the boundary is positive into the domain, split into its
    advective and diffusive part; `exchange` is what the cells gave to the field, `resupply` what
    the Dirichlet values of a tag resupplied when the cells changed the field next to them, and
    `advection_defect` the non-conservation of the velocity (zero for the conservative velocity of
    milestone 3). `storage_rate` is the change of `total_mass` over the interval divided by its
    length, and the terms add up to it exactly, up to the residuals of the linear solves.

`perfusion`
:   With perfusion: the flow through each pressure tag (`tag_flows`, um^2/s, positive into the
    domain), and the pressure islands, the connected parts of the mesh which no pressure tag
    reaches (bounded by walls alone): their pressure is 0 and they have no flow. A run with
    islands says so in one line.

`cells`
:   With cells: their number, the steps and rejected steps of their integrators summed over the
    cells and the run, the time of each process in its cells over the run (`seconds`, s, without
    the wait for the other processes: the mean and the largest over the processes, whose ratio is
    the imbalance of the work of the cells), and per output time the amount each linked field
    received from the cells since t = 0 (`exchanged`) and the amount of each species of the cell
    compartment in all cells (`amounts`), both in mM*um^2.

`phases`
:   The wall time of process 0 in each phase of the run (s), separated by barriers: `setup` (from
    the start of the process: PETSc, the study, the cell model, the mesh, the maps, the problems),
    `pressure` (the flow), `transport` (the steps of the fields), `exchange` (the means of the
    fields over the cells and the changes given back), `cells` (the steps of the cells) and
    `output` (the files and the summary values).

## Perfusion

A study with a `perfusion` section solves the Darcy flow first (maps named by the study are read
from the scene) and also writes `pressure.vtu` and `velocity.vtu`. `testdata/studies/lobule_toy`
is an example:

```bash
target/debug/slidesim run testdata/studies/lobule_toy/study.json --out out
```

## Cell models

A study with a `cells` section runs an SBML cell model on every coarse element of the mesh (the
mesh before its refinements), coupled to the fields by Strang splitting: per split step
`cells.split_step`, the cells over half of it, the fields over all of it, the cells over the
other half; the linked species of the model exchange with the means of their fields over the
element. The run also writes `cells_<index>.vtu`, and `summary.json` gets `cells`.

The model is translated to C and compiled with `$CC` (or `cc`) at the start of the run, and
cached in `$SBMLSIM_C_CACHE` (default `$XDG_CACHE_HOME/sbmlsim/c` or `~/.cache/sbmlsim/c`), so a
second run of the same model starts at once. `testdata/studies/cells_toy` has examples, a linear
model (`cells.study.json`) and the SPT liver cell model with two linked fields
(`spt.study.json`):

```bash
target/debug/slidesim run testdata/studies/cells_toy/spt.study.json --out out
```

`cells.solver` chooses the path: `batched` (the default, the cells of a process as one libCEED
operator) or `engine` (one sbmlsim-core engine instance per cell, for every model sbmlsim-core
handles).

## Meshes

`slidesim mesh` writes the mesh of a `scene/1` and its quality, without a run: the mesh a run
makes of the scene before the refinements of the study. A gmsh mesh is read and converted,
tagged polygons and lobules are meshed ([Meshing](methods/meshing.md)). It runs on one process
(under `mpiexec` with more it is a usage error).

```text
slidesim mesh <SCENE> --out <DIR>
```

```bash
target/debug/slidesim mesh testdata/studies/lobules/scene.json --out mesh
```

```text
mesh: 5902 triangles, 3093 vertices, min angle 25.46 deg, SICN min 0.641 mean 0.960, edge length / size 0.51 to 1.58
```

`DIR` is created if missing; the three files in it are overwritten.

| File | Content |
| --- | --- |
| `mesh.json` | the mesh: `vertices` (`[x, y]` in um), `triangles` (three vertex indices, counterclockwise), `triangle_tags` (the index of the area tag of each triangle), `edges` (the tagged edges, `[a, b]` with `a < b`, one entry per edge and tag), `edge_tags` (the index of the tag of each), and `tags` (`name`, `value` and `dim`: 2 for an area tag, 1 for an edge tag) |
| `mesh.vtu` | for ParaView: one cell array per area tag, named after it, with the value of the tag in its triangles and NaN elsewhere |
| `quality.json` | `triangles`, `vertices`, `min_angle_deg`, `min_sicn`, `mean_sicn`, and for a generated mesh `edge_ratio`: the smallest and largest ratio of an edge length to the size field (null for a gmsh mesh) |

## PETSc options

PETSc options follow `--` and go to PETSc's options database. The solvers of a field take the
prefix `<field id>_`:

| Option | Effect |
| --- | --- |
| `-oxygen_ts_monitor` | the time steps of the field `oxygen` |
| `-oxygen_ksp_monitor` | the residuals of its linear solves |
| `-oxygen_ksp_rtol 1e-8` | the relative tolerance of its linear solves |
| `-log_view` | PETSc's profile of the run, at its end |

```bash
target/debug/slidesim run testdata/studies/tissue_oxygen/study.json --out out \
  -- -oxygen_ts_monitor
```

Precedence of the options, lowest first: slidesim's defaults (`-ksp_rtol 1e-10`, and solvers
which do not converge are errors), `solver.petsc_options` of the study, the options after `--`.
