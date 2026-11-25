# slidesim - design

Status: draft for review, 2026-11-24.

## Intent

slidesim simulates spatial PDE-ODE systems on real tissue geometry. The geometry and the local
parameters come from whole-slide scans and spatial transcriptomics, preprocessed by slideviz.
On this geometry, PDEs describe transport (diffusion, perfusion along the sinusoids, advection),
and every mesh element carries an SBML ODE network which supplies local sources and sinks and
which changes coefficients of the PDEs. The results are explored in a web viewer which navigates
the slide pyramid, the mesh and the simulated time courses in 2D and 3D.

The scale goes from one 2D section up to a 3D stack of serial sections. A 3D stack of a
20 x 15 mm section with 100 sections of 4 um is about 120 mm^3 of tissue: about 10^9 PDE
unknowns at 5 um resolution (about 10^8 with octree adaptivity) and about 10^7 ODE instances
of tens of states each. This needs GPU clusters. The same core also runs small problems on a
plain CPU and, compiled to WebAssembly, in the browser.

## Goals

- One Rust core for the PDE-ODE simulation, with three build targets:
  - cluster: MPI with one GPU per rank (CUDA or HIP), fields resident on the device;
  - CPU: a workstation or CPU cluster, MPI optional;
  - browser: sequential WebAssembly for small grids.
- The same code and the same numerics on all targets. A small problem gives the same result
  (within solver tolerances) in the browser, on a CPU and on a GPU.
- Physics, in this order:
  1. diffusion-reaction;
  2. Darcy perfusion (a pressure solve) and advection-diffusion-reaction of the species.
- ODE networks from SBML through sbmlsim-core, with a new batched code generation target.
- Python bindings to describe, run and analyze studies.
- A web viewer for slide pyramids, meshes and simulated time courses, in 2D and 3D, with
  real-time navigation, slice planes, volume rendering and time scrubbing.
- Offline high-quality renders through ParaView/VTK.

## Non-goals

- Tissue mechanics (swelling, deformation, moving meshes). Possibly a later phase; the design
  should not block it, but nothing is built for it now.
- Image analysis, segmentation, registration and the mapping of spatial transcriptomics. These
  stay in slideviz. slidesim receives their results matched by position.
- Handling of spatial transcriptomics platforms. slideviz maps all platforms to positions.
- Reconstruction of 3D cells across serial sections.

## Anti-goals

These would make slidesim a failure even if it technically works:

- Python, or any interpreter, in the hot loop of a simulation.
- Copies of fields between host and GPU at every coupling step.
- Results which differ between targets beyond solver tolerances.
- Model-specific or tissue-specific code in the core. Models, species, fields and tags are data
  from the contracts.
- Two parallel implementations of the discretization or the solvers which drift apart.

## Context: what exists

- **slideviz** (`~/git/slideviz`): reads slides (`.czi`, `.ndpi`, OME-TIFF, OME-Zarr), does the
  image analysis (tissue, zonation, necrosis predictions, stain flattening), registration
  (`registration/`) and builds 2D meshes (`mesh/`): polygons to gmsh triangle meshes with named
  areas (`surviving`, `necrotic`, `tissue`) and edges (`vessel_wall`, `section_edge`,
  `field_cut`, `necrosis_border`) and a size field, written for DOLFINx 0.11.
- **sbmlsim-core** (`~/git/sbmlsim-core`): Rust. SBML to the intermediate representation
  `sbmlsim_ir`, code generation into `model.wasm` with analytic sparse Jacobians, an engine with
  CVODE/IDA and KLU compiled into `wasm32-wasip1`, a `batch` crate for thread pools, Python
  bindings with PyO3/maturin. It already links C libraries into `wasm32-wasip1` with wasi-sdk.
  `spike/generate_c.py` is an earlier throwaway generator of C right-hand sides and CSC
  Jacobians.
- **sbmlsim-web**: a Vue 3 renderer of sbmlsim panels. Not part of the simulation path; its
  stack (Vue, TypeScript, Vite) is the reference for the viewer's stack.
- `slidesim` itself is empty.

## Approaches considered

### PDE layer

| Option | Summary | Verdict |
|---|---|---|
| A. DOLFINx C++ through a C++ shim (`cxx`) | Mature FEM, reuses slideviz meshes, FFCx generates C kernels at build time. | Rejected. GPU assembly is experimental, fields live on the host, so GPU coupling copies every step. The shim over templated C++ must follow DOLFINx API changes. |
| **B. PETSc (DMPlex, DMForest) + libCEED, both plain C, through `bindgen`** | Matrix-free FEM operators on CPU, CUDA, HIP, SYCL. PETSc time stepping, Krylov solvers, multigrid, GPU-resident vectors. No C++ layer. libCEED has official Rust bindings. | **Chosen.** Best GPU path and scaling, plain C ABI, one stack for all targets. |
| C. Pure Rust structured or octree grid with own GPU stencils and multigrid | Fastest per unknown, aligned to pyramid tiles. | Rejected. Own solver stack at 10^9 unknowns is the least robust option, and vessel walls lose conforming boundaries. |

### Small problems in the browser

| Option | Summary | Verdict |
|---|---|---|
| **1. One stack everywhere** | PETSc + libCEED built sequentially (`--with-mpi=0`, no Fortran, `f2cblaslapack`) into `wasm32-wasip1`, as sbmlsim-core does with CVODE. FEM on Browser shows PETSc runs in the browser (via emscripten). | **Chosen, after a spike.** Same code and numbers on every target. Risk: the PETSc wasi build; binary likely 10-20 MB. |
| 2. Backend trait with a second, pure-Rust solver stack for wasm | Small clean wasm. | Fallback only, if the spike fails. Doubles mesh and solver code. |

### Viewer

| Option | Summary | Verdict |
|---|---|---|
| 1. Own WebGPU viewer streaming multiscale chunks | Real-time, full control of the look, can show live in-browser simulations. | Part of the choice. |
| 2. Remote rendering (ParaView/trame, Catalyst in situ) | Beautiful, any size. Every interaction waits for the network; weak pyramid navigation. | Offline only. |
| **3. Own WebGPU viewer, plus ParaView/VTK offline for publication renders** | | **Chosen.** The in-browser simulation needs a viewer which reads live fields from memory, which only option 1 can do. |

### Where an ODE instance lives

| Option | Verdict |
|---|---|
| **1. One ODE instance per mesh element at a chosen octree level (about 20 um)** | **Chosen.** Same rule in 2D and 3D, matches the position-based mapping from slideviz, maps one-to-one onto the batched kernel. |
| 2. One ODE instance per segmented cell | Rejected. Needs 3D cell reconstruction across sections. |
| 3. Both, chosen in the study | Rejected. Two code paths. |

## Decided design

### Architecture

```
slideviz (Python)                         slidesim
  image analysis, registration,           +-------------------------------------------+
  spatial transcriptomics mapping  -----> | scene/1    mesh + tags + position fields  |
  2D gmsh meshes, 3D label volumes        | study/1    PDEs, SBML models, coupling    |
                                          |                                           |
                                          | Rust core                                 |
                                          |   mesh: DMPlex (2D gmsh), DMForest (3D)   |
                                          |   pde:  libCEED operators, PETSc TS/KSP   |
                                          |   ode:  sbmlsim-core batched kernels      |
                                          |   coupling: Strang splitting              |
                                          |   io:   Zarr v3 multiscale, VTKHDF        |
                                          |                                           |
                                          | targets: GPU cluster | CPU | browser wasm |
                                          | Python bindings (PyO3/maturin)            |
                                          +-------------------------------------------+
                                                     | result/1 (Zarr v3)   | live fields (wasm)
                                                     v                      v
                                          web viewer (WebGPU)        ParaView/VTK (offline, VTKHDF)
```

### Meshes

- **2D**: the gmsh `.msh` meshes from slideviz, read by PETSc DMPlex with their physical tags.
- **3D**: slideviz registration stacks the sections into one 3D label volume (tissue, vessel,
  necrosis, ...). slidesim builds an adaptive octree hex mesh from it with PETSc `DMForest`
  (p4est), refined at vessel walls and tissue borders, coarse in the bulk. Meshing is parallel.
  Vessel walls are stepped at the finest level rather than conforming; at a finest size of
  1-2 um this is below cell scale.
- p4est is GPL-2.0-or-later. It is linked only in an optional `octree` feature build (decided:
  MIT core, GPL feature build). The default builds (CPU, browser, GPU without `octree`) stay MIT
  and support 2D DMPlex meshes; 3D octree meshes need the `octree` build, whose binaries are
  GPL.
- Octree levels align with the slide pyramid levels, which the viewer and the output use.
- **Mesh hierarchy in 2D** (design call): the gmsh mesh from slideviz is the coarse level.
  Uniform DMPlex refinements of it give the finer levels on which the PDE is solved and the
  levels of `PCMG` (GAMG as fallback). Triangles use libCEED non-tensor bases, without sum
  factorization. For `result/1` the 2D fields are resampled onto the raster of the slide pyramid.

### PDE solver stack

| Layer | Component |
|---|---|
| Time loop | Rust. Strang splitting: half ODE step, full PDE step, half ODE step. |
| PDE time step | PETSc `TS`, implicit for diffusion (ARKIMEX or BDF), adaptive dt. |
| Nonlinear | PETSc `SNES` only where PDE terms are nonlinear. With coefficients frozen per split step each step is linear. |
| Linear | PETSc `KSP`: CG for the symmetric problems (diffusion, Darcy pressure), GMRES for advection. |
| Preconditioner | `PCMG` geometric multigrid on the octree levels, matrix-free Chebyshev-Jacobi smoothers (diagonal from libCEED), assembled coarse level solved by GAMG or hypre BoomerAMG (on GPU where available). Pattern of Ratel and HONEE. |
| Operators | libCEED, matrix-free: mass, diffusion, advection with SUPG stabilization, Darcy. Backends `/gpu/cuda/gen`, `/gpu/hip/gen`, `/cpu/self/opt` (or libxsmm), `/cpu/self` in wasm. Hex elements use sum factorization. |
| Data | PETSc `VECKOKKOS` on GPU (decided), covering NVIDIA and AMD; the ODE kernel reads and writes the same device buffers. |

Physics milestones: diffusion and diffusion-reaction first, then Darcy perfusion (a pressure
Poisson problem on the same SPD stack, recomputed only when the ODEs change the permeability)
with advection-diffusion-reaction of the species.

Perfusion (decided): the sinusoids are a homogenized porous medium. Permeability comes from the
zonation fields of the scene, optionally anisotropic, and the ODEs may change it. Pressure
boundary conditions sit on the portal walls (inflow) and the central walls (outflow). In 2D the
flow is in-plane only. This needs slideviz to split its `vessel_wall` tag into `portal_wall` and
`central_wall`; its `analysis/zonation.py` already places each lumen beside a periportal valley
or a pericentral peak. That split is a dependency of milestone 2.

Rust binds PETSc and libCEED through `bindgen` (a `petsc-sys` crate of our own; the official
`libceed` crate). MPI on the Rust side through the `mpi` crate, sharing the communicator with
PETSc.

### ODE layer (new work in sbmlsim-core)

- A new code generation target in sbmlsim-core, from `sbmlsim_ir`: a batched right-hand side and
  analytic Jacobian over all ODE instances at once, struct-of-arrays layout, per-instance
  parameters as arrays.
- Emitted for CPU (SIMD), browser (wasm SIMD128) and GPU (CUDA/HIP kernels). The GPU kernel
  targets the vendor of the available cluster first; the second vendor follows.
- Supported SBML (decided): plain ODE models, meaning reactions, rate rules, assignment rules,
  initial assignments and `piecewise`. Models with events, delays or algebraic rules are refused
  with a clear error naming the unsupported feature. Events come in a later phase.
- A per-instance adaptive implicit integrator (Rosenbrock or SDIRK) with a small dense LU per
  instance, not one CVODE instance per element. The integrator algorithm is the same on all
  targets.
- The semantics of the expressions are those of sbmlsim-core's existing code generation (no fast
  math, libroadrunner's meaning of every operator).
- Verification: the batched kernel against sbmlsim-core's CVODE engine, per instance, within
  tolerance.
- Compilation (design call). Models are data, so kernels are compiled per study and cached by
  the hash of the model and the target:
  - browser: sbmlsim-core's codegen already emits wasm directly from Rust, so it runs in the
    browser and is extended with a batched SIMD128 emitter. No external compiler.
  - CPU: emitted C, compiled by the system C compiler at study time and loaded dynamically.
  - GPU: emitted CUDA or HIP source, compiled at runtime by NVRTC or hipRTC.

### Coupling

- One ODE instance per element of a chosen level of the mesh hierarchy: in 3D an octree level
  chosen in the study (about 20 um), in 2D the coarse gmsh mesh (element size from the slideviz
  size field). The PDE may be solved on finer levels; sources and concentrations are transferred
  between the ODE level and the PDE level by the restriction and prolongation of the hierarchy.
- Cell type, and with it the SBML model, and the parameters of each instance are sampled from the
  position fields of the scene.
- Each element runs exactly one model or none (design call). Elements without a model (lumen,
  necrotic tissue) still carry PDE coefficients. Mixtures of cell types are expressed as
  parameters of one model (for example fractions from spatial transcriptomics), not as several
  models per element. The batched kernel groups instances by model.
- Exchange per split step:
  - ODE species to PDE field: uptake or secretion flux as a source term;
  - PDE field to ODE: local concentration as an input of the model;
  - ODE variable to PDE coefficient: for example permeability or diffusion, which changes the
    operator (cheap with matrix-free operators; Darcy is recomputed when permeability changes).

### Contracts

Versioned JSON contracts, as in sbmlsim-core and sbmlsim-web. Each contract has one Rust
definition (serde) and a JSON Schema generated from it.

slidesim owns the definitions of all three contracts; slideviz writes `scene/1` against the
published JSON Schema. The 3D label volume (registered sections stacked into one volume) is new
work in slideviz and a dependency of milestone 7.

- **`scene/1`**, from slideviz: the mesh (gmsh `.msh` for 2D, label volume for 3D), region and
  edge tags, physical units and frame, and position fields (cell type, expression-derived values)
  as Zarr.
- **`study/1`**: PDE fields with diffusion, advection and boundary conditions per tag; the SBML
  model per cell type; the coupling map; the parameter mapping from scene fields; the ODE level
  of the octree; time course and outputs.
- **`result/1`**: Zarr v3, multiscale, sharded, time as an axis, aligned to the slide pyramid and
  the octree levels. VTKHDF beside it for ParaView/VTK.

### Python

PyO3/maturin bindings, as in sbmlsim-core. Build and validate `study/1`, run studies, read
results as xarray, compare simulated fields against spatial transcriptomics for validation.
Python is never in the hot loop.

### Viewer

- Own WebGPU viewer in TypeScript, Vue 3 shell and Vite, the stack of sbmlsim-web.
- Two data sources behind one interface:
  - remote: Zarr v3 chunks of the slide pyramid (OME-Zarr from slideviz) and of `result/1`;
  - live: fields from the in-browser wasm simulation in memory.
- Fetches only visible chunks at the level of detail of the view.
- 2D: pyramid navigation, layers of slide channels, mesh and fields with opacity and colormaps.
- 3D: arbitrary slice planes, volume ray marching with transfer functions, the mesh at zoomed-in
  levels.
- Time scrubbing and playback of time courses.
- Prior art: Neuroglancer (streams multiscale chunks with 2D/3D cross sections).
- Offline: ParaView/VTK renders from VTKHDF.

## Milestones (proposed order)

0. **Spike (throwaway)**: PETSc + libCEED built into `wasm32-wasip1`, a 2D diffusion example
   solved in Node; the same example natively. Decides option 1 versus the fallback.
1. 2D diffusion-reaction on a slideviz gmsh mesh on CPU, verified by manufactured solutions and
   convergence rates.
2. Darcy perfusion and advection-diffusion-reaction with SUPG, verified the same way.
3. sbmlsim-core batched codegen and integrator on CPU; Strang coupling; verified against
   sbmlsim-core's engine per instance.
4. `result/1` output and viewer v1: 2D pyramid, mesh, fields, time.
5. Browser build end to end: wasm simulation feeding the viewer live.
6. GPU: libCEED GPU backends, PETSc device vectors, the ODE GPU kernel, MPI scaling.
7. 3D: octree from label volumes, viewer slice planes and volume rendering, cluster scaling.
8. Python API completed, spatial transcriptomics validation tools, VTKHDF offline renders.

Acceptance (design call): each milestone is accepted by its verification tests (manufactured
solutions, convergence rates, agreement with sbmlsim-core, agreement across targets). The scale
of 10^8-10^9 unknowns is the goal of milestones 6 and 7, measured by weak and strong scaling
benchmarks, not a gate for the earlier milestones.

## Constraints

- Our own libraries (sbmlsim-core and others) are used at their latest release.
- PETSc, libCEED and p4est are pinned to one version per release of slidesim, built the same way
  on all targets.
- License: slidesim is MIT. Only the optional `octree` feature build, which links p4est, is GPL.
- GPU CI availability is still unknown; milestone 6 needs access to a GPU machine for
  verification.
- Documentation with Zensical, as in the other repositories.

## Left to the builder

- Repository layout of the Rust workspace and crate boundaries.
- Exact integrator scheme for the ODE instances and its error control.
- Stabilization details for advection and choice among PETSc `TS` schemes.
- Zarr chunk and shard sizes, compression.
- The internal structure of the viewer and its rendering techniques.
- Build tooling for the native and GPU stacks (spack, conda-forge/pixi or own scripts).
