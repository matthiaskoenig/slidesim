# Methods

slidesim discretizes the fields with P1 finite elements on the triangles of a slideviz mesh,
applies its operators matrix-free with libCEED or assembled from them, and integrates in time
with PETSc. The cell models run on the coarse elements of the same mesh, one model instance per
element. Every piece is written once for all targets: the same libCEED QFunctions run on the CPU
backends now and are what libCEED's JIT compiles for GPUs later.

| Layer | Component |
| --- | --- |
| Mesh | a gmsh mesh, or tagged polygons or ideal lobules meshed by slidesim (`spade` and Mmg), in PETSc DMPlex with its tags, partitioned by PT-Scotch, refined uniformly `mesh.refinements` times; the coarse mesh is the level of the cells, the finest the level of the fields |
| Maps | Zarr v3 rasters, each process reads the window over its cells, sampled bilinearly at the quadrature points of every level |
| Operators | libCEED QFunctions in C (`crates/pde/qfunctions/`): mass, tensor diffusion, reaction, advection with SUPG, Darcy; element restrictions and bases from the PETSc discretization, as Ratel and HONEE do |
| Time stepping | BDF2 of PETSc `TS`, fixed or adaptive steps |
| Linear solvers | CG for the symmetric problems (Darcy pressure, diffusion-reaction), GMRES with advection, relative tolerance `solver.linear_rtol` (1e-10 by default) |
| Preconditioner | geometric multigrid (`PCMG`) over the refinement levels, the operator discretized again on every level, Chebyshev-Jacobi smoothers, algebraic multigrid (`PCGAMG`) on the coarse level |
| Cells | sbmlsim-core: the engine (CVODE) per element on native C model code, or the model with a Rosenbrock integrator as a libCEED QFunction over all elements of a process |
| Coupling | Strang splitting of cells and fields, exchange by element means and conservative give-back |

## A run in time

<svg class="sim-diagram" viewBox="0 10 684 186" xmlns="http://www.w3.org/2000/svg" role="img"
     aria-label="The pressure is solved once. Then each split step of length dt runs the cells over the first half, every field over the whole step, and the cells over the second half. Before each cell step the means of the fields over the elements are read, after it the changes are given back to the fields.">
  <defs>
    <marker id="sd-arrow-m" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="currentColor"/>
    </marker>
  </defs>
  <g class="sd-module">
    <rect x="8" y="56" width="116" height="56" rx="6"/>
  </g>
  <text x="66" y="80">pressure</text>
  <text class="sd-sub" x="66" y="97">once, at the start</text>
  <line x1="130" y1="84" x2="152" y2="84" marker-end="url(#sd-arrow-m)"/>
  <g class="sd-instance">
    <rect x="158" y="20" width="504" height="132" rx="10"/>
  </g>
  <text class="sd-title" x="410" y="40">one split step from t to t + dt, to each output time</text>
  <g class="sd-module">
    <rect x="172" y="56" width="146" height="56" rx="6"/>
    <rect x="336" y="56" width="146" height="56" rx="6"/>
    <rect x="500" y="56" width="146" height="56" rx="6"/>
  </g>
  <text x="245" y="80">cells, dt / 2</text>
  <text class="sd-sub" x="245" y="97">means in, changes out</text>
  <text x="409" y="80">fields, dt</text>
  <text class="sd-sub" x="409" y="97">BDF2 steps of time.dt</text>
  <text x="573" y="80">cells, dt / 2</text>
  <text class="sd-sub" x="573" y="97">means in, changes out</text>
  <line x1="320" y1="84" x2="332" y2="84" marker-end="url(#sd-arrow-m)"/>
  <line x1="484" y1="84" x2="496" y2="84" marker-end="url(#sd-arrow-m)"/>
  <text class="sd-sub" x="410" y="136">without cells: the fields step alone, from output time to output time</text>
  <text class="sd-sub" x="410" y="180">two half steps of the cells with no output between them run as one cell step</text>
</svg>

The pages of this section:

- [Meshing](meshing.md): tagged polygons to a graded mesh (constrained Delaunay triangulation,
  Mmg to the metric of a size field, the boundary frozen), the ideal lobules and their zonation.
- [Transport](transport.md): diffusion-reaction and advection of the fields, SUPG, BDF2, the
  multigrid, the mass balance.
- [Perfusion](perfusion.md): the Darcy pressure with an anisotropic permeability, and the
  conservative velocity from equilibrated edge fluxes.
- [Cell models](cells.md): the engine and the batched path, the Strang splitting and the
  exchange with the blood.
- [Perfusion parameters](../parameters/m2-perfusion-mouse.md): the literature parameters of the
  mouse liver lobule behind the perfusion studies.
