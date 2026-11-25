# slidesim milestone 2 - perfusion and advection - design

Status: draft for review, 2026-11-25.

Parent design: `docs/superpowers/specs/2026-11-24-slidesim-design.md` (called "the spec" below).
This document details milestone 2 of the spec. Where both say something, this document is the
more specific one.

## Intent

Milestone 1 solves diffusion-reaction with constant coefficients on a slideviz mesh. Milestone 2
adds blood flow: the sinusoids are a homogenized porous medium, a steady Darcy pressure solve gives
the flow from the portal walls to the central walls, and the species are carried by that flow
(advection-diffusion-reaction, stabilized with SUPG).

The flow only means something on a real slide when the tissue parameters follow the zonation of
the lobule. So milestone 2 also brings the first position fields from slideviz into slidesim
(the spec's "position fields as Zarr" of `scene/1`), and the slideviz work which splits the vessel
walls into portal and central walls. Milestone 3 (ODE coupling) builds on the same position fields
to choose models and parameters per element.

In this document a position field of the scene is called a **map**, to keep it apart from the
**fields** of a study (the species which the PDEs solve for).

## Goals

1. Maps in `scene/1`: rasters written by slideviz as Zarr v3, sampled by slidesim at the
   quadrature points of every mesh level.
2. Coefficients from maps: a coefficient of the study is a number or a piecewise-linear table over
   a map.
3. Darcy perfusion: a steady pressure solve with anisotropic permeability along the gradient of
   the zonation map, and the velocity at the quadrature points.
4. Advection-diffusion-reaction of the species with SUPG, carried by the Darcy velocity, with a
   porosity.
5. slideviz exports a scene: the zonation map, the lumen classes, the wall tags `portal_wall` and
   `central_wall`, and `scene.json`.
6. Verification by manufactured solutions and convergence rates, a conservation test, agreement
   across process counts, and an end-to-end run on a real slideviz mesh.

## Non-goals

- Recomputing the pressure during a run. Nothing changes the permeability in milestone 2; the ODEs
  will in milestone 3, and the design keeps the pressure solve a step of its own so that it can be
  repeated.
- 3D, octree meshes, GPU backends, the browser build of the real crates (milestones 5-7).
- `result/1` output (milestone 4). Output stays VTU as in milestone 1, with pressure and velocity
  added.
- Maps from spatial transcriptomics. slideviz has none yet; the map mechanism is generic and takes
  them when they exist.
- Several compartments (blood space and cells) per species. One compartment, the blood space, in
  milestone 2.
- A mixed (mass-conservative) Darcy formulation. The spec decided the primal pressure Poisson
  problem.
- Elements of degree 2. Milestone 2 stays with P1 triangles, as milestone 1 (call; `Discretization`
  supports degree 1 only).

## Anti-goals

From the spec, and in particular for this milestone:

- Liver-specific code in the core. "Zonation", "portal" and "central" appear only as data: names of
  maps and tags in the scene and the study. The core knows maps, tables, tags and coefficients.
- Silent defaults where data are missing: a quadrature point outside a map, or on a no-value pixel
  of a map which a coefficient uses, is an error naming the map, never a fallback value.
- A second discretization of advection or Darcy outside libCEED.

## Constraints

- All constraints of the spec and of the milestone 1 plan hold: MIT, pinned PETSc and libCEED,
  `-ffp-contract=off -fno-fast-math`, fmt/clippy/tests clean at every commit, no agent attribution,
  no em dash.
- Units (extending milestone 1): length um, time s, concentration mM, pressure Pa, permeability
  um^2, viscosity Pa*s, velocity um/s, porosity dimensionless.
- Our own libraries at their latest release.
- Zarr: `zarr>=3.3` in slideviz (already a dependency), the `zarrs` crate in slidesim.
- The slideviz work happens in `~/git/slideviz`, in its own branch and pull request, following that
  repository's conventions (uv for the package, pixi for the mesh environment with DOLFINx 0.11).

## Context: what exists

slidesim (end of milestones 0-1):

- `contracts`: `scene/1` has dimension, units (`um` only) and a gmsh mesh path. `study/1` has fields
  with constant `diffusion`, `decay`, `source`, `initial` and boundary conditions (`dirichlet` or
  `flux`) per tag; time; mesh refinements; PETSc options.
- `pde`: one `Problem` per field, BDF2 in PETSc `TS`, libCEED operators for mass and
  diffusion-reaction (QFunctions in C headers), geometric multigrid on the refinement hierarchy,
  coefficients as scalars (`Multigrid::set_coefficients(d, k, shift)`).
- `mesh`: gmsh to DMPlex with tags, refinement, PT-Scotch partitioning.

slideviz:

- `mesh/build_mesh.py` builds the gmsh mesh from polygons. Edge tags: `vessel_wall` (11),
  `section_edge` (12), `field_cut` (13), `necrosis_border` (14); areas `surviving` (1), `necrotic`
  (2), `tissue` (3). An edge is `vessel_wall` when it bounds a hole or when the lumen mask is set
  beside its midpoint (`lumen_beside`). The lumen mask is boolean: no lumen ids, no classes.
- The mesh frame: um in H&E level 0 coordinates, x = column, y = row (pointing down). Rasters state
  `um_per_px` and `origin_um` (row, col) for the centre of pixel (0, 0).
- `src/slideviz/analysis/zonation.py`: the CYP2E1 landscape (peaks pericentral, valleys periportal,
  about 7 um grid), the position `p = dp/(dp+dc)` from 0 (periportal) to 1 (pericentral) per pixel
  (`portality()`), and the classification of lumina within 15 um of a peak (central) or a valley
  (portal); a lumen near both is neither. Both p and the classes stay in memory; only aggregates are
  saved (`.npz`).
- slideviz reads Zarr but writes none; nothing mentions slidesim or `scene/1`.

## Approaches considered

### How a map becomes a coefficient

| Option | Summary | Verdict |
|---|---|---|
| **A. Rasters in the scene, sampled at quadrature points** | slideviz writes maps as Zarr v3 rasters in the mesh frame; slidesim samples them bilinearly at the quadrature points of each level; the study maps values to coefficients by tables. | **Chosen (user).** Detail independent of the mesh, refined levels keep it, model parameters stay in the study, the same mechanism serves 3D label volumes and the cell types of milestone 3, and it is the layout the spec set for `scene/1`. |
| B. Values per element baked into the mesh by slideviz | Simple reader. | Rejected. Model parameters leak into slideviz; refinement cannot recover detail below the coarse element. |
| C. Maps interpolated once onto the mesh nodes by slidesim | No raster reads during assembly. | Rejected. Loses detail below the element size; gradients (for anisotropy) piecewise constant. |

### Direction of anisotropy

| Option | Verdict |
|---|---|
| Isotropic permeability only | Rejected: the sinusoids run from portal to central, the flow is anisotropic. |
| **Anisotropic along the gradient of a chosen map, isotropic where the gradient vanishes** | **Chosen (user)**, with the zonation map. |
| Direction from a measured orientation map (sinusoid directions from image analysis) | Rejected for now: new image analysis in slideviz. The design takes it later as another map without changing the operator. |

## Decided design

Marks: **(user)** decided by the user in the brainstorming; **(spec)** decided in the spec;
**(call)** Claude's call, open to correction in review.

### Maps in `scene/1`

- `scene/1` gains `maps`, a list of entries (call):
  - `id`: identifier, as for field ids;
  - `path`: a Zarr v3 array (relative to the scene file);
  - `units`: a string, informative (`"1"` for dimensionless);
  - `um_per_px`: pixel size, the same on both axes;
  - `origin_um`: `[x, y]` of the centre of pixel (row 0, col 0) in the mesh frame;
  - `no_value`: optional; pixels with this value (or NaN, always) carry no data.
  The array is 2D (rows = y, cols = x), float32 or float64. The contract stays `scene/1` and is
  extended in place, because no version has been released (call).
- slidesim reads maps with the `zarrs` crate, rank by rank only the chunks covering the local part
  of the mesh (call: a whole-slide map can be large; reading it all on every rank does not scale).
  The reads happen after partitioning and refinement: each rank reads the bounding box of its
  local cells on all levels plus one pixel. The reads are independent per rank, not collective
  (call).
- Sampling (user: A): bilinear at the quadrature points of every level which needs the coefficient.
  A quadrature point outside the raster, or with a no-value or NaN pixel among its four
  neighbours, is an error naming the map, the point and the area tag (call, anti-goal "no silent
  defaults"). The slideviz export makes the maps cover the meshed area with a margin.
- Gradient of a map (for anisotropy): central differences on the raster, then sampled bilinearly
  (call: continuous across pixel borders, unlike the derivative of the bilinear interpolant).

### Coefficients from maps

- In `study/1`, a coefficient is a number or a table (call):
  `{"map": "<map id>", "points": [[v0, c0], [v1, c1], ...]}`, values `v` strictly increasing,
  linear between points, constant beyond the first and last point (call).
- Coefficients which take a table: `diffusion`, `decay`, `source` and `porosity` of a field, and
  `k_along` and `k_across` of the perfusion (call: one mechanism for all of them; milestone 3 needs
  spatial parameters anyway).
- A table naming a map which the scene does not have is an error before any solve, listing the
  maps (as unknown tags are today).
- Coefficients become arrays at the quadrature points; the operators and the multigrid levels take
  arrays instead of scalars. On coarse multigrid levels the coefficients are sampled from the maps
  at the coarse quadrature points (call): the coarse operators are a rediscretization, as in
  milestone 1, and serve only as preconditioner; the fine operator alone determines the result, so
  the parity across process counts is unaffected.

### Darcy perfusion

- `study/1` gains an optional `perfusion` section (call for its shape):
  - `viscosity` (Pa*s);
  - `k_along`, `k_across` (um^2): coefficients, numbers or tables;
  - `direction`: `{"map": "<map id>", "g0": <1/um>}`;
  - `boundaries`: a list of `{"tag": ..., "pressure": <Pa>}`. Every other boundary of the domain is
    no-flow.
  Without `perfusion` there is no flow and the run is as in milestone 1.
- Equation (spec): steady `-div(K grad p / mu) = 0`, pressure fixed on the listed tags. At least one
  pressure boundary is required (else the pressure is not determined): an error otherwise.
- Permeability (user: anisotropic along the zonation gradient; formula is a call). The literature
  (`docs/parameters/m2-perfusion-mouse.md`) finds the in-plane permeability of the lobule close to
  isotropic (radial / circumferential about 0.9, human); the mechanism stays, and strong radial
  anisotropy is a sensitivity case only.
  `K = k_across I + w (k_along - k_across) t t^T` with `t = g/|g|`, `g` the gradient of the
  direction map, and `w = |g|^2 / (|g|^2 + g0^2)`, a smooth blend to isotropic where the gradient
  vanishes (peaks and valleys of the zonation). The tensor is symmetric positive definite whenever
  `k_along, k_across > 0` (checked; a non-positive value is an error naming the tag or map).
- Solver (spec): CG with the geometric multigrid of milestone 1 on the same hierarchy.
- Velocity (spec, call on details): `u = -K grad p / mu` at the quadrature points of the PDE level,
  stored as a libCEED vector at quadrature points. No projection onto a finite element space.
- Pressure is solved once, before the first time step. The solve is a separate step with its own
  entry point, so that milestone 3 can repeat it when the ODEs change the permeability.
- Output: pressure (nodal) and velocity (projected to nodes for output only, call) as VTU next to
  the fields.

### Advection-diffusion-reaction

- A field gains `advection: true|false` (call; default false) and `porosity` (default 1).
- Equation (call): `phi dc/dt + u . grad c - div(phi D grad c) + phi k c = phi s`. With
  `div u = 0` (Darcy without sources) this equals the conservative form. Species live in the
  blood space (call, accepted by the user in the playback).
- SUPG stabilization (spec), with the residual-based test function term; the stabilization
  parameter tau is left to the builder (see below).
- Time integration: BDF2 in PETSc `TS` as in milestone 1. Linear solver GMRES (spec) with the
  geometric multigrid; on coarse levels the SUPG operator is rediscretized (how the coarse velocity
  is obtained is left to the builder).
- Boundaries (call):
  - `dirichlet` and `flux` as in milestone 1 (the flux is the diffusive flux);
  - inflow and outflow are decided per tag, by the net flux `integral of u . n` over the tag after
    the pressure solve (call). A tag with net inflow needs a `dirichlet` condition for every field
    with advection: an error before the first time step otherwise, naming the tag and the field;
  - in the weak form the advective boundary term is integrated per quadrature point, so local
    backflow on an outflow tag is handled consistently (call);
  - on outflow boundaries (`u . n > 0`) without a condition: free outflow (zero diffusive flux,
    advection leaves the domain).
  In a liver study this means a fixed concentration on `portal_wall` and free outflow on
  `central_wall`; the core knows only tags.

### slideviz export

All in `~/git/slideviz`, own branch and pull request (call for the file layout).

- Persist what `zonation.py` computes on the landscape grid:
  - the zonation map: p from 0 (periportal) to 1 (pericentral) (`Portality.points`), NaN outside
    the section outline;
  - the lumen classes: a uint8 raster, 0 no lumen, 1 portal, 2 central, 3 unclassified.
- `mesh/build_mesh.py` samples the lumen class beside each vessel edge, as `lumen_beside` samples
  the lumen mask today, and tags it `portal_wall`, `central_wall` or `vessel_wall` (unclassified,
  or a hole without a class). `vessel_wall` keeps 11; `portal_wall` is 15 and `central_wall` 16
  (call). The split always happens: nothing else in slideviz reads `vessel_wall`, and the slidesim
  fixtures which use it are synthetic and are updated (call: a breaking change, no compatibility
  rule).
- A scene writer: `scene.json` (`scene/1`) next to the `.msh`, and the maps as Zarr v3 arrays,
  cropped to the mesh window with a margin of at least two pixels. The writer converts slideviz's
  `origin_um` (row, col) to the scene's `[x, y]`. slideviz vendors a copy of slidesim's
  `schemas/scene-1.schema.json` and validates every written scene against it in its tests (call).
- The zonation map has NaN outside the section outline; the export fills every NaN pixel inside the
  cropped window by nearest-neighbour extension (call), so that every quadrature point of the mesh
  on every level finds data. In the filled pixels the gradient is about zero, so K is isotropic
  there (near lumen walls); this is accepted (call).
- The zonation map is `Portality.points` on the working grid of the readout (1.8 um, one pixel
  size for both axes): a ratio of distance transforms to the portal and central seeds, continuous
  and without the pixel noise of the stain, so its gradient is used without further smoothing
  (call).

### Verification and acceptance

Each item is a test; the milestone is accepted when all pass (spec: acceptance by verification
tests).

1. Maps: bilinear sampling exact for bilinear functions; gradient exact for linear functions;
   out-of-raster and no-value points are errors naming the map; chunked reads on several ranks give
   the same samples as one rank.
2. Darcy, manufactured solutions on the unit square and on a slideviz-like tagged mesh, isotropic
   and anisotropic K (direction from an analytic map): pressure O(h^2) in L2 (P1), velocity O(h) in
   L2, on 4 refinement levels.
3. Advection-diffusion-reaction with SUPG, manufactured solutions: the expected rates from a cell
   Peclet number below 1 to an advection-dominated case (Peclet about 100); no oscillations above a
   tolerance in an advection-dominated front test.
4. Conservation: the boundary fluxes are computed as consistent fluxes from the residual of the
   discrete equations (call), so the change of total mass equals inflow minus outflow plus sources
   to 1e-8 relative. The non-conservation of the primal velocity (from the jumps of its normal
   component between elements) is reported, not gated.
5. Parallel: the same results on 1, 2 and 3 processes within 1e-10 relative (as milestone 1).
6. End to end: a real slideviz mesh with `portal_wall` and `central_wall` and the zonation map,
   exported by the new slideviz writer, run by `slidesim run`: flow goes from portal to central walls
   (positive net outflow through every central wall), and a tracer front entering at the portal walls
   reaches the central walls. Rendered images are inspected. The study uses the literature
   parameter set of `docs/parameters/m2-perfusion-mouse.md` (baseline and its sensitivity cases);
   its interstitial velocity (about 73 um/s) is checked against the mouse sinusoidal velocities
   recorded there (69-180 um/s).

### Order of work

1. slidesim: maps in `scene/1`, sampling, tables in `study/1`; coefficients as arrays.
2. slidesim: Darcy with anisotropy, verified on synthetic meshes.
3. slidesim: advection with SUPG and porosity, verified.
4. slideviz: zonation map and lumen classes persisted, wall tags, scene writer.
5. End-to-end run on a real slide.

## Left to the builder

- The SUPG stabilization parameter tau (for P1 triangles, with the transient term) and the element
  length used in it.
- Treatment of the outflow boundary terms in the weak form, as long as the boundary behaviour above
  holds.
- Zarr chunk sizes and compression of the maps written by slideviz.
- Whether velocity is recomputed on each multigrid level from the pressure of that level or
  restricted from the fine level, as long as the convergence and parity tests hold.
- The VTU layout of pressure and velocity output.
- The structure of the slideviz scene writer within that repository.
