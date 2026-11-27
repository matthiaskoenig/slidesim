# Lobule demo

[Open the lobule demo](demo/lobule/index.html){ .md-button .md-button--primary }

The demo runs slidesim in the browser: the browser build of slidesim (`slidesim.wasm`, PETSc,
libCEED and Mmg in WebAssembly) meshes a parametric liver lobule, solves the perfusion and steps
the SPT study on it, the transport of the substrate S and the product P with the SPT cell model
in every element. Nothing is sent to a server: the page compiles the cell model from its SBML with
sbmlsim-core's compiler, and the mesh, the flow, the fields and the cells are computed on your
computer, in a worker beside the page.

## What it shows

- **Lobule**: the radius of the hexagonal lobule, the rings of neighbouring lobules (0 to 4: 1,
  7, 19, 37 or 61 lobules), the displacement of the veins (`jitter`, a fraction of the radius,
  with its `seed`) and the radii of the central vein and of the portal veins at the corners
  ([Meshing](methods/meshing.md#ideal-lobules), the `lobules` source of
  [scene/1](contracts.md#mesh-sources)). The page names the tags itself: the area `tissue`, the
  walls `central_wall` and `portal_wall`, the outer boundary `field_cut` and the zonation map
  `zonation`.
- **Mesh**: the element size at the vein walls and in the interior. The default (8 and 30 µm,
  about 3500 triangles) keeps a run light; the examples use 5 and 20 µm.
- **Simulation**: the simulated time. The study is that of
  `testdata/studies/lobules/spt_batched.study.json` ([Example studies](examples.md)): S enters
  at 5 mM through the portal veins, the flow carries it to the central vein, the cells take it up
  and give P back; the cell model runs on the batched kernel with a split step of 30 s, the linear
  solves to `1e-6`, an output every minute.

Start meshes and runs; Stop ends the run at once. **Show** selects a field (P1 values at the
vertices, interpolated) or a value of the cells (one per element: the states of the model, its
parameters such as `position` from the zonation, and `protein`). The colour scale spans the values
of the current output time. **Totals** are those of the run's `summary.json`
([Running a study](usage.md#summaryjson)): the amount of each field in the lobule and what it
received from the cells since the start (negative: taken up), the number of cells and the inflow
through the portal veins.

On a laptop the default lobule simulates 30 minutes in under ten seconds; each ring of neighbours
multiplies the mesh about seven times, and the time with it. Errors of slidesim (veins which would
touch, a mesh the mesher cannot make) appear under the drawing with slidesim's message.

## The API under it

The page uses the live API of the browser build (`wasm/js/slidesim.mjs`, declared in
`wasm/js/slidesim.d.ts`, see [Contracts](contracts.md#live-api) and
[Running a study](usage.md#in-the-browser)): `lobuleScene` builds the scene of the parameters,
`createSimulation` creates the simulation, and each `step()` advances to the next output time,
whose `field(id)` and `cellValues(name)` the worker posts to the page.

## Building it locally

The site's build in CI puts the browser build beside the page. Locally:

```bash
scripts/wasm/build.sh --browser              # slidesim.wasm and sbmlsim-core's modules
scripts/wasm/bundle.sh target/wasm/demo/lobule
python3 -m http.server --directory target/wasm/demo 8000
# open http://localhost:8000/lobule/
```

A copy of the site built without the bundle (`zensical serve`) has the page but not the modules:
it says so when Start is pressed.
