// The simulation of the lobule demo, off the page's thread: the page posts the compiled modules,
// the SBML text and the scene and study; the worker creates the simulation with slidesim.mjs and
// posts the mesh, then every output time with all fields and cell values, then the end. The page
// stops a run by terminating the worker, also in the middle of a step.

import { createSimulation } from "./slidesim.mjs";

self.onmessage = async ({ data }) => {
  const started = performance.now();
  let sim;
  try {
    const { slidesim, compiler, mathlib, sbml, scene, study } = data;
    sim = await createSimulation({ slidesim, compiler, mathlib, sbml, scene, study });
    const meta = sim.meta;
    const mesh = sim.mesh;
    self.postMessage({
      type: "mesh",
      meta,
      times: sim.times,
      vertices: mesh.vertices,
      triangles: mesh.triangles,
      element: mesh.element,
      tags: mesh.tags,
      cellIds: sim.cellIds,
      quality: sim.meshQuality(),
      seconds: (performance.now() - started) / 1000,
    });
    for (let index = 0; index !== null; index = sim.step()) {
      const fields = {};
      for (const { id } of meta.fields) fields[id] = sim.field(id);
      const cells = {};
      for (const name of meta.cell_values) cells[name] = sim.cellValues(name);
      const transfer = [...Object.values(fields), ...Object.values(cells)].map((a) => a.buffer);
      self.postMessage(
        { type: "frame", index, time: sim.time, fields, cells, summary: sim.summary() },
        transfer,
      );
    }
    self.postMessage({ type: "done", seconds: (performance.now() - started) / 1000 });
  } catch (error) {
    self.postMessage({ type: "error", message: error instanceof Error ? error.message : String(error) });
  } finally {
    sim?.free();
  }
};
