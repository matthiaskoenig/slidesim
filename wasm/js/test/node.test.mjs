// The JavaScript glue (slidesim.mjs) in Node: the lobule study runs, agrees bit for bit with the
// wasmtime host of wasm/host, and refusals, errors and traps reach the caller as `Error`s.

import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { test } from "node:test";

import { createSimulation, lobuleScene } from "../slidesim.mjs";
import {
  ROOT,
  TAGS,
  artifacts,
  balanceResidual,
  bits,
  diffusionStudy,
  modules,
  readJson,
  sptStudy,
} from "./artifacts.mjs";

/** An empty directory of the test `name` under target/tmp. */
function freshDir(name) {
  const dir = join(ROOT, "target/tmp/wasm-js", name);
  rmSync(dir, { recursive: true, force: true });
  mkdirSync(dir, { recursive: true });
  return dir;
}

/** Runs a program to its end (at most `seconds`), failing the test on a failure of it. */
function runProgram(program, args, seconds) {
  const result = spawnSync(program, args, {
    encoding: "utf8",
    timeout: seconds * 1000,
    killSignal: "SIGKILL",
    env: { ...process.env, HWLOC_COMPONENTS: "-gl" },
  });
  assert.equal(result.error, undefined, `${program}: ${result.error}`);
  assert.equal(result.status, 0, `${program} ${args.join(" ")}: ${result.stderr}`);
}

test("lobule_scene_has_the_callers_names_and_slidesims_defaults", () => {
  assert.deepEqual(lobuleScene({}, TAGS), {
    version: "scene/1",
    dimension: 2,
    units: { length: "um" },
    mesh: {
      format: "lobules",
      centre: { tag: "central_wall" },
      corner: { tag: "portal_wall" },
      outer_tag: "field_cut",
      area_tag: "tissue",
      zonation: { map: "zonation" },
    },
  });
  const scene = lobuleScene(
    {
      lobuleRadiusUm: 250,
      rings: 1,
      jitter: 0.1,
      seed: 7,
      centreRadiusUm: 20,
      cornerRadiusUm: 15,
      wallSizeUm: 6,
      interiorSizeUm: 25,
    },
    TAGS,
  );
  assert.deepEqual(scene.mesh, {
    format: "lobules",
    lobule_radius_um: 250,
    rings: 1,
    jitter: 0.1,
    seed: 7,
    centre: { radius_um: 20, tag: "central_wall" },
    corner: { radius_um: 15, tag: "portal_wall" },
    outer_tag: "field_cut",
    area_tag: "tissue",
    size: { wall_um: 6, interior_um: 25 },
    zonation: { map: "zonation" },
  });
  assert.throws(() => lobuleScene({ radius: 3 }, TAGS), /no parameter radius/);
  assert.throws(() => lobuleScene({}, { ...TAGS, zonation: undefined }), /tags\.zonation/);
});

test("lobule_runs_in_node", async () => {
  const study = await sptStudy(1800, [0, 900, 1800]);
  const sim = await createSimulation({
    ...(await modules()),
    scene: lobuleScene({}, TAGS),
    study,
  });
  // Right after createSimulation: output index 0, the initial state.
  assert.deepEqual(Array.from(sim.times), [0, 900, 1800]);
  assert.equal(sim.time, sim.times[0]);
  assert.equal(sim.meta.written, 1);
  assert.equal(sim.meta.complete, false);
  const quality = sim.meshQuality();
  assert.ok(quality.min_angle_deg >= 20, JSON.stringify(quality));
  assert.match(sim.log(), /^mesh: \d+ triangles/);
  assert.deepEqual(Object.keys(sim.mesh.tags).sort(), [
    "central_wall",
    "field_cut",
    "portal_wall",
    "tissue",
  ]);
  const vertices = sim.mesh.vertices.length / 2;
  assert.equal(sim.mesh.triangles.length / 3, sim.mesh.element.length);
  assert.ok(sim.cellIds.length > 0);
  const { fields, cell_values: cellValues } = sim.meta;

  const visited = [];
  for (let index = 0; ; ) {
    assert.equal(sim.time, sim.times[index]);
    assert.equal(sim.meta.written, index + 1);
    for (const { id } of fields) {
      const values = sim.field(id);
      assert.equal(values.length, vertices);
      assert.ok(values.every(Number.isFinite), `field ${id} at index ${index}`);
    }
    for (const name of cellValues) {
      const values = sim.cellValues(name);
      assert.equal(values.length, sim.cellIds.length);
      assert.ok(values.every(Number.isFinite), `cells ${name} at index ${index}`);
    }
    visited.push(index);
    const next = sim.step();
    if (next === null) break;
    assert.equal(next, index + 1);
    index = next;
  }
  assert.deepEqual(visited, [0, 1, 2]);
  assert.equal(sim.meta.complete, true);
  assert.equal(sim.step(), null, "done stays done");

  const summary = sim.summary();
  for (const { id } of fields) {
    const balances = summary.fields[id].mass_balance;
    assert.equal(balances.length, 3);
    assert.equal(balances[0], null);
    for (const balance of balances.slice(1)) {
      const { residual, scale } = balanceResidual(balance);
      assert.ok(
        Math.abs(residual) <= 1e-8 * scale,
        `field ${id}: residual ${residual} of ${scale}: ${JSON.stringify(balance)}`,
      );
    }
  }
  sim.free();
  for (const call of [() => sim.field("S"), () => sim.step(), () => sim.mesh, () => sim.time]) {
    assert.throws(call, /the simulation was freed: it is unusable/);
  }
  sim.free(); // twice is no error
});

test("node_matches_wasmtime", async () => {
  const a = artifacts();
  assert.ok(a.SLIDESIM_NATIVE && a.SLIDESIM_HOST, "run scripts/wasm/build.sh");
  const dir = freshDir("node_matches_wasmtime");
  const scenePath = join(ROOT, "testdata/studies/lobules/scene.json");
  // The shared mesh: the native mesher's mesh.json, which both hosts run on.
  runProgram(a.SLIDESIM_NATIVE, ["mesh", scenePath, "--out", join(dir, "mesh")], 120);
  const mesh = JSON.parse(readFileSync(join(dir, "mesh/mesh.json"), "utf8"));
  const scene = await readJson("testdata/studies/lobules/scene.json");
  const study = await sptStudy(1800, [0, 900, 1800]);
  writeFileSync(join(dir, "inputs.json"), JSON.stringify({ scene, study, mesh }));
  runProgram(
    a.SLIDESIM_HOST,
    [
      a.SLIDESIM_WASM,
      join(a.SBMLSIM_KERNELS, "spt_liver_cell"),
      a.SBMLSIM_MATHLIB,
      join(dir, "inputs.json"),
      join(dir, "wasmtime"),
    ],
    300,
  );
  const wasmtime = (name) => readFileSync(join(dir, "wasmtime", name));
  /** Asserts that the bytes of `array` (a typed array of Node) are those wasmtime wrote. */
  const same = (name, array) => {
    const expected = wasmtime(name);
    const actual = Buffer.from(array.buffer, array.byteOffset, array.byteLength);
    assert.ok(actual.equals(expected), `${name}: the bytes differ (${actual.length} vs ${expected.length})`);
  };

  const sim = await createSimulation({ ...(await modules()), scene, study, mesh });
  assert.equal(sim.meshQuality(), null, "a mesh of the inputs");
  assert.equal(sim.log(), wasmtime("log.txt").toString("utf8"));
  same("time", sim.times);
  same("mesh/vertices", sim.mesh.vertices);
  same("mesh/triangles", sim.mesh.triangles);
  same("mesh/element", sim.mesh.element);
  for (const [name, values] of Object.entries(sim.mesh.tags)) same(`mesh/tags/${name}`, values);
  same("cells/element", sim.cellIds);
  const { fields, cell_values: cellValues } = sim.meta;
  let compared = 0;
  for (let index = 0; index !== null; index = sim.step()) {
    const at = `outputs/${index}`;
    same(`${at}/time`, Float64Array.of(sim.time));
    assert.deepEqual(sim.meta, JSON.parse(wasmtime(`${at}/meta.json`)), `${at}/meta.json`);
    for (const { id } of fields) same(`${at}/fields/${id}`, sim.field(id));
    for (const name of cellValues) same(`${at}/cells/${name}`, sim.cellValues(name));
    compared += 1;
  }
  assert.equal(compared, 3);
  assert.deepEqual(sim.meta, JSON.parse(wasmtime("meta.json")));
  // The summary but its wall times.
  const timeless = (summary) => {
    const { phases, cells, ...rest } = summary;
    const { seconds, ...counts } = cells;
    return { ...rest, cells: counts };
  };
  assert.deepEqual(timeless(sim.summary()), timeless(JSON.parse(wasmtime("summary.json"))));
  const S = sim.field("S");
  console.log(
    `node_matches_wasmtime: ${sim.mesh.triangles.length / 3} triangles, ${cellValues.length} ` +
      `cell values, 3 output times bitwise equal; |S|^2 bits ${bits([S.reduce((s, v) => s + v * v, 0)])}`,
  );
  sim.free();
});

test("engine_solver_is_refused_in_the_browser", async () => {
  const study = await sptStudy(1800, [0, 1800]);
  study.cells.solver = "engine";
  await assert.rejects(
    createSimulation({ ...(await modules()), scene: lobuleScene({}, TAGS), study }),
    (error) => {
      assert.ok(error instanceof Error);
      assert.match(
        error.message,
        /cells\.solver "engine" needs native code; the browser build runs "batched"/,
      );
      return true;
    },
  );
});

test("file_scene_is_refused_in_the_browser", async () => {
  const scene = {
    version: "scene/1",
    dimension: 2,
    units: { length: "um" },
    mesh: { format: "gmsh", path: "lobule_toy.msh" },
  };
  await assert.rejects(
    createSimulation({ ...(await modules()), scene, study: diffusionStudy() }),
    /lobule_toy\.msh/,
  );
});

test("trap_is_an_error", async () => {
  const m = await modules();
  const maximumPages = 1024; // 64 MiB
  // 61 lobules (rings 4, the most) do not fit: the mesher's PETSc runs out of memory.
  await assert.rejects(
    createSimulation({
      ...m,
      scene: lobuleScene({ rings: 4 }, TAGS),
      study: diffusionStudy(),
      memory: { maximumPages },
    }),
    (error) => {
      assert.ok(error instanceof Error);
      assert.match(error.message, /needs more memory than its maximum of 1024 wasm pages \(64 MiB/);
      console.log(`rings 4 in 64 MiB: ${error.message.split("\n")[0]}`);
      return true;
    },
  );
  // One lobule with its cells does not fit either: an allocation of Rust or libCEED fails, a trap.
  await assert.rejects(
    createSimulation({
      ...m,
      scene: lobuleScene({}, TAGS),
      study: await sptStudy(60, [0, 60]),
      memory: { maximumPages },
    }),
    (error) => {
      assert.ok(error instanceof Error);
      assert.match(error.message, /needs more memory than its maximum of 1024 wasm pages/);
      assert.match(error.message, /trapped in slidesim_create/);
      return true;
    },
  );
  // The next createSimulation instantiates afresh and runs.
  const sim = await createSimulation({ ...m, scene: lobuleScene({}, TAGS), study: diffusionStudy() });
  assert.equal(sim.step(), 1);
  assert.ok(sim.field("S").some((v) => v > 0));
  sim.free();
});

test("trapped_simulation_is_unusable", async () => {
  const m = await modules();
  const sim = await createSimulation({ ...m, scene: lobuleScene({}, TAGS), study: diffusionStudy() });
  const other = await createSimulation({
    ...m,
    scene: lobuleScene({}, TAGS),
    study: diffusionStudy(),
  });
  // A host function which fails inside a call of slidesim (the clock slidesim reads as it steps)
  // traps that call.
  const now = performance.now;
  performance.now = () => {
    throw new Error("the clock failed");
  };
  try {
    assert.throws(
      () => sim.step(),
      (error) => {
        assert.ok(error instanceof Error);
        assert.match(error.message, /slidesim\.wasm trapped in slidesim_next_output \(the clock failed\)/);
        return true;
      },
    );
  } finally {
    performance.now = now;
  }
  // Every later call throws, the getters included.
  for (const call of [
    () => sim.step(),
    () => sim.field("S"),
    () => sim.summary(),
    () => sim.log(),
    () => sim.mesh,
    () => sim.times,
    () => sim.time,
    () => sim.meta,
    () => sim.cellIds,
  ]) {
    assert.throws(call, /the simulation is unusable: slidesim\.wasm trapped before/);
  }
  sim.free();
  // Another simulation has its own instance: untouched.
  assert.equal(other.step(), 1);
  other.free();
});

test("error_is_not_a_trap", async () => {
  const sim = await createSimulation({
    ...(await modules()),
    scene: lobuleScene({}, TAGS),
    study: diffusionStudy(),
  });
  assert.throws(() => sim.field("nope"), (error) => {
    assert.ok(error instanceof Error);
    assert.match(error.message, /the study has no field 'nope'/);
    return true;
  });
  assert.throws(() => sim.cellValues("protein"), /the study has no cells|no value 'protein'/);
  assert.equal(sim.step(), 1, "the simulation goes on after an error");
  assert.equal(sim.step(), null);
  assert.ok(sim.field("S").some((v) => v > 0));
  sim.free();
});

test("memory_option_is_checked", async () => {
  const m = await modules();
  const options = { ...m, scene: lobuleScene({}, TAGS), study: diffusionStudy() };
  for (const maximumPages of [0, 1.5, 65537, "1024"]) {
    await assert.rejects(createSimulation({ ...options, memory: { maximumPages } }), RangeError);
  }
  // Below the memory slidesim imports: refused before it runs.
  await assert.rejects(
    createSimulation({ ...options, memory: { maximumPages: 1 } }),
    /needs at least \d+ wasm pages of memory, more than memory\.maximumPages 1/,
  );
});

test("compiled_module_runs_capped_simulations", async () => {
  // A page compiles slidesim.wasm once and creates many simulations from it, each with the cap:
  // the memory is the host's (slidesim imports it), so a compiled module is capped like bytes.
  const m = await modules();
  const slidesim = await WebAssembly.compile(m.slidesim);
  const options = { ...m, slidesim, scene: lobuleScene({}, TAGS), study: diffusionStudy() };
  const memory = { maximumPages: 4096 }; // 256 MiB
  const first = await createSimulation({ ...options, memory });
  const second = await createSimulation({ ...options, memory });
  assert.equal(first.step(), 1);
  assert.equal(second.step(), 1);
  assert.equal(bits(first.field("S")), bits(second.field("S")));
  first.free();
  second.free();
  // The cap holds for a compiled module: 61 lobules do not fit in 64 MiB.
  await assert.rejects(
    createSimulation({
      ...options,
      scene: lobuleScene({ rings: 4 }, TAGS),
      memory: { maximumPages: 1024 },
    }),
    /needs more memory than its maximum of 1024 wasm pages \(64 MiB/,
  );
  // Without a cap too.
  const uncapped = await createSimulation(options);
  assert.equal(uncapped.step(), 1);
  uncapped.free();
});
