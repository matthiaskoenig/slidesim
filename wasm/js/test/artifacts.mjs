// What the tests of wasm/js share: the artifacts of scripts/wasm/build.sh (their paths in
// target/wasm/artifacts.env), the inputs of the lobule studies and helpers of the comparisons.

import { readFileSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

/** The repository. */
export const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "../../..");

/** The paths of target/wasm/artifacts.env, by key. */
export function artifacts() {
  const path = join(ROOT, "target/wasm/artifacts.env");
  let text;
  try {
    text = readFileSync(path, "utf8");
  } catch (error) {
    throw new Error(`${path}: ${error.message}; run scripts/wasm/build.sh`);
  }
  const values = {};
  for (const line of text.split("\n")) {
    if (line.startsWith("#")) continue;
    const at = line.indexOf("=");
    if (at > 0) values[line.slice(0, at)] = line.slice(at + 1);
  }
  for (const key of ["SLIDESIM_WASM", "SBMLSIM_COMPILER", "SBMLSIM_MATHLIB"]) {
    if (!values[key]) throw new Error(`${path} has no ${key}; run scripts/wasm/build.sh`);
  }
  return values;
}

/** The modules and the SPT model, as `createSimulation` takes them. */
export async function modules() {
  const a = artifacts();
  return {
    slidesim: await readFile(a.SLIDESIM_WASM),
    compiler: await readFile(a.SBMLSIM_COMPILER),
    mathlib: await readFile(a.SBMLSIM_MATHLIB),
    sbml: await readFile(join(ROOT, "testdata/models/spt_liver_cell.xml"), "utf8"),
  };
}

/** The tag and map names of the lobule demo (and of testdata/studies/lobules). */
export const TAGS = {
  centre: "central_wall",
  corner: "portal_wall",
  outer: "field_cut",
  area: "tissue",
  zonation: "zonation",
};

/** A JSON file of the repository. */
export async function readJson(path) {
  return JSON.parse(await readFile(join(ROOT, path), "utf8"));
}

/** The SPT study of the lobule on the batched path, shortened to `end` s with `outputs`. */
export async function sptStudy(end, outputs) {
  const study = await readJson("testdata/studies/lobules/spt_1h.study.json");
  study.time.end = end;
  study.time.outputs = outputs;
  return study;
}

/** A study of one field on the lobule, without flow and cells, of one step: cheap. */
export function diffusionStudy() {
  return {
    version: "study/1",
    scene: "scene.json",
    fields: [
      {
        id: "S",
        diffusion: 700.0,
        decay: 0.0,
        source: 0.0,
        porosity: 0.15,
        advection: false,
        initial: 0.0,
        boundaries: [{ tag: "portal_wall", dirichlet: 5.0 }],
      },
    ],
    time: { end: 15.0, dt: 15.0, adaptive: false, outputs: [0.0, 15.0] },
    mesh: { refinements: 0 },
  };
}

/** The bits of the floats of `values` as one string, for an exact comparison. */
export function bits(values) {
  const array = Float64Array.from(values);
  return Buffer.from(array.buffer).toString("hex");
}

/**
 * The terms of a mass balance of `summary.json` and the residual of `storage_rate = source +
 * inflows + exchange - advection_defect`.
 */
export function balanceResidual(balance) {
  const terms = [
    balance.storage_rate,
    balance.source,
    balance.advection_defect,
    balance.exchange,
  ];
  let inflows = 0;
  for (const inflow of Object.values(balance.boundary)) {
    for (const part of ["advective", "diffusive", "resupply"]) {
      terms.push(inflow[part]);
      inflows += inflow[part];
    }
  }
  const residual =
    balance.storage_rate - balance.source - inflows - balance.exchange + balance.advection_defect;
  const scale = terms.reduce((m, x) => Math.max(m, Math.abs(x)), 0);
  return { residual, scale };
}
