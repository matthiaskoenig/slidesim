// The lobule demo: the parameters of a parametric liver lobule and the SPT study on it, run by
// slidesim's browser build in a worker (worker.mjs), the field or cell value of the current output
// time drawn on a canvas. The page needs the bundle beside it (scripts/wasm/bundle.sh):
// slidesim.mjs with sbmlsim-core/, slidesim.wasm, compiler.wasm, mathlib.wasm and the SBML of the
// SPT model.

import { lobuleScene } from "./slidesim.mjs";

/** The tag and map names of the demo's lobules: slidesim fixes none. */
const TAGS = {
  centre: "central_wall",
  corner: "portal_wall",
  outer: "field_cut",
  area: "tissue",
  zonation: "zonation",
};

/** Seconds between the output times (the frames). */
const OUTPUT_EVERY_S = 60;

/** Viridis at nine evenly spaced stops. */
const VIRIDIS = [
  [68, 1, 84], [71, 44, 122], [59, 81, 139], [44, 113, 142], [33, 144, 141],
  [39, 173, 129], [92, 200, 99], [170, 220, 50], [253, 231, 37],
];

/** 256 colours of viridis as RGBA words of an `ImageData` (little endian: ABGR). */
const LUT = (() => {
  const lut = new Uint32Array(256);
  for (let k = 0; k < 256; k++) {
    const x = (k / 255) * (VIRIDIS.length - 1);
    const i = Math.min(Math.floor(x), VIRIDIS.length - 2);
    const f = x - i;
    const [r, g, b] = [0, 1, 2].map((c) => Math.round(VIRIDIS[i][c] + f * (VIRIDIS[i + 1][c] - VIRIDIS[i][c])));
    lut[k] = (255 << 24) | (b << 16) | (g << 8) | r;
  }
  return lut;
})();

/**
 * The SPT study of the lobule on the batched path (testdata/studies/lobules/spt_batched),
 * `minutes` long with an output every minute.
 */
export function demoStudy(minutes) {
  const end = minutes * 60;
  const outputs = [];
  for (let t = 0; t < end; t += OUTPUT_EVERY_S) outputs.push(t);
  outputs.push(end);
  const field = (id, dirichlet) => ({
    id,
    diffusion: 700.0,
    decay: 0.0,
    source: 0.0,
    porosity: 0.15,
    advection: true,
    initial: 0.0,
    boundaries: [{ tag: "portal_wall", dirichlet }],
  });
  return {
    version: "study/1",
    scene: "scene.json",
    fields: [field("S", 5.0), field("P", 0.0)],
    perfusion: {
      viscosity: 0.0024,
      k_along: 0.0156,
      k_across: 0.0175,
      direction: { map: "zonation", g0: 0.00043 },
      boundaries: [
        { tag: "portal_wall", pressure: 390.0 },
        { tag: "central_wall", pressure: 0.0 },
      ],
    },
    cells: {
      model: "spt_liver_cell.xml",
      solver: "batched",
      time_unit_s: 60.0,
      species: [
        { species: "S_ext", field: "S" },
        { species: "P_ext", field: "P" },
      ],
      volumes: { blood: "Vext", cells: "Vli" },
      parameters: [
        { id: "pattern", value: 7 },
        { id: "position", value: { map: "zonation", points: [[0, 0], [1, 1]] } },
        { id: "protein_ext", value: { map: "zonation", points: [[0, 0.1], [1, 1]] } },
      ],
      split_step: 30.0,
      outputs: ["protein"],
    },
    solver: { linear_rtol: 1e-6 },
    time: { end, dt: 15.0, adaptive: false, outputs },
    mesh: { refinements: 0 },
  };
}

const $ = (id) => document.getElementById(id);
const form = $("parameters");
const startButton = $("start");
const stopButton = $("stop");
const show = $("show");
const canvas = $("field");
const status = $("status");
const errorBox = $("error");
const progress = $("progress");
const timeOutput = $("time");

/** What the tests (and a curious reader) can see of the run: its inputs and the last frame. */
const state = { inputs: null, frame: null, done: false };
window.lobuleDemo = state;

/** A number for people: 3 significant digits, exponents for the very large and small. */
function format(x) {
  if (!Number.isFinite(x)) return String(x);
  if (x === 0) return "0";
  const a = Math.abs(x);
  if (a >= 1e5 || a < 1e-3) return x.toExponential(2).replace("e+", "e");
  return String(Number(x.toPrecision(3)));
}

/** The modules and the model, fetched and compiled once. */
let assets = null;
function loadAssets() {
  assets ??= (async () => {
    const fetchOk = async (name) => {
      const response = await fetch(name);
      if (!response.ok) {
        throw new Error(
          `${name}: ${response.status} ${response.statusText}. This copy of the site has no ` +
            "browser build: scripts/wasm/build.sh --browser, then scripts/wasm/bundle.sh into " +
            "this folder.",
        );
      }
      return response;
    };
    const compile = async (name) =>
      WebAssembly.compile(await (await fetchOk(name)).arrayBuffer());
    const [slidesim, compiler, mathlib, sbml] = await Promise.all([
      compile("slidesim.wasm"),
      compile("compiler.wasm"),
      compile("mathlib.wasm"),
      fetchOk("spt_liver_cell.xml").then((r) => r.text()),
    ]);
    return { slidesim, compiler, mathlib, sbml };
  })();
  assets.catch(() => {
    assets = null;
  });
  return assets;
}

/** The lobule parameters of the form, `null` with the first invalid input focused. */
function readParameters() {
  const ids = [
    "lobuleRadiusUm", "rings", "jitter", "seed", "centreRadiusUm", "cornerRadiusUm",
    "wallSizeUm", "interiorSizeUm", "minutes",
  ];
  const values = {};
  for (const id of ids) {
    const input = $(id);
    if (!input.checkValidity() || input.value === "") {
      input.reportValidity();
      input.focus();
      return null;
    }
    values[id] = Number(input.value);
  }
  const { minutes, ...lobule } = values;
  return { lobule, minutes };
}

/** The current run: its worker, mesh and drawing, `null` before the first. */
let run = null;

function setRunning(running) {
  startButton.disabled = running;
  stopButton.disabled = !running;
  for (const input of form.querySelectorAll("input")) input.disabled = running;
}

function showError(message) {
  errorBox.textContent = message;
  errorBox.hidden = false;
}

function start() {
  const parameters = readParameters();
  if (!parameters) return;
  stop(false);
  errorBox.hidden = true;
  const scene = lobuleScene(parameters.lobule, TAGS);
  const study = demoStudy(parameters.minutes);
  state.inputs = { scene, study };
  state.frame = null;
  state.done = false;
  const current = {
    worker: null,
    mesh: null,
    frame: null,
    started: performance.now(),
    end: study.time.end,
  };
  run = current;
  setRunning(true);
  progress.hidden = false;
  progress.removeAttribute("value");
  status.textContent = "Loading slidesim…";
  loadAssets()
    .then((modules) => {
      if (run !== current) return;
      status.textContent = "Meshing the lobule and setting up the simulation…";
      const worker = new Worker(new URL("./worker.mjs", import.meta.url), { type: "module" });
      current.worker = worker;
      worker.onmessage = ({ data }) => receive(current, data);
      worker.onerror = (event) => {
        event.preventDefault();
        fail(current, event.message || "the worker failed");
      };
      worker.postMessage({ ...modules, scene, study });
    })
    .catch((error) => fail(current, error.message));
}

function fail(current, message) {
  if (run !== current) return;
  current.worker?.terminate();
  current.worker = null;
  setRunning(false);
  progress.hidden = true;
  status.textContent = "The run failed.";
  showError(message);
}

/** Stops the run (if `say`, telling where). */
function stop(say = true) {
  if (!run?.worker) return;
  run.worker.terminate();
  run.worker = null;
  setRunning(false);
  if (say) {
    progress.value = run.frame ? run.frame.time / run.end : 0;
    status.textContent = run.frame
      ? `Stopped at t = ${run.frame.time} s.`
      : "Stopped before the first output time.";
  }
}

function receive(current, data) {
  if (run !== current) return;
  if (data.type === "mesh") {
    current.mesh = prepareMesh(data);
    current.fieldIds = data.meta.fields.map((f) => f.id);
    fillSelect(data.meta);
    const q = data.quality;
    status.textContent = `Meshed in ${data.seconds.toFixed(1)} s: ${data.triangles.length / 3} triangles, min angle ${q.min_angle_deg.toFixed(1)}°. Running…`;
    $("placeholder").hidden = true;
  } else if (data.type === "frame") {
    current.frame = data;
    $("legend").hidden = false;
    $("summary").hidden = false;
    const S = data.fields.S;
    let squares = 0;
    for (const v of S) squares += v * v;
    state.frame = {
      index: data.index,
      time: data.time,
      l2: data.summary.fields.S.l2_norm.at(-1),
      squares,
    };
    timeOutput.textContent = `t = ${data.time} s`;
    progress.value = data.time / current.end;
    showTotals(current, data.summary);
    scheduleDraw();
  } else if (data.type === "done") {
    current.worker?.terminate();
    current.worker = null;
    setRunning(false);
    progress.value = 1;
    state.done = true;
    status.textContent = `Done: ${current.end / 60} min simulated in ${data.seconds.toFixed(1)} s.`;
  } else if (data.type === "error") {
    fail(current, data.message);
  }
}

/** The mesh of a run with what drawing it needs. */
function prepareMesh(data) {
  const { vertices, triangles, element, cellIds, tags } = data;
  let [xMin, yMin, xMax, yMax] = [Infinity, Infinity, -Infinity, -Infinity];
  for (let k = 0; k < vertices.length; k += 2) {
    xMin = Math.min(xMin, vertices[k]);
    xMax = Math.max(xMax, vertices[k]);
    yMin = Math.min(yMin, vertices[k + 1]);
    yMax = Math.max(yMax, vertices[k + 1]);
  }
  const cellOf = new Map();
  cellIds.forEach((id, k) => cellOf.set(id, k));
  const triangleCell = new Int32Array(element.length);
  element.forEach((id, t) => {
    triangleCell[t] = cellOf.get(id) ?? -1;
  });
  const walls = [TAGS.centre, TAGS.corner, TAGS.outer].map((name) => tags[name]).filter(Boolean);
  canvas.style.aspectRatio = `${xMax - xMin} / ${yMax - yMin}`;
  return { vertices, triangles, triangleCell, walls, box: [xMin, yMin, xMax, yMax] };
}

function fillSelect(meta) {
  const previous = show.value;
  show.replaceChildren();
  const group = (label, items, prefix) => {
    const g = document.createElement("optgroup");
    g.label = label;
    for (const name of items) {
      const option = document.createElement("option");
      option.value = `${prefix}:${name}`;
      option.textContent = name;
      g.append(option);
    }
    show.append(g);
  };
  group("Fields (mM)", meta.fields.map((f) => f.id), "field");
  group("Cell values", meta.cell_values, "cell");
  show.value = [...show.options].some((o) => o.value === previous) ? previous : "field:S";
  show.disabled = false;
}

function showTotals(current, summary) {
  const items = [];
  // In the order of the study's fields.
  const ids = current.fieldIds;
  for (const id of ids) {
    items.push([`Amount of ${id}`, `${format(summary.fields[id].total_mass.at(-1))} mM·µm²`]);
  }
  for (const id of ids) {
    const exchanged = summary.cells?.exchanged?.[id];
    if (exchanged) items.push([`${id} from the cells`, `${format(exchanged.at(-1))} mM·µm²`]);
  }
  if (summary.cells) items.push(["Cells", String(summary.cells.count)]);
  const inflow = summary.perfusion?.tag_flows?.[TAGS.corner];
  if (inflow !== undefined) items.push(["Portal inflow", `${format(inflow)} µm²/s`]);
  $("totals").replaceChildren(
    ...items.map(([term, value]) => {
      const div = document.createElement("div");
      const dt = document.createElement("dt");
      const dd = document.createElement("dd");
      dt.textContent = term;
      dd.textContent = value;
      div.append(dt, dd);
      return div;
    }),
  );
}

let drawPending = false;
function scheduleDraw() {
  if (drawPending) return;
  drawPending = true;
  requestAnimationFrame(() => {
    drawPending = false;
    draw();
  });
}

/** Draws the selected field (P1, interpolated) or cell value (per element) of the last frame. */
function draw() {
  const current = run;
  if (!current?.mesh || !current.frame) return;
  const [kind, name] = show.value.split(/:(.*)/s);
  const values = kind === "field" ? current.frame.fields[name] : current.frame.cells[name];
  if (!values) return;
  // The colour scale spans the values of this output time: the pattern over the lobule, not the
  // change in time, which the totals show.
  const range = { low: Infinity, high: -Infinity };
  for (const v of values) {
    if (v < range.low) range.low = v;
    if (v > range.high) range.high = v;
  }
  const low = range.low;
  const high = range.high > range.low ? range.high : range.low + 1;
  $("low").textContent = format(range.low);
  $("high").textContent = format(range.high);
  $("units").textContent = kind === "field" ? `${name} in mM` : `${name} per element`;
  canvas.setAttribute(
    "aria-label",
    `${name} on the lobule at t = ${current.frame.time} s, from ${format(range.low)} to ${format(range.high)}`,
  );

  const { vertices, triangles, triangleCell, walls, box } = current.mesh;
  const dpr = window.devicePixelRatio || 1;
  const width = Math.max(1, Math.round(canvas.clientWidth * dpr));
  const height = Math.max(1, Math.round(canvas.clientHeight * dpr));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  const [xMin, yMin, xMax, yMax] = box;
  const scale = Math.min(width / (xMax - xMin), height / (yMax - yMin));
  const ox = (width - scale * (xMax - xMin)) / 2;
  const oy = (height - scale * (yMax - yMin)) / 2;
  const n = vertices.length / 2;
  const px = new Float64Array(n);
  const py = new Float64Array(n);
  for (let k = 0; k < n; k++) {
    px[k] = ox + (vertices[2 * k] - xMin) * scale;
    py[k] = oy + (yMax - vertices[2 * k + 1]) * scale;
  }
  const image = new ImageData(width, height);
  const pixels = new Uint32Array(image.data.buffer);
  const level = (v) => {
    const k = Math.round(((v - low) / (high - low)) * 255);
    return LUT[k < 0 ? 0 : k > 255 ? 255 : k];
  };
  const eps = -1e-9;
  for (let t = 0; t < triangles.length / 3; t++) {
    const a = triangles[3 * t];
    const b = triangles[3 * t + 1];
    const c = triangles[3 * t + 2];
    const [ax, ay, bx, by, cx, cy] = [px[a], py[a], px[b], py[b], px[c], py[c]];
    const area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
    if (area === 0) continue;
    let flat = 0;
    let va = 0;
    let vb = 0;
    let vc = 0;
    if (kind === "field") {
      va = values[a];
      vb = values[b];
      vc = values[c];
    } else {
      const cell = triangleCell[t];
      if (cell < 0) continue;
      flat = level(values[cell]);
    }
    const x0 = Math.max(0, Math.floor(Math.min(ax, bx, cx)));
    const x1 = Math.min(width - 1, Math.ceil(Math.max(ax, bx, cx)));
    const y0 = Math.max(0, Math.floor(Math.min(ay, by, cy)));
    const y1 = Math.min(height - 1, Math.ceil(Math.max(ay, by, cy)));
    for (let y = y0; y <= y1; y++) {
      const sy = y + 0.5;
      for (let x = x0; x <= x1; x++) {
        const sx = x + 0.5;
        const wa = ((bx - sx) * (cy - sy) - (by - sy) * (cx - sx)) / area;
        const wb = ((cx - sx) * (ay - sy) - (cy - sy) * (ax - sx)) / area;
        const wc = 1 - wa - wb;
        if (wa < eps || wb < eps || wc < eps) continue;
        pixels[y * width + x] = kind === "field" ? level(wa * va + wb * vb + wc * vc) : flat;
      }
    }
  }
  const context = canvas.getContext("2d");
  context.putImageData(image, 0, 0);
  // The walls: the veins and the outer boundary.
  context.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue("--wall");
  context.lineWidth = Math.max(1, dpr);
  context.beginPath();
  for (const edges of walls) {
    for (let k = 0; k < edges.length; k += 2) {
      context.moveTo(px[edges[k]], py[edges[k]]);
      context.lineTo(px[edges[k + 1]], py[edges[k + 1]]);
    }
  }
  context.stroke();
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  start();
});
stopButton.addEventListener("click", () => stop());
show.addEventListener("change", scheduleDraw);
new ResizeObserver(scheduleDraw).observe(canvas);
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", scheduleDraw);
