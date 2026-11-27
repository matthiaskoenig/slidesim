// The lobule demo in headless Chromium (Playwright): the bundle of scripts/wasm/bundle.sh served
// statically, a run of 10 min through the page, and its result against the same run in Node.

import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { createReadStream, rmSync, statSync } from "node:fs";
import { createServer } from "node:http";
import { extname, join, normalize, sep } from "node:path";
import { after, before, test } from "node:test";

import { chromium } from "playwright";

import { createSimulation } from "../slidesim.mjs";
import { ROOT, bits, modules } from "./artifacts.mjs";

const TYPES = {
  ".html": "text/html; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".wasm": "application/wasm",
  ".xml": "application/xml",
};

let server;
let browser;
let origin;

/** Serves the files of `dir` (GET only, no listings). */
function serve(dir) {
  return createServer((request, response) => {
    const path = decodeURIComponent(new URL(request.url, "http://localhost").pathname);
    let file = normalize(join(dir, path));
    if (!file.startsWith(dir + sep) && file !== dir) {
      response.writeHead(403).end();
      return;
    }
    try {
      if (statSync(file).isDirectory()) file = join(file, "index.html");
      statSync(file);
    } catch {
      response.writeHead(404).end();
      return;
    }
    response.writeHead(200, { "content-type": TYPES[extname(file)] ?? "application/octet-stream" });
    createReadStream(file).pipe(response);
  });
}

before(async () => {
  const dir = join(ROOT, "target/tmp/wasm-js/browser/lobule");
  rmSync(dir, { recursive: true, force: true });
  execFileSync(join(ROOT, "scripts/wasm/bundle.sh"), [dir], { stdio: "pipe" });
  server = serve(dir);
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  origin = `http://127.0.0.1:${server.address().port}`;
  browser = await chromium.launch({ headless: true });
});

after(async () => {
  await browser?.close();
  await new Promise((resolve) => (server ? server.close(resolve) : resolve()));
});

/** The number of distinct colours of the opaque pixels of the canvas. */
const canvasColours = () =>
  new Promise((resolve) => {
    const canvas = document.getElementById("field");
    const data = canvas.getContext("2d").getImageData(0, 0, canvas.width, canvas.height).data;
    const colours = new Set();
    for (let k = 0; k < data.length; k += 4) {
      if (data[k + 3] === 255) colours.add((data[k] << 16) | (data[k + 1] << 8) | data[k + 2]);
    }
    resolve(colours.size);
  });

test("lobule_demo_runs_in_chromium", { timeout: 240_000 }, async () => {
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
  const errors = [];
  page.on("console", (message) => {
    if (message.type() === "error") errors.push(message.text());
  });
  page.on("pageerror", (error) => errors.push(error.message));
  page.on("response", (response) => {
    if (response.status() >= 400) errors.push(`${response.status()} ${response.url()}`);
  });
  await page.goto(`${origin}/index.html`);
  await page.fill("#minutes", "10");
  await page.click("#start");
  await page.waitForFunction(
    () => window.lobuleDemo.done || !document.getElementById("error").hidden,
    null,
    { timeout: 200_000, polling: 250 },
  );
  assert.equal(await page.isHidden("#error"), true, await page.textContent("#error"));
  assert.equal(await page.textContent("#time"), "t = 600 s");
  assert.match(await page.textContent("#status"), /^Done: 10 min simulated in [\d.]+ s\.$/);
  // Drawing settles on the next animation frame.
  await page.evaluate(() => new Promise((resolve) => requestAnimationFrame(() => resolve())));
  const colours = await page.evaluate(canvasColours);
  assert.ok(colours > 50, `the canvas has ${colours} colours`);
  await page.selectOption("#show", "cell:protein");
  await page.evaluate(() => new Promise((resolve) => requestAnimationFrame(() => resolve())));
  const cellColours = await page.evaluate(canvasColours);
  assert.ok(cellColours > 20, `the canvas of protein has ${cellColours} colours`);

  const { inputs, frame } = await page.evaluate(() => window.lobuleDemo);
  assert.equal(frame.time, 600);
  // The same run in Node: the L2 norm of S (slidesim's) and the sum of the squares of its nodal
  // values (the page's arithmetic) bit for bit.
  const sim = await createSimulation({ ...(await modules()), ...inputs });
  while (sim.step() !== null);
  assert.equal(sim.time, 600);
  const l2 = sim.summary().fields.S.l2_norm.at(-1);
  let squares = 0;
  for (const v of sim.field("S")) squares += v * v;
  sim.free();
  assert.equal(bits([frame.l2]), bits([l2]), `L2 norm of S: page ${frame.l2}, Node ${l2}`);
  assert.equal(bits([frame.squares]), bits([squares]));
  console.log(`lobule demo: page and Node L2 norm of S at 600 s ${l2} (bits ${bits([l2])})`);
  assert.deepEqual(errors, [], "no console errors");
  await page.close();
});

test("stop_while_loading_cancels_the_run", { timeout: 120_000 }, async () => {
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  page.on("console", (message) => {
    if (message.type() === "error") errors.push(message.text());
  });
  // slidesim.wasm arrives only when the test lets it: the run is still loading.
  let release;
  const released = new Promise((resolve) => (release = resolve));
  await page.route("**/slidesim.wasm", async (route) => {
    await released;
    await route.continue();
  });
  const workers = [];
  page.on("worker", (worker) => workers.push(worker.url()));
  await page.goto(`${origin}/index.html`);
  await page.click("#start");
  await page.waitForFunction(() => document.getElementById("status").textContent.startsWith("Loading"));
  assert.equal(await page.isEnabled("#stop"), true);
  await page.click("#stop");
  assert.equal(await page.textContent("#status"), "Stopped while loading slidesim.");
  assert.equal(await page.isEnabled("#start"), true);
  assert.equal(await page.isEnabled("#stop"), false);
  assert.equal(await page.isHidden("#progress"), true);
  assert.equal(await page.isEnabled("#minutes"), true);
  // The load finishes: the stopped run starts no worker.
  release();
  await page.waitForResponse("**/slidesim.wasm");
  await page.waitForTimeout(500);
  assert.deepEqual(workers, []);
  assert.equal(await page.textContent("#status"), "Stopped while loading slidesim.");
  assert.equal(await page.isHidden("#time"), true, "no time before a frame");
  // A new run starts and draws.
  await page.fill("#minutes", "1");
  await page.click("#start");
  await page.waitForFunction(() => window.lobuleDemo.done, null, { timeout: 100_000 });
  assert.equal(await page.textContent("#time"), "t = 60 s");
  assert.equal(workers.length, 1);
  assert.deepEqual(errors, []);
  await page.close();
});
