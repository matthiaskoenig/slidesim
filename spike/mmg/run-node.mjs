// Run a wasm32-wasip1 command module in Node with <dir> preopened as /out:
// node run-node.mjs <dir> <module.wasm> [args...]
// Prints the compile time of the module to stderr; the exit code is the module's.
import { readFile } from "node:fs/promises";
import { argv, exit, stderr } from "node:process";
import { WASI } from "node:wasi";

const [dir, path, ...args] = argv.slice(2);
const wasi = new WASI({
  version: "preview1",
  args: [path, ...args],
  env: {},
  preopens: { "/out": dir },
  returnOnExit: true,
});
const bytes = await readFile(path);
const t0 = performance.now();
const module = await WebAssembly.compile(bytes);
stderr.write(`node: WebAssembly.compile ${(performance.now() - t0).toFixed(1)} ms\n`);
const instance = await WebAssembly.instantiate(module, wasi.getImportObject());
exit(wasi.start(instance));
