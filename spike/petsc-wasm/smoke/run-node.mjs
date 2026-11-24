// Run a wasm32-wasip1 command module in Node: node run-node.mjs <module.wasm> [args...]
import { readFile } from "node:fs/promises";
import { argv, exit } from "node:process";
import { WASI } from "node:wasi";

const [path, ...args] = argv.slice(2);
const wasi = new WASI({ version: "preview1", args: [path, ...args], env: {}, returnOnExit: true });
const module = await WebAssembly.compile(await readFile(path));
const instance = await WebAssembly.instantiate(module, wasi.getImportObject());
exit(wasi.start(instance));
