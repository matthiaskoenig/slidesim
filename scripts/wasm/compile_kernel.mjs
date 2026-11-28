// Compiles the batched kernel (kernel/1) of an SBML model with compiler.wasm of sbmlsim-core, as
// a page does: `compiler.compileKernel` of sbmlsim-core's js/compile.mjs, which gives the bytes
// of `sbmlsim-core compile --kernel`.
//
// Usage: node compile_kernel.mjs <compile.mjs> <compiler.wasm> <model.xml> <out dir>
// Writes <out dir>/kernel.wasm and <out dir>/kernel.json.

import { mkdir, readFile, writeFile } from "node:fs/promises";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";

const [compileModule, compilerWasm, model, out] = process.argv.slice(2);
if (!out) {
  console.error("usage: node compile_kernel.mjs <compile.mjs> <compiler.wasm> <model.xml> <out dir>");
  process.exit(2);
}
const { createCompiler } = await import(pathToFileURL(resolve(compileModule)).href);
const compiler = await createCompiler({ compiler: await readFile(compilerWasm) });
const { wasm, description } = compiler.compileKernel(await readFile(model, "utf8"));
await mkdir(out, { recursive: true });
await writeFile(join(out, "kernel.wasm"), wasm);
await writeFile(join(out, "kernel.json"), JSON.stringify(description, null, 2) + "\n");
