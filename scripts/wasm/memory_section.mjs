// Records the memory slidesim.wasm imports in a custom section of the module, so that a host
// which has only the compiled module (`WebAssembly.Module.customSections`) knows the least
// memory to create for it: the JavaScript API cannot read the limits of an import.
//
// slidesim.wasm imports its memory (`env.memory`, linked with `--import-memory`): the host
// creates it, with the maximum it allows, and gives the same memory to slidesim and the kernel.
// The section `slidesim.memory` holds the JSON `{"minimum_pages": <n>}`, `n` the minimum of the
// import in 64 KiB pages. An existing section of that name is replaced.
//
// Usage: node memory_section.mjs <slidesim.wasm>

import { readFile, writeFile } from "node:fs/promises";

const SECTION = "slidesim.memory";

/** The unsigned LEB128 at `offset` of `bytes`: `[value, offset after it]`. */
function readLeb(bytes, offset) {
  let value = 0;
  let shift = 0;
  for (;;) {
    const byte = bytes[offset++];
    if (byte === undefined) throw new Error("truncated LEB128");
    value += (byte & 0x7f) * 2 ** shift;
    if ((byte & 0x80) === 0) return [value, offset];
    shift += 7;
  }
}

function leb(value) {
  const out = [];
  do {
    let byte = value % 128;
    value = Math.floor(value / 128);
    if (value > 0) byte |= 0x80;
    out.push(byte);
  } while (value > 0);
  return out;
}

function readName(bytes, offset) {
  let length;
  [length, offset] = readLeb(bytes, offset);
  return [new TextDecoder().decode(bytes.subarray(offset, offset + length)), offset + length];
}

const [path] = process.argv.slice(2);
if (!path) {
  console.error("usage: node memory_section.mjs <slidesim.wasm>");
  process.exit(2);
}
const bytes = new Uint8Array(await readFile(path));
const kept = [bytes.subarray(0, 8)];
let minimum = null;
for (let offset = 8; offset < bytes.length; ) {
  const start = offset;
  const id = bytes[offset++];
  let size;
  [size, offset] = readLeb(bytes, offset);
  const end = offset + size;
  if (id === 0 && readName(bytes, offset)[0] === SECTION) {
    offset = end;
    continue;
  }
  if (id === 2) {
    let count;
    let p;
    [count, p] = readLeb(bytes, offset);
    for (let k = 0; k < count; k++) {
      let module;
      let name;
      [module, p] = readName(bytes, p);
      [name, p] = readName(bytes, p);
      const kind = bytes[p++];
      if (kind === 0) {
        [, p] = readLeb(bytes, p); // a function: its type index
      } else if (kind === 1) {
        p += 1; // a table: its element type, then limits
        const flags = bytes[p++];
        [, p] = readLeb(bytes, p);
        if (flags & 1) [, p] = readLeb(bytes, p);
      } else if (kind === 2) {
        const flags = bytes[p++];
        let min;
        [min, p] = readLeb(bytes, p);
        if (flags & 1) [, p] = readLeb(bytes, p);
        if (module === "env" && name === "memory") minimum = min;
      } else if (kind === 3) {
        p += 2; // a global: its value type and mutability
      } else {
        throw new Error(`${path}: an import of kind ${kind}`);
      }
    }
  }
  kept.push(bytes.subarray(start, end));
  offset = end;
}
if (minimum === null) {
  throw new Error(`${path} does not import env.memory: link it with --import-memory`);
}
const name = new TextEncoder().encode(SECTION);
const payload = new TextEncoder().encode(JSON.stringify({ minimum_pages: minimum }));
const content = [...leb(name.length), ...name, ...payload];
kept.push(Uint8Array.from([0, ...leb(content.length), ...content]));
await writeFile(path, Buffer.concat(kept));
console.log(`${path}: imports a memory of at least ${minimum} pages (section ${SECTION})`);
