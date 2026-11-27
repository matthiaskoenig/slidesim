// The live browser API of slidesim (declared in `slidesim.d.ts`, described in docs/contracts.md,
// "Live API"): `slidesim.wasm` (crates/slidesim-wasm, a WASI preview 1 reactor) in a JavaScript
// host, with the batched kernel of the cell model compiled in the page by sbmlsim-core.
//
// Plain ECMAScript without dependencies and without APIs of Node, so it runs in browsers, in
// workers and in Node alike. sbmlsim-core's `compile.mjs` and `kernel.mjs` lie beside it in
// `sbmlsim-core/` (copied there by scripts/wasm/build.sh).
//
// The modules and how they are linked, as in the wasmtime host of `wasm/host`:
//
// - `slidesim.wasm` imports its memory (`env.memory`), which this file creates for each
//   simulation (at the minimum the module records in its custom section `slidesim.memory`, with
//   `memory.maximumPages` as its maximum), a few functions of WASI preview 1, which this file
//   implements over nothing (no files, no environment, no arguments; what slidesim writes to
//   stdout and stderr is kept for the messages of a trap), and `slidesim_kernel.init`,
//   `slidesim_kernel.advance` and `slidesim_kernel.assign`, which are bound late, to the kernel
//   once it is instantiated.
// - The kernel (`compileKernel` of the SBML text with `compiler.wasm`) is instantiated after
//   slidesim, with the same memory and `mathlib.wasm`.
//
// Each simulation has its own instance of `slidesim.wasm`: after a trap (a panic, an allocation
// past the memory cap) the instance is in the middle of a call and is discarded with its
// simulation, which is marked unusable; the next `createSimulation` instantiates afresh. An
// error of slidesim (a refused input, a failed step) is not a trap: it is thrown as an `Error`
// with slidesim's message and the simulation stays usable.

import { createCompiler } from "./sbmlsim-core/compile.mjs";
import { instantiateKernel } from "./sbmlsim-core/kernel.mjs";

/** The bytes of a wasm page. */
const PAGE = 65536;

/** The most wasm pages of a wasm32 memory (4 GiB). */
const MAX_PAGES = 65536;

/** The most bytes of slidesim's stdout and stderr each which are kept. */
const STDIO_CAPACITY = 1 << 20;

/**
 * A `scene/1` document of parametric lobules (`mesh.format` `lobules`, docs/contracts.md).
 * Parameters which are not given are left out, so that slidesim's defaults apply; the tag and
 * map names are the caller's.
 *
 * @param {import("./slidesim.d.ts").LobuleParams} params
 * @param {import("./slidesim.d.ts").LobuleTags} tags
 * @returns {object}
 */
export function lobuleScene(params, tags) {
  if (params === null || typeof params !== "object") {
    throw new TypeError("lobuleScene needs the parameters as an object");
  }
  if (tags === null || typeof tags !== "object") {
    throw new TypeError("lobuleScene needs the tags as an object");
  }
  for (const key of ["centre", "corner", "outer", "area", "zonation"]) {
    if (typeof tags[key] !== "string") {
      throw new TypeError(`lobuleScene needs the name tags.${key}`);
    }
  }
  const known = new Set([
    "lobuleRadiusUm",
    "rings",
    "jitter",
    "seed",
    "centreRadiusUm",
    "cornerRadiusUm",
    "wallSizeUm",
    "interiorSizeUm",
  ]);
  for (const key of Object.keys(params)) {
    if (!known.has(key)) throw new TypeError(`lobuleScene has no parameter ${key}`);
  }
  /** `object` without the keys whose value is undefined. */
  const defined = (object) =>
    Object.fromEntries(Object.entries(object).filter(([, value]) => value !== undefined));
  const size = defined({ wall_um: params.wallSizeUm, interior_um: params.interiorSizeUm });
  const mesh = defined({
    format: "lobules",
    lobule_radius_um: params.lobuleRadiusUm,
    rings: params.rings,
    jitter: params.jitter,
    seed: params.seed,
    centre: defined({ radius_um: params.centreRadiusUm, tag: tags.centre }),
    corner: defined({ radius_um: params.cornerRadiusUm, tag: tags.corner }),
    outer_tag: tags.outer,
    area_tag: tags.area,
    size: Object.keys(size).length > 0 ? size : undefined,
    zonation: { map: tags.zonation },
  });
  return { version: "scene/1", dimension: 2, units: { length: "um" }, mesh };
}

/** A module, compiled when it is given as bytes. */
async function toModule(source, what) {
  if (source instanceof WebAssembly.Module) return source;
  try {
    return await WebAssembly.compile(source);
  } catch (error) {
    throw new Error(`${what} does not compile: ${error.message}`);
  }
}

/** The custom section in which scripts/wasm/build.sh records the memory slidesim imports. */
const MEMORY_SECTION = "slidesim.memory";

/**
 * The least memory `module` (slidesim.wasm) imports, in wasm pages: from its custom section
 * `slidesim.memory`, since the JavaScript API cannot read the limits of an import.
 */
function minimumPages(module) {
  const imports = WebAssembly.Module.imports(module);
  if (!imports.some((i) => i.module === "env" && i.name === "memory" && i.kind === "memory")) {
    throw new Error("slidesim.wasm does not import env.memory: build it with scripts/wasm/build.sh");
  }
  const [section] = WebAssembly.Module.customSections(module, MEMORY_SECTION);
  const pages = section && JSON.parse(new TextDecoder().decode(section)).minimum_pages;
  if (!Number.isInteger(pages) || pages < 1) {
    throw new Error(
      `slidesim.wasm has no custom section ${MEMORY_SECTION} with its least memory: build it ` +
        "with scripts/wasm/build.sh",
    );
  }
  return pages;
}

/** A `proc_exit` of slidesim: it ends the instance like a trap. */
class WasiExit extends Error {
  constructor(code) {
    super(`slidesim.wasm exited with code ${code}`);
    this.name = "WasiExit";
  }
}

/** WASI preview 1 errno values. */
const ERRNO = { success: 0, badf: 8, inval: 28, nosys: 52 };

/**
 * The functions of WASI preview 1 which slidesim imports, over nothing: no files (no preopened
 * directories, every descriptor but 0, 1 and 2 is bad), no environment, the clocks of the host,
 * random bytes of `crypto`; stdout and stderr are kept as text (up to `STDIO_CAPACITY` bytes).
 */
class Wasi {
  constructor() {
    this.memory = null;
    this.out = { 1: [], 2: [] };
    this.length = { 1: 0, 2: 0 };
    const view = () => new DataView(this.memory.buffer);
    const bytes = () => new Uint8Array(this.memory.buffer);
    const nosys = () => ERRNO.nosys;
    const badf = () => ERRNO.badf;
    this.imports = {
      args_get: () => ERRNO.success,
      args_sizes_get: (count, size) => {
        view().setUint32(count, 0, true);
        view().setUint32(size, 0, true);
        return ERRNO.success;
      },
      environ_get: () => ERRNO.success,
      environ_sizes_get: (count, size) => {
        view().setUint32(count, 0, true);
        view().setUint32(size, 0, true);
        return ERRNO.success;
      },
      clock_res_get: (id, resolution) => {
        view().setBigUint64(resolution, 1000n, true);
        return ERRNO.success;
      },
      clock_time_get: (id, precision, time) => {
        let ns;
        if (id === 0) ns = BigInt(Date.now()) * 1000000n;
        else if (id === 1 || id === 2 || id === 3) {
          ns = BigInt(Math.round(performance.now() * 1e6));
        } else return ERRNO.inval;
        view().setBigUint64(time, ns, true);
        return ERRNO.success;
      },
      random_get: (buf, len) => {
        const target = bytes().subarray(buf, buf + len);
        for (let k = 0; k < len; k += 65536) {
          crypto.getRandomValues(target.subarray(k, Math.min(len, k + 65536)));
        }
        return ERRNO.success;
      },
      fd_write: (fd, iovs, count, written) => {
        if (fd !== 1 && fd !== 2) return ERRNO.badf;
        const v = view();
        let total = 0;
        for (let k = 0; k < count; k++) {
          const ptr = v.getUint32(iovs + 8 * k, true);
          const len = v.getUint32(iovs + 8 * k + 4, true);
          const keep = Math.min(len, STDIO_CAPACITY - this.length[fd]);
          if (keep > 0) {
            this.out[fd].push(bytes().slice(ptr, ptr + keep));
            this.length[fd] += keep;
          }
          total += len;
        }
        v.setUint32(written, total, true);
        return ERRNO.success;
      },
      fd_fdstat_get: (fd, stat) => {
        if (fd > 2) return ERRNO.badf;
        const v = view();
        // A character device without rights to seek: a terminal, as the descriptors of a
        // process without files.
        v.setUint8(stat, 2);
        v.setUint16(stat + 2, 0, true);
        v.setBigUint64(stat + 8, fd === 0 ? 2n : 64n, true);
        v.setBigUint64(stat + 16, 0n, true);
        return ERRNO.success;
      },
      fd_fdstat_set_flags: badf,
      fd_filestat_get: badf,
      fd_prestat_get: badf,
      fd_prestat_dir_name: badf,
      fd_read: badf,
      fd_readdir: badf,
      fd_seek: badf,
      fd_close: badf,
      path_create_directory: badf,
      path_filestat_get: badf,
      path_open: badf,
      path_remove_directory: badf,
      path_unlink_file: badf,
      poll_oneoff: nosys,
      sched_yield: () => ERRNO.success,
      proc_exit: (code) => {
        throw new WasiExit(code);
      },
    };
  }

  /** What slidesim wrote to the descriptor `fd` (1 or 2). */
  text(fd) {
    const all = new Uint8Array(this.length[fd]);
    let offset = 0;
    for (const chunk of this.out[fd]) {
      all.set(chunk, offset);
      offset += chunk.length;
    }
    return new TextDecoder().decode(all);
  }
}

/**
 * What a failed allocation writes: Rust's allocation error (then a trap), libCEED's (then a
 * trap), PETSc's (an error of slidesim).
 */
const ALLOCATION_FAILED = /memory allocation of \d+ bytes failed|failed to allocate|out of memory/i;

/** `bytes` in MiB, rounded. */
const mebibytes = (bytes) => Math.round(bytes / (1 << 20));

/**
 * When `text` (a message or stderr of slidesim) tells of a failed allocation: what the memory
 * of `instance` was, against its cap if it has one; else `null`.
 */
function memoryNote(instance, text) {
  if (!ALLOCATION_FAILED.test(text)) return null;
  const used = mebibytes(instance.memory.buffer.byteLength);
  const pages = instance.maximumPages;
  return pages !== undefined
    ? `slidesim.wasm needs more memory than its maximum of ${pages} wasm pages ` +
        `(${mebibytes(pages * PAGE)} MiB, memory.maximumPages): an allocation failed with ` +
        `${used} MiB in use`
    : `slidesim.wasm ran out of memory: an allocation failed with ${used} MiB in use`;
}

/** The `Error` of a trap of `instance` in the call `what` (`cause` what the engine threw). */
function trapError(instance, what, cause) {
  const stderr = instance.wasi.text(2).trim();
  const parts = [];
  const note = memoryNote(instance, stderr);
  if (note) parts.push(note);
  const message = cause instanceof Error ? cause.message : String(cause);
  parts.push(`slidesim.wasm trapped in ${what} (${message}) and was discarded`);
  if (stderr) parts.push(`it wrote to stderr: ${stderr}`);
  return new Error(parts.join("; "), { cause });
}

/** An instance of `slidesim.wasm` with its kernel, and the calls of its C ABI. */
class Instance {
  constructor(exports, memory, wasi, maximumPages) {
    this.exports = exports;
    this.memory = memory;
    this.wasi = wasi;
    this.maximumPages = maximumPages;
    /** The `Error` of the trap which ended this instance, `null` while it lives. */
    this.trap = null;
  }

  /** Calls the export `name`; a trap ends the instance and is thrown as an `Error`. */
  call(name, ...args) {
    if (this.trap) throw this.unusable();
    try {
      return this.exports[name](...args);
    } catch (cause) {
      this.trap = trapError(this, name, cause);
      throw this.trap;
    }
  }

  /** The `Error` of a call after the trap. */
  unusable() {
    return new Error(
      `the simulation is unusable: slidesim.wasm trapped before and was discarded (${this.trap.message})`,
    );
  }

  /** `bytes` copied into memory slidesim allocated: its address (free with `release`). */
  write(bytes) {
    const ptr = bytes.length === 0 ? 0 : this.call("slidesim_alloc", bytes.length) >>> 0;
    if (ptr === 0 && bytes.length > 0) {
      throw new Error(`slidesim_alloc of ${bytes.length} bytes failed`);
    }
    new Uint8Array(this.memory.buffer).set(bytes, ptr);
    return ptr;
  }

  release(ptr, length) {
    if (length > 0) this.call("slidesim_free", ptr, length);
  }

  /**
   * slidesim's message of the last error, after the note on the memory when it tells of a
   * failed allocation.
   */
  error() {
    const ptr = this.call("slidesim_error_ptr") >>> 0;
    const len = this.call("slidesim_error_len") >>> 0;
    const message = new TextDecoder().decode(new Uint8Array(this.memory.buffer, ptr, len));
    const note = memoryNote(this, message);
    return note ? `${note}: ${message}` : message;
  }

  /** A copy of the bytes of the array `name` of the simulation `handle`. */
  array(handle, name) {
    const bytes = new TextEncoder().encode(name);
    const namePtr = this.write(bytes);
    let ptr;
    try {
      ptr = this.call("slidesim_array", handle, namePtr, bytes.length) >>> 0;
    } finally {
      if (!this.trap) this.release(namePtr, bytes.length);
    }
    if (ptr === 0) throw new Error(this.error());
    const len = this.call("slidesim_array_len") >>> 0;
    return this.memory.buffer.slice(ptr, ptr + len);
  }

  json(handle, name) {
    return JSON.parse(new TextDecoder().decode(this.array(handle, name)));
  }
}

/** A running simulation (see `Simulation` in `slidesim.d.ts`). */
class Simulation {
  #instance;
  #handle;
  #freed = false;
  #done = false;
  #time;
  #meta;
  #mesh;
  #times;
  #cellIds;

  constructor(instance, handle) {
    this.#instance = instance;
    this.#handle = handle;
    this.#meta = instance.json(handle, "meta");
    this.#times = new Float64Array(instance.array(handle, "time"));
    const tags = {};
    for (const { name } of this.#meta.tags) {
      tags[name] = new Uint32Array(instance.array(handle, `mesh/tags/${name}`));
    }
    this.#mesh = Object.freeze({
      vertices: new Float64Array(instance.array(handle, "mesh/vertices")),
      triangles: new Uint32Array(instance.array(handle, "mesh/triangles")),
      element: new BigUint64Array(instance.array(handle, "mesh/element")),
      tags: Object.freeze(tags),
    });
    this.#cellIds = new BigUint64Array(instance.array(handle, "cells/element"));
    this.#advance();
  }

  /** The instance, unless the simulation was freed or its instance trapped. */
  #live() {
    if (this.#freed) throw new Error("the simulation was freed: it is unusable");
    if (this.#instance.trap) throw this.#instance.unusable();
    return this.#instance;
  }

  /** `slidesim_next_output`: the index reached, `null` at the end. */
  #advance() {
    const instance = this.#instance;
    const index = instance.call("slidesim_next_output", this.#handle);
    if (index === -2) throw new Error(instance.error());
    if (index === -1) {
      this.#done = true;
      return null;
    }
    if (index < 0) throw new Error(`slidesim_next_output gave ${index}`);
    this.#time = instance.call("slidesim_time", this.#handle);
    this.#meta = instance.json(this.#handle, "meta");
    return index;
  }

  /** The mesh: the same arrays on every access (see `SimulationMesh` in the declarations). */
  get mesh() {
    this.#live();
    return this.#mesh;
  }

  get times() {
    this.#live();
    return this.#times.slice();
  }

  get time() {
    this.#live();
    return this.#time;
  }

  get meta() {
    this.#live();
    return structuredClone(this.#meta);
  }

  get cellIds() {
    this.#live();
    return this.#cellIds.slice();
  }

  step() {
    this.#live();
    if (this.#done) return null;
    return this.#advance();
  }

  field(id) {
    return new Float64Array(this.#live().array(this.#handle, `fields/${id}`));
  }

  cellValues(name) {
    return new Float64Array(this.#live().array(this.#handle, `cells/${name}`));
  }

  summary() {
    return this.#live().json(this.#handle, "summary");
  }

  meshQuality() {
    return this.#live().json(this.#handle, "quality");
  }

  log() {
    return new TextDecoder().decode(this.#live().array(this.#handle, "log"));
  }

  free() {
    if (this.#freed) return;
    this.#freed = true;
    if (!this.#instance.trap) this.#instance.call("slidesim_destroy", this.#handle);
    this.#instance = null;
  }
}

/**
 * Creates a simulation (see `createSimulation` in `slidesim.d.ts`).
 *
 * @param {import("./slidesim.d.ts").CreateOptions} options
 * @returns {Promise<import("./slidesim.d.ts").Simulation>}
 */
export async function createSimulation(options) {
  if (options === null || typeof options !== "object") {
    throw new TypeError("createSimulation needs its options");
  }
  const { slidesim, compiler, mathlib, sbml, scene, study, mesh, memory } = options;
  for (const [name, value] of Object.entries({ slidesim, compiler, mathlib })) {
    if (!value) throw new TypeError(`createSimulation needs ${name}`);
  }
  if (scene === null || typeof scene !== "object") {
    throw new TypeError("createSimulation needs the scene as an object");
  }
  if (study === null || typeof study !== "object") {
    throw new TypeError("createSimulation needs the study as an object");
  }
  const cells = study.cells !== undefined && study.cells !== null;
  if (cells && typeof sbml !== "string") {
    throw new TypeError("createSimulation needs the SBML text of the cell model");
  }
  let maximumPages;
  if (memory !== undefined) {
    maximumPages = memory?.maximumPages;
    if (!Number.isInteger(maximumPages) || maximumPages < 1 || maximumPages > MAX_PAGES) {
      throw new RangeError(
        `memory.maximumPages must be an integer within 1 and ${MAX_PAGES}, got ${maximumPages}`,
      );
    }
  }

  const slidesimModule = await toModule(slidesim, "slidesim.wasm");
  // The memory slidesim imports and the kernel works in: created here, capped at maximumPages.
  const initial = minimumPages(slidesimModule);
  if (maximumPages !== undefined && initial > maximumPages) {
    throw new Error(
      `slidesim.wasm needs at least ${initial} wasm pages of memory, more than ` +
        `memory.maximumPages ${maximumPages}`,
    );
  }
  const wasmMemory = new WebAssembly.Memory(
    maximumPages === undefined ? { initial } : { initial, maximum: maximumPages },
  );
  const mathlibModule = await toModule(mathlib, "mathlib.wasm");
  let kernel = null;
  if (cells) {
    const sbmlCompiler = await createCompiler({ compiler: await toModule(compiler, "compiler.wasm") });
    kernel = sbmlCompiler.compileKernel(sbml);
  }

  const wasi = new Wasi();
  let bound = null;
  const kernelCall = (name) => {
    if (!bound) throw new Error(`slidesim_kernel.${name} called before the kernel is instantiated`);
    return bound[name];
  };
  const imports = {
    env: { memory: wasmMemory },
    wasi_snapshot_preview1: wasi.imports,
    slidesim_kernel: {
      init: (n, parameters, state) => kernelCall("init")(n, parameters, state),
      advance: (n, state, parameters, control, counts, context, scratch) =>
        kernelCall("advance")(n, state, parameters, control, counts, context, scratch),
      assign: (n, state, parameters, control) =>
        kernelCall("assign")(n, state, parameters, control),
    },
  };
  let exports;
  try {
    ({ exports } = await WebAssembly.instantiate(slidesimModule, imports));
  } catch (error) {
    throw new Error(`cannot instantiate slidesim.wasm: ${error.message}`, { cause: error });
  }
  wasi.memory = wasmMemory;
  const instance = new Instance(exports, wasmMemory, wasi, maximumPages);
  if (typeof exports._initialize === "function") instance.call("_initialize");
  if (kernel) {
    bound = await instantiateKernel({
      wasm: kernel.wasm,
      mathlib: mathlibModule,
      memory: wasmMemory,
    });
  }

  const request = { scene, study };
  if (kernel) request.kernel = kernel.description;
  if (mesh !== undefined) request.mesh = mesh;
  const bytes = new TextEncoder().encode(JSON.stringify(request));
  const ptr = instance.write(bytes);
  const handle = instance.call("slidesim_create", ptr, bytes.length);
  instance.release(ptr, bytes.length);
  if (handle < 0) throw new Error(instance.error());
  return new Simulation(instance, handle);
}
