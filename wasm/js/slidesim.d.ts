// Declarations of the live browser API of slidesim (implemented by `slidesim.mjs`).
// The arrays are those of the `result/1` store at the current time index, see docs/contracts.md.
// Units: length um, time s, concentration mM.
//
// Errors and traps: an error of slidesim (a refused input, a failed step, an unknown array) is
// thrown, or rejected by `createSimulation`, as an `Error` with slidesim's message; the
// simulation stays usable. A trap of `slidesim.wasm` (a panic, an allocation past the memory
// cap, a failure of a host function) leaves the module in the middle of a call: its instance is
// discarded with the simulation in it, the call which trapped throws (or rejects) an `Error`
// naming the trap (and the memory, when an allocation failed), and every later call of that
// `Simulation`, its getters included, throws an `Error` saying that it is unusable (as after
// `free()`). Each simulation has its own
// instance, so other simulations and the next `createSimulation` are not affected.

/** Parameters of a lobule scene; the defaults are those of the Rust `LobuleSpec`. */
export interface LobuleParams {
  /** Circumradius of the hexagonal lobule in um, > 0 (default 300). */
  lobuleRadiusUm?: number;
  /** Rings of neighbouring lobules, an integer within 0 and 4 (default 0). */
  rings?: number;
  /** Vein displacement as a fraction of the radius, 0 <= jitter <= 0.25 (default 0). */
  jitter?: number;
  /** Seed of the displacements (default 0); above 2^53 it loses precision in JS. */
  seed?: number;
  /** Radius of the central veins in um (default 30). */
  centreRadiusUm?: number;
  /** Radius of the corner veins in um (default 25). */
  cornerRadiusUm?: number;
  /** Element size near the walls in um (default 5), at most `interiorSizeUm`. */
  wallSizeUm?: number;
  /** Element size in the interior in um (default 20). */
  interiorSizeUm?: number;
}

/** Tag and map names of a lobule scene; supplied by the caller, slidesim fixes none. */
export interface LobuleTags {
  centre: string;
  corner: string;
  outer: string;
  area: string;
  zonation: string;
}

/** A `scene/1` document for a lobule mesh. */
export function lobuleScene(params: LobuleParams, tags: LobuleTags): object;

export interface ResultField {
  id: string;
  /** `mM`. */
  units: string;
}

export interface ResultTag {
  name: string;
  /** 2 for an area (triangle indices), 1 for an edge (vertex index pairs). */
  dim: number;
}

export interface RasterMeta {
  um_per_px: number;
  origin_um: [number, number];
  levels: number;
  /** `[rows, cols]` per level, level 0 first. */
  shapes: Array<[number, number]>;
}

/** The root attributes of `result.zarr`: the JSON of the Rust `ResultMeta`. */
export interface ResultMeta {
  version: "result/1";
  study: string;
  scene: string;
  refinements: number;
  times: number[];
  written: number;
  complete: boolean;
  fields: ResultField[];
  cell_values: string[];
  tags: ResultTag[];
  raster?: RasterMeta;
}

/**
 * The mesh of a simulation: unlike the other arrays these are not copies but the simulation's
 * own, the same on every access of `Simulation.mesh` and never changed by it. Do not modify them,
 * and copy one before transferring its buffer to a worker (a transfer detaches it).
 */
export interface SimulationMesh {
  /** `[V, 2]` row-major, um. */
  vertices: Float64Array;
  /** `[N, 3]` row-major vertex indices, counterclockwise. */
  triangles: Uint32Array;
  /** `[N]` coarse element id of each triangle. */
  element: BigUint64Array;
  /** Area tags: triangle indices `[k]`; edge tags: vertex index pairs `[k, 2]` flattened. */
  tags: Record<string, Uint32Array>;
}

/**
 * A running simulation. After `createSimulation` resolves it is at output index 0, the initial
 * state: `time === times[0]`, `field()` and `cellValues()` return the values at index 0 and
 * `meta.written === 1`. Each `step()` advances to the next output index and returns it (the
 * first call returns 1), or returns `null` when there is none left, after which
 * `meta.complete` is true. `written` is the number of output times reached so far, as in the
 * `result/1` store where an index `t < written` is complete.
 */
export interface Simulation {
  readonly mesh: SimulationMesh;
  /** All output times in s. */
  readonly times: Float64Array;
  /** The current time in s. */
  readonly time: number;
  readonly meta: ResultMeta;
  /** Advance to the next output time and return its index, `null` when done. */
  step(): number | null;
  /** P1 nodal values `[V]` of a field at the current time, a copy the caller owns. */
  field(id: string): Float64Array;
  /** Coarse element ids `[C]` holding cells. */
  readonly cellIds: BigUint64Array;
  /** Per-element values `[C]` of a cell value at the current time, a copy the caller owns. */
  cellValues(name: string): Float64Array;
  summary(): object;
  /** The quality report of a generated mesh, `null` for a loaded mesh. */
  meshQuality(): object | null;
  /**
   * What the simulation reported (the array `log`), one line each: the quality line of a
   * generated mesh, which `slidesim run` prints; empty for a loaded mesh.
   */
  log(): string;
  /** Release the wasm memory; the simulation is unusable afterwards. */
  free(): void;
}

export interface CreateOptions {
  slidesim: WebAssembly.Module | BufferSource;
  compiler: WebAssembly.Module | BufferSource;
  mathlib: WebAssembly.Module | BufferSource;
  /** The SBML model text. */
  sbml: string;
  /** A `scene/1` document. */
  scene: object;
  /** A `study/1` document. */
  study: object;
  /** A prebuilt mesh, when the scene's source is not generated in the browser. */
  mesh?: object;
  /**
   * Caps the wasm memory of the simulation, in 64 KiB wasm pages (an integer within 1 and 65536,
   * 4 GiB): the maximum of the memory the host creates for it (slidesim.wasm imports its memory),
   * for a compiled module and bytes alike. Exceeding it is an `Error` naming the cap: an error of
   * slidesim when the allocation which failed was PETSc's (the simulation stays usable), a trap
   * when it was Rust's or libCEED's (the simulation is unusable afterwards). Mmg, the remesher of
   * the mesher, caps itself at 2 GiB in wasm (a constant of the Mmg patch of the wasi build), so
   * a cap below that is the one which holds; without a cap the browser's limit holds.
   */
  memory?: { maximumPages: number };
}

/**
 * Compiles the kernel of the cell model (with `compiler`; not for a study without `cells`),
 * instantiates `slidesim.wasm`, the kernel and `mathlib` and creates the simulation: the checks,
 * the mesh, the maps, the flow and the setup, at output index 0. Errors of slidesim and traps
 * reject with an `Error` (see the top of this file).
 */
export function createSimulation(options: CreateOptions): Promise<Simulation>;
