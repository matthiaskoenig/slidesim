// Declarations of the live browser API of slidesim (implemented by `slidesim.mjs`).
// The arrays are those of the `result/1` store at the current time index, see docs/contracts.md.
// Units: length um, time s, concentration mM.

/** Parameters of a lobule scene; the defaults are those of the Rust `LobuleSpec`. */
export interface LobuleParams {
  /** Circumradius of the hexagonal lobule in um, > 0 (default 300). */
  lobuleRadiusUm?: number;
  /** Rings of neighbouring lobules (default 0). */
  rings?: number;
  /** Vein displacement as a fraction of the radius, 0 <= jitter <= 0.25 (default 0). */
  jitter?: number;
  /** Seed of the displacements (default 0). */
  seed?: number;
  /** Radius of the central veins in um (default 30). */
  centreRadiusUm?: number;
  /** Radius of the corner veins in um (default 25). */
  cornerRadiusUm?: number;
  /** Element size near the walls in um (default 5). */
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
  version: string;
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
  /** Caps the wasm memory; exceeding it is an `Error`. */
  memory?: { maximumPages: number };
}

/** Errors of slidesim are thrown or rejected as `Error` with slidesim's message. */
export function createSimulation(options: CreateOptions): Promise<Simulation>;
