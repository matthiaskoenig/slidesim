//! slidesim in wasm32-wasip1, the browser build: a WASI reactor (no `main`; a host calls
//! `_initialize` first) which runs a `study/1` as `slidesim run` does, in its own memory, and
//! exports a C ABI over that memory. Built by `scripts/wasm/build.sh` against the wasi builds of
//! PETSc (MPIUNI, one process), libCEED and Mmg; `wasm/host` is a wasmtime host of it, Task 17's
//! JavaScript the browser's.
//!
//! The cells run on the batched kernel of sbmlsim-core (`kernel/1`, `cells::KernelCells`): the
//! host compiles the kernel of the cell model (`kernel.wasm` and its `kernel.json`, which the
//! inputs carry), instantiates it with slidesim's memory and mathlib, and binds
//! slidesim's imports `slidesim_kernel.init`, `slidesim_kernel.advance` and
//! `slidesim_kernel.assign` to its `sbmlsim_kernel_init`, `sbmlsim_kernel_advance` and
//! `sbmlsim_kernel_assign`. Only `"solver": "batched"` runs; the
//! engine path needs native code.
//!
//! The build reads and writes no files: the inputs are JSON, the results arrays in memory. A
//! scene whose mesh or maps are files is refused, naming the file, unless the inputs carry the
//! mesh (`mesh`, the `mesh.json` of `slidesim mesh`).
//!
//! The memory is imported (`env.memory`, linked with `--import-memory` by `build.rs`) and
//! exported again (`memory`, for WASI hosts): the host creates it at the minimum of the import,
//! which `scripts/wasm/build.sh` records in the custom section `slidesim.memory`
//! (`{"minimum_pages": n}`), with its cap as the maximum, and gives the same memory to the kernel.
//!
//! Exports (pointers and lengths are `u32` byte offsets and counts in slidesim's memory):
//!
//! - `slidesim_alloc(len) -> ptr`, `slidesim_free(ptr, len)`: memory for the caller's bytes
//!   (the inputs, the name of an array).
//! - `slidesim_create(ptr, len) -> i32`: the simulation of the JSON
//!   `{"scene", "study", "kernel", "mesh"?}` (`scene` a `scene/1`, `study` a `study/1`, `kernel`
//!   the `kernel.json` of the cell model, required with `cells`, `mesh` a `mesh.json`): its
//!   handle (>= 0), or -1 with the error. Every check, the mesh, the maps, the flow and the setup
//!   of the fields and the cells (`simulation::Simulation::new`).
//! - `slidesim_next_output(h) -> i32`: steps to the next output time and gives its index (the
//!   first at time 0 without a step), -1 when there is none left (without a step: the last
//!   output time is `time.end`, and the arrays stay those of it), -2 with the error.
//! - `slidesim_time(h) -> f64`: the time the simulation reached (s).
//! - `slidesim_array(h, name_ptr, name_len) -> ptr` and `slidesim_array_len() -> u32`: the bytes
//!   of the array `name` now, valid until the next call of `slidesim_array`; the pointer 0 with
//!   the error for a name there is none of. The arrays are those of `result/1`
//!   (`docs/contracts.md`), little endian: `time` (f64), `mesh/vertices` (f64, `[V, 2]`),
//!   `mesh/triangles` (u32, `[N, 3]`), `mesh/element` (u64), `mesh/tags/<name>` (u32),
//!   `fields/<id>` (f64, `[V]`: the row of the current output time; the field now), `cells/element`
//!   (u64), `cells/<value>` (f64, `[C]`, the row now) and, with `result.raster`,
//!   `raster/<field>/<level>` (f32, `[rows, cols]`, the row now); and as JSON `meta` (the
//!   `ResultMeta`: `written` the output times reached, `complete` after the last), `summary`
//!   (`summary.json` of `slidesim run`) and `quality` (of a generated mesh, `null` for a mesh of
//!   the inputs); and as UTF-8 text `log`, what the simulation reported (the quality of a
//!   generated mesh), one line each.
//! - `slidesim_error_ptr() -> ptr`, `slidesim_error_len() -> u32`: the message of the last
//!   error (UTF-8), the error and its causes as `slidesim run` prints them.
//! - `slidesim_destroy(h)`: frees the simulation.
//!
//! slidesim prints nothing to stdout or stderr (a panic, a bug, writes its message to stderr and
//! traps). PETSc is initialized once, at the first `slidesim_create`, and stays.
//!
//! After a trap (a panic, an allocation past the memory the host allows, any other) the instance
//! must be discarded, with every simulation in it: the trap leaves the state of slidesim as it was
//! in the middle of a call (its `RefCell` borrowed, PETSc's objects half updated), so that every
//! later call would trap again or worse. A host instantiates `slidesim.wasm`, mathlib and the
//! kernel again for the next simulation. An error (-1, -2, a null array) is not a trap: the
//! instance and its other simulations stay usable.
//!
//! A pointer with a length of 0 is never read, so it may be anything, 0 included (0 is a valid
//! address in wasm, and the host need not allocate for an empty name or input).
//!
//! Memory: a wasm32 memory has at most 4 GiB; the host may cap it lower (the maximum of the
//! memory it creates: `wasm/host`, `CreateOptions.memory` of the JavaScript API), and an
//! allocation past the cap fails the call (a trap, or an error of PETSc or Mmg). Mmg, the
//! remesher of the mesher, caps itself at 2 GiB in wasm (the constant of
//! `scripts/deps/patches/mmg-0001-memory-size-on-wasm.patch`: half the address space by Mmg's own
//! rule): PETSc's adaptor sets no `MMG2D_IPARAM_mem`, so the cap of the host cannot reach Mmg's
//! own limit.

use std::cell::RefCell;
use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result, anyhow, bail};
use cells::KernelLayout;
use contracts::{MeshSource, Scene, Study};
use mesher::{Quality, TaggedMesh};
use petsc::Petsc;
use serde_json::Value;
use simulation::{Inputs, ResultLayout, Simulation, causes};

/// The name of the study in `meta.study`: the inputs carry its JSON, not a file.
const STUDY: &str = "study.json";

/// A simulation and what it serves besides the values of the output times.
struct Run {
    simulation: Simulation<'static>,
    layout: ResultLayout,
    quality: Option<Quality>,
    log: String,
}

#[derive(Default)]
struct State {
    petsc: Option<&'static Petsc>,
    /// The simulations, by their handles; `None` once destroyed.
    runs: Vec<Option<Run>>,
    error: String,
    array: Vec<u8>,
}

thread_local! {
    static STATE: RefCell<State> = RefCell::new(State::default());
}

fn with_state<T>(f: impl FnOnce(&mut State) -> T) -> T {
    STATE.with(|state| f(&mut state.borrow_mut()))
}

impl State {
    fn petsc(&mut self) -> Result<&'static Petsc> {
        if let Some(petsc) = self.petsc {
            return Ok(petsc);
        }
        let petsc = Petsc::init(&[]).context("cannot initialize PETSc")?;
        let petsc: &'static Petsc = Box::leak(Box::new(petsc));
        self.petsc = Some(petsc);
        Ok(petsc)
    }

    fn run(&mut self, handle: i32) -> Result<&mut Run> {
        usize::try_from(handle)
            .ok()
            .and_then(|h| self.runs.get_mut(h))
            .and_then(Option::as_mut)
            .ok_or_else(|| anyhow!("no simulation {handle}"))
    }

    /// The simulation of the JSON `bytes` (see the crate documentation): its handle.
    fn create(&mut self, bytes: &[u8]) -> Result<i32> {
        let petsc = self.petsc()?;
        let mut request: Value =
            serde_json::from_slice(bytes).context("the inputs are not JSON")?;
        let mut take = |key: &str| request.get_mut(key).map(Value::take);
        let study = take("study").context("the inputs have no study")?;
        let scene = take("scene").context("the inputs have no scene")?;
        let kernel = take("kernel")
            .map(serde_json::from_value::<KernelLayout>)
            .transpose()
            .context("the kernel is not a kernel.json")?;
        let mesh = take("mesh")
            .map(serde_json::from_value::<TaggedMesh>)
            .transpose()
            .context("the mesh is not a mesh.json")?;
        let study = Study::from_value(study, Path::new(STUDY)).context("cannot load the study")?;
        let scene = Scene::from_value(scene, &study.scene).context("cannot load the scene")?;
        no_files(&scene, mesh.is_some())?;
        let inputs = Inputs {
            study,
            scene,
            kernel,
            mesh,
        };
        let mut log = String::new();
        let simulation = Simulation::new(petsc, inputs, Instant::now(), &mut |line| {
            log.push_str(line);
            log.push('\n');
        })?;
        simulation.check_result_options()?;
        let layout =
            ResultLayout::new(&simulation, Path::new(STUDY))?.expect("one process, process 0");
        let quality = simulation.mesh().generated().map(|(_, q)| q.clone());
        let handle = i32::try_from(self.runs.len()).context("too many simulations")?;
        self.runs.push(Some(Run {
            simulation,
            layout,
            quality,
            log,
        }));
        Ok(handle)
    }

    /// Steps the simulation `handle` to its next output time: its index, `None` at the end.
    fn next_output(&mut self, handle: i32) -> Result<Option<usize>> {
        let run = self.run(handle)?;
        let index = run.simulation.next_output()?;
        if let Some(index) = index {
            let meta = &mut run.layout.meta;
            meta.written = index + 1;
            meta.complete = meta.written == meta.times.len();
        }
        Ok(index)
    }

    /// The bytes of the array `name` of the simulation `handle` (see the crate documentation).
    fn array(&mut self, handle: i32, name: &str) -> Result<Vec<u8>> {
        let run = self.run(handle)?;
        let simulation = &run.simulation;
        let layout = &run.layout;
        let mesh = &layout.canonical;
        Ok(match name {
            "time" => le(layout.meta.times.iter(), f64::to_le_bytes),
            "mesh/vertices" => le(mesh.vertices.iter().flatten(), f64::to_le_bytes),
            "mesh/triangles" => le(mesh.triangles.iter().flatten(), u32::to_le_bytes),
            "mesh/element" => le(mesh.element.iter(), u64::to_le_bytes),
            "cells/element" => le(layout.cell_elements.iter(), u64::to_le_bytes),
            "meta" => json(&layout.meta)?,
            "summary" => json(simulation.summary())?,
            "quality" => json(&run.quality)?,
            "log" => run.log.as_bytes().to_vec(),
            _ => {
                if let Some(tag) = name.strip_prefix("mesh/tags/") {
                    let tag = mesh
                        .tags
                        .iter()
                        .find(|t| t.name == tag && matches!(t.dim, 1 | 2))
                        .ok_or_else(|| anyhow!("the mesh has no tag '{tag}'"))?;
                    if tag.dim == 2 {
                        le(tag.triangles.iter(), u32::to_le_bytes)
                    } else {
                        le(tag.edges.iter().flatten(), u32::to_le_bytes)
                    }
                } else if let Some(id) = name.strip_prefix("fields/") {
                    le(field_row(simulation, id)?.iter(), f64::to_le_bytes)
                } else if let Some(value) = name.strip_prefix("cells/") {
                    le(
                        cell_row(simulation, layout, value)?.iter(),
                        f64::to_le_bytes,
                    )
                } else if let Some(rest) = name.strip_prefix("raster/") {
                    let (id, level) = rest
                        .rsplit_once('/')
                        .ok_or_else(|| anyhow!("no array '{name}'"))?;
                    let raster = layout
                        .raster
                        .as_ref()
                        .ok_or_else(|| anyhow!("the study has no result.raster"))?;
                    let level: usize = level
                        .parse()
                        .ok()
                        .filter(|&k| k < raster.meta().levels as usize)
                        .ok_or_else(|| anyhow!("the raster has no level '{level}'"))?;
                    let pyramid = raster.pyramid(&field_row(simulation, id)?);
                    le(pyramid[level].iter(), f32::to_le_bytes)
                } else {
                    bail!("no array '{name}'")
                }
            }
        })
    }
}

/// Values as their little endian bytes, one after the other.
fn le<'a, T: Copy + 'a, const N: usize>(
    values: impl Iterator<Item = &'a T>,
    bytes: fn(T) -> [u8; N],
) -> Vec<u8> {
    values.flat_map(|&v| bytes(v)).collect()
}

/// `value` as JSON.
fn json<T: serde::Serialize + ?Sized>(value: &T) -> Result<Vec<u8>> {
    Ok(serde_json::to_vec(value)?)
}

/// The field `id` now at the vertices of the canonical mesh.
fn field_row(simulation: &Simulation, id: &str) -> Result<Vec<f64>> {
    let field = simulation
        .field_ids()
        .iter()
        .position(|f| *f == id)
        .ok_or_else(|| anyhow!("the study has no field '{id}'"))?;
    let values = simulation.field_values(field)?;
    Ok(simulation
        .mesh()
        .gather_vertex_values(&values)?
        .expect("one process, process 0"))
}

/// The cell value `name` now, per cell of `cells/element`.
fn cell_row(simulation: &Simulation, layout: &ResultLayout, name: &str) -> Result<Vec<f64>> {
    let names = simulation.cell_value_names();
    let k = names
        .iter()
        .position(|n| n == name)
        .ok_or_else(|| anyhow!("the cells have no value '{name}'"))?;
    let (ids, table) = simulation
        .cell_values()
        .ok_or_else(|| anyhow!("the study has no cells"))?;
    let (ids, table) = simulation
        .mesh()
        .gather_element_values(ids, &table, names.len())?
        .expect("one process, process 0");
    debug_assert_eq!(ids, layout.cell_elements);
    Ok(table.iter().skip(k).step_by(names.len()).copied().collect())
}

/// An error unless the simulation of `scene` reads no file: the browser build has none. With a
/// mesh in the inputs the mesh source of the scene is not read.
fn no_files(scene: &Scene, has_mesh: bool) -> Result<()> {
    let file = match &scene.mesh {
        _ if has_mesh => None,
        MeshSource::Gmsh { path } => Some(("the gmsh mesh", path)),
        MeshSource::Polygons { path, .. } => Some(("the polygons", path)),
        MeshSource::Lobules(_) => None,
    };
    if let Some((what, path)) = file {
        bail!(
            "the browser build reads no files, and the mesh of the scene is {what} {}: give the \
             mesh in the inputs (`mesh`, the mesh.json of `slidesim mesh`)",
            path.display()
        );
    }
    if let Some(map) = scene.maps.first() {
        bail!(
            "the browser build reads no files, and the map '{}' of the scene is the file {}",
            map.id,
            map.path.display()
        );
    }
    Ok(())
}

/// Memory of `len` bytes for the caller: its address.
#[unsafe(no_mangle)]
pub extern "C" fn slidesim_alloc(len: usize) -> *mut u8 {
    let mut bytes = Vec::<u8>::with_capacity(len);
    let ptr = bytes.as_mut_ptr();
    std::mem::forget(bytes);
    ptr
}

/// Frees the memory of [`slidesim_alloc`] at `ptr` of `len` bytes.
///
/// # Safety
/// `ptr` and `len` are those of a [`slidesim_alloc`] not freed yet.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn slidesim_free(ptr: *mut u8, len: usize) {
    if len == 0 {
        // Nothing was allocated: `slidesim_alloc(0)` gives a dangling pointer.
        return;
    }
    // SAFETY: the memory of a `Vec<u8>` of the capacity `len` (the caller's contract).
    drop(unsafe { Vec::from_raw_parts(ptr, 0, len) });
}

/// The `len` bytes at `ptr`; none, whatever `ptr` is (0 included), for a length of 0.
///
/// # Safety
/// For a length above 0, `ptr` points to `len` readable bytes which live for `'a`.
unsafe fn bytes<'a>(ptr: *const u8, len: usize) -> &'a [u8] {
    if len == 0 {
        return &[];
    }
    // SAFETY: the caller's contract; not null, since `len` bytes are there.
    unsafe { std::slice::from_raw_parts(ptr, len) }
}

/// The simulation of the inputs at `ptr` of `len` bytes: its handle, or -1 with the error.
///
/// # Safety
/// `ptr` points to `len` readable bytes (any pointer for 0 bytes).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn slidesim_create(ptr: *const u8, len: usize) -> i32 {
    // SAFETY: the caller's contract.
    let bytes = unsafe { bytes(ptr, len) };
    with_state(|state| {
        state.create(bytes).unwrap_or_else(|e| {
            state.error = causes(&e);
            -1
        })
    })
}

/// Steps the simulation `handle` to its next output time: its index, -1 at the end, -2 with the
/// error.
#[unsafe(no_mangle)]
pub extern "C" fn slidesim_next_output(handle: i32) -> i32 {
    with_state(|state| match state.next_output(handle) {
        Ok(Some(index)) => i32::try_from(index).unwrap_or(i32::MAX),
        Ok(None) => -1,
        Err(e) => {
            state.error = causes(&e);
            -2
        }
    })
}

/// The time the simulation `handle` reached (s); NaN for no simulation.
#[unsafe(no_mangle)]
pub extern "C" fn slidesim_time(handle: i32) -> f64 {
    with_state(|state| {
        state
            .run(handle)
            .map_or(f64::NAN, |run| run.simulation.time())
    })
}

/// The bytes of the array named by the `name_len` bytes at `name_ptr` of the simulation `handle`
/// (see the crate documentation), valid until the next call; null with the error.
///
/// # Safety
/// `name_ptr` points to `name_len` readable bytes (any pointer for 0 bytes).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn slidesim_array(
    handle: i32,
    name_ptr: *const u8,
    name_len: usize,
) -> *const u8 {
    // SAFETY: the caller's contract.
    let name = unsafe { bytes(name_ptr, name_len) };
    with_state(|state| {
        let array = std::str::from_utf8(name)
            .map_err(|_| anyhow!("the name of an array is not UTF-8"))
            .and_then(|name| state.array(handle, name));
        match array {
            Ok(bytes) => {
                state.array = bytes;
                state.array.as_ptr()
            }
            Err(e) => {
                state.array.clear();
                state.error = causes(&e);
                std::ptr::null()
            }
        }
    })
}

/// The length in bytes of the array of the last [`slidesim_array`].
#[unsafe(no_mangle)]
pub extern "C" fn slidesim_array_len() -> usize {
    with_state(|state| state.array.len())
}

/// The message of the last error (UTF-8).
#[unsafe(no_mangle)]
pub extern "C" fn slidesim_error_ptr() -> *const u8 {
    with_state(|state| state.error.as_ptr())
}

/// The length in bytes of the message of the last error.
#[unsafe(no_mangle)]
pub extern "C" fn slidesim_error_len() -> usize {
    with_state(|state| state.error.len())
}

/// Frees the simulation `handle`; nothing for no simulation.
#[unsafe(no_mangle)]
pub extern "C" fn slidesim_destroy(handle: i32) {
    with_state(|state| {
        if let Some(slot) = usize::try_from(handle)
            .ok()
            .and_then(|h| state.runs.get_mut(h))
        {
            *slot = None;
        }
    })
}
