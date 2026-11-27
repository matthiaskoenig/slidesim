//! A wasmtime host of `slidesim.wasm` (`crates/slidesim-wasm`, built by `scripts/wasm/build.sh`):
//! [`run`] runs a study in the browser build of slidesim as a page does, without a browser, and
//! gives every array of every output time ([`Outputs`]), so that its tests compare it with the
//! native build.
//!
//! The modules and how they are linked:
//!
//! - `slidesim.wasm`, a WASI preview 1 reactor (its `_initialize` is called first), imports
//!   `slidesim_kernel.init`, `slidesim_kernel.advance` and `slidesim_kernel.assign`, the functions
//!   of the batched kernel (`kernel/1` of sbmlsim-core: `sbmlsim_kernel_init(n, parameters,
//!   state) -> i32`, `sbmlsim_kernel_advance(n, state, parameters, control, counts, context,
//!   scratch) -> i32` and `sbmlsim_kernel_assign(n, state, parameters, control) -> i32`, byte
//!   offsets into the memory of slidesim). A kernel without `sbmlsim_kernel_assign` (before it
//!   was added to sbmlsim-core) is linked too: `assign` is then an error if called, which slidesim
//!   never does for such a kernel (it refuses outputs of rule targets without `assigned_ids`). The kernel imports the memory it works in, so it
//!   is instantiated after slidesim, with slidesim's exported memory: the imports of slidesim
//!   are bound late, to host functions which call the kernel once it is there.
//! - `mathlib.wasm` of sbmlsim-core (no imports), whose functions the kernel imports as `math.*`.
//! - `kernel.wasm` with `env.memory` (slidesim's memory) and `math.*`.
//!
//! The inputs are the JSON `{"scene", "study", "mesh"?}` of `slidesim_create` without the
//! `kernel`, which [`run`] adds from `kernel.json` beside `kernel.wasm`. WASI gets no files, no
//! environment and no arguments; what slidesim writes to its stdout and stderr is kept in
//! [`Outputs`] (it writes nothing there unless it panics: its log is the array `log`).
//!
//! Memory: [`run`] can cap each linear memory of the store at a number of wasm pages (64 KiB),
//! with a wasmtime [`ResourceLimiter`]: a growth past the cap is refused (`memory.grow` gives -1,
//! so a C `malloc` returns NULL and Rust's allocator aborts, a trap), and any failure of the run
//! after a refused growth is reported as an error naming the cap. The host process lives on:
//! a trap of slidesim is an error of [`run`], not a crash of the host. Every [`run`] instantiates the
//! modules afresh, so an instance that trapped is dropped with its store, as it must be (see
//! `crates/slidesim-wasm`: after a trap its state is that of the middle of a call).

use std::collections::BTreeMap;
use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result, anyhow, bail, ensure};
use serde_json::Value;
use wasmtime::{
    Caller, Engine, Instance, Linker, Memory, Module, ResourceLimiter, Store, TypedFunc,
};
use wasmtime_wasi::WasiCtxBuilder;
use wasmtime_wasi::p1::{self, WasiP1Ctx};
use wasmtime_wasi::p2::pipe::MemoryOutputPipe;

/// The bytes of a wasm page.
pub const PAGE: u64 = 64 * 1024;

/// The most bytes of slidesim's stdout and stderr each which [`Outputs`] keeps.
const STDIO_CAPACITY: usize = 1 << 20;

/// The arrays `mesh/*` of `result/1`.
#[derive(Debug, Clone, PartialEq)]
pub struct MeshArrays {
    /// `mesh/vertices`, `[V, 2]` row by row (um).
    pub vertices: Vec<f64>,
    /// `mesh/triangles`, `[N, 3]` row by row.
    pub triangles: Vec<u32>,
    /// `mesh/element`, `[N]`.
    pub element: Vec<u64>,
    /// `mesh/tags/<name>` of each tag of `meta.tags`.
    pub tags: BTreeMap<String, Vec<u32>>,
}

/// The arrays of one output time.
#[derive(Debug, Clone, PartialEq)]
pub struct Output {
    /// Its index in `time`, as `slidesim_next_output` gave it.
    pub index: usize,
    /// `slidesim_time` there (s).
    pub time: f64,
    /// `fields/<id>` of each field: its row `[V]`.
    pub fields: BTreeMap<String, Vec<f64>>,
    /// `cells/<value>` of each cell value: its row `[C]`.
    pub cells: BTreeMap<String, Vec<f64>>,
    /// `raster/<field>/<level>` of each field, with a raster: per level its `[rows, cols]` row.
    pub rasters: BTreeMap<String, Vec<Vec<f32>>>,
    /// `meta` there.
    pub meta: Value,
}

/// Everything a run of `slidesim.wasm` gives ([`run`]).
#[derive(Debug, Clone)]
pub struct Outputs {
    /// `mesh/*`.
    pub mesh: MeshArrays,
    /// `time`.
    pub time: Vec<f64>,
    /// `cells/element`.
    pub cell_elements: Vec<u64>,
    /// Each output time, in turn.
    pub outputs: Vec<Output>,
    /// `meta` at the end.
    pub meta: Value,
    /// `summary` at the end.
    pub summary: Value,
    /// `quality`: of a generated mesh, `null` for a mesh of the inputs.
    pub quality: Value,
    /// `log`: the lines slidesim reported.
    pub log: String,
    /// What slidesim wrote to its stdout and stderr.
    pub stdout: String,
    pub stderr: String,
    /// The wall time of `slidesim_create` (s): the checks, the mesh, the maps, the flow and the
    /// setup of the fields and the cells.
    pub create_seconds: f64,
}

/// Caps each linear memory of a store at a number of bytes, and remembers a refused growth.
struct Limiter {
    max_bytes: Option<usize>,
    /// The size a refused growth asked for (bytes).
    refused: Option<usize>,
}

impl ResourceLimiter for Limiter {
    fn memory_growing(
        &mut self,
        _current: usize,
        desired: usize,
        _maximum: Option<usize>,
    ) -> wasmtime::Result<bool> {
        if self.max_bytes.is_some_and(|max| desired > max) {
            self.refused = Some(desired);
            return Ok(false);
        }
        Ok(true)
    }

    fn table_growing(
        &mut self,
        _current: usize,
        _desired: usize,
        _maximum: Option<usize>,
    ) -> wasmtime::Result<bool> {
        Ok(true)
    }
}

/// The arguments of `sbmlsim_kernel_advance`: `n`, then the offsets of the state, the
/// parameters, the control, the counts, the context and the scratch.
type AdvanceArgs = (i32, i32, i32, i32, i32, i32, i32);

/// The functions of the kernel, once it is instantiated.
#[derive(Clone)]
struct Kernel {
    init: TypedFunc<(i32, i32, i32), i32>,
    advance: TypedFunc<AdvanceArgs, i32>,
    /// None for a kernel before `sbmlsim_kernel_assign`.
    assign: Option<TypedFunc<(i32, i32, i32, i32), i32>>,
}

struct State {
    wasi: WasiP1Ctx,
    limiter: Limiter,
    kernel: Option<Kernel>,
}

/// The exports of `slidesim.wasm`.
struct Slidesim {
    memory: Memory,
    alloc: TypedFunc<i32, i32>,
    free: TypedFunc<(i32, i32), ()>,
    create: TypedFunc<(i32, i32), i32>,
    next_output: TypedFunc<i32, i32>,
    time: TypedFunc<i32, f64>,
    array: TypedFunc<(i32, i32, i32), i32>,
    array_len: TypedFunc<(), i32>,
    error_ptr: TypedFunc<(), i32>,
    error_len: TypedFunc<(), i32>,
    destroy: TypedFunc<i32, ()>,
}

/// A store with slidesim, mathlib and the kernel instantiated and linked.
struct Host {
    store: Store<State>,
    slidesim: Slidesim,
    stdout: MemoryOutputPipe,
    stderr: MemoryOutputPipe,
    max_pages: Option<u64>,
}

/// Runs `inputs` (the JSON `{"scene", "study", "mesh"?}`) in `slidesim.wasm` at `slidesim`, with
/// the kernel `kernel.wasm` and `kernel.json` in `kernel_dir` and sbmlsim-core's `mathlib.wasm` at
/// `mathlib`, each linear memory capped at `max_pages` wasm pages if given: creates the
/// simulation, reads the arrays of the mesh, steps to every output time and reads its arrays,
/// then the summary, the log and the meta (see the crate documentation).
///
/// Errors: a module which cannot be read, compiled or linked; an error of slidesim (its
/// message); a trap of slidesim (with what it wrote to stderr); a memory growth past
/// `max_pages` (an error naming the cap, whatever failed after it).
pub fn run(
    slidesim: &Path,
    kernel_dir: &Path,
    mathlib: &Path,
    inputs: &str,
    max_pages: Option<u64>,
) -> Result<Outputs> {
    let kernel_json = kernel_dir.join("kernel.json");
    let kernel = std::fs::read_to_string(&kernel_json)
        .with_context(|| format!("cannot read {}", kernel_json.display()))?;
    let inputs = with_kernel(inputs, &kernel)
        .with_context(|| format!("the inputs with {}", kernel_json.display()))?;
    let mut host = Host::new(
        slidesim,
        &kernel_dir.join("kernel.wasm"),
        mathlib,
        max_pages,
    )?;
    let result = host.simulate(inputs.as_bytes());
    host.finish(result)
}

/// The JSON object `inputs` with the key `kernel` set to the JSON `kernel`. The values pass
/// through as they are: a float of a mesh.json keeps its bits (serde_json reads floats correctly
/// rounded with the feature `float_roundtrip` and writes the shortest text which reads back).
fn with_kernel(inputs: &str, kernel: &str) -> Result<String> {
    let mut inputs: serde_json::Map<String, Value> =
        serde_json::from_str(inputs).context("the inputs are not a JSON object")?;
    let kernel: Value = serde_json::from_str(kernel).context("the kernel is not JSON")?;
    inputs.insert("kernel".into(), kernel);
    Ok(serde_json::to_string(&inputs)?)
}

impl Host {
    fn new(slidesim: &Path, kernel: &Path, mathlib: &Path, max_pages: Option<u64>) -> Result<Host> {
        let engine = Engine::default();
        let module = |path: &Path| {
            Module::from_file(&engine, path)
                .map_err(anyhow::Error::from)
                .with_context(|| format!("cannot load {}", path.display()))
        };
        let (slidesim_module, mathlib_module, kernel_module) =
            (module(slidesim)?, module(mathlib)?, module(kernel)?);
        let stdout = MemoryOutputPipe::new(STDIO_CAPACITY);
        let stderr = MemoryOutputPipe::new(STDIO_CAPACITY);
        let wasi = WasiCtxBuilder::new()
            .stdout(stdout.clone())
            .stderr(stderr.clone())
            .build_p1();
        let max_bytes = max_pages
            .map(|pages| usize::try_from(pages.saturating_mul(PAGE)).unwrap_or(usize::MAX));
        let mut store = Store::new(
            &engine,
            State {
                wasi,
                limiter: Limiter {
                    max_bytes,
                    refused: None,
                },
                kernel: None,
            },
        );
        store.limiter(|state| &mut state.limiter);

        let mut linker: Linker<State> = Linker::new(&engine);
        p1::add_to_linker_sync(&mut linker, |state| &mut state.wasi)?;
        linker.func_wrap(
            "slidesim_kernel",
            "init",
            |mut caller: Caller<'_, State>, n: i32, p: i32, y: i32| -> wasmtime::Result<i32> {
                let kernel = kernel_of(&caller)?;
                kernel.init.call(&mut caller, (n, p, y))
            },
        )?;
        linker.func_wrap(
            "slidesim_kernel",
            "advance",
            |mut caller: Caller<'_, State>,
             n: i32,
             y: i32,
             p: i32,
             control: i32,
             counts: i32,
             context: i32,
             scratch: i32|
             -> wasmtime::Result<i32> {
                let kernel = kernel_of(&caller)?;
                kernel
                    .advance
                    .call(&mut caller, (n, y, p, control, counts, context, scratch))
            },
        )?;
        linker.func_wrap(
            "slidesim_kernel",
            "assign",
            |mut caller: Caller<'_, State>,
             n: i32,
             y: i32,
             p: i32,
             control: i32|
             -> wasmtime::Result<i32> {
                let assign = kernel_of(&caller)?.assign.ok_or_else(|| {
                    wasmtime::Error::msg("kernel.wasm has no sbmlsim_kernel_assign")
                })?;
                assign.call(&mut caller, (n, y, p, control))
            },
        )?;
        let host_error = |store: &mut Store<State>, e: wasmtime::Error, what: &str| {
            memory_error(store.data(), max_pages)
                .unwrap_or_else(|| anyhow::Error::from(e).context(what.to_string()))
        };
        let instance = match linker.instantiate(&mut store, &slidesim_module) {
            Ok(instance) => instance,
            Err(e) => {
                return Err(host_error(
                    &mut store,
                    e,
                    "cannot instantiate slidesim.wasm",
                ));
            }
        };
        if let Ok(initialize) = instance.get_typed_func::<(), ()>(&mut store, "_initialize") {
            initialize
                .call(&mut store, ())
                .map_err(|e| host_error(&mut store, e, "_initialize of slidesim.wasm failed"))?;
        }
        let slidesim = Slidesim::new(&mut store, &instance)?;
        let math = Linker::new(&engine)
            .instantiate(&mut store, &mathlib_module)
            .map_err(|e| host_error(&mut store, e, "cannot instantiate mathlib.wasm"))?;
        let mut kernel_linker: Linker<State> = Linker::new(&engine);
        kernel_linker.define(&store, "env", "memory", slidesim.memory)?;
        kernel_linker.instance(&mut store, "math", math)?;
        let kernel = kernel_linker
            .instantiate(&mut store, &kernel_module)
            .map_err(|e| host_error(&mut store, e, "cannot instantiate kernel.wasm"))?;
        let kernel = Kernel {
            init: kernel.get_typed_func(&mut store, "sbmlsim_kernel_init")?,
            advance: kernel.get_typed_func(&mut store, "sbmlsim_kernel_advance")?,
            assign: kernel
                .get_typed_func(&mut store, "sbmlsim_kernel_assign")
                .ok(),
        };
        store.data_mut().kernel = Some(kernel);
        Ok(Host {
            store,
            slidesim,
            stdout,
            stderr,
            max_pages,
        })
    }

    /// Creates the simulation of `inputs` and reads everything (see [`run`]); the arrays without
    /// the outputs of stdio.
    fn simulate(&mut self, inputs: &[u8]) -> Result<Outputs> {
        let started = Instant::now();
        let handle = {
            let ptr = self.write(inputs)?;
            let len = i32::try_from(inputs.len()).context("inputs of more than 2 GiB")?;
            let handle = self.slidesim.create.call(&mut self.store, (ptr, len))?;
            self.slidesim.free.call(&mut self.store, (ptr, len))?;
            handle
        };
        if handle < 0 {
            bail!("{}", self.error()?);
        }
        let create_seconds = started.elapsed().as_secs_f64();
        let meta = self.json(handle, "meta")?;
        let strings = |key: &str| -> Result<Vec<String>> {
            meta[key]
                .as_array()
                .with_context(|| format!("meta.{key} is not an array"))?
                .iter()
                .map(|v| {
                    v.as_str()
                        .or_else(|| v["id"].as_str())
                        .or_else(|| v["name"].as_str())
                        .map(String::from)
                        .with_context(|| format!("meta.{key} holds {v}"))
                })
                .collect()
        };
        let (field_ids, cell_values, tags) = (
            strings("fields")?,
            strings("cell_values")?,
            strings("tags")?,
        );
        let raster_levels = meta["raster"]["levels"].as_u64();
        let mesh = MeshArrays {
            vertices: f64s(&self.array(handle, "mesh/vertices")?),
            triangles: u32s(&self.array(handle, "mesh/triangles")?),
            element: u64s(&self.array(handle, "mesh/element")?),
            tags: tags
                .iter()
                .map(|name| {
                    Ok((
                        name.clone(),
                        u32s(&self.array(handle, &format!("mesh/tags/{name}"))?),
                    ))
                })
                .collect::<Result<_>>()?,
        };
        let time = f64s(&self.array(handle, "time")?);
        let cell_elements = u64s(&self.array(handle, "cells/element")?);
        let quality = self.json(handle, "quality")?;
        let mut outputs = Vec::new();
        loop {
            let index = self.slidesim.next_output.call(&mut self.store, handle)?;
            match index {
                -1 => break,
                -2 => bail!("{}", self.error()?),
                index if index >= 0 => {}
                index => bail!("slidesim_next_output gave {index}"),
            }
            let mut output = Output {
                index: index as usize,
                time: self.slidesim.time.call(&mut self.store, handle)?,
                fields: BTreeMap::new(),
                cells: BTreeMap::new(),
                rasters: BTreeMap::new(),
                meta: self.json(handle, "meta")?,
            };
            for id in &field_ids {
                output.fields.insert(
                    id.clone(),
                    f64s(&self.array(handle, &format!("fields/{id}"))?),
                );
                if let Some(levels) = raster_levels {
                    let levels = (0..levels)
                        .map(|k| Ok(f32s(&self.array(handle, &format!("raster/{id}/{k}"))?)))
                        .collect::<Result<_>>()?;
                    output.rasters.insert(id.clone(), levels);
                }
            }
            for name in &cell_values {
                output.cells.insert(
                    name.clone(),
                    f64s(&self.array(handle, &format!("cells/{name}"))?),
                );
            }
            outputs.push(output);
        }
        let summary = self.json(handle, "summary")?;
        let meta = self.json(handle, "meta")?;
        let log = String::from_utf8(self.array(handle, "log")?).context("the log is not UTF-8")?;
        self.slidesim.destroy.call(&mut self.store, handle)?;
        Ok(Outputs {
            mesh,
            time,
            cell_elements,
            outputs,
            meta,
            summary,
            quality,
            log,
            stdout: String::new(),
            stderr: String::new(),
            create_seconds,
        })
    }

    /// `result` with what slidesim wrote to its stdout and stderr; an error after a refused
    /// growth of a memory as an error naming the cap, a trap with slidesim's stderr.
    fn finish(self, result: Result<Outputs>) -> Result<Outputs> {
        let text = |pipe: &MemoryOutputPipe| String::from_utf8_lossy(&pipe.contents()).into_owned();
        let (stdout, stderr) = (text(&self.stdout), text(&self.stderr));
        match result {
            Ok(outputs) => Ok(Outputs {
                stdout,
                stderr,
                ..outputs
            }),
            Err(e) => {
                if let Some(e) = memory_error(self.store.data(), self.max_pages) {
                    return Err(e);
                }
                if stderr.is_empty() {
                    Err(e)
                } else {
                    Err(e.context(format!(
                        "slidesim.wasm wrote to stderr: {}",
                        stderr.trim_end()
                    )))
                }
            }
        }
    }

    /// `bytes` copied into memory slidesim allocated (`slidesim_alloc`): its address.
    fn write(&mut self, bytes: &[u8]) -> Result<i32> {
        let len = i32::try_from(bytes.len()).context("more than 2 GiB to write")?;
        let ptr = self.slidesim.alloc.call(&mut self.store, len)?;
        ensure!(ptr != 0 || len == 0, "slidesim_alloc of {len} bytes failed");
        self.slidesim
            .memory
            .write(&mut self.store, ptr as u32 as usize, bytes)?;
        Ok(ptr)
    }

    /// The bytes at `ptr` of length `len` (both `u32` in wasm).
    fn read(&mut self, ptr: i32, len: i32) -> Result<Vec<u8>> {
        let mut bytes = vec![0; len as u32 as usize];
        self.slidesim
            .memory
            .read(&self.store, ptr as u32 as usize, &mut bytes)?;
        Ok(bytes)
    }

    /// slidesim's error message (`slidesim_error_ptr`, `slidesim_error_len`).
    fn error(&mut self) -> Result<String> {
        let ptr = self.slidesim.error_ptr.call(&mut self.store, ())?;
        let len = self.slidesim.error_len.call(&mut self.store, ())?;
        let bytes = self.read(ptr, len)?;
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    /// The bytes of the array `name` of the simulation `handle` (`slidesim_array`); an error with
    /// slidesim's message if it has none.
    fn array(&mut self, handle: i32, name: &str) -> Result<Vec<u8>> {
        let name_ptr = self.write(name.as_bytes())?;
        let name_len = name.len() as i32;
        let ptr = self
            .slidesim
            .array
            .call(&mut self.store, (handle, name_ptr, name_len))?;
        self.slidesim
            .free
            .call(&mut self.store, (name_ptr, name_len))?;
        if ptr == 0 {
            return Err(anyhow!("{}", self.error()?)).with_context(|| format!("the array {name}"));
        }
        let len = self.slidesim.array_len.call(&mut self.store, ())?;
        self.read(ptr, len)
    }

    /// The array `name` as JSON.
    fn json(&mut self, handle: i32, name: &str) -> Result<Value> {
        serde_json::from_slice(&self.array(handle, name)?)
            .with_context(|| format!("the array {name} is not JSON"))
    }
}

impl Slidesim {
    fn new(store: &mut Store<State>, instance: &Instance) -> Result<Slidesim> {
        let memory = instance
            .get_memory(&mut *store, "memory")
            .context("slidesim.wasm exports no memory")?;
        Ok(Slidesim {
            memory,
            alloc: instance.get_typed_func(&mut *store, "slidesim_alloc")?,
            free: instance.get_typed_func(&mut *store, "slidesim_free")?,
            create: instance.get_typed_func(&mut *store, "slidesim_create")?,
            next_output: instance.get_typed_func(&mut *store, "slidesim_next_output")?,
            time: instance.get_typed_func(&mut *store, "slidesim_time")?,
            array: instance.get_typed_func(&mut *store, "slidesim_array")?,
            array_len: instance.get_typed_func(&mut *store, "slidesim_array_len")?,
            error_ptr: instance.get_typed_func(&mut *store, "slidesim_error_ptr")?,
            error_len: instance.get_typed_func(&mut *store, "slidesim_error_len")?,
            destroy: instance.get_typed_func(&mut *store, "slidesim_destroy")?,
        })
    }
}

/// The kernel of the store of `caller`, once it is instantiated.
fn kernel_of(caller: &Caller<'_, State>) -> wasmtime::Result<Kernel> {
    caller
        .data()
        .kernel
        .clone()
        .ok_or_else(|| wasmtime::Error::msg("the kernel is not instantiated"))
}

/// After a refused growth of a memory, the error which names the cap.
fn memory_error(state: &State, max_pages: Option<u64>) -> Option<anyhow::Error> {
    let desired = state.limiter.refused?;
    let pages = max_pages.unwrap_or(0);
    Some(anyhow!(
        "slidesim.wasm needs more memory than its maximum of {pages} wasm pages ({} MiB): a \
         memory was to grow to {} MiB",
        pages * PAGE / (1 << 20),
        desired as u64 / (1 << 20)
    ))
}

fn f64s(bytes: &[u8]) -> Vec<f64> {
    bytes
        .as_chunks::<8>()
        .0
        .iter()
        .map(|b| f64::from_le_bytes(*b))
        .collect()
}

fn f32s(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}

fn u32s(bytes: &[u8]) -> Vec<u32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| u32::from_le_bytes(*b))
        .collect()
}

fn u64s(bytes: &[u8]) -> Vec<u64> {
    bytes
        .as_chunks::<8>()
        .0
        .iter()
        .map(|b| u64::from_le_bytes(*b))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Floats pass [`with_kernel`] bit for bit: 10^4 of them spread over the bits of f64.
    #[test]
    fn inputs_keep_their_floats() {
        let mut x: u64 = 0x9e37_79b9_7f4a_7c15;
        let values: Vec<f64> = (0..10_000)
            .map(|_| {
                // xorshift64: arbitrary bits, of finite floats.
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                let v = f64::from_bits(x);
                if v.is_finite() { v } else { 1.0 / 3.0 }
            })
            .collect();
        let inputs = serde_json::json!({"mesh": {"vertices": values}}).to_string();
        let out: Value = serde_json::from_str(&with_kernel(&inputs, "{}").unwrap()).unwrap();
        let back: Vec<f64> = out["mesh"]["vertices"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        let differ = values
            .iter()
            .zip(&back)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert_eq!(differ, 0, "{differ} of {} floats changed", values.len());
        assert_eq!(out["kernel"], serde_json::json!({}));
    }
}
