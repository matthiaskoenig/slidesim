//! `result/1`: the store `<out>/result.zarr` of a run, mesh native ([`ResultWriter`]), and its
//! reader ([`read_result`]).
//!
//! The store is the Zarr v3 group of the contract `result/1` (see `contracts::ResultMeta` and
//! `docs/contracts.md`): the canonical finest mesh ([`mesh::Mesh::canonical`]: `mesh/vertices`,
//! `mesh/triangles`, `mesh/element`, `mesh/tags/<name>`), the output times (`time`), every field
//! at the canonical vertices per output time (`fields/<id>`, one chunk `[1, V]` per time) and,
//! with cells, the element ids of the cells (`cells/element`, ascending) and each cell value per
//! output time (`cells/<value>`, one chunk `[1, C]` per time), zstd compressed. The root
//! attributes are the [`ResultMeta`].
//!
//! - `study` and `scene` are the file names of the study and its scene (the last component of
//!   their paths, without directories): the store names no path of the machine it was written
//!   on. The study's path is an argument of [`ResultWriter::create`] (a [`Simulation`] holds the
//!   study, not the file it was read from).
//! - A tag of dimension 2 is `mesh/tags/<name>` of shape `[k]`, the indices of its triangles
//!   ascending; a tag of dimension 1 is of shape `[k, 2]`, its edges as pairs of vertex indices,
//!   the smaller first, the pairs ascending. A tag of another dimension (points) is not written,
//!   in neither the arrays nor `tags`.
//! - A study without cells has an empty `cells/element` (shape `[0]`), no `cells/<value>` and an
//!   empty `cell_values`.
//! - The arrays of the output times hold NaN (their fill value) until written. Before the
//!   first output time `written` is 0; [`ResultWriter::write`] writes the chunks of an output
//!   time first and then raises `written` (the root attributes are replaced atomically, by a
//!   rename), so a reader of a store being written sees every index below `written` complete.
//!   The last output time sets `complete`. A run which ends early leaves `complete` false.
//!
//! Process 0 writes the store; the others take part in the gathers ([`mesh::Mesh::canonical`],
//! [`mesh::Mesh::gather_vertex_values`], [`mesh::Mesh::gather_element_values`]). Every error of
//! the writer is met by every process and carries the context [`OnEveryProcess`] (process 0
//! tells the others when it cannot write). The writer only reads the simulation: a run with a
//! store has the same results as one without.

use std::path::Path;
#[cfg(not(target_os = "wasi"))]
use std::sync::Arc;

use anyhow::{Context, Result, anyhow, ensure};
use contracts::{RESULT_VERSION, ResultField, ResultMeta, ResultTag};
#[cfg(not(target_os = "wasi"))]
use mesh::CanonicalTag;
use mesh::{CanonicalMesh, MeshError};
use petsc::Comm;
#[cfg(not(target_os = "wasi"))]
use serde_json::Value;
#[cfg(not(target_os = "wasi"))]
use zarrs::array::codec::ZstdCodec;
#[cfg(not(target_os = "wasi"))]
use zarrs::array::{Array, ArrayBuilder, ArraySubset, CodecOptions, DataType, data_type};
#[cfg(not(target_os = "wasi"))]
use zarrs::filesystem::FilesystemStore;
#[cfg(not(target_os = "wasi"))]
use zarrs::group::{Group, GroupBuilder};

use crate::raster::Raster;
use crate::{OnEveryProcess, Simulation};

/// The name of the store in the output directory.
#[cfg(not(target_os = "wasi"))]
pub const RESULT_STORE: &str = "result.zarr";

/// The units of every field.
const FIELD_UNITS: &str = "mM";

/// The zstd level of every array: fast, about as small as higher levels for these values.
#[cfg(not(target_os = "wasi"))]
const ZSTD_LEVEL: i32 = 3;

/// The side of the chunks of the raster arrays, in pixels.
#[cfg(not(target_os = "wasi"))]
const RASTER_CHUNK: u64 = 256;

/// The context of every error of the layout and the writer: of the store of `slidesim run`, or
/// of the arrays in memory of the browser build, which writes no store.
#[cfg(not(target_os = "wasi"))]
pub(crate) const CANNOT_WRITE: OnEveryProcess = OnEveryProcess("cannot write result.zarr");
#[cfg(target_os = "wasi")]
pub(crate) const CANNOT_WRITE: OnEveryProcess = OnEveryProcess("cannot set up the result arrays");

/// Writes the store of a run, output time by output time (see the module).
#[cfg(not(target_os = "wasi"))]
pub struct ResultWriter {
    /// The store, on process 0.
    store: Option<Store>,
    /// The output times, `time.outputs` of the study.
    times: Vec<f64>,
    /// The index of the output time written next.
    next: usize,
}

/// The open store on process 0.
#[cfg(not(target_os = "wasi"))]
struct Store {
    /// The directory of the store.
    dir: std::path::PathBuf,
    meta: ResultMeta,
    /// `fields/<id>` of each field, in the order of the fields.
    fields: Vec<Array<FilesystemStore>>,
    /// `cells/<value>` of each cell value, in the order of the values.
    cells: Vec<Array<FilesystemStore>>,
    /// The element ids of the cells, `cells/element`.
    cell_elements: Vec<u64>,
    /// The raster pyramid, with its arrays `raster/<field>/<level>` of each field (in the order
    /// of the fields) and level.
    raster: Option<(Raster, Vec<Vec<Array<FilesystemStore>>>)>,
}

/// Everything in a store ([`read_result`]).
#[cfg(not(target_os = "wasi"))]
#[derive(Debug, Clone, PartialEq)]
pub struct ResultData {
    /// The root attributes.
    pub meta: ResultMeta,
    /// `time`.
    pub time: Vec<f64>,
    /// `mesh/*`: the canonical mesh, with the tags of `meta.tags` (each with its triangles or
    /// its edges).
    pub mesh: CanonicalMesh,
    /// `fields/<id>` of each field of `meta.fields`: its id and its `[T, V]` values, row by row.
    pub fields: Vec<(String, Vec<f64>)>,
    /// `cells/element`.
    pub cell_elements: Vec<u64>,
    /// `cells/<value>` of each name of `meta.cell_values`: the name and the `[T, C]` values, row
    /// by row.
    pub cells: Vec<(String, Vec<f64>)>,
    /// `raster/<field>/<level>` of each field of `meta.fields` when `meta.raster` is set (else
    /// empty): the id and, per level, the `[T, rows, cols]` values, row by row.
    pub rasters: Vec<(String, Vec<Vec<f32>>)>,
}

/// What `result/1` holds of a simulation apart from the values of its output times, on process 0:
/// the root attributes before the first output time (`written` 0), the canonical finest mesh,
/// the element ids of the cells (`cells/element`) and the raster of `result.raster` (if any).
/// [`ResultWriter`] writes it into the store; the browser build serves it from memory.
#[derive(Debug, Clone)]
pub struct ResultLayout {
    pub meta: ResultMeta,
    pub canonical: CanonicalMesh,
    pub cell_elements: Vec<u64>,
    pub raster: Option<Raster>,
}

impl ResultLayout {
    /// The layout of `simulation`, run from the study file `study` (only its file name is kept,
    /// see the module): `Some` on process 0, `None` on the others. Makes the canonical mesh,
    /// which the gathers of the vertex values need first. Collective.
    ///
    /// Errors (on every process): a tag or a cell value whose name cannot name an array (empty,
    /// with a `/`, `.` or `..`, starting with `__`, or a cell value `element`), the gathers, a
    /// raster which cannot be made.
    pub fn new(simulation: &Simulation, study: &Path) -> Result<Option<ResultLayout>> {
        let comm = simulation.mesh().finest().comm();
        let names = simulation.cell_value_names();
        check_names(simulation.mesh().tags(), &names).map_err(|e| e.context(CANNOT_WRITE))?;
        let canonical = simulation.mesh().canonical().map_err(mesh_error)?;
        let cell_elements = match simulation.cell_values() {
            Some((ids, table)) => simulation
                .mesh()
                .gather_element_values(ids, &table, names.len())
                .map_err(mesh_error)?
                .map(|(ids, _)| ids),
            None => Some(Vec::new()),
        };
        let file_name = |path: &Path| {
            path.file_name().map_or_else(
                || path.display().to_string(),
                |n| n.to_string_lossy().into_owned(),
            )
        };
        let study_data = simulation.study();
        let times = study_data.time.outputs.clone();
        let layout = match (canonical, cell_elements) {
            (Some(canonical), Some(cell_elements)) => {
                let raster = study_data
                    .result
                    .as_ref()
                    .and_then(|r| r.raster.as_ref())
                    .map(|r| Raster::new(&canonical, r.um_per_px, r.origin_um, r.levels))
                    .transpose()
                    .context("cannot make the raster");
                Some(raster.map(|raster| {
                    let meta = ResultMeta {
                        version: RESULT_VERSION.to_string(),
                        study: file_name(study),
                        scene: file_name(&study_data.scene),
                        refinements: study_data.mesh.refinements,
                        complete: times.is_empty(),
                        times,
                        written: 0,
                        fields: simulation
                            .field_ids()
                            .into_iter()
                            .map(|id| ResultField {
                                id: id.to_string(),
                                units: FIELD_UNITS.to_string(),
                            })
                            .collect(),
                        cell_values: names,
                        tags: canonical
                            .tags
                            .iter()
                            .filter(|t| matches!(t.dim, 1 | 2))
                            .map(|t| ResultTag {
                                name: t.name.clone(),
                                dim: u32::from(t.dim),
                            })
                            .collect(),
                        raster: raster.as_ref().map(Raster::meta),
                    };
                    ResultLayout {
                        meta,
                        canonical,
                        cell_elements,
                        raster,
                    }
                }))
            }
            _ => None,
        };
        on_every_process(&comm, layout.transpose())
    }
}

#[cfg(not(target_os = "wasi"))]
impl ResultWriter {
    /// Makes the store `<out>/result.zarr` of `simulation`, run from the study file `study`
    /// (only its file name is kept): an earlier store there is removed whole, then the mesh, the
    /// output times and the arrays of the fields and the cells are written, no output time yet
    /// (`written` 0). Makes the canonical mesh, which the gathers of the vertex values need
    /// first. `out` is created if missing. Collective.
    ///
    /// Errors (on every process): a tag or a cell value whose name cannot name an array (empty,
    /// with a `/`, `.` or `..`, starting with `__`, or a cell value `element`), the gathers, the
    /// store cannot be written.
    pub fn create(out: &Path, simulation: &Simulation, study: &Path) -> Result<ResultWriter> {
        let comm = simulation.mesh().finest().comm();
        let times = simulation.study().time.outputs.clone();
        let store = ResultLayout::new(simulation, study)?.map(|layout| {
            Store::create(
                &out.join(RESULT_STORE),
                layout.meta,
                &layout.canonical,
                layout.cell_elements,
                layout.raster,
            )
        });
        let store = on_every_process(&comm, store.transpose())?;
        Ok(ResultWriter {
            store,
            times,
            next: 0,
        })
    }

    /// Writes the output time `index` of `simulation`, which is there (the index
    /// [`Simulation::next_output`] returned last): each field at the canonical vertices and each
    /// cell value, then `written = index + 1`, and `complete` with the last output time. The
    /// output times are written in turn, from 0. Collective.
    ///
    /// Errors (on every process): `index` is not the output time to write next or the
    /// simulation is not at it, the gathers, the store cannot be written.
    pub fn write(&mut self, simulation: &Simulation, index: usize) -> Result<()> {
        if index != self.next {
            return Err(cannot_write(format!(
                "the output time {index} is not the one to write next, {}",
                self.next
            )));
        }
        if self.times.get(index) != Some(&simulation.time()) {
            return Err(cannot_write(format!(
                "the simulation is not at the output time {index} (it is at {} s)",
                simulation.time()
            )));
        }
        let comm = simulation.mesh().finest().comm();
        let mesh = simulation.mesh();
        // Process 0 writes what it gets until it fails; every process takes part in every
        // gather regardless, and they agree at the end.
        let mut written: Result<()> = Ok(());
        let t = index as u64;
        for field in 0..simulation.field_ids().len() {
            let values = simulation.field_values(field)?;
            let gathered = mesh.gather_vertex_values(&values).map_err(mesh_error)?;
            if let (Some(values), Some(store), Ok(())) = (gathered, &self.store, &written) {
                written = store_row(&store.fields[field], t, &values)
                    .and_then(|()| store.write_raster(field, t, &values));
            }
        }
        if let Some((ids, table)) = simulation.cell_values() {
            let width = simulation.cell_value_names().len();
            let gathered = mesh
                .gather_element_values(ids, &table, width)
                .map_err(mesh_error)?;
            if let (Some((ids, table)), Some(store), Ok(())) = (gathered, &self.store, &written) {
                written = store.write_cells(t, &ids, &table);
            }
        }
        if let (Some(store), Ok(())) = (&mut self.store, &written) {
            store.meta.written = index + 1;
            store.meta.complete = store.meta.written == store.meta.times.len();
            written = store.write_meta();
        }
        on_every_process(&comm, written.map(|()| None::<()>))?;
        self.next += 1;
        Ok(())
    }
}

/// An error of the writer of the message `message`, which every process has.
#[cfg(not(target_os = "wasi"))]
fn cannot_write(message: String) -> anyhow::Error {
    anyhow!(message).context(CANNOT_WRITE)
}

#[cfg(not(target_os = "wasi"))]
impl Store {
    /// Makes the store in the directory `dir` (removed first, if there): the root attributes
    /// `meta`, the mesh `canonical`, the output times and the arrays of the fields and of the
    /// cells of the elements `cell_elements`.
    fn create(
        dir: &Path,
        meta: ResultMeta,
        canonical: &CanonicalMesh,
        cell_elements: Vec<u64>,
        raster: Option<Raster>,
    ) -> Result<Store> {
        if dir.is_dir() {
            std::fs::remove_dir_all(dir)
        } else if dir.exists() {
            std::fs::remove_file(dir)
        } else {
            Ok(())
        }
        .with_context(|| format!("cannot remove the earlier {}", dir.display()))?;
        std::fs::create_dir_all(dir).with_context(|| format!("cannot create {}", dir.display()))?;
        let storage = Arc::new(FilesystemStore::new(dir)?);
        let store = Store {
            dir: dir.to_path_buf(),
            meta,
            fields: Vec::new(),
            cells: Vec::new(),
            cell_elements,
            raster: None,
        };
        store.write_meta()?;
        let groups = ["/mesh", "/mesh/tags", "/fields", "/cells"];
        for group in groups
            .into_iter()
            .chain(raster.is_some().then_some("/raster"))
        {
            GroupBuilder::new()
                .build(storage.clone(), group)?
                .store_metadata()?;
        }
        let whole = |storage: &Arc<FilesystemStore>, path: &str, shape: Vec<u64>, data_type| {
            new_array(storage, path, shape.clone(), shape, data_type)
        };
        let t = store.meta.times.len() as u64;
        whole(&storage, "/time", vec![t], data_type::float64())?
            .store_chunk(&[0], store.meta.times.as_slice())?;
        let v = canonical.vertices.len() as u64;
        let vertices: Vec<f64> = canonical.vertices.iter().flatten().copied().collect();
        whole(&storage, "/mesh/vertices", vec![v, 2], data_type::float64())?
            .store_chunk(&[0, 0], vertices)?;
        let n = canonical.triangles.len() as u64;
        let triangles: Vec<u32> = canonical.triangles.iter().flatten().copied().collect();
        whole(&storage, "/mesh/triangles", vec![n, 3], data_type::uint32())?
            .store_chunk(&[0, 0], triangles)?;
        whole(&storage, "/mesh/element", vec![n], data_type::uint64())?
            .store_chunk(&[0], canonical.element.as_slice())?;
        for tag in &canonical.tags {
            let path = format!("/mesh/tags/{}", tag.name);
            match tag.dim {
                2 => {
                    let k = tag.triangles.len() as u64;
                    let array = whole(&storage, &path, vec![k], data_type::uint32())?;
                    if k > 0 {
                        array.store_chunk(&[0], tag.triangles.as_slice())?;
                    }
                }
                1 => {
                    let k = tag.edges.len() as u64;
                    let edges: Vec<u32> = tag.edges.iter().flatten().copied().collect();
                    let array = whole(&storage, &path, vec![k, 2], data_type::uint32())?;
                    if k > 0 {
                        array.store_chunk(&[0, 0], edges)?;
                    }
                }
                _ => {}
            }
        }
        let c = store.cell_elements.len() as u64;
        let elements = whole(&storage, "/cells/element", vec![c], data_type::uint64())?;
        if c > 0 {
            elements.store_chunk(&[0], store.cell_elements.as_slice())?;
        }
        let per_time = |path: String, n: u64| {
            new_array(
                &storage,
                &path,
                vec![t, n],
                vec![1, n],
                data_type::float64(),
            )
        };
        let fields = store
            .meta
            .fields
            .iter()
            .map(|f| per_time(format!("/fields/{}", f.id), v))
            .collect::<Result<_>>()?;
        let cells = store
            .meta
            .cell_values
            .iter()
            .map(|name| per_time(format!("/cells/{name}"), c))
            .collect::<Result<_>>()?;
        let raster = match raster {
            Some(raster) => {
                let mut arrays = Vec::new();
                for field in &store.meta.fields {
                    GroupBuilder::new()
                        .build(storage.clone(), &format!("/raster/{}", field.id))?
                        .store_metadata()?;
                    let mut levels = Vec::new();
                    for (k, [rows, cols]) in raster.meta().shapes.into_iter().enumerate() {
                        levels.push(new_array(
                            &storage,
                            &format!("/raster/{}/{k}", field.id),
                            vec![t, rows, cols],
                            vec![1, RASTER_CHUNK, RASTER_CHUNK],
                            data_type::float32(),
                        )?);
                    }
                    arrays.push(levels);
                }
                Some((raster, arrays))
            }
            None => None,
        };
        Ok(Store {
            fields,
            cells,
            raster,
            ..store
        })
    }

    /// Writes the pyramid of the field `field` at the output time `t` from its nodal `values`
    /// (nothing without a raster).
    fn write_raster(&self, field: usize, t: u64, values: &[f64]) -> Result<()> {
        let Some((raster, arrays)) = &self.raster else {
            return Ok(());
        };
        for (array, level) in arrays[field].iter().zip(raster.pyramid(values)) {
            let shape = array.shape();
            let subset = ArraySubset::new_with_ranges(&[t..t + 1, 0..shape[1], 0..shape[2]]);
            array
                .store_array_subset_opt(&subset, level, &one_thread())
                .with_context(|| format!("cannot write {} at {t}", array.path()))?;
        }
        Ok(())
    }

    /// Writes the rows `table` of the cells `ids` (one value per cell value) as the output time
    /// `t` of each `cells/<value>`.
    fn write_cells(&self, t: u64, ids: &[u64], table: &[f64]) -> Result<()> {
        ensure!(
            ids == self.cell_elements,
            "the cells are not those of cells/element"
        );
        let width = self.cells.len();
        for (k, array) in self.cells.iter().enumerate() {
            let column: Vec<f64> = table.iter().skip(k).step_by(width).copied().collect();
            store_row(array, t, &column)?;
        }
        Ok(())
    }

    /// Writes the root attributes, the [`ResultMeta`], atomically: into a file beside
    /// `zarr.json`, renamed to it.
    fn write_meta(&self) -> Result<()> {
        self.meta.check()?;
        let root = serde_json::json!({
            "zarr_format": 3,
            "node_type": "group",
            "attributes": self.meta,
        });
        let path = self.dir.join("zarr.json");
        let partial = self.dir.join(".zarr.json.partial");
        let mut text = serde_json::to_vec_pretty(&root)?;
        text.push(b'\n');
        std::fs::write(&partial, text)
            .with_context(|| format!("cannot write {}", partial.display()))?;
        std::fs::rename(&partial, &path)
            .with_context(|| format!("cannot write {}", path.display()))?;
        Ok(())
    }
}

/// `values`, the row `t` of the `[T, n]` array `array` (its chunk `[t, 0]`), on this thread.
#[cfg(not(target_os = "wasi"))]
fn store_row(array: &Array<FilesystemStore>, t: u64, values: &[f64]) -> Result<()> {
    if values.is_empty() {
        return Ok(());
    }
    array
        .store_chunk_opt(&[t, 0], values, &one_thread())
        .with_context(|| format!("cannot write {} at {t}", array.path()))
}

/// The codec options of the writer: one thread, beside the processes of the run on their cores.
#[cfg(not(target_os = "wasi"))]
fn one_thread() -> CodecOptions {
    CodecOptions::default().with_concurrent_target(1)
}

/// The array `path` of `storage` of shape `shape`, chunks `chunks` (each at least 1) and data
/// type `data_type`, zstd compressed, its metadata written; floats filled with NaN, integers
/// with 0.
#[cfg(not(target_os = "wasi"))]
fn new_array(
    storage: &Arc<FilesystemStore>,
    path: &str,
    shape: Vec<u64>,
    chunks: Vec<u64>,
    data_type: DataType,
) -> Result<Array<FilesystemStore>> {
    let chunks: Vec<u64> = chunks.into_iter().map(|c| c.max(1)).collect();
    let mut builder = if data_type == data_type::float64() {
        ArrayBuilder::new(shape, chunks, data_type, f64::NAN)
    } else if data_type == data_type::float32() {
        ArrayBuilder::new(shape, chunks, data_type, f32::NAN)
    } else if data_type == data_type::uint32() {
        ArrayBuilder::new(shape, chunks, data_type, 0u32)
    } else {
        ArrayBuilder::new(shape, chunks, data_type, 0u64)
    };
    let array = builder
        .bytes_to_bytes_codecs(vec![Arc::new(ZstdCodec::new(ZSTD_LEVEL, false))])
        .build(storage.clone(), path)
        .with_context(|| format!("cannot make the array {path}"))?;
    array
        .store_metadata()
        .with_context(|| format!("cannot write the array {path}"))?;
    Ok(array)
}

/// Checks that every tag (of dimension 1 or 2) and every cell value can name an array: not
/// empty, no `/`, not `.` or `..`, not starting with `__` (reserved by Zarr), and no cell value
/// `element` (the array of the element ids). The same on every process.
fn check_names(tags: &[mesh::Tag], cell_values: &[String]) -> Result<()> {
    let bad = |name: &str| {
        name.is_empty()
            || name.contains('/')
            || name == "."
            || name == ".."
            || name.starts_with("__")
    };
    for tag in tags {
        ensure!(
            !matches!(tag.dim, 1 | 2) || !bad(&tag.name),
            "the tag '{}' cannot name an array of mesh/tags (empty, a '/', '.', '..' or a \
             leading '__')",
            tag.name
        );
    }
    for name in cell_values {
        ensure!(
            !bad(name) && name != "element",
            "the cell value '{name}' cannot name an array of cells ('element', empty, a '/', \
             '.', '..' or a leading '__')"
        );
    }
    Ok(())
}

/// A [`MeshError`] of a gather of the writer, with the context [`OnEveryProcess`] if every
/// process has it.
fn mesh_error(e: MeshError) -> anyhow::Error {
    let everywhere = e.is_collective();
    let e = anyhow::Error::new(e);
    if everywhere {
        e.context(CANNOT_WRITE)
    } else {
        e.context(CANNOT_WRITE.0)
    }
}

/// The outcome `local` of process 0 (`None` on the others) on every process: `Ok` with the
/// value of process 0 (`None` on the others), or the error of process 0 on every process, with
/// the context [`OnEveryProcess`]. Collective.
fn on_every_process<T>(comm: &Comm, local: Result<Option<T>>) -> Result<Option<T>> {
    let mut message = match &local {
        Err(e) => format!("{e:#}").into_bytes(),
        Ok(_) => Vec::new(),
    };
    let failed = comm.max_int(local.is_err().into())?;
    if failed == 0 {
        return local;
    }
    comm.broadcast_bytes(0, &mut message)?;
    Err(anyhow!(String::from_utf8_lossy(&message).into_owned()).context(CANNOT_WRITE))
}

/// Everything in the store `path` (`<out>/result.zarr`), for tests and tools: the root
/// attributes, checked ([`ResultMeta::check`]), and every array they name. Not collective.
///
/// Errors: the store cannot be read, or does not hold what its attributes name.
#[cfg(not(target_os = "wasi"))]
pub fn read_result(path: &Path) -> Result<ResultData> {
    let context = || format!("cannot read {}", path.display());
    let storage = Arc::new(FilesystemStore::new(path).with_context(context)?);
    let root = Group::open(storage.clone(), "/").with_context(context)?;
    let meta: ResultMeta =
        serde_json::from_value(Value::Object(root.attributes().clone())).with_context(context)?;
    meta.check().with_context(context)?;
    let t = Some(meta.times.len() as u64);
    let time: Vec<f64> = read_all(&storage, "/time", &[t])?;
    let vertices: Vec<f64> = read_all(&storage, "/mesh/vertices", &[None, Some(2)])?;
    let triangles: Vec<u32> = read_all(&storage, "/mesh/triangles", &[None, Some(3)])?;
    let n = Some(triangles.len() as u64 / 3);
    let element: Vec<u64> = read_all(&storage, "/mesh/element", &[n])?;
    let v = Some(vertices.len() as u64 / 2);
    let tags = meta
        .tags
        .iter()
        .map(|tag| {
            let path = format!("/mesh/tags/{}", tag.name);
            let shape: &[Option<u64>] = if tag.dim == 2 {
                &[None]
            } else {
                &[None, Some(2)]
            };
            let values: Vec<u32> = read_all(&storage, &path, shape)?;
            Ok(CanonicalTag {
                name: tag.name.clone(),
                dim: tag.dim as u8,
                triangles: if tag.dim == 2 {
                    values.clone()
                } else {
                    Vec::new()
                },
                edges: if tag.dim == 2 {
                    Vec::new()
                } else {
                    values.as_chunks::<2>().0.to_vec()
                },
            })
        })
        .collect::<Result<_>>()?;
    let fields = meta
        .fields
        .iter()
        .map(|f| {
            Ok((
                f.id.clone(),
                read_all(&storage, &format!("/fields/{}", f.id), &[t, v])?,
            ))
        })
        .collect::<Result<_>>()?;
    let cell_elements: Vec<u64> = read_all(&storage, "/cells/element", &[None])?;
    let c = Some(cell_elements.len() as u64);
    let cells = meta
        .cell_values
        .iter()
        .map(|name| {
            Ok((
                name.clone(),
                read_all(&storage, &format!("/cells/{name}"), &[t, c])?,
            ))
        })
        .collect::<Result<_>>()?;
    let rasters = match &meta.raster {
        Some(raster) => meta
            .fields
            .iter()
            .map(|f| {
                let levels = (0..raster.levels)
                    .map(|k| {
                        let [rows, cols] = raster.shapes[k as usize];
                        let shape = [t, Some(rows), Some(cols)];
                        read_all(&storage, &format!("/raster/{}/{k}", f.id), &shape)
                    })
                    .collect::<Result<_>>()?;
                Ok((f.id.clone(), levels))
            })
            .collect::<Result<_>>()?,
        None => Vec::new(),
    };
    Ok(ResultData {
        meta,
        time,
        mesh: CanonicalMesh {
            vertices: vertices.as_chunks::<2>().0.to_vec(),
            triangles: triangles.as_chunks::<3>().0.to_vec(),
            element,
            tags,
        },
        fields,
        cell_elements,
        cells,
        rasters,
    })
}

/// The values of the whole array `path` of `storage`, row by row; its shape is `shape`, an extent
/// of each dimension (`None`: any).
#[cfg(not(target_os = "wasi"))]
fn read_all<T: zarrs::array::ElementOwned>(
    storage: &Arc<FilesystemStore>,
    path: &str,
    shape: &[Option<u64>],
) -> Result<Vec<T>> {
    let array = Array::open(storage.clone(), path)
        .with_context(|| format!("cannot open the array {path}"))?;
    let found = array.shape();
    ensure!(
        found.len() == shape.len()
            && found
                .iter()
                .zip(shape)
                .all(|(n, s)| s.is_none_or(|s| s == *n)),
        "the array {path} has the shape {found:?}, not {}",
        shape_text(shape)
    );
    let values: Vec<T> = array
        .retrieve_array_subset(&array.subset_all())
        .with_context(|| format!("cannot read the array {path}"))?;
    Ok(values)
}

/// `shape` as text, `_` for any extent: `[4, _]`.
#[cfg(not(target_os = "wasi"))]
fn shape_text(shape: &[Option<u64>]) -> String {
    let extents: Vec<String> = shape
        .iter()
        .map(|s| s.map_or_else(|| "_".to_string(), |n| n.to_string()))
        .collect();
    format!("[{}]", extents.join(", "))
}

#[cfg(test)]
mod tests {
    use super::check_names;

    fn tag(name: &str, dim: u8) -> mesh::Tag {
        mesh::Tag {
            name: name.to_string(),
            dim,
        }
    }

    /// Tags and cell values name arrays: a name Zarr refuses, or the cell value `element`, is
    /// an error; a tag of points is not written, so its name does not matter.
    #[test]
    fn names_of_arrays_are_checked() {
        let values = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        check_names(
            &[tag("portal wall", 1), tag("x/y", 0)],
            &values(&["A", "_b"]),
        )
        .unwrap();
        for bad in ["a/b", ".", "..", "__x", ""] {
            let e = check_names(&[tag(bad, 2)], &[]).unwrap_err();
            assert!(e.to_string().contains(&format!("tag '{bad}'")), "{e}");
            let e = check_names(&[], &values(&[bad])).unwrap_err();
            assert!(
                e.to_string().contains(&format!("cell value '{bad}'")),
                "{e}"
            );
        }
        let e = check_names(&[], &values(&["element"])).unwrap_err();
        assert!(e.to_string().contains("'element'"), "{e}");
    }
}
