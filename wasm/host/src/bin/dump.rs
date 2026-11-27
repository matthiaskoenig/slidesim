//! Runs inputs in `slidesim.wasm` with [`slidesim_host::run`] and writes every array it gives
//! into a directory, for a comparison with another host (the Node tests of `wasm/js` compare the
//! JavaScript glue with this host bit for bit).
//!
//! Usage: `dump <slidesim.wasm> <kernel dir> <mathlib.wasm> <inputs.json> <out dir>`
//!
//! `<inputs.json>` is the JSON `{"scene", "study", "mesh"?}` of [`slidesim_host::run`]. The out
//! directory gets the arrays of `result/1` under their names, as little endian bytes (`time`,
//! `mesh/vertices`, `mesh/triangles`, `mesh/element`, `mesh/tags/<name>`, `cells/element`), each
//! output time `k` in `outputs/<k>/` (`time`, the f64 of `slidesim_time`, `fields/<id>`,
//! `cells/<value>`, `raster/<field>/<level>` (f32), `meta.json`), and `meta.json`, `summary.json`, `quality.json` and `log.txt`
//! of the end. An existing out directory is replaced.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use slidesim_host::run;

fn main() -> Result<()> {
    let args: Vec<PathBuf> = std::env::args_os().skip(1).map(PathBuf::from).collect();
    let [slidesim, kernel, mathlib, inputs, out] = args.as_slice() else {
        bail!("usage: dump <slidesim.wasm> <kernel dir> <mathlib.wasm> <inputs.json> <out dir>");
    };
    let text = std::fs::read_to_string(inputs)
        .with_context(|| format!("cannot read {}", inputs.display()))?;
    let outputs = run(slidesim, kernel, mathlib, &text, None)?;

    if out.exists() {
        std::fs::remove_dir_all(out).with_context(|| format!("cannot remove {}", out.display()))?;
    }
    let write = |path: PathBuf, bytes: &[u8]| -> Result<()> {
        std::fs::create_dir_all(path.parent().expect("a file in the out directory"))?;
        std::fs::write(&path, bytes).with_context(|| format!("cannot write {}", path.display()))
    };
    let f64s = |values: &[f64]| {
        values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>()
    };
    let u32s = |values: &[u32]| {
        values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>()
    };
    let u64s = |values: &[u64]| {
        values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>()
    };
    let at = |name: &str| -> PathBuf { Path::new(out).join(name) };

    write(at("time"), &f64s(&outputs.time))?;
    write(at("mesh/vertices"), &f64s(&outputs.mesh.vertices))?;
    write(at("mesh/triangles"), &u32s(&outputs.mesh.triangles))?;
    write(at("mesh/element"), &u64s(&outputs.mesh.element))?;
    for (name, values) in &outputs.mesh.tags {
        write(at(&format!("mesh/tags/{name}")), &u32s(values))?;
    }
    write(at("cells/element"), &u64s(&outputs.cell_elements))?;
    for output in &outputs.outputs {
        let dir = format!("outputs/{}", output.index);
        write(at(&format!("{dir}/time")), &output.time.to_le_bytes())?;
        write(
            at(&format!("{dir}/meta.json")),
            output.meta.to_string().as_bytes(),
        )?;
        for (id, values) in &output.fields {
            write(at(&format!("{dir}/fields/{id}")), &f64s(values))?;
        }
        for (name, values) in &output.cells {
            write(at(&format!("{dir}/cells/{name}")), &f64s(values))?;
        }
        for (id, levels) in &output.rasters {
            for (level, values) in levels.iter().enumerate() {
                let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
                write(at(&format!("{dir}/raster/{id}/{level}")), &bytes)?;
            }
        }
    }
    write(at("meta.json"), outputs.meta.to_string().as_bytes())?;
    write(at("summary.json"), outputs.summary.to_string().as_bytes())?;
    write(at("quality.json"), outputs.quality.to_string().as_bytes())?;
    write(at("log.txt"), outputs.log.as_bytes())?;
    Ok(())
}
