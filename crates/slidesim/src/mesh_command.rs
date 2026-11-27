//! `slidesim mesh`: the mesh of a scene, to look at and to reuse.
//!
//! Makes the 2D mesh of a `scene/1` on one process ([`mesh::mesh_serial`]: a gmsh file read
//! and converted, or the polygons or lobules meshed) and writes into the output directory:
//!
//! - `mesh.json`: the [`TaggedMesh`] as JSON (`tags`, `vertices`, `triangles`,
//!   `triangle_tags`, `edges`, `edge_tags`), the coarse mesh a run makes of the scene;
//! - `mesh.vtu`: the mesh for ParaView, one cell array per area tag, named after it, with the
//!   value of the tag in its triangles and NaN elsewhere;
//! - `quality.json`: its [`Quality`]: the numbers of triangles and vertices, the smallest angle
//!   in degrees, the smallest and mean SICN and, for a generated mesh, the smallest and largest
//!   ratio of an edge length to the size field.
//!
//! The quality is printed in one line ([`quality_line`]), as a run prints it for a generated
//! mesh.

use std::path::Path;

use anyhow::{Context, Result};
use contracts::Scene;
use mesh::Mesh;
use mesher::{Quality, TaggedMesh};
use petsc::Petsc;

use crate::coupled::CellWriter;

/// The quality of a mesh in one line.
pub(crate) fn quality_line(q: &Quality) -> String {
    let mut line = format!(
        "mesh: {} triangles, {} vertices, min angle {:.2} deg, SICN min {:.3} mean {:.3}",
        q.triangles, q.vertices, q.min_angle_deg, q.min_sicn, q.mean_sicn
    );
    if let Some([lo, hi]) = q.edge_ratio {
        line.push_str(&format!(", edge length / size {lo:.2} to {hi:.2}"));
    }
    line
}

/// Writes the mesh of the scene `scene_path` into the directory `out` (created if missing; see
/// the module documentation) and prints its quality. On one process.
pub(crate) fn mesh(petsc: &Petsc, scene_path: &Path, out: &Path) -> Result<()> {
    let scene = Scene::load(scene_path).context("cannot load the scene")?;
    let (tagged, quality) = mesh::mesh_serial(petsc, &scene).context("cannot make the mesh")?;
    std::fs::create_dir_all(out)
        .with_context(|| format!("cannot create the output directory {}", out.display()))?;
    let write = |name: &str, text: String| {
        let path = out.join(name);
        std::fs::write(&path, text).with_context(|| format!("cannot write {}", path.display()))
    };
    write("mesh.json", serde_json::to_string(&tagged)? + "\n")?;
    write(
        "quality.json",
        serde_json::to_string_pretty(&quality)? + "\n",
    )?;
    let path = out.join("mesh.vtu");
    write_areas(petsc, &tagged, &path)
        .with_context(|| format!("cannot write {}", path.display()))?;
    println!("{}", quality_line(&quality));
    Ok(())
}

/// Writes `tagged` as the VTU file `path`: the coarse mesh a run makes of it, one cell array per
/// area tag with the value of the tag in its triangles, NaN elsewhere.
fn write_areas(petsc: &Petsc, tagged: &TaggedMesh, path: &Path) -> Result<()> {
    let mesh = Mesh::from_tagged(petsc, Some(tagged), 0)?;
    let coarse = &mesh.levels()[0];
    let names: Vec<String> = mesh
        .tags()
        .iter()
        .filter(|t| t.dim == 2)
        .map(|t| t.name.clone())
        .collect();
    let cells = coarse.height_stratum(0)?;
    let mut table = vec![f64::NAN; cells.len() * names.len()];
    for (k, name) in names.iter().enumerate() {
        for value in coarse.label_values(name)? {
            for p in coarse.stratum_points(name, value)? {
                if cells.contains(&p) {
                    table[(p - cells.start) as usize * names.len() + k] = value as f64;
                }
            }
        }
    }
    let local: Vec<usize> = (0..cells.len()).collect();
    CellWriter::new(coarse, names)?.write(&local, &table, path)
}
