//! What several test binaries of `pde` share: the unit square mesh, the libCEED context, and
//! test maps over the unit square. Each binary uses some of it.
//!
//! The maps are test Zarr arrays over `unit_square.msh` with pixels of [`PIXEL`] um, the centre
//! of pixel (row 0, col 0) at ([`ORIGIN`], [`ORIGIN`]): [`PIXELS`] pixels cover [-0.05, 1.05] in
//! each direction. Every process writes its own copy into a directory of its own.

#![allow(dead_code)]

use std::path::{Path, PathBuf};

use ceed::{Ceed, DEFAULT_RESOURCE};
use contracts::{LengthUnit, MapSpec, MeshSource, SCENE_VERSION, Scene, Units};
use maps::Maps;
use mesh::Mesh;
use petsc::Petsc;

/// The pixel size of the maps, in um.
pub const PIXEL: f64 = 0.01;
/// The centre of pixel (row 0, col 0), in x and in y.
pub const ORIGIN: f64 = -0.05;
/// The pixels which cover [-0.05, 1.05].
pub const PIXELS: usize = 111;

pub fn fixture(file: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../testdata/meshes")
        .join(file)
}

/// `unit_square.msh` (the square [0, 1] x [0, 1] um, area `domain`, edges `left`, `right`,
/// `top`, `bottom`), refined `refinements` times.
pub fn unit_square(p: &Petsc, refinements: u32) -> Mesh {
    fixture_mesh(p, "unit_square.msh", refinements)
}

/// The mesh `file` of `testdata/meshes`, refined `refinements` times.
pub fn fixture_mesh(p: &Petsc, file: &str, refinements: u32) -> Mesh {
    let scene = Scene {
        version: SCENE_VERSION.to_string(),
        dimension: 2,
        units: Units {
            length: LengthUnit::Um,
        },
        mesh: MeshSource::Gmsh {
            path: fixture(file),
        },
        maps: Vec::new(),
    };
    Mesh::load(p, &scene, refinements).unwrap()
}

pub fn ceed() -> Ceed {
    Ceed::new(DEFAULT_RESOURCE).unwrap()
}

/// A directory of this process for the maps of a test, removed when dropped.
pub struct Scratch(PathBuf);

impl Scratch {
    pub fn new(test: &str) -> Scratch {
        let dir = std::env::temp_dir().join(format!("slidesim-pde-{test}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        Scratch(dir)
    }

    /// The map `id` of `f` at the pixel centres, `PIXELS` rows (y) and `cols` columns (x) from
    /// the origin.
    pub fn map(&self, id: &str, cols: usize, f: impl Fn([f64; 2]) -> f64) -> MapSpec {
        self.map_of_shape(id, PIXELS, cols, f)
    }

    /// The map `id` of `f` at the pixel centres, `rows` rows (y) and `cols` columns (x) from
    /// the origin.
    pub fn map_of_shape(
        &self,
        id: &str,
        rows: usize,
        cols: usize,
        f: impl Fn([f64; 2]) -> f64,
    ) -> MapSpec {
        self.map_on_grid(id, [rows, cols], PIXEL, [ORIGIN, ORIGIN], f)
    }

    /// The map `id` of `f` at the pixel centres of `rows` rows (y) and `cols` columns (x) of
    /// `pixel` um, the centre of pixel (row 0, col 0) at `origin` (`[x, y]`, um).
    pub fn map_on_grid(
        &self,
        id: &str,
        [rows, cols]: [usize; 2],
        pixel: f64,
        origin: [f64; 2],
        f: impl Fn([f64; 2]) -> f64,
    ) -> MapSpec {
        let values: Vec<f64> = (0..rows)
            .flat_map(|r| (0..cols).map(move |c| (r, c)))
            .map(|(r, c)| f([origin[0] + c as f64 * pixel, origin[1] + r as f64 * pixel]))
            .collect();
        let path = self.0.join(format!("{id}.zarr"));
        let mut spec = maps::write_test_map(&path, &values, rows, cols, pixel, origin);
        spec.id = id.to_string();
        spec
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// The maps `specs`, read for the cells of this process.
pub fn read_maps(specs: &[MapSpec], mesh: &Mesh) -> Maps {
    let ids: Vec<&str> = specs.iter().map(|s| s.id.as_str()).collect();
    Maps::read(
        specs,
        &ids,
        mesh.local_bounds().expect("cells on every process"),
    )
    .unwrap()
}
