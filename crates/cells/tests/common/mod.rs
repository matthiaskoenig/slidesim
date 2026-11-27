//! The helpers of the tests of the cells: the cache of the compiled models, the test data, the
//! meshes and the specs of the test models.

#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::sync::Once;

use ceed::{Ceed, DEFAULT_RESOURCE};
use contracts::{
    CellParameter, CellSolverKind, CellsSpec, Coefficient, LengthUnit, MeshSource, SCENE_VERSION,
    Scene, SpeciesLink, Table, Units, VolumeLink,
};
use mesh::Mesh;
use pde::Discretization;
use petsc::Petsc;

/// Sets the cache of the compiled models to a directory of this test binary, once per process
/// (before any thread of the test reads the environment).
pub fn c_cache() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let dir = std::env::temp_dir().join("slidesim-cells-tests-c-cache");
        // SAFETY: called first in each test, before the test starts threads or reads the
        // environment elsewhere.
        unsafe { std::env::set_var("SBMLSIM_C_CACHE", dir) };
    });
}

pub fn testdata(path: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../testdata")
        .join(path)
}

pub fn model(file: &str) -> String {
    std::fs::read_to_string(testdata(&format!("models/{file}"))).unwrap()
}

pub fn mesh(p: &Petsc, file: &str, refinements: u32) -> Mesh {
    let scene = Scene {
        version: SCENE_VERSION.to_string(),
        dimension: 2,
        units: Units {
            length: LengthUnit::Um,
        },
        mesh: MeshSource::Gmsh {
            path: testdata(&format!("meshes/{file}")),
        },
        maps: Vec::new(),
    };
    Mesh::load(p, &scene, refinements).unwrap()
}

pub fn ceed() -> Ceed {
    Ceed::new(DEFAULT_RESOURCE).unwrap()
}

pub fn table(map: &str, points: &[[f64; 2]]) -> Coefficient {
    Coefficient::Table(Table {
        map: map.to_string(),
        points: points.to_vec(),
    })
}

pub fn parameter(id: &str, value: Coefficient) -> CellParameter {
    CellParameter {
        id: id.to_string(),
        value,
        replace: None,
    }
}

/// The cells of `linear_uptake.xml` in the area `tissue`, with `parameters`.
pub fn linear_spec(parameters: Vec<CellParameter>) -> CellsSpec {
    CellsSpec {
        model: PathBuf::from("linear_uptake.xml"),
        solver: CellSolverKind::Engine,
        time_unit_s: 1.0,
        areas: Some(vec!["tissue".to_string()]),
        species: vec![SpeciesLink {
            species: "A_ext".to_string(),
            field: "a".to_string(),
        }],
        volumes: VolumeLink {
            blood: "Vext".to_string(),
            cells: "Vli".to_string(),
        },
        parameters,
        split_step: 1.0,
        rtol: 1e-8,
        atol: 1e-12,
        outputs: vec!["uptake_rate".to_string()],
        balance: true,
    }
}

/// The discretization of the finest level of `mesh`, as the fields have it.
pub fn finest(mesh: &Mesh, ceed: &Ceed) -> Discretization {
    Discretization::new(mesh, mesh.levels().len() - 1, ceed, 1).unwrap()
}

/// The cells of the SPT model (`spt_liver_cell.xml`) on every area, of the solver `solver`: the
/// pattern 7, the position and the protein from the map `zonation`, the rule target `protein`
/// as an output.
pub fn spt_spec(solver: CellSolverKind) -> CellsSpec {
    CellsSpec {
        model: PathBuf::from("spt_liver_cell.xml"),
        solver,
        time_unit_s: 60.0,
        areas: None,
        species: ["S", "P"]
            .map(|s| SpeciesLink {
                species: format!("{s}_ext"),
                field: s.to_string(),
            })
            .to_vec(),
        volumes: VolumeLink {
            blood: "Vext".to_string(),
            cells: "Vli".to_string(),
        },
        parameters: vec![
            parameter("pattern", Coefficient::Constant(7.0)),
            parameter("position", table("zonation", &[[0.0, 0.0], [1.0, 1.0]])),
            parameter("protein_ext", table("zonation", &[[0.0, 0.3], [1.0, 1.0]])),
        ],
        split_step: 60.0,
        rtol: 1e-6,
        atol: 1e-9,
        outputs: vec!["protein".to_string()],
        balance: true,
    }
}

/// `linear_uptake.xml` with an oscillator of the angular frequency `w` (1/s, default 1):
/// `x' = w y`, `y' = -w x` from `x = 1`, `y = 0`. The tolerances set the steps per period, so the
/// step attempts of a cell over an interval grow in proportion to `w`: the parameter `w` sets the
/// work of a cell.
pub fn oscillator_model() -> String {
    let math =
        |body: &str| format!(r#"<math xmlns="http://www.w3.org/1998/Math/MathML">{body}</math>"#);
    let rules = format!(
        r#"<listOfRules><rateRule variable="x">{}</rateRule><rateRule variable="y">{}</rateRule>"#,
        math("<apply><times/><ci> w </ci><ci> y </ci></apply>"),
        math("<apply><minus/><apply><times/><ci> w </ci><ci> x </ci></apply></apply>"),
    );
    model("linear_uptake.xml")
        .replace("<listOfRules>", &rules)
        .replace(
            "</listOfParameters>",
            r#"<parameter id="w" value="1" constant="true"/><parameter id="x" value="1" constant="false"/><parameter id="y" value="0" constant="false"/></listOfParameters>"#,
        )
}
