//! Tests of the cells on the coarse elements of a mesh: their parameters from the maps
//! ([`CellSetup`]) and their models on sbmlsim-core's engine ([`EngineCells`]). nextest runs
//! each test in its own process, so every test initializes PETSc once.
//! `parallel_on_2_processes` and `parallel_on_3_processes` run some of them again under
//! `mpiexec`.
//!
//! The compiled models go to a cache directory of this test binary in the temporary directory
//! (`SBMLSIM_C_CACHE`), not to the cache of the user.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Once};

use ceed::{Ceed, DEFAULT_RESOURCE};
use cells::{CellSetup, CellSolver, CellsError, EngineCells};
use contracts::{
    CellParameter, CellSolverKind, CellsSpec, Coefficient, LengthUnit, MapSpec, MeshFormat,
    MeshRef, Replace, SCENE_VERSION, Scene, SpeciesLink, Study, Table, Units, VolumeLink,
};
use maps::Maps;
use mesh::Mesh;
use pde::Discretization;
use petsc::Petsc;
use sbmlsim_abi::{IntegratorSettings, Tolerance};
use sbmlsim_engine::Stepper;

/// The pixel size of the test maps over `two_areas.msh` (um).
const PIXEL: f64 = 0.01;

/// Sets the cache of the compiled models to a directory of this test binary, once per process
/// (before any thread of the test reads the environment).
fn c_cache() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let dir = std::env::temp_dir().join("slidesim-cells-tests-c-cache");
        // SAFETY: called first in each test, before the test starts threads or reads the
        // environment elsewhere.
        unsafe { std::env::set_var("SBMLSIM_C_CACHE", dir) };
    });
}

fn testdata(path: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../testdata")
        .join(path)
}

fn model(file: &str) -> String {
    std::fs::read_to_string(testdata(&format!("models/{file}"))).unwrap()
}

fn mesh(p: &Petsc, file: &str, refinements: u32) -> Mesh {
    let scene = Scene {
        version: SCENE_VERSION.to_string(),
        dimension: 2,
        units: Units {
            length: LengthUnit::Um,
        },
        mesh: MeshRef {
            format: MeshFormat::Gmsh,
            path: testdata(&format!("meshes/{file}")),
        },
        maps: Vec::new(),
    };
    Mesh::load(p, &scene, refinements).unwrap()
}

fn ceed() -> Ceed {
    Ceed::new(DEFAULT_RESOURCE).unwrap()
}

/// A directory of this process for the maps of a test, removed when dropped.
struct Scratch(PathBuf);

impl Scratch {
    fn new(test: &str) -> Scratch {
        let dir =
            std::env::temp_dir().join(format!("slidesim-cells-{test}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        Scratch(dir)
    }

    /// The map `id` of `f` over `two_areas.msh` (the rectangle [0, 2] x [0, 1] um) and a margin,
    /// at the pixel centres.
    fn map(&self, id: &str, f: impl Fn([f64; 2]) -> f64) -> MapSpec {
        let origin = [-0.05, -0.05];
        let (rows, cols) = (111, 211);
        let values: Vec<f64> = (0..rows)
            .flat_map(|r| (0..cols).map(move |c| (r, c)))
            .map(|(r, c)| f([origin[0] + c as f64 * PIXEL, origin[1] + r as f64 * PIXEL]))
            .collect();
        let path = self.0.join(format!("{id}.zarr"));
        let mut spec = maps::write_test_map(&path, &values, rows, cols, PIXEL, origin);
        spec.id = id.to_string();
        spec
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn read_maps(specs: &[MapSpec], mesh: &Mesh) -> Maps {
    let ids: Vec<&str> = specs.iter().map(|s| s.id.as_str()).collect();
    Maps::read(specs, &ids, mesh.local_bounds().unwrap()).unwrap()
}

fn table(map: &str, points: &[[f64; 2]]) -> Coefficient {
    Coefficient::Table(Table {
        map: map.to_string(),
        points: points.to_vec(),
    })
}

fn parameter(id: &str, value: Coefficient) -> CellParameter {
    CellParameter {
        id: id.to_string(),
        value,
        replace: None,
    }
}

/// The cells of `linear_uptake.xml` in the area `tissue`, with `parameters`.
fn linear_spec(parameters: Vec<CellParameter>) -> CellsSpec {
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
    }
}

/// The value of parameter `id` of cell `i` of `setup`.
fn value(setup: &CellSetup, i: usize, id: &str) -> f64 {
    setup.parameters[i]
        .iter()
        .find(|(p, _)| p == id)
        .unwrap_or_else(|| panic!("no parameter {id} in {:?}", setup.parameters[i]))
        .1
}

/// The discretization of the finest level of `mesh`, as the fields have it.
fn finest(mesh: &Mesh, ceed: &Ceed) -> Discretization {
    Discretization::new(mesh, mesh.levels().len() - 1, ceed, 1).unwrap()
}

/// The local coarse cells of `mesh` in the area `tissue`, by the area tags of level 0.
fn tissue_cells(mesh: &Mesh, ceed: &Ceed) -> Vec<usize> {
    let coarse = Discretization::new(mesh, 0, ceed, 1).unwrap();
    let n = mesh.levels()[0].height_stratum(0).unwrap().len();
    (0..n)
        .filter(|&c| coarse.cell_area_tag(c).as_deref() == Some("tissue"))
        .collect()
}

/// The weighted mean of `f` over the quadrature points of the finest cells in each coarse cell
/// of `mesh` (`weights` and `points` of the finest level).
fn child_means(mesh: &Mesh, fine: &Discretization, f: impl Fn([f64; 2]) -> f64) -> Vec<f64> {
    let n = mesh.levels()[0].height_stratum(0).unwrap().len();
    let finest = mesh.levels().len() - 1;
    let q = fine.quadrature_points().len() / mesh.finest().height_stratum(0).unwrap().len();
    let mut sums = vec![[0.0; 2]; n];
    for (k, (x, w)) in fine
        .quadrature_points()
        .iter()
        .zip(fine.quadrature_weights())
        .enumerate()
    {
        let c = mesh.coarse_cell(finest, k / q);
        sums[c][0] += w * f(*x);
        sums[c][1] += w;
    }
    sums.iter().map(|[s, w]| s / w).collect()
}

/// The centroids of the local coarse cells of `mesh`, from the quadrature of level 0.
fn centroids(mesh: &Mesh, ceed: &Ceed) -> Vec<[f64; 2]> {
    let coarse = Discretization::new(mesh, 0, ceed, 1).unwrap();
    let x = child_means_on(&coarse, |x| x[0]);
    let y = child_means_on(&coarse, |x| x[1]);
    x.into_iter().zip(y).map(|(x, y)| [x, y]).collect()
}

/// The weighted mean of `f` over the quadrature points of each local cell of `disc`.
fn child_means_on(disc: &Discretization, f: impl Fn([f64; 2]) -> f64) -> Vec<f64> {
    let n = disc.dm().height_stratum(0).unwrap().len();
    let q = disc.quadrature_points().len() / n;
    let pairs: Vec<(f64, f64)> = disc
        .quadrature_points()
        .iter()
        .zip(disc.quadrature_weights())
        .map(|(x, w)| (w * f(*x), *w))
        .collect();
    pairs
        .chunks_exact(q)
        .map(|c| c.iter().map(|p| p.0).sum::<f64>() / c.iter().map(|p| p.1).sum::<f64>())
        .collect()
}

/// A parameter from a map of `x` over a mesh of two areas, the model in one of them: the cells
/// are the coarse elements of that area, each parameter the mean of `x` over the quadrature
/// points of its finest children (the centroid's `x`, for a linear map), the volumes the mean
/// porosity and its complement.
#[test]
fn parameters_are_area_means() {
    c_cache();
    let p = Petsc::init(&[]).unwrap();
    let mesh = mesh(&p, "two_areas.msh", 2);
    let ceed = ceed();
    let fine = finest(&mesh, &ceed);
    let scratch = Scratch::new("area-means");
    let maps = read_maps(&[scratch.map("x", |x| x[0])], &mesh);
    let spec = linear_spec(vec![parameter("k", table("x", &[[0.0, 0.0], [2.0, 2.0]]))]);
    let porosity = table("x", &[[0.0, 0.1], [2.0, 0.3]]);
    let setup = CellSetup::new(&fine, &mesh, &spec, &maps, &porosity).unwrap();

    let expected_cells = tissue_cells(&mesh, &ceed);
    assert_eq!(setup.cells, expected_cells);
    let total = p
        .comm()
        .sum_int(setup.cells.len().try_into().unwrap())
        .unwrap();
    assert!(total > 0);
    let means = child_means(&mesh, &fine, |x| x[0]);
    let centroids = centroids(&mesh, &ceed);
    let numbering = mesh.levels()[0].cell_numbering().unwrap();
    assert_eq!(setup.parameters.len(), setup.cells.len());
    for (i, &c) in setup.cells.iter().enumerate() {
        let k = value(&setup, i, "k");
        assert!(
            (k - means[c]).abs() <= 1e-12,
            "cell {c}: {k} against {}",
            means[c]
        );
        assert!((k - centroids[c][0]).abs() <= 1e-12, "cell {c}: {k}");
        assert!(k < 1.0 + 1e-12, "cell {c} at x = {k} is not in the tissue");
        let porosity = 0.1 + 0.1 * centroids[c][0];
        assert!((value(&setup, i, "Vext") - porosity).abs() <= 1e-12);
        assert!((value(&setup, i, "Vli") - (1.0 - porosity)).abs() <= 1e-12);
        assert_eq!(setup.parameters[i].len(), 3, "{:?}", setup.parameters[i]);
        assert_eq!(setup.global[i], i64::from(numbering[c]));
        for axis in 0..2 {
            assert!((setup.centroids[i][axis] - centroids[c][axis]).abs() <= 1e-12);
        }
    }
}

/// Where the mean of the map of a replacement rule is above its threshold, the parameter is
/// the mean of the replacement value instead.
#[test]
fn replace_rule_switches_value() {
    c_cache();
    let p = Petsc::init(&[]).unwrap();
    let mesh = mesh(&p, "two_areas.msh", 1);
    let ceed = ceed();
    let fine = finest(&mesh, &ceed);
    let scratch = Scratch::new("replace");
    let flag = |x: [f64; 2]| if x[1] > 0.5 { 1.0 } else { 0.0 };
    let maps = read_maps(
        &[scratch.map("x", |x| x[0]), scratch.map("flag", flag)],
        &mesh,
    );
    let spec = linear_spec(vec![CellParameter {
        id: "k".to_string(),
        value: Coefficient::Constant(1.0),
        replace: Some(Replace {
            map: "flag".to_string(),
            above: 0.5,
            value: table("x", &[[0.0, 10.0], [2.0, 12.0]]),
        }),
    }]);
    let setup = CellSetup::new(&fine, &mesh, &spec, &maps, &Coefficient::Constant(0.2)).unwrap();
    let flag_map = maps.get("flag").unwrap();
    let flags = child_means(&mesh, &fine, |x| flag_map.sample(x).unwrap());
    let centroids = centroids(&mesh, &ceed);
    let mut replaced = 0;
    for (i, &c) in setup.cells.iter().enumerate() {
        let k = value(&setup, i, "k");
        if flags[c] > 0.5 {
            replaced += 1;
            let expected = 10.0 + centroids[c][0];
            assert!(
                (k - expected).abs() <= 1e-12,
                "cell {c}: {k} against {expected}"
            );
        } else {
            assert_eq!(k, 1.0, "cell {c}");
        }
        assert_eq!(value(&setup, i, "Vext"), 0.2);
    }
    // Both: the areas are split along x, the flag along y.
    let comm = p.comm();
    let replaced = comm.sum_int(replaced).unwrap();
    let total = comm.sum_int(setup.cells.len().try_into().unwrap()).unwrap();
    assert!(replaced > 0 && replaced < total, "{replaced} of {total}");
}

/// A no-value pixel in the tissue of a cell is an error on every process, naming the map and
/// the element; in the area without cells it is not.
#[test]
fn no_value_in_tissue_is_an_error() {
    c_cache();
    let p = Petsc::init(&[]).unwrap();
    let mesh = mesh(&p, "two_areas.msh", 1);
    let ceed = ceed();
    let fine = finest(&mesh, &ceed);
    let scratch = Scratch::new("no-value");
    // No value in the lumen (x > 1): it holds no cells.
    let lumen = |x: [f64; 2]| if x[0] > 1.05 { f64::NAN } else { x[0] };
    let hole = |x: [f64; 2]| {
        if (x[0] - 0.5).powi(2) + (x[1] - 0.5).powi(2) < 0.01 {
            f64::NAN
        } else {
            lumen(x)
        }
    };
    let maps = read_maps(
        &[scratch.map("lumen", lumen), scratch.map("hole", hole)],
        &mesh,
    );
    let porosity = Coefficient::Constant(0.2);
    let spec = linear_spec(vec![parameter(
        "k",
        table("lumen", &[[0.0, 0.0], [1.0, 1.0]]),
    )]);
    CellSetup::new(&fine, &mesh, &spec, &maps, &porosity).unwrap();
    let spec = linear_spec(vec![parameter(
        "k",
        table("hole", &[[0.0, 0.0], [1.0, 1.0]]),
    )]);
    match CellSetup::new(&fine, &mesh, &spec, &maps, &porosity) {
        Err(CellsError::Element {
            cell,
            centroid,
            message,
        }) => {
            assert!(cell >= 0);
            let distance = (centroid[0] - 0.5).hypot(centroid[1] - 0.5);
            assert!(distance < 0.1 + 0.25, "{centroid:?}");
            assert!(message.contains("map 'hole'"), "{message}");
            assert!(message.contains("'k'"), "{message}");
        }
        other => panic!("expected an error naming the element, got {other:?}"),
    }
}

/// The checks at study start: unknown areas, ids which the model does not have, tolerances.
#[test]
fn study_start_checks() {
    c_cache();
    let p = Petsc::init(&[]).unwrap();
    let mesh = mesh(&p, "two_areas.msh", 1);
    let ceed = ceed();
    let fine = finest(&mesh, &ceed);
    let maps = Maps::default();
    let porosity = Coefficient::Constant(0.2);
    let sbml = model("linear_uptake.xml");
    let base = linear_spec(vec![parameter("k", Coefficient::Constant(0.5))]);
    let setup = CellSetup::new(&fine, &mesh, &base, &maps, &porosity).unwrap();
    // The default: every area tag.
    let all = CellsSpec {
        areas: None,
        ..base.clone()
    };
    let n = mesh.levels()[0].height_stratum(0).unwrap().len();
    assert_eq!(
        CellSetup::new(&fine, &mesh, &all, &maps, &porosity)
            .unwrap()
            .cells
            .len(),
        n
    );
    let unknown_area = CellsSpec {
        areas: Some(vec!["tissue".into(), "stroma".into()]),
        ..base.clone()
    };
    let e = CellSetup::new(&fine, &mesh, &unknown_area, &maps, &porosity).unwrap_err();
    assert!(e.to_string().contains("'stroma'"), "{e}");
    assert!(e.to_string().contains("lumen, tissue"), "{e}");

    // States and assignment rule targets are outputs.
    let outputs = CellsSpec {
        outputs: vec!["A".into(), "uptake_rate".into()],
        ..base.clone()
    };
    EngineCells::new(&setup, &outputs, &sbml).unwrap();
    let cases: Vec<(CellsSpec, &str)> = vec![
        (
            CellsSpec {
                outputs: vec!["k".into()],
                ..base.clone()
            },
            "'k'",
        ),
        (
            CellsSpec {
                species: vec![SpeciesLink {
                    species: "B_ext".into(),
                    field: "a".into(),
                }],
                ..base.clone()
            },
            "'B_ext'",
        ),
        (
            CellsSpec {
                species: vec![SpeciesLink {
                    species: "A".into(),
                    field: "a".into(),
                }],
                ..base.clone()
            },
            "'A'",
        ),
        (
            CellsSpec {
                volumes: VolumeLink {
                    blood: "Vb".into(),
                    cells: "Vli".into(),
                },
                ..base.clone()
            },
            "'Vb'",
        ),
        (
            CellsSpec {
                volumes: VolumeLink {
                    blood: "Vext".into(),
                    cells: "k".into(),
                },
                ..base.clone()
            },
            "'k'",
        ),
        (
            CellsSpec {
                parameters: vec![parameter("kk", Coefficient::Constant(1.0))],
                ..base.clone()
            },
            "'kk'",
        ),
        (
            CellsSpec {
                rtol: 0.0,
                ..base.clone()
            },
            "rtol",
        ),
        (
            CellsSpec {
                atol: -1.0,
                ..base.clone()
            },
            "atol",
        ),
    ];
    for (spec, named) in cases {
        let setup = CellSetup::new(&fine, &mesh, &spec, &maps, &porosity).unwrap();
        let e = EngineCells::new(&setup, &spec, &sbml).unwrap_err();
        assert!(e.to_string().contains(named), "{named}: {e}");
    }
}

/// The SPT model on 10 coarse elements of `lobule_toy`, the blood held fixed: each cell is the
/// standalone stepper of its parameters (the same code, so the same values).
#[test]
fn engine_cells_match_stepper() {
    c_cache();
    let p = Petsc::init(&[]).unwrap();
    let mesh = mesh(&p, "lobule_toy.msh", 0);
    let ceed = ceed();
    let fine = finest(&mesh, &ceed);
    let study = Study::load(testdata("studies/lobule_toy/study.json")).unwrap();
    let scene = Scene::load(&study.scene).unwrap();
    let maps = Maps::read(&scene.maps, &["zonation"], mesh.local_bounds().unwrap()).unwrap();
    let spec = CellsSpec {
        model: PathBuf::from("spt_liver_cell.xml"),
        solver: CellSolverKind::Engine,
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
    };
    let porosity = Coefficient::Constant(0.15);
    let mut setup = CellSetup::new(&fine, &mesh, &spec, &maps, &porosity).unwrap();
    let n = 10;
    assert!(setup.cells.len() >= n);
    setup.cells.truncate(n);
    setup.parameters.truncate(n);
    setup.global.truncate(n);
    setup.centroids.truncate(n);
    let sbml = model("spt_liver_cell.xml");
    let mut cells = EngineCells::new(&setup, &spec, &sbml).unwrap();
    assert_eq!(cells.n_cells(), n);
    assert_eq!(cells.species(), ["S_ext", "P_ext"]);

    let source = sbmlsim_c::generate(&sbml).unwrap();
    let library = sbmlsim_c::Library::build(&source.code).unwrap();
    let model = sbmlsim_c::CModel::new(library, &source.description).unwrap();
    let description = Arc::new(source.description);
    let settings = IntegratorSettings {
        relative_tolerance: spec.rtol,
        absolute_tolerance: Tolerance::Scalar(spec.atol),
        ..IntegratorSettings::default()
    };
    let ids: Vec<String> = description.states.iter().map(|s| s.id.clone()).collect();
    assert_eq!(cells.state_ids(), ids.as_slice());
    let ns = ids.len();
    let state = |id: &str| ids.iter().position(|s| s == id).unwrap();
    let vext = description
        .parameters
        .iter()
        .position(|p| p.id == "Vext")
        .unwrap();
    let mut steppers: Vec<Stepper> = setup
        .parameters
        .iter()
        .map(|changes| {
            Stepper::new(
                Box::new(model.clone()),
                Arc::clone(&description),
                settings.clone(),
                changes,
            )
            .unwrap()
        })
        .collect();
    let fixed: Vec<f64> = (0..n).flat_map(|i| [0.5 + 0.4 * i as f64, 0.1]).collect();
    let mut blood = vec![0.0; 2 * n];
    let mut steps = 0;
    for step in 0..6 {
        let (t0, t1) = (600.0 * step as f64, 600.0 * (step + 1) as f64);
        blood.copy_from_slice(&fixed);
        let stats = cells.advance(t0, t1, &mut blood).unwrap();
        steps += stats.steps;
        assert!(stats.seconds >= 0.0);
        for (i, stepper) in steppers.iter_mut().enumerate() {
            let volume = stepper.parameters()[vext];
            let mut y = stepper.state().to_vec();
            y[state("S_ext")] = fixed[2 * i] * volume;
            y[state("P_ext")] = fixed[2 * i + 1] * volume;
            stepper.advance(&y, t0 / 60.0, t1 / 60.0).unwrap();
            let expected = stepper.state();
            let actual = &cells.states()[i * ns..(i + 1) * ns];
            for (k, (a, e)) in actual.iter().zip(expected).enumerate() {
                assert!(
                    (a - e).abs() <= 1e-12 * e.abs() + 1e-300,
                    "cell {i}, {} at {t1} s: {a} against {e}",
                    ids[k]
                );
            }
            for (k, species) in ["S_ext", "P_ext"].iter().enumerate() {
                let e = expected[state(species)] / volume;
                let a = blood[2 * i + k];
                assert!((a - e).abs() <= 1e-12 * e.abs(), "{a} against {e}");
            }
        }
    }
    assert!(steps > 0);
    // The outputs: the states, species as concentrations, then the rule target protein, which
    // pattern 7 sets to protein_ext.
    let values = cells.output_values();
    let names: Vec<&str> = values.iter().map(|v| v.id.as_str()).collect();
    let mut expected_names: Vec<&str> = ids.iter().map(String::as_str).collect();
    expected_names.push("protein");
    assert_eq!(names, expected_names);
    for (v, compartment) in [("S_ext", "Vext"), ("S", "Vli"), ("T", "Vli")] {
        let value = &values[names.iter().position(|n| *n == v).unwrap()];
        assert_eq!(value.compartment.as_deref(), Some(compartment), "{v}");
    }
    assert_eq!(values.last().unwrap().compartment, None);
    let mut outputs = Vec::new();
    cells.outputs(&mut outputs);
    let no = names.len();
    assert_eq!(outputs.len(), n * no);
    let vli = description
        .parameters
        .iter()
        .position(|p| p.id == "Vli")
        .unwrap();
    for (i, stepper) in steppers.iter().enumerate() {
        let row = &outputs[i * no..(i + 1) * no];
        let s = stepper.state()[state("S")] / stepper.parameters()[vli];
        assert!((row[state("S")] - s).abs() <= 1e-14 * s.abs(), "{row:?}");
        let protein = setup.parameters[i]
            .iter()
            .find(|(id, _)| id == "protein_ext")
            .unwrap()
            .1;
        assert!((row[no - 1] - protein).abs() <= 1e-14, "cell {i}: {row:?}");
    }
    // The substrate is taken up: S_ext falls in every cell over a step.
    for i in 0..n {
        assert!(blood[2 * i] < fixed[2 * i], "cell {i}");
    }
    let per_cell = cells.memory_bytes() / n;
    println!("memory per cell: {per_cell} bytes");
    assert!(per_cell > 1000 && per_cell < 100_000, "{per_cell}");
}

/// A cell model which fails on one process (a decay rate whose growth overflows) stops every
/// process with the same error, naming the element.
#[test]
fn failure_on_one_process_stops_all() {
    c_cache();
    let p = Petsc::init(&[]).unwrap();
    let comm = p.comm();
    let mesh = mesh(&p, "lobule_toy.msh", 0);
    let ceed = ceed();
    let fine = finest(&mesh, &ceed);
    let spec = CellsSpec {
        areas: None,
        ..linear_spec(vec![
            parameter("k", Coefficient::Constant(0.1)),
            parameter("d", Coefficient::Constant(0.01)),
        ])
    };
    let mut setup = CellSetup::new(
        &fine,
        &mesh,
        &spec,
        &Maps::default(),
        &Coefficient::Constant(0.2),
    )
    .unwrap();
    let failing = comm.size() - 1;
    if comm.rank() == failing {
        for (id, value) in &mut setup.parameters[0] {
            if id == "d" {
                *value = -1e6;
            }
        }
    }
    let element = comm
        .max_int(if comm.rank() == failing {
            setup.global[0].try_into().unwrap()
        } else {
            -1
        })
        .unwrap();
    let element = i64::from(element);
    let mut cells = EngineCells::new(&setup, &spec, &model("linear_uptake.xml")).unwrap();
    let mut blood = vec![1.0; cells.n_cells()];
    match cells.advance(0.0, 600.0, &mut blood) {
        Err(CellsError::Element { cell, message, .. }) => {
            println!("{message}");
            assert_eq!(cell, element, "{message}");
            assert!(message.contains("600"), "{message}");
        }
        other => panic!("expected an error naming the element, got {other:?}"),
    }
}

/// A setup whose lists differ in length on one process only: an error on every process, none
/// waits in the compilation.
#[test]
fn invalid_setup_on_one_process_is_an_error_everywhere() {
    c_cache();
    let p = Petsc::init(&[]).unwrap();
    let comm = p.comm();
    let mesh = mesh(&p, "lobule_toy.msh", 0);
    let ceed = ceed();
    let fine = finest(&mesh, &ceed);
    let spec = CellsSpec {
        areas: None,
        ..linear_spec(Vec::new())
    };
    let mut setup = CellSetup::new(
        &fine,
        &mesh,
        &spec,
        &Maps::default(),
        &Coefficient::Constant(0.2),
    )
    .unwrap();
    if comm.rank() == comm.size() - 1 {
        setup.centroids.pop();
    }
    let e = EngineCells::new(&setup, &spec, &model("linear_uptake.xml")).unwrap_err();
    assert!(e.to_string().contains("centroids"), "{e}");
}

#[test]
fn mpi_world_has_the_expected_size() {
    let p = Petsc::init(&[]).unwrap();
    assert_eq!(p.comm().size(), test_support::ranks());
}

/// The tests `parallel_on_*_processes` run on several processes: the paths a single process does
/// not take (cells on every process, errors one process finds).
const PARALLEL_TESTS: [&str; 7] = [
    "mpi_world_has_the_expected_size",
    "parameters_are_area_means",
    "replace_rule_switches_value",
    "no_value_in_tissue_is_an_error",
    "study_start_checks",
    "failure_on_one_process_stops_all",
    "invalid_setup_on_one_process_is_an_error_everywhere",
];

#[test]
fn parallel_on_2_processes() {
    test_support::parallel(2, &PARALLEL_TESTS);
}

#[test]
fn parallel_on_3_processes() {
    test_support::parallel(3, &PARALLEL_TESTS);
}
