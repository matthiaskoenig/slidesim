//! Tests of the batched path: the cells of a process as one libCEED operator of the QFunction of
//! the model ([`BatchedCells`]), against the engine path, on two backends and on several
//! processes. nextest runs each test in its own process, so every test initializes PETSc once.
//! `parallel_on_2_processes` and `parallel_on_3_processes` compare the cells on several
//! processes with one and run some of the tests again under `mpiexec`.
//!
//! The compiled models go to a cache directory of this test binary in the temporary directory
//! (`SBMLSIM_C_CACHE`), not to the cache of the user.

mod common;

use std::collections::HashMap;

use ceed::Ceed;
use cells::{BatchedCells, CellModel, CellSetup, CellSolver, Cells, CellsError, EngineCells};
use common::{
    c_cache, ceed, finest, linear_spec, mesh, model, oscillator_model, parameter, spt_spec,
    testdata,
};
use contracts::{CellSolverKind, CellsSpec, Coefficient, Scene, Study};
use maps::Maps;
use mesh::Mesh;
use petsc::Petsc;

/// The bound of the states of the QFunction at the tolerances 1e-6 and 1e-9 against the engine
/// at 1e-12 and 1e-14: `RELATIVE |y| + ABSOLUTE`, the bound of `rosenbrock_matches_stepper` of
/// sbmlsim-c (its QFunction against its stepper).
const RELATIVE: f64 = 2e-6;
const ABSOLUTE: f64 = 2e-9;

/// The results of one process for the runs on several processes (a file of them).
const REFERENCE: &str = "SLIDESIM_TEST_BATCHED_REFERENCE";

/// SplitMix64, a seeded generator of uniform numbers in `[0, 1)`, as sbmlsim-c's tests have it.
struct Random(u64);

impl Random {
    fn uniform(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^= z >> 31;
        (z >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// The 50 parameter sets of the SPT model of sbmlsim-c's tests of the QFunction: the pattern,
/// the protein and the position of the cell, and the concentration of the external substrate.
fn cell_parameter_sets() -> Vec<Vec<(String, f64)>> {
    let mut random = Random(7);
    (0..50)
        .map(|_| {
            vec![
                ("pattern".to_string(), 7.0),
                ("protein_ext".to_string(), random.uniform()),
                ("position".to_string(), random.uniform()),
                ("[S_ext]".to_string(), 5.0 * random.uniform()),
            ]
        })
        .collect()
}

/// The cells of the SPT model of `solver` on every coarse element of `lobule_toy`, the
/// parameters from the map `zonation`, and the mesh (whose communicator the setup uses).
fn spt_setup(p: &Petsc, ceed: &Ceed, solver: CellSolverKind) -> (Mesh, CellSetup, CellsSpec) {
    let mesh = mesh(p, "lobule_toy.msh", 0);
    let fine = finest(&mesh, ceed);
    let study = Study::load(testdata("studies/lobule_toy/study.json")).unwrap();
    let scene = Scene::load(&study.scene).unwrap();
    let maps = Maps::read(&scene.maps, &["zonation"], mesh.local_bounds().unwrap()).unwrap();
    let spec = spt_spec(solver);
    let porosity = Coefficient::Constant(0.15);
    let setup = CellSetup::new(&fine, &mesh, &spec, &maps, &porosity).unwrap();
    (mesh, setup, spec)
}

/// The first 50 cells of [`spt_setup`] with the 50 parameter sets (and the volumes of their
/// elements), and the blood of each: the substrate of its set, no product.
fn fifty_cells(p: &Petsc, ceed: &Ceed) -> (Mesh, CellSetup, CellsSpec, Vec<f64>) {
    let (mesh, mut setup, spec) = spt_setup(p, ceed, CellSolverKind::Batched);
    let sets = cell_parameter_sets();
    let n = sets.len();
    assert!(setup.cells.len() >= n, "{} cells", setup.cells.len());
    setup.cells.truncate(n);
    setup.parameters.truncate(n);
    setup.elements.truncate(n);
    setup.centroids.truncate(n);
    for (cell, set) in setup.parameters.iter_mut().zip(&sets) {
        let volumes: Vec<(String, f64)> = cell
            .iter()
            .filter(|(id, _)| id == "Vext" || id == "Vli")
            .cloned()
            .collect();
        assert_eq!(volumes.len(), 2);
        *cell = set.clone();
        cell.extend(volumes);
    }
    let blood = sets.iter().flat_map(|set| [set[3].1, 0.0]).collect();
    (mesh, setup, spec, blood)
}

/// Whether `a` and `b` agree within `relative` (and the smallest positive number, for zeros).
fn close(a: f64, b: f64, relative: f64) -> bool {
    (a - b).abs() <= relative * b.abs() + f64::MIN_POSITIVE
}

/// What [`day`] gives per solver.
struct Day {
    worst: Vec<Vec<f64>>,
    steps: Vec<(u64, u64)>,
    blood: Vec<Vec<f64>>,
}

/// Advances each of `cells` over 1440 min in steps of 60 min, the blood held at `fixed`: the
/// largest difference of the states of each from those of the first, per state, in multiples of
/// the bound `RELATIVE |y| + ABSOLUTE`, the steps and rejected steps of each, and the blood of
/// each after the last step.
fn day(cells: &mut [&mut dyn CellSolver], fixed: &[f64]) -> Day {
    let ns = cells[0].state_ids().len();
    let mut worst = vec![vec![0.0; ns]; cells.len()];
    let mut steps = vec![(0, 0); cells.len()];
    let mut blood = vec![fixed.to_vec(); cells.len()];
    for k in 0..24 {
        let (t0, t1) = (3600.0 * k as f64, 3600.0 * (k + 1) as f64);
        for (c, cells) in cells.iter_mut().enumerate() {
            blood[c].copy_from_slice(fixed);
            let stats = cells.advance(t0, t1, &mut blood[c]).unwrap();
            assert!(stats.seconds >= 0.0);
            steps[c].0 += stats.steps;
            steps[c].1 += stats.rejected;
        }
        for c in 1..cells.len() {
            for (j, (a, b)) in cells[c].states().iter().zip(cells[0].states()).enumerate() {
                let ratio = (a - b).abs() / (RELATIVE * b.abs() + ABSOLUTE);
                worst[c][j % ns] = f64::max(worst[c][j % ns], ratio);
            }
        }
    }
    Day {
        worst,
        steps,
        blood,
    }
}

/// The largest difference of the batched path from the reference at the switch of the necrosis
/// of [`batched_matches_engine`], in multiples of the bound: about twice the 473 measured (for
/// `P_ext`), so that a change of the batched path which makes it worse there fails.
const SWITCH_CAP: f64 = 1000.0;

/// The SPT model with the 50 parameter sets as cells on `lobule_toy`, the blood held fixed,
/// 1440 min in steps of 60 min: the batched path at the tolerances of the spec agrees with the
/// engine at tight tolerances within the bound of sbmlsim-c's QFunction against its stepper;
/// both start from the same initial state; the rule target `protein` (pattern 7 sets it to
/// `protein_ext`) is evaluated per cell.
///
/// The bound holds away from the switch of the necrosis (the threshold of the toxic product `T`
/// raised to 20 mM, above the 12 mM the cells reach). With the threshold of the model (5 mM),
/// the cells which reach it amplify the differences of their trajectories, so neither path
/// keeps the bound there (measured: the batched path up to 473 times it, the engine at the
/// tolerances of the spec up to 7818 times); the batched path is within the bound or the closer
/// one for every state, which the test prints; it asserts the cap [`SWITCH_CAP`].
#[test]
fn batched_matches_engine() {
    c_cache();
    let p = Petsc::init(&[]).unwrap();
    let ceed = ceed();
    let (_mesh, setup, spec, fixed) = fifty_cells(&p, &ceed);
    let n = setup.cells.len();
    let sbml = model("spt_liver_cell.xml");
    let reference = CellsSpec {
        solver: CellSolverKind::Engine,
        rtol: 1e-12,
        atol: 1e-14,
        ..spec.clone()
    };

    // Away from the switch: within the bound.
    let mut off = setup.clone();
    for cell in &mut off.parameters {
        cell.push(("T_threshold".to_string(), 20.0));
    }
    let mut engine = EngineCells::new(&off, &reference, &sbml).unwrap();
    let mut batched = BatchedCells::new(&off, &spec, &sbml, &ceed).unwrap();
    assert_eq!(batched.n_cells(), n);
    assert_eq!(batched.species(), ["S_ext", "P_ext"]);
    assert_eq!(batched.state_ids(), engine.state_ids());
    assert_eq!(batched.states(), engine.states());
    let ids = batched.state_ids().to_vec();
    let Day {
        worst,
        steps,
        blood,
    } = day(&mut [&mut engine, &mut batched], &fixed);
    for (id, w) in ids.iter().zip(&worst[1]) {
        eprintln!("{id}: the largest difference is {w:.3} of the bound");
        assert!(*w <= 1.0, "{id}: {w} of the bound");
    }
    eprintln!(
        "{:.1} steps and {:.2} rejected per cell and 60 min",
        steps[1].0 as f64 / (n * 24) as f64,
        steps[1].1 as f64 / (n * 24) as f64
    );
    assert!(steps[1].0 > 0);
    // The outputs: those of the engine path; the blood the linked species of the states; the
    // rule target protein = protein_ext.
    assert_eq!(batched.output_values(), engine.output_values());
    let values = batched.output_values();
    let no = values.len();
    assert_eq!(values.last().unwrap().id, "protein");
    let (mut a, mut b) = (Vec::new(), Vec::new());
    batched.outputs(&mut a);
    engine.outputs(&mut b);
    assert_eq!(a.len(), n * no);
    for (i, cell) in setup.parameters.iter().enumerate() {
        for (s, species) in ["S_ext", "P_ext"].iter().enumerate() {
            let column = ids.iter().position(|id| id == species).unwrap();
            assert!(
                close(blood[1][2 * i + s], a[i * no + column], 1e-15),
                "cell {i}, {species}"
            );
        }
        let protein = cell.iter().find(|(id, _)| id == "protein_ext").unwrap().1;
        assert!(close(a[i * no + no - 1], protein, 1e-14), "cell {i}");
        for k in 0..no {
            let (x, y) = (a[i * no + k], b[i * no + k]);
            assert!(
                (x - y).abs() <= RELATIVE * y.abs() + ABSOLUTE / 0.15,
                "cell {i}, {}: batched {x}, engine {y}",
                values[k].id
            );
        }
    }

    // At the switch: within the bound or closer than the engine at the same tolerances.
    let mut engine = EngineCells::new(&setup, &reference, &sbml).unwrap();
    let loose = CellsSpec {
        solver: CellSolverKind::Engine,
        ..spec.clone()
    };
    let mut engine_loose = EngineCells::new(&setup, &loose, &sbml).unwrap();
    let mut batched = BatchedCells::new(&setup, &spec, &sbml, &ceed).unwrap();
    let worst = day(&mut [&mut engine, &mut batched, &mut engine_loose], &fixed).worst;
    for (s, id) in ids.iter().enumerate() {
        let (w_batched, w_engine) = (worst[1][s], worst[2][s]);
        // The engine at the same tolerances for comparison only: its differences are no bound.
        eprintln!("{id} at the switch: batched {w_batched:.3}, engine {w_engine:.3} of the bound");
        assert!(
            w_batched <= SWITCH_CAP,
            "{id}: batched {w_batched} of the bound (engine {w_engine})"
        );
    }
}

/// The 50 cells of [`batched_matches_engine`] on the reference backend and on the blocked one
/// (blocks of 8 points, the last one padded): the same steps, the states within 1e-12.
#[test]
fn same_on_backends() {
    c_cache();
    let p = Petsc::init(&[]).unwrap();
    let ceed = ceed();
    let (_mesh, setup, spec, fixed) = fifty_cells(&p, &ceed);
    let sbml = model("spt_liver_cell.xml");
    let run = |resource: &str| {
        let ceed = Ceed::new(resource).unwrap();
        let mut cells = BatchedCells::new(&setup, &spec, &sbml, &ceed).unwrap();
        let mut steps = Vec::new();
        for k in 0..6 {
            let (t0, t1) = (3600.0 * k as f64, 3600.0 * (k + 1) as f64);
            let stats = cells.advance(t0, t1, &mut fixed.clone()).unwrap();
            steps.push((stats.steps, stats.rejected));
        }
        (cells.states().to_vec(), steps)
    };
    let (reference, steps) = run("/cpu/self/ref/serial");
    let (blocked, blocked_steps) = run("/cpu/self/opt/blocked");
    assert_eq!(steps, blocked_steps);
    assert_eq!(reference.len(), blocked.len());
    for (k, (a, b)) in blocked.iter().zip(&reference).enumerate() {
        assert!(
            close(*a, *b, 1e-12),
            "value {k}: blocked {a}, reference {b}"
        );
    }
}

/// The states of cells by the keys of their centroids.
type ByCentroid = Vec<([i64; 2], Vec<f64>)>;

/// The key of a centroid (um), the same on any number of processes: in nm.
fn key(centroid: [f64; 2]) -> [i64; 2] {
    centroid.map(|x| (x * 1e3).round() as i64)
}

/// The SPT model on every coarse element of `lobule_toy` (the parameters from the map), the
/// blood of each a function of its centroid, 6 h in steps of 1 h: the states of the cells of
/// this process by the keys of their centroids, and the steps of every process.
fn every_cell(p: &Petsc) -> (ByCentroid, u64) {
    let ceed = ceed();
    let (_mesh, setup, spec) = spt_setup(p, &ceed, CellSolverKind::Batched);
    let sbml = model("spt_liver_cell.xml");
    let mut cells = BatchedCells::new(&setup, &spec, &sbml, &ceed).unwrap();
    let fixed: Vec<f64> = setup
        .centroids
        .iter()
        .flat_map(|[x, y]| {
            [
                2.5 + 2.0 * (0.01 * x + 0.02 * y).sin(),
                0.1 + 0.05 * (0.015 * x).cos(),
            ]
        })
        .collect();
    let mut steps = 0;
    for k in 0..6 {
        let (t0, t1) = (3600.0 * k as f64, 3600.0 * (k + 1) as f64);
        steps += cells.advance(t0, t1, &mut fixed.clone()).unwrap().steps;
    }
    let steps = p.comm().sum_int(steps.try_into().unwrap()).unwrap();
    let ns = cells.state_ids().len();
    let states = setup
        .centroids
        .iter()
        .zip(cells.states().chunks(ns))
        .map(|(c, s)| (key(*c), s.to_vec()))
        .collect();
    (states, steps.try_into().unwrap())
}

/// Runs [`every_cell`] on one process, then the test `test` itself on `ranks` processes with
/// its results in a file named by [`REFERENCE`], where the cells of every process compare their
/// states with those of one process (within 1e-12) and the steps agree. Then the other tests
/// of [`PARALLEL_TESTS`] on `ranks` processes.
fn parity(ranks: usize, test: &str) {
    c_cache();
    let p = Petsc::init(&[]).unwrap();
    let (states, steps) = every_cell(&p);
    match std::env::var(REFERENCE) {
        Err(_) => {
            assert_eq!(test_support::ranks(), 1);
            // MPI is finalized before mpiexec starts.
            drop(p);
            let mut text = format!("{steps}\n");
            for (key, values) in &states {
                let values: Vec<String> = values.iter().map(|v| format!("{v:e}")).collect();
                text.push_str(&format!("{} {} {}\n", key[0], key[1], values.join(" ")));
            }
            let path = std::env::temp_dir().join(format!(
                "slidesim-cells-batched-{ranks}-{}.txt",
                std::process::id()
            ));
            std::fs::write(&path, text).unwrap();
            println!("{} cells on one process, {steps} steps", states.len());
            test_support::parallel_with_env(ranks, &[test], &[(REFERENCE, path.to_str().unwrap())]);
            let _ = std::fs::remove_file(&path);
            test_support::parallel(ranks, &PARALLEL_TESTS);
        }
        Ok(path) => {
            let text = std::fs::read_to_string(path).unwrap();
            let mut lines = text.lines();
            let reference_steps: u64 = lines.next().unwrap().parse().unwrap();
            let reference: HashMap<[i64; 2], Vec<f64>> = lines
                .map(|line| {
                    let words: Vec<&str> = line.split(' ').collect();
                    let key = [words[0].parse().unwrap(), words[1].parse().unwrap()];
                    let values = words[2..].iter().map(|w| w.parse().unwrap()).collect();
                    (key, values)
                })
                .collect();
            assert_eq!(steps, reference_steps);
            let count = p.comm().sum_int(states.len().try_into().unwrap()).unwrap();
            assert_eq!(usize::try_from(count).unwrap(), reference.len());
            for (key, values) in &states {
                let one = &reference[key];
                for (k, (a, b)) in values.iter().zip(one).enumerate() {
                    assert!(
                        close(*a, *b, 1e-12),
                        "cell at {key:?} nm, state {k}: {a} on {} processes, {b} on one",
                        test_support::ranks()
                    );
                }
            }
        }
    }
}

/// A cell model which fails on one process (a decay rate whose growth overflows) stops every
/// process with the same error, naming the element; the cells are not advanced after it. The
/// last cell of that process fails too: the error names the first, also where the balance of
/// the work has another process advance the last (on 3 processes, the last of 359 cells goes to
/// the first process, of 357).
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
        solver: CellSolverKind::Batched,
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
        let last = setup.parameters.len() - 1;
        for i in [0, last] {
            for (id, value) in &mut setup.parameters[i] {
                if id == "d" {
                    *value = -1e6;
                }
            }
        }
    }
    // The element id of the first cell of the failing process (partition independent).
    assert_eq!(setup.elements[0], mesh.element_ids()[setup.cells[0]]);
    let element = comm
        .max_int(if comm.rank() == failing {
            setup.elements[0].try_into().unwrap()
        } else {
            -1
        })
        .unwrap();
    let element = u64::try_from(element).unwrap();
    let mut cells = BatchedCells::new(&setup, &spec, &model("linear_uptake.xml"), &ceed).unwrap();
    let mut blood = vec![1.0; cells.n_cells()];
    match cells.advance(0.0, 600.0, &mut blood) {
        Err(e @ CellsError::Element { .. }) => {
            let text = e.to_string();
            println!("{text}");
            assert!(
                text.starts_with(&format!("element {element} at (")),
                "{text}"
            );
            let CellsError::Element {
                element: id,
                message,
                ..
            } = e
            else {
                unreachable!()
            };
            assert_eq!(id, element, "{message}");
            assert!(message.contains("600"), "{message}");
        }
        other => panic!("expected an error naming the element, got {other:?}"),
    }
    // The cells are not advanced again: the same error, the states as they were.
    let states = cells.states().to_vec();
    let mut blood = vec![1.0; cells.n_cells()];
    match cells.advance(600.0, 1200.0, &mut blood) {
        Err(CellsError::Element { element: id, .. }) => assert_eq!(id, element),
        other => panic!("expected the error again, got {other:?}"),
    }
    assert_eq!(cells.states(), states.as_slice());
}

/// An event of `linear_uptake.xml`: `A` is set to 0 at t = 1.
const EVENT: &str = r#"<listOfEvents><event id="reset" useValuesFromTriggerTime="true"><trigger initialValue="false" persistent="true"><math xmlns="http://www.w3.org/1998/Math/MathML"><apply><gt/><csymbol encoding="text" definitionURL="http://www.sbml.org/sbml/symbols/time">t</csymbol><cn>1</cn></apply></math></trigger><listOfEventAssignments><eventAssignment variable="A"><math xmlns="http://www.w3.org/1998/Math/MathML"><cn>0</cn></math></eventAssignment></listOfEventAssignments></event></listOfEvents>"#;

/// A model with events on the batched path: an error naming the feature and the engine path,
/// at the start of a study ([`CellModel::new`]) and for the cells; on the engine path the same
/// model runs. [`Cells::new`] chooses the path of the spec.
#[test]
fn batched_refuses_events() {
    c_cache();
    let p = Petsc::init(&[]).unwrap();
    let comm = p.comm();
    let mesh = mesh(&p, "lobule_toy.msh", 0);
    let ceed = ceed();
    let fine = finest(&mesh, &ceed);
    let batched = CellsSpec {
        areas: None,
        solver: CellSolverKind::Batched,
        ..linear_spec(Vec::new())
    };
    let setup = CellSetup::new(
        &fine,
        &mesh,
        &batched,
        &Maps::default(),
        &Coefficient::Constant(0.2),
    )
    .unwrap();
    let sbml = model("linear_uptake.xml").replace("</model>", &format!("{EVENT}</model>"));
    let refused = |e: CellsError| {
        assert!(matches!(e, CellsError::Unsupported { .. }), "{e:?}");
        let message = e.to_string();
        assert!(message.contains("events"), "{message}");
        assert!(message.contains("\"engine\""), "{message}");
    };
    refused(BatchedCells::new(&setup, &batched, &sbml, &ceed).unwrap_err());
    refused(CellModel::new(&comm, &batched, &sbml).unwrap_err());

    let engine = CellsSpec {
        solver: CellSolverKind::Engine,
        ..batched.clone()
    };
    let model_with_events = CellModel::new(&comm, &engine, &sbml).unwrap();
    let mut cells = Cells::new(&setup, &engine, &model_with_events, &ceed).unwrap();
    assert!(matches!(cells, Cells::Engine(_)));
    let mut blood = vec![1.0; cells.n_cells()];
    cells.advance(0.0, 2.0, &mut blood).unwrap();
    let linear = model("linear_uptake.xml");
    let model = CellModel::new(&comm, &batched, &linear).unwrap();
    let cells = Cells::new(&setup, &batched, &model, &ceed).unwrap();
    assert!(matches!(cells, Cells::Batched(_)));
    assert_eq!(cells.n_cells(), setup.cells.len());
}

/// `linear_uptake.xml` with the rule target `scaled = k^0.75 [A]`, whose `k^0.75` the C target
/// hoists into a constant.
fn hoisted_model() -> String {
    let rule = r#"<assignmentRule variable="scaled"><math xmlns="http://www.w3.org/1998/Math/MathML"><apply><times/><apply><power/><ci> k </ci><cn> 0.75 </cn></apply><ci> A </ci></apply></math></assignmentRule>"#;
    model("linear_uptake.xml")
        .replace("<listOfRules>", &format!("<listOfRules>{rule}"))
        .replace(
            "</listOfParameters>",
            r#"<parameter id="scaled" constant="false"/></listOfParameters>"#,
        )
}

/// The outputs of a rule which reads a constant of the model (a subexpression of parameters
/// alone, which the compiled model computes once): on both paths, the rule of the state of each
/// cell, with the constant of the cell's parameters.
#[test]
fn rule_outputs_read_constants() {
    c_cache();
    let p = Petsc::init(&[]).unwrap();
    let mesh = mesh(&p, "lobule_toy.msh", 0);
    let ceed = ceed();
    let fine = finest(&mesh, &ceed);
    let sbml = hoisted_model();
    let source = sbmlsim_c::generate(&sbml).unwrap();
    assert!(source.description.n_constants > 0);
    for solver in [CellSolverKind::Engine, CellSolverKind::Batched] {
        let spec = CellsSpec {
            areas: None,
            solver,
            outputs: vec!["uptake_rate".to_string(), "scaled".to_string()],
            ..linear_spec(vec![parameter("k", Coefficient::Constant(0.3))])
        };
        let setup = CellSetup::new(
            &fine,
            &mesh,
            &spec,
            &Maps::default(),
            &Coefficient::Constant(0.2),
        )
        .unwrap();
        let model = CellModel::new(&p.comm(), &spec, &sbml).unwrap();
        let mut cells = Cells::new(&setup, &spec, &model, &ceed).unwrap();
        let mut blood = vec![1.0; cells.n_cells()];
        cells.advance(0.0, 10.0, &mut blood).unwrap();
        let names: Vec<&str> = cells
            .output_values()
            .iter()
            .map(|v| v.id.as_str())
            .collect();
        assert_eq!(names, ["A_ext", "A", "uptake_rate", "scaled"], "{solver:?}");
        let mut outputs = Vec::new();
        cells.outputs(&mut outputs);
        for (i, row) in outputs.chunks(4).enumerate() {
            let (a_ext, a) = (row[0], row[1]);
            assert!(a > 0.0, "{solver:?}, cell {i}: {row:?}");
            assert!(
                close(row[2], 0.3 * (a_ext - a), 1e-14),
                "{solver:?}: {row:?}"
            );
            assert!(
                close(row[3], 0.3f64.powf(0.75) * a, 1e-14),
                "{solver:?}: {row:?}"
            );
        }
    }
}

/// What [`balanced_run`] gives: per advance the steps and rejected steps of this process, then
/// the states, the outputs and the blood of its cells at the end.
struct BalancedRun {
    local: Vec<(u64, u64)>,
    states: Vec<f64>,
    outputs: Vec<f64>,
    blood: Vec<f64>,
}

/// The SPT model on every coarse element of `lobule_toy` (the parameters from the map), the
/// blood at first a function of the centroid, then as the cells leave it, 4 h in steps of 1 h,
/// with the balance of the work `balance`.
fn balanced_run(p: &Petsc, ceed: &Ceed, balance: bool) -> BalancedRun {
    let (_mesh, setup, spec) = spt_setup(p, ceed, CellSolverKind::Batched);
    let spec = CellsSpec { balance, ..spec };
    let mut cells = BatchedCells::new(&setup, &spec, &model("spt_liver_cell.xml"), ceed).unwrap();
    let mut blood: Vec<f64> = setup
        .centroids
        .iter()
        .flat_map(|[x, y]| [2.5 + 2.0 * (0.01 * x + 0.02 * y).sin(), 0.0])
        .collect();
    let local = (0..4)
        .map(|k| {
            let (t0, t1) = (3600.0 * k as f64, 3600.0 * (k + 1) as f64);
            let stats = cells.advance(t0, t1, &mut blood).unwrap();
            (stats.steps, stats.rejected)
        })
        .collect();
    let mut outputs = Vec::new();
    cells.outputs(&mut outputs);
    BalancedRun {
        local,
        states: cells.states().to_vec(),
        outputs,
        blood,
    }
}

/// On 2 and 3 processes, the cells advance alike with and without the balance of their work
/// (`cells.balance`): the states, the outputs and the blood of every cell bit for bit equal, and
/// the steps and rejected steps of every advance, summed over the processes. The balance moved
/// work: the steps of a process differ in some advance.
#[test]
fn balanced_cells_match_unbalanced() {
    for ranks in [2, 3] {
        test_support::parallel(ranks, &["balanced_cells_match_unbalanced"]);
    }
    c_cache();
    let p = Petsc::init(&[]).unwrap();
    let comm = p.comm();
    let ceed = ceed();
    let (off, on) = (
        balanced_run(&p, &ceed, false),
        balanced_run(&p, &ceed, true),
    );
    let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&off.states), bits(&on.states));
    assert_eq!(bits(&off.outputs), bits(&on.outputs));
    assert_eq!(bits(&off.blood), bits(&on.blood));
    let sum = |x: u64| comm.sum_int(x.try_into().unwrap()).unwrap();
    for (a, b) in off.local.iter().zip(&on.local) {
        assert_eq!((sum(a.0), sum(a.1)), (sum(b.0), sum(b.1)));
    }
    let moved = off.local.iter().zip(&on.local).any(|(a, b)| a != b);
    let moved = comm.max_int(moved.into()).unwrap() > 0;
    println!(
        "{} processes: work moved {moved}, {:?} against {:?}",
        comm.size(),
        on.local,
        off.local
    );
    assert_eq!(moved, comm.size() > 1);
}

/// On 2 and 3 processes, the oscillator model with the cells of process 0 at 10 times the
/// frequency of the others (10 times their step attempts): with the balance, the step attempts
/// of each process in the second advance (after the first, whose attempts the balance reads)
/// are within 10 % of their mean; without it, process 0 makes far more.
#[test]
fn balance_evens_the_work() {
    for ranks in [2, 3] {
        test_support::parallel(ranks, &["balance_evens_the_work"]);
    }
    if test_support::ranks() == 1 {
        return;
    }
    c_cache();
    let p = Petsc::init(&[]).unwrap();
    let comm = p.comm();
    let mesh = mesh(&p, "lobule_toy.msh", 0);
    let ceed = ceed();
    let fine = finest(&mesh, &ceed);
    let sbml = oscillator_model();
    for balance in [false, true] {
        let spec = CellsSpec {
            areas: None,
            solver: CellSolverKind::Batched,
            balance,
            ..linear_spec(vec![parameter("w", Coefficient::Constant(1.0))])
        };
        let mut setup = CellSetup::new(
            &fine,
            &mesh,
            &spec,
            &Maps::default(),
            &Coefficient::Constant(0.2),
        )
        .unwrap();
        if comm.rank() == 0 {
            for cell in &mut setup.parameters {
                cell.iter_mut().find(|(id, _)| id == "w").unwrap().1 = 10.0;
            }
        }
        let mut cells = BatchedCells::new(&setup, &spec, &sbml, &ceed).unwrap();
        let mut blood = vec![1.0; cells.n_cells()];
        cells.advance(0.0, 6.0, &mut blood).unwrap();
        let stats = cells.advance(6.0, 12.0, &mut blood).unwrap();
        let attempts = (stats.steps + stats.rejected) as f64;
        let max = comm.max_real(attempts).unwrap();
        let mean = comm.sum_real(attempts).unwrap() / comm.size() as f64;
        println!(
            "{} processes, balance {balance}: {} cells, attempts {attempts} here, largest over \
             mean {}",
            comm.size(),
            cells.n_cells(),
            max / mean
        );
        if balance {
            assert!(max / mean <= 1.1, "{}", max / mean);
        } else {
            assert!(max / mean > 1.5, "{}", max / mean);
        }
    }
}

/// On 3 processes with one cell each, the cell of process 0 at 100 times the frequency of the
/// others: after the first advance the balance sends that cell away (its weight is more than
/// twice those of the others together), so process 0 advances no cell, and its statistics count
/// no step (not those of the advance before, when it advanced its cell). The steps of each
/// advance, summed over the processes, are those without the balance.
#[test]
fn process_without_points_counts_nothing() {
    if test_support::ranks() == 1 {
        test_support::parallel(3, &["process_without_points_counts_nothing"]);
        return;
    }
    c_cache();
    let p = Petsc::init(&[]).unwrap();
    let comm = p.comm();
    let mesh = mesh(&p, "lobule_toy.msh", 0);
    let ceed = ceed();
    let fine = finest(&mesh, &ceed);
    let sbml = oscillator_model();
    let mut runs = Vec::new();
    for balance in [false, true] {
        let spec = CellsSpec {
            areas: None,
            solver: CellSolverKind::Batched,
            balance,
            ..linear_spec(vec![parameter("w", Coefficient::Constant(1.0))])
        };
        let mut setup = CellSetup::new(
            &fine,
            &mesh,
            &spec,
            &Maps::default(),
            &Coefficient::Constant(0.2),
        )
        .unwrap();
        assert!(!setup.cells.is_empty(), "a process without cells");
        setup.cells.truncate(1);
        setup.parameters.truncate(1);
        setup.elements.truncate(1);
        setup.centroids.truncate(1);
        if comm.rank() == 0 {
            let w = setup.parameters[0]
                .iter_mut()
                .find(|(id, _)| id == "w")
                .unwrap();
            w.1 = 100.0;
        }
        let mut cells = BatchedCells::new(&setup, &spec, &sbml, &ceed).unwrap();
        let mut blood = vec![1.0; cells.n_cells()];
        let steps: Vec<u64> = [(0.0, 6.0), (6.0, 12.0), (12.0, 18.0)]
            .into_iter()
            .map(|(t0, t1)| cells.advance(t0, t1, &mut blood).unwrap().steps)
            .collect();
        runs.push(steps);
    }
    let (off, on) = (&runs[0], &runs[1]);
    println!(
        "process {}: steps {on:?} with the balance, {off:?} without",
        comm.rank()
    );
    let sum = |x: u64| comm.sum_int(x.try_into().unwrap()).unwrap();
    for (a, b) in off.iter().zip(on) {
        assert_eq!(sum(*a), sum(*b), "{off:?} against {on:?}");
    }
    assert!(off[0] > 0);
    if comm.rank() == 0 {
        assert_eq!(on[0], off[0], "nothing moves in the first advance");
        assert_eq!(
            on[1..],
            [0, 0],
            "process 0 advances no cell after the first"
        );
    }
}

#[test]
fn mpi_world_has_the_expected_size() {
    let p = Petsc::init(&[]).unwrap();
    assert_eq!(p.comm().size(), test_support::ranks());
}

/// The tests [`parity`] runs again on several processes: errors one process finds, the choice
/// of the path, the outputs.
const PARALLEL_TESTS: [&str; 4] = [
    "mpi_world_has_the_expected_size",
    "failure_on_one_process_stops_all",
    "batched_refuses_events",
    "rule_outputs_read_constants",
];

#[test]
fn parallel_on_2_processes() {
    parity(2, "parallel_on_2_processes");
}

#[test]
fn parallel_on_3_processes() {
    parity(3, "parallel_on_3_processes");
}
