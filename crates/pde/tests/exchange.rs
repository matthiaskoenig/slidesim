//! Tests of the [`Exchange`] between a field and cell models on the coarse elements of the mesh,
//! and of [`Problem::modify`], on `lobule_toy.msh` refined once with the porosity a table over
//! its `zonation` map: the means of the field over the elements, the increment given back to
//! the field (the amount of substance changes by exactly its sum, nodal values stay
//! nonnegative), and the restart of the time integration after a change of the solution.
//! nextest runs each test in its own process, so every test initializes PETSc once.
//! `parallel_on_2_processes` runs them again under `mpiexec`.

mod common;

use std::path::Path;

use common::{ceed, fixture_mesh};
use contracts::{BoundarySpec, Coefficient, Condition, FieldSpec, Scene, Study, Table, TimeSpec};
use maps::Maps;
use mesh::Mesh;
use pde::{Exchange, Inputs, PdeError, Problem};
use petsc::Petsc;

/// The time step of the problems (s).
const DT: f64 = 0.5;

/// The mesh of the study `lobule_toy`, refined once.
fn lobule(p: &Petsc) -> Mesh {
    fixture_mesh(p, "lobule_toy.msh", 1)
}

/// The maps of the study `lobule_toy` (its `zonation`), read for the cells of this process.
fn lobule_maps(mesh: &Mesh) -> Maps {
    let path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/studies/lobule_toy/study.json");
    let study = Study::load(&path).unwrap();
    let scene = Scene::load(&study.scene).unwrap();
    Maps::read(&scene.maps, &["zonation"], mesh.local_bounds().unwrap()).unwrap()
}

/// The porosity of the tests: 0.1 at the portal end of the zonation, 0.3 at the central end.
fn porosity() -> Coefficient {
    Coefficient::Table(Table {
        map: "zonation".to_string(),
        points: vec![[0.0, 0.1], [1.0, 0.3]],
    })
}

/// A field `s` without advection, of the porosity of [`porosity`], initial value `initial` and
/// the Dirichlet value `portal` on the portal wall.
fn field(initial: f64, portal: f64) -> FieldSpec {
    FieldSpec {
        id: "s".to_string(),
        diffusion: Coefficient::Constant(700.0),
        decay: Coefficient::Constant(0.0),
        source: Coefficient::Constant(0.0),
        porosity: porosity(),
        advection: false,
        initial,
        boundaries: vec![BoundarySpec {
            tag: "portal_wall".to_string(),
            condition: Condition::Dirichlet { value: portal },
        }],
    }
}

fn problem(mesh: &Mesh, maps: &Maps, field: &FieldSpec) -> Problem {
    let time = TimeSpec {
        end: 10.0,
        dt: DT,
        adaptive: false,
        outputs: Vec::new(),
    };
    let inputs = Inputs { maps, flow: None };
    Problem::new(mesh, field, &time, &ceed(), "", &inputs).unwrap()
}

/// The exchange of every local coarse cell of `mesh` with the field of `problem`.
fn exchange(mesh: &Mesh, maps: &Maps, problem: &Problem) -> Exchange {
    let cells: Vec<usize> = (0..mesh.levels()[0].height_stratum(0).unwrap().len()).collect();
    Exchange::new(problem.discretization(), mesh, &cells, &porosity(), maps).unwrap()
}

/// SplitMix64: uniform numbers in `[0, 1)`.
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

/// `sum_i delta_i V_i` over the cells of every process.
fn exchanged(exchange: &Exchange, delta: &[f64], mesh: &Mesh) -> f64 {
    let local: f64 = delta
        .iter()
        .zip(exchange.volumes())
        .map(|(d, v)| d * v)
        .sum();
    mesh.finest().comm().sum_real(local).unwrap()
}

/// The number of nodal values of the solution below zero, on every process.
fn negative_values(problem: &Problem, mesh: &Mesh) -> usize {
    let negative = problem
        .solution()
        .with_array(|u| u.iter().filter(|&&u| u < 0.0).count())
        .unwrap();
    let negative = petsc::sys::PetscInt::try_from(negative).unwrap();
    usize::try_from(mesh.finest().comm().sum_int(negative).unwrap()).unwrap()
}

/// The mean of a uniform field over every element is the value, with the Dirichlet values
/// (the same) included and the porosity varying.
#[test]
fn means_of_constant_field_are_the_constant() {
    let p = Petsc::init(&[]).unwrap();
    let mesh = lobule(&p);
    let maps = lobule_maps(&mesh);
    let problem = problem(&mesh, &maps, &field(0.7, 0.7));
    let exchange = exchange(&mesh, &maps, &problem);
    assert_eq!(exchange.volumes().len(), exchange.len());
    let mut means = vec![0.0; exchange.len()];
    exchange.means(&problem, &mut means).unwrap();
    for (i, mean) in means.iter().enumerate() {
        assert!((mean - 0.7).abs() <= 1e-12 * 0.7, "cell {i}: {mean}");
    }
    // The blood volume of an element is its area times its mean porosity, in [0.1, 0.3].
    let area: f64 = mesh.levels()[0]
        .height_stratum(0)
        .unwrap()
        .map(|c| mesh.levels()[0].cell_volume(c).unwrap())
        .collect::<Vec<_>>()
        .iter()
        .sum();
    let volume: f64 = exchange.volumes().iter().sum();
    assert!(
        0.1 * area < volume && volume < 0.3 * area,
        "{volume} of {area}"
    );
}

/// The resupply of the Dirichlet tags of `exchange`, summed: the same on every process.
fn total_resupply(exchange: &Exchange, resupply: &[(String, f64)]) -> f64 {
    let tags: Vec<&String> = resupply.iter().map(|(t, _)| t).collect();
    assert_eq!(tags, exchange.dirichlet_tags().iter().collect::<Vec<_>>());
    resupply.iter().map(|(_, r)| r).sum()
}

/// Random increments, up to removing all of an element's mean: the amount of substance
/// `int phi u` changes by `sum_i delta_i V_i` plus the resupply of the Dirichlet values on the
/// portal wall, which stay.
#[test]
fn give_back_conserves_total() {
    let p = Petsc::init(&[]).unwrap();
    let mesh = lobule(&p);
    let maps = lobule_maps(&mesh);
    let mut problem = problem(&mesh, &maps, &field(0.0, 2.0));
    problem
        .set_initial_condition(|[x, y]| 1.0 + (x / 100.0).sin() * (y / 70.0).cos())
        .unwrap();
    problem.step_to(DT).unwrap();
    let exchange = exchange(&mesh, &maps, &problem);
    let mut means = vec![0.0; exchange.len()];
    exchange.means(&problem, &mut means).unwrap();
    let rank = mesh.finest().comm().rank() as u64;
    let mut random = Random(11 + rank);
    let delta: Vec<f64> = means
        .iter()
        .map(|m| (2.0 * random.uniform() - 1.0) * m)
        .collect();
    let before = problem.total_mass().unwrap();
    let resupply = exchange.give_back(&mut problem, &delta).unwrap();
    let resupply = total_resupply(&exchange, &resupply);
    assert!(resupply != 0.0);
    let after = problem.total_mass().unwrap();
    let expected = exchanged(&exchange, &delta, &mesh);
    let error = ((after - before - resupply) - expected).abs();
    assert!(
        error <= 1e-12 * before.abs().max(expected.abs()),
        "change {} - resupply {resupply} against {expected}: error {error:e}",
        after - before
    );
    // The time integration goes on from the changed field.
    problem.step_to(2.0 * DT).unwrap();
}

/// Uptake of 99.9 % of every element's blood, of a field which is zero on a part of the domain,
/// next to its Dirichlet boundary of value 2 as well: every nodal value stays nonnegative, and
/// the amount changes by exactly the sum plus the resupply of the Dirichlet values.
#[test]
fn steep_uptake_stays_nonnegative() {
    let p = Petsc::init(&[]).unwrap();
    let mesh = lobule(&p);
    let maps = lobule_maps(&mesh);
    let mut problem = problem(&mesh, &maps, &field(0.0, 2.0));
    problem
        .set_initial_condition(|[x, y]| if x + y > 400.0 { 3.0 + x / 400.0 } else { 0.0 })
        .unwrap();
    let exchange = exchange(&mesh, &maps, &problem);
    let mut means = vec![0.0; exchange.len()];
    exchange.means(&problem, &mut means).unwrap();
    let delta: Vec<f64> = means.iter().map(|m| -0.999 * m).collect();
    let before = problem.total_mass().unwrap();
    let resupply = exchange.give_back(&mut problem, &delta).unwrap();
    let resupply = total_resupply(&exchange, &resupply);
    let after = problem.total_mass().unwrap();
    assert_eq!(negative_values(&problem, &mesh), 0);
    // The portal wall resupplies the share of its nodes.
    assert!(resupply > 0.0, "{resupply}");
    let expected = exchanged(&exchange, &delta, &mesh);
    let error = ((after - before - resupply) - expected).abs();
    assert!(
        error <= 1e-12 * before,
        "change {} - resupply {resupply} against {expected}: error {error:e}",
        after - before
    );
}

/// A uniform field, its Dirichlet value the same, and a uniform change of every element: the
/// free nodal values change by the same, and the Dirichlet tag resupplies (or takes away) the
/// share of its nodes; field change minus resupply is the exchanged amount.
#[test]
fn uniform_change_with_resupply() {
    let p = Petsc::init(&[]).unwrap();
    let mesh = lobule(&p);
    let maps = lobule_maps(&mesh);
    let mut problem = problem(&mesh, &maps, &field(1.0, 1.0));
    let exchange = exchange(&mesh, &maps, &problem);
    for (change, value) in [(-0.5, 0.5), (0.25, 0.75)] {
        let delta = vec![change; exchange.len()];
        let before = problem.total_mass().unwrap();
        let resupply = exchange.give_back(&mut problem, &delta).unwrap();
        assert_eq!(resupply.len(), 1);
        assert_eq!(resupply[0].0, "portal_wall");
        let resupply = resupply[0].1;
        let after = problem.total_mass().unwrap();
        // Out of the reservoir for an uptake, into it for a release.
        assert!(resupply * change < 0.0, "{change}: {resupply}");
        let expected = exchanged(&exchange, &delta, &mesh);
        let error = ((after - before - resupply) - expected).abs();
        assert!(error <= 1e-12 * before, "{change}: error {error:e}");
        problem
            .solution()
            .with_array(|u| {
                for u in u {
                    assert!((u - value).abs() <= 1e-12, "{u} against {value}");
                }
            })
            .unwrap();
    }
}

/// A change of the solution between steps starts the time integration again (BDF2 starts with
/// its first step again), and the rate of change is unknown until the next step.
#[test]
fn modify_restarts_the_time_integration() {
    let p = Petsc::init(&[]).unwrap();
    let mesh = lobule(&p);
    let maps = lobule_maps(&mesh);
    let mut problem = problem(&mesh, &maps, &field(1.0, 1.0));
    problem.step_to(3.0 * DT).unwrap();
    let start = problem.jacobian_shifts().to_vec();
    let uniform = problem.total_mass().unwrap();
    problem
        .modify(|u| {
            u.with_array_mut(|u| u.iter_mut().for_each(|u| *u += 0.5))?;
            Ok(())
        })
        .unwrap();
    assert!(matches!(
        problem.mass_balance(),
        Err(PdeError::NotStepped { .. })
    ));
    problem.step_to(6.0 * DT).unwrap();
    let shifts = problem.jacobian_shifts();
    // The three steps after the change repeat the shifts of the first three: the start of
    // BDF2, then BDF2.
    assert_eq!(&shifts[start.len()..], &start[..], "{shifts:?}");
    assert!(problem.mass_balance().is_ok());
    // The change is in the solution: the field relaxes towards the Dirichlet value 1.
    let mass = problem.total_mass().unwrap();
    assert!(mass > uniform, "{mass} against {uniform}");
}

/// An exchange for another discretization than the problem's is refused.
#[test]
fn other_discretization_is_an_error() {
    let p = Petsc::init(&[]).unwrap();
    let mesh = lobule(&p);
    let maps = lobule_maps(&mesh);
    let first = problem(&mesh, &maps, &field(1.0, 1.0));
    let exchange = exchange(&mesh, &maps, &first);
    drop(first);
    let second = problem(&mesh, &maps, &field(1.0, 1.0));
    let mut means = vec![0.0; exchange.len()];
    assert!(matches!(
        exchange.means(&second, &mut means),
        Err(PdeError::Unsupported(_))
    ));
}

/// Changes of the wrong length on one process only: an error on every process, none waits.
#[test]
fn wrong_length_on_one_process_is_an_error_everywhere() {
    let p = Petsc::init(&[]).unwrap();
    let mesh = lobule(&p);
    let maps = lobule_maps(&mesh);
    let mut problem = problem(&mesh, &maps, &field(1.0, 1.0));
    let exchange = exchange(&mesh, &maps, &problem);
    let comm = p.comm();
    let n = exchange.len() + usize::from(comm.rank() == comm.size() - 1);
    let mut means = vec![0.0; n];
    assert!(matches!(
        exchange.means(&problem, &mut means),
        Err(PdeError::Unsupported(_))
    ));
    assert!(matches!(
        exchange.give_back(&mut problem, &means),
        Err(PdeError::Unsupported(_))
    ));
}

#[test]
fn mpi_world_has_the_expected_size() {
    let p = Petsc::init(&[]).unwrap();
    assert_eq!(p.comm().size(), test_support::ranks());
}

#[test]
fn parallel_on_2_processes() {
    test_support::parallel(
        2,
        &[
            "mpi_world_has_the_expected_size",
            "means_of_constant_field_are_the_constant",
            "give_back_conserves_total",
            "steep_uptake_stays_nonnegative",
            "uniform_change_with_resupply",
            "modify_restarts_the_time_integration",
            "other_discretization_is_an_error",
            "wrong_length_on_one_process_is_an_error_everywhere",
        ],
    );
}
