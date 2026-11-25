//! Tests of finite elements (`PetscFE`) and of discretizations on a DMPlex: fields, essential
//! boundaries, sections, cell restrictions and the scatters between local and global vectors.
//! nextest runs each test in its own process, so every test initializes PETSc once.

use std::collections::BTreeSet;
use std::path::Path;

use petsc::sys;
use petsc::{Dm, Fe, InsertMode, Ksp, Petsc, Vector};

/// The gmsh square [0, 1] x [0, 1] of `testdata/meshes`: 31 vertices (16 on the boundary), 44
/// triangles, boundary edges in the label `Face Sets` with the values 11 to 14 (one per side).
fn unit_square(p: &Petsc) -> Dm {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/meshes/unit_square.msh");
    Dm::plex_from_file(&p.comm(), &path, "square", true).unwrap()
}

const VERTICES: usize = 31;
const BOUNDARY_VERTICES: usize = 16;
const CELLS: usize = 44;

/// `unit_square` with the P1 field and affine coordinates, and essential boundaries on the label
/// values `essential` of `Face Sets`.
fn p1(p: &Petsc, essential: &[sys::PetscInt]) -> Dm {
    p1_on(&unit_square(p), essential)
}

/// A clone of `mesh` with the P1 field and affine coordinates, and essential boundaries on the
/// label values `essential` of `Face Sets`.
fn p1_on(mesh: &Dm, essential: &[sys::PetscInt]) -> Dm {
    let mut dm = mesh.clone_mesh().unwrap();
    dm.set_coordinate_disc(None, false).unwrap();
    dm.add_field(&Fe::lagrange(2, 1, true, 1, 2).unwrap())
        .unwrap();
    dm.create_ds().unwrap();
    if !essential.is_empty() {
        dm.add_essential_boundary("wall", "Face Sets", essential, 0)
            .unwrap();
    }
    dm
}

#[test]
fn lagrange_p1_on_the_reference_triangle() {
    let _p = Petsc::init(&[]).unwrap();
    let fe = Fe::lagrange(2, 1, true, 1, 2).unwrap();
    let q = fe.quadrature().unwrap();
    assert_eq!(q.dim, 2);
    assert_eq!(q.points.len(), 2 * q.weights.len());
    // The reference triangle (-1, -1), (1, -1), (-1, 1) has area 2.
    assert!((q.weights.iter().sum::<f64>() - 2.0).abs() < 1e-14);
    let t = fe.tabulation().unwrap();
    assert_eq!(
        (t.num_points, t.num_basis, t.num_comp, t.dim),
        (q.weights.len(), 3, 1, 2)
    );
    // The basis functions of the vertices in order, -(x + y) / 2, (1 + x) / 2, (1 + y) / 2.
    let gradients = [[-0.5, -0.5], [0.5, 0.0], [0.0, 0.5]];
    for i in 0..t.num_points {
        let (x, y) = (q.points[2 * i], q.points[2 * i + 1]);
        let expected = [-(x + y) / 2.0, (1.0 + x) / 2.0, (1.0 + y) / 2.0];
        for b in 0..3 {
            assert!((t.values[i * 3 + b] - expected[b]).abs() < 1e-14);
            for d in 0..2 {
                assert!((t.derivatives[(i * 3 + b) * 2 + d] - gradients[b][d]).abs() < 1e-14);
            }
        }
    }
}

#[test]
fn clone_mesh_has_its_own_labels() {
    let p = Petsc::init(&[]).unwrap();
    let dm = unit_square(&p);
    let mut clone = dm.clone_mesh().unwrap();
    assert_ne!(clone.as_raw(), dm.as_raw());
    assert_eq!(
        clone.height_stratum(0).unwrap(),
        dm.height_stratum(0).unwrap()
    );
    assert!(clone.is_simplex().unwrap());
    let left = dm.stratum_points("Face Sets", 11).unwrap();
    assert_eq!(clone.stratum_points("Face Sets", 11).unwrap(), left);
    // An essential boundary completes the clone's label with the vertices, not the original's.
    clone.set_coordinate_disc(None, false).unwrap();
    clone
        .add_field(&Fe::lagrange(2, 1, true, 1, 2).unwrap())
        .unwrap();
    clone.create_ds().unwrap();
    clone
        .add_essential_boundary("left", "Face Sets", &[11], 0)
        .unwrap();
    assert_eq!(clone.stratum_points("Face Sets", 11).unwrap().len(), 4 + 5);
    assert_eq!(dm.stratum_points("Face Sets", 11).unwrap(), left);
}

#[test]
fn label_values_on_every_process() {
    let p = Petsc::init(&[]).unwrap();
    let dm = unit_square(&p);
    assert_eq!(
        dm.label_values_global("Face Sets").unwrap(),
        [11, 12, 13, 14]
    );
    assert!(dm.label_values_global("no such label").unwrap().is_empty());
}

#[test]
fn cell_restrictions_of_the_field_and_the_coordinates() {
    let p = Petsc::init(&[]).unwrap();
    let dm = p1(&p, &[]);
    let u = dm.local_offsets(0).unwrap();
    assert_eq!(
        (u.num_cells, u.cell_size, u.num_comp, u.local_size),
        (CELLS, 3, 1, VERTICES)
    );
    assert_eq!(u.offsets.len(), CELLS * 3);
    let used: BTreeSet<_> = u.offsets.iter().copied().collect();
    assert_eq!(used, (0..VERTICES as sys::PetscInt).collect());
    // A cell has three distinct nodes, the vertices of its closure.
    let cells = dm.height_stratum(0).unwrap();
    for (e, c) in cells.enumerate() {
        let nodes: BTreeSet<_> = u.offsets[3 * e..3 * e + 3].iter().copied().collect();
        let vertices = dm.depth_stratum(0).unwrap();
        let closure: BTreeSet<_> = dm
            .closure(c)
            .unwrap()
            .into_iter()
            .filter(|q| vertices.contains(q))
            .map(|q| dm.local_dofs(q).unwrap().start as sys::PetscInt)
            .collect();
        assert_eq!(nodes, closure);
    }
    let x = dm.coordinate_dm().unwrap().local_offsets(0).unwrap();
    assert_eq!((x.num_cells, x.cell_size, x.num_comp), (CELLS, 3, 2));
    let coordinates = dm.local_coordinates().unwrap();
    assert_eq!(coordinates.len(), x.local_size);
    assert_eq!(coordinates.len(), 2 * VERTICES);
    assert!(coordinates.iter().all(|c| (0.0..=1.0).contains(c)));
}

#[test]
fn essential_boundaries_constrain_dofs() {
    let p = Petsc::init(&[]).unwrap();
    let dm = p1(&p, &[11, 12, 13, 14]);
    let global = Vector::create_global(&dm).unwrap();
    assert_eq!(global.size().unwrap(), VERTICES - BOUNDARY_VERTICES);
    let local = Vector::create_local(&dm).unwrap();
    assert_eq!(local.size().unwrap(), VERTICES);
    let ltog = dm.local_to_global_indices().unwrap();
    assert_eq!(ltog.len(), VERTICES);
    assert_eq!(ltog.iter().filter(|&&g| g < 0).count(), BOUNDARY_VERTICES);
    let mut owned: Vec<_> = ltog.iter().copied().filter(|&g| g >= 0).collect();
    owned.sort_unstable();
    assert_eq!(
        owned,
        (0..(VERTICES - BOUNDARY_VERTICES) as i32).collect::<Vec<_>>()
    );
    let constrained = dm
        .depth_stratum(0)
        .unwrap()
        .filter(|&v| dm.constrained_dofs(v).unwrap() == 1)
        .count();
    assert_eq!(constrained, BOUNDARY_VERTICES);
    // The closure of an edge: the edge and its two vertices.
    let edge = dm.depth_stratum(1).unwrap().start;
    let closure = dm.closure(edge).unwrap();
    assert_eq!(closure.len(), 3);
    assert_eq!(closure[0], edge);
}

#[test]
fn scatters_between_local_and_global_vectors() {
    let p = Petsc::init(&[]).unwrap();
    let dm = p1(&p, &[11]);
    let mut global = Vector::create_global(&dm).unwrap();
    global.set(1.0).unwrap();
    let mut local = Vector::create_local(&dm).unwrap();
    local.set(7.0).unwrap();
    dm.global_to_local(&global, &mut local).unwrap();
    // The 5 constrained entries (the left side) keep their value.
    let values = local.with_array(<[f64]>::to_vec).unwrap();
    assert_eq!(values.iter().filter(|&&v| v == 7.0).count(), 5);
    assert_eq!(values.iter().filter(|&&v| v == 1.0).count(), VERTICES - 5);
    // Adding the local entries into the global vector (one process: each entry once).
    dm.local_to_global(&local, InsertMode::Add, &mut global)
        .unwrap();
    global
        .with_array(|g| assert!(g.iter().all(|&v| v == 2.0)))
        .unwrap();
    dm.local_to_global(&local, InsertMode::Insert, &mut global)
        .unwrap();
    global
        .with_array(|g| assert!(g.iter().all(|&v| v == 1.0)))
        .unwrap();
}

#[test]
fn essential_boundary_on_a_missing_label_is_error() {
    let p = Petsc::init(&[]).unwrap();
    let mut dm = p1(&p, &[]);
    let e = dm
        .add_essential_boundary("wall", "no such label", &[1], 0)
        .unwrap_err();
    assert_eq!(e.code, sys::PETSC_ERR_ARG_WRONG);
    assert!(e.message.contains("no such label"), "{e}");
}

#[test]
fn local_offsets_without_a_field_is_error() {
    let p = Petsc::init(&[]).unwrap();
    let dm = unit_square(&p);
    assert!(dm.local_offsets(0).is_err());
}

/// The global vector of the P1 field of `dm` (one process) with the values of `f` at the nodes.
fn nodal(dm: &Dm, f: impl Fn(f64, f64) -> f64) -> Vector {
    let coordinates = dm.local_coordinates().unwrap();
    let mut global = Vector::create_global(dm).unwrap();
    global
        .with_array_mut(|values| {
            for (i, g) in dm
                .local_to_global_indices()
                .unwrap()
                .into_iter()
                .enumerate()
            {
                if g >= 0 {
                    values[g as usize] = f(coordinates[2 * i], coordinates[2 * i + 1]);
                }
            }
        })
        .unwrap();
    global
}

#[test]
fn interpolation_to_a_refinement() {
    let p = Petsc::init(&[]).unwrap();
    let mesh = unit_square(&p);
    let refined = mesh.refine().unwrap();
    // Natural boundaries: P1 interpolation reproduces linear functions.
    let coarse = p1_on(&mesh, &[]);
    let mut fine = p1_on(&refined, &[]);
    fine.set_coarse_dm(&coarse).unwrap();
    let interpolation = coarse.create_interpolation(&fine).unwrap();
    let linear = |x: f64, y: f64| 1.0 + 2.0 * x - 3.0 * y;
    let mut u = Vector::create_global(&fine).unwrap();
    interpolation.mult(&nodal(&coarse, linear), &mut u).unwrap();
    u.axpy(-1.0, &nodal(&fine, linear)).unwrap();
    assert!(u.norm2().unwrap() < 1e-13, "{}", u.norm2().unwrap());
    // Essential boundaries on all sides: the constrained dofs are left out on both levels. A hat
    // function of the coarse level is 1 at its vertex, 1/2 at the midpoints of its edges.
    let coarse = p1_on(&mesh, &[11, 12, 13, 14]);
    let mut fine = p1_on(&refined, &[11, 12, 13, 14]);
    fine.set_coarse_dm(&coarse).unwrap();
    let interpolation = coarse.create_interpolation(&fine).unwrap();
    let mut hat = Vector::create_global(&coarse).unwrap();
    hat.set(0.0).unwrap();
    hat.with_array_mut(|h| h[0] = 1.0).unwrap();
    let mut u = Vector::create_global(&fine).unwrap();
    interpolation.mult(&hat, &mut u).unwrap();
    let u = u.with_array(<[f64]>::to_vec).unwrap();
    assert_eq!(u.len(), 105 - 32);
    assert_eq!(u.iter().filter(|&&v| v == 1.0).count(), 1);
    let halves = u.iter().filter(|&&v| v == 0.5).count();
    assert!((3..=8).contains(&halves), "{halves}");
    assert!(u.iter().all(|&v| v == 0.0 || v == 0.5 || v == 1.0));
}

#[test]
fn ksp_takes_an_active_or_inactive_dm() {
    let p = Petsc::init(&[]).unwrap();
    let dm = p1(&p, &[]);
    let b = Vector::create_global(&dm).unwrap();
    let mut x = Vector::create_global(&dm).unwrap();
    let mut message = |active: bool| {
        let mut ksp = Ksp::create(&p.comm()).unwrap();
        ksp.set_dm(&dm, active).unwrap();
        let mut raw = std::ptr::null_mut();
        assert_eq!(unsafe { sys::KSPGetDM(ksp.as_raw(), &mut raw) }, 0);
        assert_eq!(raw, dm.as_raw());
        // Without operators either way: the error tells which.
        ksp.solve(&b, &mut x).unwrap_err().message
    };
    // Active, the DM should compute the operators.
    assert!(message(true).contains("DMKSPSetComputeOperators"));
    // Inactive, the operators are the caller's to set.
    assert!(!message(false).contains("DMKSPSetComputeOperators"));
}
