//! Spike: -Δu = 2π² sin(πx) sin(πy) on the unit square, u = 0 on the boundary, with P1 elements
//! on a DMPlex box mesh of n x n squares split into triangles (n = 64, option `-n`). The
//! operator is a libCEED operator (gallery `Poisson2DBuild` / `Poisson2DApply`) on the
//! `/cpu/self/opt/blocked` backend applied through a PETSc `MATSHELL`; CG with Jacobi solves it.
//! The exact solution is sin(πx) sin(πy).

mod ffi {
    #![allow(
        non_upper_case_globals,
        non_camel_case_types,
        non_snake_case,
        dead_code
    )]
    #![allow(clippy::all, unsafe_op_in_unsafe_fn, improper_ctypes, unused_imports)]
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

use std::f64::consts::PI;
use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::ptr::{null, null_mut};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

// Note: `Vec` below is PETSc's vector (the glob import shadows the prelude); Rust vectors are
// written `std::vec::Vec`.
use ffi::*;

/// The libCEED backend, always the full resource (see PATCHES.md): `/cpu/self` resolves to
/// different backends natively (avx) and in wasm (opt).
const CEED_RESOURCE: &CStr = c"/cpu/self/opt/blocked";
/// Quadrature degree of the PetscFE (operator, right-hand side and L2 error).
const QUAD_ORDER: PetscInt = 3;

macro_rules! petsc {
    ($e:expr) => {{
        let ierr = unsafe { $e };
        if ierr != 0 {
            panic!("PETSc error {} in `{}`", ierr, stringify!($e));
        }
    }};
}

macro_rules! ceed {
    ($e:expr) => {{
        let ierr = unsafe { $e };
        if ierr != 0 {
            panic!("libCEED error {} in `{}`", ierr, stringify!($e));
        }
    }};
}

/// `-rust_libm`: sin() of the `libm` crate (a port of musl's, the same code as wasi-libc's) on
/// every target instead of the platform's (glibc natively, wasi-libc in wasm).
static RUST_LIBM: AtomicBool = AtomicBool::new(false);

fn sin(x: f64) -> f64 {
    if RUST_LIBM.load(Ordering::Relaxed) {
        libm::sin(x)
    } else {
        x.sin()
    }
}

fn exact(x: f64, y: f64) -> f64 {
    sin(PI * x) * sin(PI * y)
}

fn forcing(x: f64, y: f64) -> f64 {
    2.0 * PI * PI * exact(x, y)
}

/// PETSc point function of the exact solution (for the L2 error).
unsafe extern "C" fn exact_fn(
    _dim: PetscInt,
    _t: PetscReal,
    x: *const PetscReal,
    _nc: PetscInt,
    u: *mut PetscScalar,
    _ctx: *mut c_void,
) -> PetscErrorCode {
    unsafe { *u = exact(*x, *x.add(1)) };
    0
}

/// PETSc point function of the boundary value.
unsafe extern "C" fn zero_fn(
    _dim: PetscInt,
    _t: PetscReal,
    _x: *const PetscReal,
    _nc: PetscInt,
    u: *mut PetscScalar,
    _ctx: *mut c_void,
) -> PetscErrorCode {
    unsafe { *u = 0.0 };
    0
}

/// libCEED QFunction of the right-hand side: v = w det(J) f(x).
/// Inputs: x [2][Q] (interp of the coordinates), dx [2][2][Q] (dx_c/dX_j at [j][c]), w [Q].
unsafe extern "C" fn rhs_qf(
    _ctx: *mut c_void,
    q: CeedInt,
    input: *const *const CeedScalar,
    output: *const *mut CeedScalar,
) -> c_int {
    let q = q as usize;
    let (x, dx, w, v) = unsafe {
        (
            std::slice::from_raw_parts(*input, 2 * q),
            std::slice::from_raw_parts(*input.add(1), 4 * q),
            std::slice::from_raw_parts(*input.add(2), q),
            std::slice::from_raw_parts_mut(*output, q),
        )
    };
    for i in 0..q {
        let det = dx[i] * dx[3 * q + i] - dx[q + i] * dx[2 * q + i];
        if det <= 0.0 {
            // Poisson2DBuild divides by det(J) without abs(): the mesh must be positively oriented.
            return CEED_ERROR_DIMENSION as c_int;
        }
        v[i] = w[i] * det * forcing(x[i], x[q + i]);
    }
    0
}

/// PETSc stores callbacks type-erased (`PetscVoidFn`, `PetscErrorCodeFn`) and casts them back to
/// their real type before the call (which wasm's typed `call_indirect` requires).
fn erase(f: *const ()) -> unsafe extern "C" fn() {
    unsafe { std::mem::transmute::<*const (), unsafe extern "C" fn()>(f) }
}

/// State of the MATSHELL: y = A x with A the libCEED operator on the local vectors.
struct Shell {
    dm: DM,
    x_loc: Vec,
    y_loc: Vec,
    x_ceed: CeedVector,
    y_ceed: CeedVector,
    op: CeedOperator,
    diag: Vec,
}

/// y_loc = op(x_loc) with libCEED vectors borrowing the PETSc arrays.
unsafe fn apply_local(s: &Shell, x_loc: Vec, y_loc: Vec) {
    let mut x: *const PetscScalar = null();
    let mut y: *mut PetscScalar = null_mut();
    petsc!(VecGetArrayRead(x_loc, &mut x));
    petsc!(VecGetArrayWrite(y_loc, &mut y));
    ceed!(CeedVectorSetArray(
        s.x_ceed,
        CEED_MEM_HOST,
        CEED_USE_POINTER,
        x as *mut _
    ));
    ceed!(CeedVectorSetArray(
        s.y_ceed,
        CEED_MEM_HOST,
        CEED_USE_POINTER,
        y
    ));
    ceed!(CeedOperatorApply(
        s.op,
        s.x_ceed,
        s.y_ceed,
        CEED_REQUEST_IMMEDIATE
    ));
    let mut p: *mut CeedScalar = null_mut();
    ceed!(CeedVectorTakeArray(s.x_ceed, CEED_MEM_HOST, &mut p));
    ceed!(CeedVectorTakeArray(s.y_ceed, CEED_MEM_HOST, &mut p));
    petsc!(VecRestoreArrayRead(x_loc, &mut x));
    petsc!(VecRestoreArrayWrite(y_loc, &mut y));
}

unsafe fn shell(a: Mat) -> &'static Shell {
    let mut ctx: *mut c_void = null_mut();
    petsc!(MatShellGetContext(
        a,
        &mut ctx as *mut *mut c_void as *mut c_void
    ));
    unsafe { &*(ctx as *const Shell) }
}

unsafe extern "C" fn shell_mult(a: Mat, x: Vec, y: Vec) -> PetscErrorCode {
    let s = unsafe { shell(a) };
    // Constrained (boundary) dofs are not set by DMGlobalToLocal and stay zero.
    petsc!(VecZeroEntries(s.x_loc));
    petsc!(DMGlobalToLocal(s.dm, x, INSERT_VALUES, s.x_loc));
    unsafe { apply_local(s, s.x_loc, s.y_loc) };
    petsc!(VecZeroEntries(y));
    petsc!(DMLocalToGlobal(s.dm, s.y_loc, ADD_VALUES, y));
    0
}

unsafe extern "C" fn shell_diagonal(a: Mat, d: Vec) -> PetscErrorCode {
    let s = unsafe { shell(a) };
    petsc!(VecCopy(s.diag, d));
    0
}

/// libCEED H1 basis from the PetscFE tabulation (nodes in closure order, matching
/// DMPlexGetLocalOffsets), as Ratel and HONEE do.
unsafe fn basis_from_fe(ceed: Ceed, fe: PetscFE, num_comp: CeedInt) -> CeedBasis {
    let mut quad: PetscQuadrature = null_mut();
    let (mut dim, mut nc, mut nq) = (0, 0, 0);
    let (mut points, mut weights): (*const PetscReal, *const PetscReal) = (null(), null());
    petsc!(PetscFEGetQuadrature(fe, &mut quad));
    petsc!(PetscQuadratureGetData(
        quad,
        &mut dim,
        &mut nc,
        &mut nq,
        &mut points,
        &mut weights
    ));
    let mut tab: PetscTabulation = null_mut();
    petsc!(PetscFEGetCellTabulation(fe, 1, &mut tab));
    let t = unsafe { &*tab };
    assert!(
        t.Nc == 1 && t.cdim == dim && t.Np == nq,
        "scalar FE expected"
    );
    let (d, q, p) = (dim as usize, nq as usize, t.Nb as usize);
    let (t0, t1) = unsafe {
        (
            std::slice::from_raw_parts(*t.T, q * p),
            std::slice::from_raw_parts(*t.T.add(1), q * p * d),
        )
    };
    // PETSc: T0[q][p], T1[q][p][d]; libCEED: interp[q][p], grad[d][q][p].
    let mut grad = vec![0.0; d * q * p];
    for (k, g) in grad.iter_mut().enumerate() {
        let (j, qi, pi) = (k / (q * p), (k / p) % q, k % p);
        *g = t1[(qi * p + pi) * d + j];
    }
    let mut basis: CeedBasis = null_mut();
    ceed!(CeedBasisCreateH1(
        ceed,
        CEED_TOPOLOGY_TRIANGLE,
        num_comp,
        p as CeedInt,
        q as CeedInt,
        t0.as_ptr(),
        grad.as_ptr(),
        points,
        weights,
        &mut basis
    ));
    basis
}

/// libCEED restriction of field 0 of `dm` (cells) from DMPlexGetLocalOffsets.
unsafe fn restriction_from_plex(ceed: Ceed, dm: DM) -> (CeedElemRestriction, PetscInt, PetscInt) {
    let (mut ncell, mut cell_size, mut ncomp, mut lsize) = (0, 0, 0, 0);
    let mut offsets: *mut PetscInt = null_mut();
    petsc!(DMPlexGetLocalOffsets(
        dm,
        null_mut(),
        0,
        0,
        0,
        &mut ncell,
        &mut cell_size,
        &mut ncomp,
        &mut lsize,
        &mut offsets
    ));
    let n = (ncell * cell_size) as usize;
    let off: std::vec::Vec<CeedInt> = unsafe { std::slice::from_raw_parts(offsets, n) }
        .iter()
        .map(|&o| CeedInt::try_from(o).expect("offset fits CeedInt"))
        .collect();
    petsc!(PetscTrFree.unwrap()(
        offsets as *mut c_void,
        line!() as c_int,
        c"restriction_from_plex".as_ptr(),
        c"main.rs".as_ptr()
    ));
    let mut r: CeedElemRestriction = null_mut();
    ceed!(CeedElemRestrictionCreate(
        ceed,
        ncell as CeedInt,
        cell_size as CeedInt,
        ncomp as CeedInt,
        1,
        lsize as CeedSize,
        CEED_MEM_HOST,
        CEED_COPY_VALUES,
        off.as_ptr(),
        &mut r
    ));
    (r, ncell, lsize)
}

/// CeedVector holding a copy of a PETSc vector.
unsafe fn ceed_copy(ceed: Ceed, v: Vec) -> CeedVector {
    let mut n: PetscInt = 0;
    let mut a: *const PetscScalar = null();
    let mut c: CeedVector = null_mut();
    petsc!(VecGetLocalSize(v, &mut n));
    ceed!(CeedVectorCreate(ceed, n as CeedSize, &mut c));
    petsc!(VecGetArrayRead(v, &mut a));
    ceed!(CeedVectorSetArray(
        c,
        CEED_MEM_HOST,
        CEED_COPY_VALUES,
        a as *mut _
    ));
    petsc!(VecRestoreArrayRead(v, &mut a));
    c
}

/// Global vector of a local CeedVector (local to global with ADD_VALUES), destroys `c`.
unsafe fn to_global(dm: DM, c: &mut CeedVector, loc: Vec, glob: Vec) {
    let mut a: *mut PetscScalar = null_mut();
    petsc!(VecGetArrayWrite(loc, &mut a));
    let mut src: *const CeedScalar = null();
    ceed!(CeedVectorGetArrayRead(*c, CEED_MEM_HOST, &mut src));
    let mut n: PetscInt = 0;
    petsc!(VecGetLocalSize(loc, &mut n));
    unsafe { std::ptr::copy_nonoverlapping(src, a, n as usize) };
    ceed!(CeedVectorRestoreArrayRead(*c, &mut src));
    petsc!(VecRestoreArrayWrite(loc, &mut a));
    ceed!(CeedVectorDestroy(c));
    petsc!(VecZeroEntries(glob));
    petsc!(DMLocalToGlobal(dm, loc, ADD_VALUES, glob));
}

fn main() {
    let t_start = Instant::now();

    // PETSc options from the command line.
    let args: std::vec::Vec<CString> = std::env::args().map(|a| CString::new(a).unwrap()).collect();
    let mut argv: std::vec::Vec<*mut c_char> =
        args.iter().map(|a| a.as_ptr() as *mut c_char).collect();
    argv.push(null_mut());
    let mut argc = args.len() as c_int;
    let mut argv_ptr = argv.as_mut_ptr();
    petsc!(PetscInitialize(&mut argc, &mut argv_ptr, null(), null()));
    let comm = unsafe { PETSC_COMM_WORLD };

    let mut n: PetscInt = 64;
    petsc!(PetscOptionsGetInt(
        null_mut(),
        null(),
        c"-n".as_ptr(),
        &mut n,
        null_mut()
    ));
    let mut rust_libm = false;
    petsc!(PetscOptionsGetBool(
        null_mut(),
        null(),
        c"-rust_libm".as_ptr(),
        &mut rust_libm,
        null_mut()
    ));
    RUST_LIBM.store(rust_libm, Ordering::Relaxed);

    // libCEED, the explicit backend.
    let mut ceed: Ceed = null_mut();
    ceed!(CeedInit(CEED_RESOURCE.as_ptr(), &mut ceed));
    let mut resource: *const c_char = null();
    ceed!(CeedGetResource(ceed, &mut resource));
    let resource = unsafe { CStr::from_ptr(resource) };
    println!("libCEED resource     {}", resource.to_str().unwrap());
    assert_eq!(
        resource, CEED_RESOURCE,
        "libCEED resolved a different backend"
    );

    // Mesh: n x n squares, each split into two triangles. DMPlexCreateBoxMesh with simplex = true
    // needs a mesh generator (Triangle) in 2D, which neither build has; the box of quadrilaterals
    // goes through the refine_tosimplex transform (one diagonal per square) instead.
    let t_setup = Instant::now();
    let mut quads: DM = null_mut();
    let mut dm: DM = null_mut();
    let faces = [n, n];
    let (lower, upper) = ([0.0, 0.0], [1.0, 1.0]);
    petsc!(DMPlexCreateBoxMesh(
        comm,
        2,
        false,
        faces.as_ptr(),
        lower.as_ptr(),
        upper.as_ptr(),
        null(),
        true,
        0,
        false,
        &mut quads
    ));
    let mut tr: DMPlexTransform = null_mut();
    petsc!(DMPlexTransformCreate(comm, &mut tr));
    petsc!(DMPlexTransformSetDM(tr, quads));
    petsc!(DMPlexTransformSetType(
        tr,
        DMPLEXREFINETOSIMPLEX.as_ptr() as *const c_char
    ));
    petsc!(DMPlexTransformSetUp(tr));
    petsc!(DMPlexTransformApply(tr, quads, &mut dm));
    petsc!(DMPlexTransformDestroy(&mut tr));
    petsc!(DMDestroy(&mut quads));

    // P1 for the solution and the coordinates (same quadrature).
    let mut fe: PetscFE = null_mut();
    let mut fe_coord: PetscFE = null_mut();
    petsc!(PetscFECreateLagrange(
        comm, 2, 1, true, 1, QUAD_ORDER, &mut fe
    ));
    petsc!(PetscFECreateLagrange(
        comm,
        2,
        2,
        true,
        1,
        QUAD_ORDER,
        &mut fe_coord
    ));
    petsc!(DMSetCoordinateDisc(dm, fe_coord, false, true));
    petsc!(DMSetField(dm, 0, null_mut(), fe as PetscObject));
    petsc!(DMCreateDS(dm));

    // Zero Dirichlet on the boundary: constrained dofs of the section.
    let mut label: DMLabel = null_mut();
    petsc!(DMCreateLabel(dm, c"boundary".as_ptr()));
    petsc!(DMGetLabel(dm, c"boundary".as_ptr(), &mut label));
    petsc!(DMPlexMarkBoundaryFaces(dm, 1, label));
    petsc!(DMPlexLabelComplete(dm, label));
    let ids: [PetscInt; 1] = [1];
    let bc = erase(zero_fn as *const ());
    petsc!(DMAddBoundary(
        dm,
        DM_BC_ESSENTIAL,
        c"wall".as_ptr(),
        label,
        1,
        ids.as_ptr(),
        0,
        0,
        null(),
        Some(bc),
        None,
        null_mut(),
        null_mut()
    ));

    let mut u: Vec = null_mut();
    let mut b: Vec = null_mut();
    let mut x_loc: Vec = null_mut();
    let mut y_loc: Vec = null_mut();
    let mut diag: Vec = null_mut();
    petsc!(DMCreateGlobalVector(dm, &mut u));
    petsc!(VecDuplicate(u, &mut b));
    petsc!(VecDuplicate(u, &mut diag));
    petsc!(DMCreateLocalVector(dm, &mut x_loc));
    petsc!(VecDuplicate(x_loc, &mut y_loc));
    let (mut n_glob, mut n_loc) = (0, 0);
    petsc!(VecGetSize(u, &mut n_glob));
    petsc!(VecGetLocalSize(x_loc, &mut n_loc));

    // libCEED objects: restrictions and bases from the DMPlex / PetscFE.
    let (mut r_u, ncell, lsize) = unsafe { restriction_from_plex(ceed, dm) };
    assert_eq!(lsize, n_loc);
    let mut cdm: DM = null_mut();
    let mut coords: Vec = null_mut();
    petsc!(DMGetCoordinateDM(dm, &mut cdm));
    petsc!(DMGetCoordinatesLocal(dm, &mut coords));
    let (mut r_x, ncell_x, _) = unsafe { restriction_from_plex(ceed, cdm) };
    assert_eq!(ncell, ncell_x);
    let mut basis_u = unsafe { basis_from_fe(ceed, fe, 1) };
    let mut basis_x = unsafe { basis_from_fe(ceed, fe, 2) };
    let mut nq: CeedInt = 0;
    ceed!(CeedBasisGetNumQuadraturePoints(basis_u, &mut nq));
    let mut x_coord = unsafe { ceed_copy(ceed, coords) };

    // Geometric data: Poisson2DBuild, stored per quadrature point (3 components).
    let ncell = ncell as CeedInt;
    let qdata_len = (ncell * nq * 3) as CeedSize;
    let mut r_q: CeedElemRestriction = null_mut();
    let mut qdata: CeedVector = null_mut();
    ceed!(CeedElemRestrictionCreateStrided(
        ceed,
        ncell,
        nq,
        3,
        qdata_len,
        CEED_STRIDES_BACKEND.as_ptr(),
        &mut r_q
    ));
    ceed!(CeedVectorCreate(ceed, qdata_len, &mut qdata));
    let mut qf_build: CeedQFunction = null_mut();
    let mut qf_apply: CeedQFunction = null_mut();
    let mut qf_rhs: CeedQFunction = null_mut();
    ceed!(CeedQFunctionCreateInteriorByName(
        ceed,
        c"Poisson2DBuild".as_ptr(),
        &mut qf_build
    ));
    ceed!(CeedQFunctionCreateInteriorByName(
        ceed,
        c"Poisson2DApply".as_ptr(),
        &mut qf_apply
    ));
    let mut op_build: CeedOperator = null_mut();
    ceed!(CeedOperatorCreate(
        ceed,
        qf_build,
        CEED_QFUNCTION_NONE,
        CEED_QFUNCTION_NONE,
        &mut op_build
    ));
    ceed!(CeedOperatorSetField(
        op_build,
        c"dx".as_ptr(),
        r_x,
        basis_x,
        CEED_VECTOR_ACTIVE
    ));
    ceed!(CeedOperatorSetField(
        op_build,
        c"weights".as_ptr(),
        CEED_ELEMRESTRICTION_NONE,
        basis_x,
        CEED_VECTOR_NONE
    ));
    ceed!(CeedOperatorSetField(
        op_build,
        c"qdata".as_ptr(),
        r_q,
        CEED_BASIS_NONE,
        CEED_VECTOR_ACTIVE
    ));
    ceed!(CeedOperatorApply(
        op_build,
        x_coord,
        qdata,
        CEED_REQUEST_IMMEDIATE
    ));

    // The diffusion operator.
    let mut op: CeedOperator = null_mut();
    ceed!(CeedOperatorCreate(
        ceed,
        qf_apply,
        CEED_QFUNCTION_NONE,
        CEED_QFUNCTION_NONE,
        &mut op
    ));
    ceed!(CeedOperatorSetField(
        op,
        c"du".as_ptr(),
        r_u,
        basis_u,
        CEED_VECTOR_ACTIVE
    ));
    ceed!(CeedOperatorSetField(
        op,
        c"qdata".as_ptr(),
        r_q,
        CEED_BASIS_NONE,
        qdata
    ));
    ceed!(CeedOperatorSetField(
        op,
        c"dv".as_ptr(),
        r_u,
        basis_u,
        CEED_VECTOR_ACTIVE
    ));

    // Right-hand side: a QFunction written in Rust.
    let qf_rhs_fn: CeedQFunctionUser = Some(rhs_qf);
    ceed!(CeedQFunctionCreateInterior(
        ceed,
        1,
        qf_rhs_fn,
        c"".as_ptr(),
        &mut qf_rhs
    ));
    ceed!(CeedQFunctionAddInput(
        qf_rhs,
        c"x".as_ptr(),
        2,
        CEED_EVAL_INTERP
    ));
    ceed!(CeedQFunctionAddInput(
        qf_rhs,
        c"dx".as_ptr(),
        4,
        CEED_EVAL_GRAD
    ));
    ceed!(CeedQFunctionAddInput(
        qf_rhs,
        c"w".as_ptr(),
        1,
        CEED_EVAL_WEIGHT
    ));
    ceed!(CeedQFunctionAddOutput(
        qf_rhs,
        c"v".as_ptr(),
        1,
        CEED_EVAL_INTERP
    ));
    let mut op_rhs: CeedOperator = null_mut();
    ceed!(CeedOperatorCreate(
        ceed,
        qf_rhs,
        CEED_QFUNCTION_NONE,
        CEED_QFUNCTION_NONE,
        &mut op_rhs
    ));
    ceed!(CeedOperatorSetField(
        op_rhs,
        c"x".as_ptr(),
        r_x,
        basis_x,
        x_coord
    ));
    ceed!(CeedOperatorSetField(
        op_rhs,
        c"dx".as_ptr(),
        r_x,
        basis_x,
        x_coord
    ));
    ceed!(CeedOperatorSetField(
        op_rhs,
        c"w".as_ptr(),
        CEED_ELEMRESTRICTION_NONE,
        basis_x,
        CEED_VECTOR_NONE
    ));
    ceed!(CeedOperatorSetField(
        op_rhs,
        c"v".as_ptr(),
        r_u,
        basis_u,
        CEED_VECTOR_ACTIVE
    ));
    let mut rhs_ceed: CeedVector = null_mut();
    ceed!(CeedVectorCreate(ceed, n_loc as CeedSize, &mut rhs_ceed));
    ceed!(CeedOperatorApply(
        op_rhs,
        CEED_VECTOR_NONE,
        rhs_ceed,
        CEED_REQUEST_IMMEDIATE
    ));
    unsafe { to_global(dm, &mut rhs_ceed, x_loc, b) };

    // Jacobi: the assembled diagonal of the libCEED operator.
    let mut diag_ceed: CeedVector = null_mut();
    ceed!(CeedVectorCreate(ceed, n_loc as CeedSize, &mut diag_ceed));
    ceed!(CeedOperatorLinearAssembleDiagonal(
        op,
        diag_ceed,
        CEED_REQUEST_IMMEDIATE
    ));
    unsafe { to_global(dm, &mut diag_ceed, x_loc, diag) };

    // MATSHELL around the libCEED operator.
    let mut x_ceed: CeedVector = null_mut();
    let mut y_ceed: CeedVector = null_mut();
    ceed!(CeedVectorCreate(ceed, n_loc as CeedSize, &mut x_ceed));
    ceed!(CeedVectorCreate(ceed, n_loc as CeedSize, &mut y_ceed));
    let state = Box::new(Shell {
        dm,
        x_loc,
        y_loc,
        x_ceed,
        y_ceed,
        op,
        diag,
    });
    let mut a: Mat = null_mut();
    petsc!(MatCreateShell(
        comm,
        n_glob,
        n_glob,
        n_glob,
        n_glob,
        &*state as *const Shell as *mut c_void,
        &mut a
    ));
    petsc!(MatShellSetOperation(
        a,
        MATOP_MULT,
        Some(erase(shell_mult as *const ()))
    ));
    petsc!(MatShellSetOperation(
        a,
        MATOP_GET_DIAGONAL,
        Some(erase(shell_diagonal as *const ()))
    ));
    let setup_s = t_setup.elapsed().as_secs_f64();

    // CG + Jacobi.
    let mut ksp: KSP = null_mut();
    let mut pc: PC = null_mut();
    petsc!(KSPCreate(comm, &mut ksp));
    petsc!(KSPSetOperators(ksp, a, a));
    petsc!(KSPSetType(ksp, KSPCG.as_ptr() as *const c_char));
    petsc!(KSPGetPC(ksp, &mut pc));
    petsc!(PCSetType(pc, PCJACOBI.as_ptr() as *const c_char));
    petsc!(KSPSetTolerances(
        ksp,
        1e-12,
        PETSC_CURRENT as PetscReal,
        PETSC_CURRENT as PetscReal,
        PETSC_CURRENT as PetscInt
    ));
    petsc!(KSPSetFromOptions(ksp));
    let t_solve = Instant::now();
    petsc!(KSPSolve(ksp, b, u));
    let solve_s = t_solve.elapsed().as_secs_f64();
    let mut reason: KSPConvergedReason = 0;
    let mut its: PetscInt = 0;
    petsc!(KSPGetConvergedReason(ksp, &mut reason));
    petsc!(KSPGetIterationNumber(ksp, &mut its));
    assert!(reason > 0, "KSP diverged: reason {reason}");

    // Results: true residual, L2 error against the exact solution, 2-norm of the solution.
    let mut r: Vec = null_mut();
    let (mut res, mut bnorm, mut unorm, mut l2) = (0.0, 0.0, 0.0, 0.0);
    petsc!(VecDuplicate(u, &mut r));
    petsc!(MatMult(a, u, r));
    petsc!(VecAYPX(r, -1.0, b));
    petsc!(VecNorm(r, NORM_2, &mut res));
    petsc!(VecNorm(b, NORM_2, &mut bnorm));
    petsc!(VecNorm(u, NORM_2, &mut unorm));
    let mut funcs = [Some(
        exact_fn as unsafe extern "C" fn(_, _, _, _, _, _) -> _,
    )];
    let mut ctxs = [null_mut::<c_void>()];
    petsc!(DMComputeL2Diff(
        dm,
        0.0,
        funcs.as_mut_ptr(),
        ctxs.as_mut_ptr(),
        u,
        &mut l2
    ));

    // FNV-1a over the bits of every entry: bitwise identical solutions have the same hash.
    let mut ua: *const PetscScalar = null();
    petsc!(VecGetArrayRead(u, &mut ua));
    let hash = unsafe { std::slice::from_raw_parts(ua, n_glob as usize) }
        .iter()
        .fold(0xcbf29ce484222325u64, |h, v| {
            v.to_bits()
                .to_le_bytes()
                .iter()
                .fold(h, |h, &b| (h ^ b as u64).wrapping_mul(0x100000001b3))
        });
    petsc!(VecRestoreArrayRead(u, &mut ua));

    println!(
        "sin()                {}",
        if rust_libm {
            "libm crate"
        } else {
            "platform libm"
        }
    );
    println!(
        "mesh                 {n} x {n} squares, {ncell} triangles, P1, {nq} quadrature points"
    );
    println!("unknowns             {n_glob} (local vector {n_loc})");
    println!("CG + Jacobi          {its} iterations, rtol 1e-12");
    println!("relative residual    {:.16e}", res / bnorm);
    println!("L2 error             {l2:.16e}");
    println!("solution 2-norm      {unorm:.16e}");
    println!("solution hash        {hash:016x}");
    println!("time setup           {:.3} ms", setup_s * 1e3);
    println!("time solve           {:.3} ms", solve_s * 1e3);

    // Teardown.
    petsc!(KSPDestroy(&mut ksp));
    petsc!(MatDestroy(&mut a));
    petsc!(VecDestroy(&mut r));
    let mut s = state;
    ceed!(CeedVectorDestroy(&mut s.x_ceed));
    ceed!(CeedVectorDestroy(&mut s.y_ceed));
    ceed!(CeedOperatorDestroy(&mut s.op));
    ceed!(CeedOperatorDestroy(&mut op_rhs));
    ceed!(CeedOperatorDestroy(&mut op_build));
    ceed!(CeedQFunctionDestroy(&mut qf_rhs));
    ceed!(CeedQFunctionDestroy(&mut qf_apply));
    ceed!(CeedQFunctionDestroy(&mut qf_build));
    ceed!(CeedVectorDestroy(&mut qdata));
    ceed!(CeedVectorDestroy(&mut x_coord));
    ceed!(CeedElemRestrictionDestroy(&mut r_q));
    ceed!(CeedElemRestrictionDestroy(&mut r_x));
    ceed!(CeedElemRestrictionDestroy(&mut r_u));
    ceed!(CeedBasisDestroy(&mut basis_x));
    ceed!(CeedBasisDestroy(&mut basis_u));
    ceed!(CeedDestroy(&mut ceed));
    petsc!(VecDestroy(&mut s.diag));
    petsc!(VecDestroy(&mut s.y_loc));
    petsc!(VecDestroy(&mut s.x_loc));
    petsc!(VecDestroy(&mut b));
    petsc!(VecDestroy(&mut u));
    petsc!(PetscFEDestroy(&mut fe_coord));
    petsc!(PetscFEDestroy(&mut fe));
    petsc!(DMDestroy(&mut s.dm));
    petsc!(PetscFinalize());
    println!(
        "time total           {:.3} ms",
        t_start.elapsed().as_secs_f64() * 1e3
    );
}
