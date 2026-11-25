//! An rpath to the PETSc and libCEED libraries the binary links (`$PETSC_DIR/lib` and
//! `$CEED_DIR/lib`, the installations `petsc-sys` and `ceed-sys` build against), and to PETSc's
//! PT-Scotch, which `petsc-sys` links too and which may live elsewhere (its `PTSCOTCH_LIB` in
//! PETSc's `petscvariables`): `slidesim` runs without `LD_LIBRARY_PATH`. The link arguments of a
//! build script apply to its own package, so this is the one place for them (see
//! `crates/petsc-sys/build.rs`).

use std::env;
use std::path::PathBuf;

// Shared with other build scripts, each using part of it.
#[allow(dead_code)]
#[path = "../petsc-sys/petscvariables.rs"]
mod petscvariables;

/// The installation directory in the environment variable `var`.
fn installation(var: &str) -> PathBuf {
    println!("cargo:rerun-if-env-changed={var}");
    env::var(var)
        .map(PathBuf::from)
        .unwrap_or_else(|_| panic!("{var} unset: source deps/install/<config>/env.sh"))
}

fn main() {
    let petsc_dir = installation("PETSC_DIR");
    let mut libs = vec![petsc_dir.join("lib"), installation("CEED_DIR").join("lib")];
    for file in petscvariables::files(&petsc_dir) {
        println!("cargo:rerun-if-changed={}", file.display());
    }
    // petsc-sys fails the build if PETSc has PT-Scotch without its flags, and reports why.
    if let Ok(Some(ptscotch)) = petscvariables::package(&petsc_dir, "PTSCOTCH") {
        let dirs = ptscotch.lib.rpaths.iter().chain(&ptscotch.lib.link_dirs);
        libs.extend(dirs.map(PathBuf::from));
    }
    // One installation directory usually holds several of them.
    let mut seen: Vec<&PathBuf> = Vec::new();
    for lib in &libs {
        if !seen.contains(&lib) {
            println!("cargo:rustc-link-arg-bins=-Wl,-rpath,{}", lib.display());
            seen.push(lib);
        }
    }
}
