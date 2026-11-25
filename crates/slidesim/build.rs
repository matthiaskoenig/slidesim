//! An rpath to the PETSc and libCEED libraries the binary links (`$PETSC_DIR/lib` and
//! `$CEED_DIR/lib`, the installations `petsc-sys` and `ceed-sys` build against): `slidesim`
//! runs without `LD_LIBRARY_PATH`. The link arguments of a build script apply to its own
//! package, so this is the one place for them (see `crates/petsc-sys/build.rs`).

use std::env;
use std::path::PathBuf;

fn main() {
    let mut libs: Vec<PathBuf> = Vec::new();
    for var in ["PETSC_DIR", "CEED_DIR"] {
        println!("cargo:rerun-if-env-changed={var}");
        let dir = env::var(var)
            .map(PathBuf::from)
            .unwrap_or_else(|_| panic!("{var} unset: source deps/install/<config>/env.sh"));
        let lib = dir.join("lib");
        // One installation directory usually holds both.
        if !libs.contains(&lib) {
            println!("cargo:rustc-link-arg-bins=-Wl,-rpath,{}", lib.display());
            libs.push(lib);
        }
    }
}
