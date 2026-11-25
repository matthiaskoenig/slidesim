//! Compiles the QFunctions of `qfunctions/` (`CEED_QFUNCTION` style C headers, exported by
//! `qfunctions/qfunctions.c`) against libCEED's headers (`DEP_CEED_INCLUDE`, from `ceed-sys`),
//! and exports the absolute path of their directory as `SLIDESIM_PDE_QFUNCTIONS`: libCEED
//! backends which compile QFunctions at run time (GPU JIT) read the sources from there.

use std::env;
use std::path::PathBuf;

/// C flags of every C file in slidesim (reproducible floating point on every target).
const C_FLAGS: [&str; 2] = ["-ffp-contract=off", "-fno-fast-math"];

fn main() {
    let dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("qfunctions");
    let dir = dir.canonicalize().expect("qfunctions directory");
    println!("cargo:rerun-if-changed={}", dir.display());
    println!("cargo:rustc-env=SLIDESIM_PDE_QFUNCTIONS={}", dir.display());

    let includes = env::var_os("DEP_CEED_INCLUDE").expect("ceed-sys exports DEP_CEED_INCLUDE");
    let mut build = cc::Build::new();
    build
        .file(dir.join("qfunctions.c"))
        .includes(env::split_paths(&includes))
        .opt_level(3)
        .define("NDEBUG", None)
        .warnings(true)
        // A QFunction takes a context argument whether it uses one or not.
        .flag("-Wno-unused-parameter");
    for flag in C_FLAGS {
        build.flag(flag);
    }
    build.compile("slidesim_pde_qfunctions");
}
