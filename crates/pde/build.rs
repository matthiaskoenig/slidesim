//! Compiles the QFunctions of `qfunctions/` (`CEED_QFUNCTION` style C headers, exported by
//! `qfunctions/qfunctions.c`) against libCEED's headers (`DEP_CEED_INCLUDE`, from `ceed-sys`),
//! and exports the absolute path of their directory as `SLIDESIM_PDE_QFUNCTIONS`: libCEED
//! backends which compile QFunctions at run time (GPU JIT) read the sources from there.
//!
//! For wasm32-wasip1 the QFunctions are compiled with wasi-sdk's clang, its sysroot and the flags
//! of the wasi libraries (`crates/petsc-sys/wasi_sdk.rs`).

use std::env;
use std::path::PathBuf;

#[path = "../petsc-sys/wasi_sdk.rs"]
mod wasi_sdk;

/// C flags of every C file in slidesim (reproducible floating point on every target).
const C_FLAGS: [&str; 2] = ["-ffp-contract=off", "-fno-fast-math"];

fn main() {
    let dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("qfunctions");
    let dir = dir.canonicalize().expect("qfunctions directory");
    println!("cargo:rerun-if-changed={}", dir.display());
    println!("cargo:rustc-env=SLIDESIM_PDE_QFUNCTIONS={}", dir.display());

    let includes = env::var_os("DEP_CEED_INCLUDE").expect("ceed-sys exports DEP_CEED_INCLUDE");
    let mut build = cc::Build::new();
    if env::var("TARGET").unwrap() == wasi_sdk::WASI_TARGET {
        println!("cargo:rerun-if-env-changed={}", wasi_sdk::VAR);
        let sdk = wasi_sdk::locate(env::var_os(wasi_sdk::VAR), env::var_os("HOME"))
            .unwrap_or_else(|e| panic!("{e}"));
        build
            .compiler(wasi_sdk::clang(&sdk))
            .archiver(wasi_sdk::archiver(&sdk));
        for flag in wasi_sdk::wasi_c_flags(&sdk) {
            build.flag(flag);
        }
    }
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
