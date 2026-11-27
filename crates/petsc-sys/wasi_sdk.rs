//! Where wasi-sdk is: bindgen uses its clang and libclang for the bindings of every target
//! (`$WASI_SDK_PATH`, default `~/opt/wasi-sdk`).
//!
//! For wasm32-wasip1 also the C toolchain: wasi-sdk's clang and llvm-ar, and the flags of every
//! C file compiled for it ([`wasi_c_flags`]).
//!
//! Standard library only: the build scripts of petsc-sys, ceed-sys and pde include it (`#[path]`),
//! each using part of it, and `tests/wasi_sdk.rs` tests it.

#![allow(dead_code)]

use std::ffi::OsString;
use std::path::{Path, PathBuf};

/// The environment variable which names the wasi-sdk directory.
pub const VAR: &str = "WASI_SDK_PATH";

/// The wasi-sdk directory without [`VAR`], relative to the home directory.
pub const DEFAULT: &str = "opt/wasi-sdk";

/// The wasi-sdk directory: `path` (the value of [`VAR`]), else [`DEFAULT`] in `home`. An error (a
/// message which says where it looked and how to point elsewhere) unless the directory has
/// `bin/clang` and libclang in `lib`.
pub fn locate(path: Option<OsString>, home: Option<OsString>) -> Result<PathBuf, String> {
    let (dir, origin) = match (path, home) {
        (Some(path), _) => (PathBuf::from(path), format!("from {VAR}")),
        (None, Some(home)) => (
            Path::new(&home).join(DEFAULT),
            format!("the default; {VAR} is unset"),
        ),
        (None, None) => {
            return Err(format!(
                "wasi-sdk is required (bindgen uses its clang and libclang), but neither {VAR} \
                 nor HOME is set: set {VAR} to the wasi-sdk directory"
            ));
        }
    };
    let clang = dir.join("bin/clang");
    let missing = if !clang.is_file() {
        Some("bin/clang")
    } else if !has_libclang(&dir.join("lib")) {
        Some("lib/libclang")
    } else {
        None
    };
    match missing {
        None => Ok(dir),
        Some(missing) => Err(format!(
            "wasi-sdk is required (bindgen uses its clang and libclang), but {} ({origin}) has no \
             {missing}: install wasi-sdk there (version WASI_SDK_VERSION in deps/versions.env), \
             or set {VAR} to the wasi-sdk directory",
            dir.display()
        )),
    }
}

/// Whether the directory `lib` has a libclang shared library (`libclang.so`, `libclang.dylib`,
/// `libclang.so.<version>`).
fn has_libclang(lib: &Path) -> bool {
    std::fs::read_dir(lib).is_ok_and(|entries| {
        entries.filter_map(|e| e.ok()).any(|e| {
            let name = e.file_name();
            let name = name.to_string_lossy();
            name.starts_with("libclang.so") || name == "libclang.dylib"
        })
    })
}

/// The Rust target of the wasm build.
pub const WASI_TARGET: &str = "wasm32-wasip1";

/// The flags of a C file compiled for [`WASI_TARGET`] with wasi-sdk's clang in `sdk`, as
/// `scripts/deps/build-wasi.sh` compiles PETSc, libCEED and Mmg: the target, wasi-sdk's sysroot
/// (whose wasi-libc the libraries and slidesim link against), the wasi-libc emulations of
/// `signal()`, the process clocks and `getpid()` (PETSc's headers declare them), and wasm SIMD.
pub fn wasi_c_flags(sdk: &Path) -> Vec<String> {
    vec![
        format!("--target={WASI_TARGET}"),
        format!("--sysroot={}", sdk.join("share/wasi-sysroot").display()),
        "-D_WASI_EMULATED_SIGNAL".to_string(),
        "-D_WASI_EMULATED_PROCESS_CLOCKS".to_string(),
        "-D_WASI_EMULATED_GETPID".to_string(),
        "-msimd128".to_string(),
    ]
}

/// wasi-sdk's clang in `sdk`.
pub fn clang(sdk: &Path) -> PathBuf {
    sdk.join("bin/clang")
}

/// wasi-sdk's archiver in `sdk`, which indexes wasm objects (wasm-ld needs the index).
pub fn archiver(sdk: &Path) -> PathBuf {
    sdk.join("bin/llvm-ar")
}
