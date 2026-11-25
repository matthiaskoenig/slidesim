//! Where wasi-sdk is: bindgen uses its clang and libclang for the bindings of every target
//! (`$WASI_SDK_PATH`, default `~/opt/wasi-sdk`).
//!
//! Standard library only: the build scripts of petsc-sys and ceed-sys include it (`#[path]`), and
//! `tests/wasi_sdk.rs` tests it.

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
