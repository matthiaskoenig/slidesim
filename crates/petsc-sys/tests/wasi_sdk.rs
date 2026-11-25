//! Tests of `wasi_sdk.rs`, which the build scripts of petsc-sys and ceed-sys use to find wasi-sdk.

#[path = "../wasi_sdk.rs"]
mod wasi_sdk;

use std::ffi::OsString;
use std::path::{Path, PathBuf};

/// An empty directory for the test `name`.
fn scratch(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("wasi_sdk")
        .join(name);
    if dir.exists() {
        std::fs::remove_dir_all(&dir).unwrap();
    }
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// A fake wasi-sdk in `dir`, with the files `files`.
fn sdk(dir: &Path, files: &[&str]) {
    for file in files {
        let path = dir.join(file);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, "").unwrap();
    }
}

fn os(path: &Path) -> Option<OsString> {
    Some(path.as_os_str().to_owned())
}

#[test]
fn wasi_sdk_path_takes_precedence() {
    let dir = scratch("wasi_sdk_path_takes_precedence");
    sdk(
        &dir.join("sdk"),
        &["bin/clang", "lib/libclang.so.23.1-wasi-sdk"],
    );
    let found = wasi_sdk::locate(os(&dir.join("sdk")), os(&dir.join("home"))).unwrap();
    assert_eq!(found, dir.join("sdk"));
}

#[test]
fn default_is_in_the_home_directory() {
    let home = scratch("default_is_in_the_home_directory");
    sdk(
        &home.join("opt/wasi-sdk"),
        &["bin/clang", "lib/libclang.so"],
    );
    let found = wasi_sdk::locate(None, os(&home)).unwrap();
    assert_eq!(found, home.join("opt/wasi-sdk"));
}

#[test]
fn missing_wasi_sdk_says_where_it_looked_and_what_to_set() {
    let home = scratch("missing_wasi_sdk_says_where_it_looked_and_what_to_set");
    let e = wasi_sdk::locate(None, os(&home)).unwrap_err();
    let dir = home.join("opt/wasi-sdk");
    for part in [
        "wasi-sdk is required",
        &format!("{} (the default; WASI_SDK_PATH is unset)", dir.display()),
        "has no bin/clang",
        "set WASI_SDK_PATH to the wasi-sdk directory",
    ] {
        assert!(e.contains(part), "{part:?} not in {e:?}");
    }

    let e = wasi_sdk::locate(os(&home.join("nowhere")), os(&home)).unwrap_err();
    let expected = format!(
        "{} (from WASI_SDK_PATH) has no bin/clang",
        home.join("nowhere").display()
    );
    assert!(e.contains(&expected), "{e}");

    let e = wasi_sdk::locate(None, None).unwrap_err();
    assert!(e.contains("neither WASI_SDK_PATH nor HOME is set"), "{e}");
}

#[test]
fn wasi_sdk_without_libclang_is_error() {
    let dir = scratch("wasi_sdk_without_libclang_is_error");
    sdk(&dir, &["bin/clang", "lib/libLLVM.so"]);
    let e = wasi_sdk::locate(os(&dir), None).unwrap_err();
    assert!(e.contains("has no lib/libclang"), "{e}");
}
