//! The PT-Scotch of the PETSc installation (`PETSC_DIR`) is built without threads: its threaded
//! routines may give other partitions from run to run (`scripts/deps/build-native.sh`).

#[path = "../petscvariables.rs"]
mod petscvariables;

use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::process::Command;

/// The PETSc installation the tests link: `PETSC_DIR` now, else when they were built.
fn petsc_dir() -> PathBuf {
    std::env::var_os("PETSC_DIR")
        .map(PathBuf::from)
        .or_else(|| option_env!("PETSC_DIR").map(PathBuf::from))
        .expect("PETSC_DIR unset: source deps/install/<config>/env.sh")
}

/// The file of the library `name` (`lib<name>.so`, else `lib<name>.a`) in the first of `dirs`
/// which has one.
fn library(dirs: &[String], name: &str) -> Option<PathBuf> {
    dirs.iter().find_map(|dir| {
        ["so", "a"]
            .map(|ext| Path::new(dir).join(format!("lib{name}.{ext}")))
            .into_iter()
            .find(|path| path.exists())
    })
}

/// The undefined symbols of the library `path` (`nm`), or `None` without `nm`.
fn undefined_symbols(path: &Path) -> Option<String> {
    let mut cmd = Command::new("nm");
    if path.extension().is_some_and(|ext| ext == "so") {
        cmd.arg("-D");
    }
    let out = match cmd.arg("--undefined-only").arg(path).output() {
        Ok(out) => out,
        Err(e) if e.kind() == ErrorKind::NotFound => return None,
        Err(e) => panic!("cannot run nm: {e}"),
    };
    assert!(
        out.status.success(),
        "nm {} failed: {}",
        path.display(),
        String::from_utf8_lossy(&out.stderr)
    );
    Some(String::from_utf8(out.stdout).expect("utf-8 output"))
}

#[test]
fn ptscotch_has_no_threads() {
    let petsc_dir = petsc_dir();
    let Some(ptscotch) = petscvariables::package(&petsc_dir, "PTSCOTCH").unwrap() else {
        eprintln!("skipped: PETSc in {} has no PT-Scotch", petsc_dir.display());
        return;
    };
    let dirs = &ptscotch.lib.link_dirs;
    let libs: Vec<&String> = ptscotch
        .lib
        .libs
        .iter()
        .filter(|name| name.contains("scotch"))
        .collect();
    for name in ["scotch", "ptscotch"] {
        assert!(
            libs.iter().any(|lib| *lib == name),
            "PTSCOTCH_LIB does not link lib{name}: {libs:?}"
        );
    }
    for name in libs {
        let path = library(dirs, name)
            .unwrap_or_else(|| panic!("no lib{name}.so or lib{name}.a in {dirs:?}"));
        let Some(symbols) = undefined_symbols(&path) else {
            eprintln!("skipped: no nm");
            return;
        };
        assert!(
            !symbols
                .split_whitespace()
                .any(|s| s.split('@').next() == Some("pthread_create")),
            "{} imports pthread_create: PT-Scotch was built with threads",
            path.display()
        );
    }
}
