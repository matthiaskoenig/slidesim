//! Tests of `petscvariables.rs`, the reader of PETSc's make variables which the build scripts of
//! petsc-sys and slidesim use for the flags of PT-Scotch.

#[path = "../petscvariables.rs"]
mod petscvariables;

use std::path::{Path, PathBuf};

use petscvariables::{Flags, Package, package, parse_flags, variable};

fn strings(items: &[&str]) -> Vec<String> {
    items.iter().map(|s| s.to_string()).collect()
}

#[test]
fn variable_is_the_last_definition() {
    let text = "\
PETSC_ARCH = arch\n\
PTSCOTCH_INCLUDE = -I/a/include\n\
PTSCOTCH_LIB = -L/a/lib -lscotch\n\
PTSCOTCH_LIB := -L/b/lib \\\n  -lptscotch -lscotch\n\
EMPTY =\n\
NOSPACE=-lx\n";
    assert_eq!(variable(text, "PTSCOTCH_INCLUDE").unwrap(), "-I/a/include");
    assert_eq!(
        variable(text, "PTSCOTCH_LIB").unwrap(),
        "-L/b/lib    -lptscotch -lscotch"
    );
    assert_eq!(variable(text, "EMPTY").unwrap(), "");
    assert_eq!(variable(text, "NOSPACE").unwrap(), "-lx");
    // A name is matched whole.
    assert_eq!(variable(text, "PTSCOTCH"), None);
    assert_eq!(variable(text, "MISSING"), None);
}

#[test]
fn flags_of_petsc_ptscotch_lib() {
    // As PETSc writes it for --download-ptscotch.
    let flags = parse_flags(
        "-Wl,-rpath,/p/lib -L/p/lib -lptesmumps -lptscotchparmetisv3 -lptscotch -lptscotcherr \
         -lesmumps -lscotch -lscotcherr",
    );
    assert_eq!(
        flags,
        Flags {
            include_dirs: vec![],
            link_dirs: strings(&["/p/lib"]),
            libs: strings(&[
                "ptesmumps",
                "ptscotchparmetisv3",
                "ptscotch",
                "ptscotcherr",
                "esmumps",
                "scotch",
                "scotcherr"
            ]),
            rpaths: strings(&["/p/lib"]),
            ignored: vec![],
        }
    );
}

#[test]
fn flag_forms() {
    let flags = parse_flags(
        "-I /i1 -I/i2 -L /l1 -L/l2 -lm -Wl,-rpath=/r1 -Wl,-R,/r2 -Wl,-rpath,/r3,-rpath,/r4 \
         /opt/scotch/lib/libscotch.so /opt/x/lib64/libesmumps.so.7.0 /opt/y/libptscotch.a \
         -pthread -Wl,--as-needed -l:libz.so -Wl,-rpath, -L",
    );
    assert_eq!(flags.include_dirs, strings(&["/i1", "/i2"]));
    assert_eq!(
        flags.link_dirs,
        strings(&["/l1", "/l2", "/opt/scotch/lib", "/opt/x/lib64", "/opt/y"])
    );
    assert_eq!(flags.libs, strings(&["m", "scotch", "esmumps", "ptscotch"]));
    assert_eq!(flags.rpaths, strings(&["/r1", "/r2", "/r3", "/r4"]));
    // `-L` at the end has no directory.
    assert_eq!(
        flags.ignored,
        strings(&[
            "-pthread",
            "-Wl,--as-needed",
            "-l:libz.so",
            "-Wl,-rpath,",
            "-L"
        ])
    );
    // Not library files: relative, no `lib` prefix, other extensions.
    let flags = parse_flags("lib/libscotch.so /a/scotch.so /a/libscotch.txt /a/libx.sox");
    assert!(
        flags.libs.is_empty() && flags.link_dirs.is_empty(),
        "{flags:?}"
    );
    assert_eq!(flags.ignored.len(), 4);
}

/// A PETSc installation in a new temporary directory with the given `petscconf.h` and
/// `petscvariables` (none if `None`).
fn installation(name: &str, conf: &str, variables: Option<&str>) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "petsc-sys-petscvariables-{}-{name}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("include")).unwrap();
    std::fs::create_dir_all(dir.join("lib/petsc/conf")).unwrap();
    std::fs::write(dir.join("include/petscconf.h"), conf).unwrap();
    if let Some(v) = variables {
        std::fs::write(dir.join("lib/petsc/conf/petscvariables"), v).unwrap();
    }
    dir
}

fn package_in(dir: &Path) -> Result<Option<Package>, String> {
    let result = package(dir, "PTSCOTCH");
    std::fs::remove_dir_all(dir).unwrap();
    result
}

const HAVE: &str = "#define PETSC_HAVE_MPI 1\n#define PETSC_HAVE_PTSCOTCH 1\n";

#[test]
fn package_outside_the_installation() {
    let dir = installation(
        "outside",
        HAVE,
        Some(
            "PTSCOTCH_LIB = -Wl,-rpath,/opt/scotch/lib -L/opt/scotch/lib -lptscotch -lscotch\n\
             PTSCOTCH_INCLUDE = -I/opt/scotch/include\n",
        ),
    );
    let p = package_in(&dir).unwrap().unwrap();
    assert_eq!(p.include.include_dirs, strings(&["/opt/scotch/include"]));
    assert_eq!(p.lib.link_dirs, strings(&["/opt/scotch/lib"]));
    assert_eq!(p.lib.libs, strings(&["ptscotch", "scotch"]));
    assert_eq!(p.lib.rpaths, strings(&["/opt/scotch/lib"]));
}

#[test]
fn package_in_default_directories_has_no_include_flags() {
    let dir = installation(
        "system",
        HAVE,
        Some("PTSCOTCH_INCLUDE = \nPTSCOTCH_LIB = -lptscotch -lscotch\n"),
    );
    let p = package_in(&dir).unwrap().unwrap();
    assert_eq!(p.include, Flags::default());
    assert_eq!(p.lib.libs, strings(&["ptscotch", "scotch"]));
}

#[test]
fn no_package_without_its_define() {
    // `PETSC_HAVE_PTSCOTCH_FOO` is another define; petscvariables is not needed.
    let dir = installation("without", "#define PETSC_HAVE_PTSCOTCH_FOO 1\n", None);
    assert_eq!(package_in(&dir).unwrap(), None);
}

#[test]
fn missing_flags_are_errors() {
    let dir = installation("no-variables", HAVE, None);
    let e = package_in(&dir).unwrap_err();
    assert!(
        e.starts_with("cannot read ") && e.contains("petscvariables"),
        "{e}"
    );

    let dir = installation("no-lib", HAVE, Some("PTSCOTCH_INCLUDE = -I/x\n"));
    let e = package_in(&dir).unwrap_err();
    assert!(e.contains("does not define PTSCOTCH_LIB"), "{e}");

    let dir = installation("no-include", HAVE, Some("PTSCOTCH_LIB = -lscotch\n"));
    let e = package_in(&dir).unwrap_err();
    assert!(e.contains("does not define PTSCOTCH_INCLUDE"), "{e}");

    let dir = installation(
        "no-library",
        HAVE,
        Some("PTSCOTCH_INCLUDE = -I/x\nPTSCOTCH_LIB = -L/x/lib -pthread\n"),
    );
    let e = package_in(&dir).unwrap_err();
    assert!(
        e.contains("PTSCOTCH_LIB in") && e.contains("names no library"),
        "{e}"
    );

    let dir = std::env::temp_dir().join("petsc-sys-petscvariables-no-such-installation");
    let e = package(&dir, "PTSCOTCH").unwrap_err();
    assert!(
        e.starts_with("cannot read ") && e.contains("petscconf.h"),
        "{e}"
    );
}
