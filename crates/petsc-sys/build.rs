//! Bindings and link flags for the PETSc installation in `PETSC_DIR`
//! (`deps/install/<config>/env.sh` sets it).
//!
//! Native targets link the shared `libpetsc.so`, and MPI as `mpicc -show` reports it (the Rust
//! code calls MPI directly, and PETSc's pkg-config file does not list MPI). At run time
//! `LD_LIBRARY_PATH` (set by `env.sh`) finds `libpetsc.so` for tests: an rpath from here would
//! not help, since `cargo:rustc-link-arg` of a build script applies to the targets of its own
//! package only, not to the crates depending on it. The `slidesim` binary sets its own rpath
//! (`crates/slidesim/build.rs`).
//!
//! bindgen uses wasi-sdk's libclang (`$WASI_SDK_PATH/lib`, default `~/opt/wasi-sdk`) and its
//! resource directory: one clang for the bindings of every target, and the system libclang may
//! lack its builtin headers. Without wasi-sdk the build fails, with where it looked
//! (`wasi_sdk.rs`).
//!
//! Only `-L` and `-l` of the pkg-config and `mpicc` link lines are used: anything else there is
//! a compile flag that must not reach the linker.
//!
//! If PETSc has PT-Scotch, the shim calls it too, with the include and link flags PETSc records in
//! `lib/petsc/conf/petscvariables` (`PTSCOTCH_INCLUDE`, `PTSCOTCH_LIB`; `petscvariables.rs`):
//! PT-Scotch may live outside `PETSC_DIR`.
//!
//! wasm32-wasip1 is not supported yet. It will link the static archives of `deps/install/wasi`
//! (`pkg-config --static`) plus the wasi-libc emulation libraries listed in
//! `spike/petsc-wasm/PATCHES.md`, take the sysroot from wasi-sdk instead of MPI include flags,
//! and pass `-fvisibility=default` to bindgen (see `spike/petsc-wasm/diffusion/build.rs`).

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

// Shared with other build scripts, each using part of it.
#[allow(dead_code)]
#[path = "petscvariables.rs"]
mod petscvariables;
use petscvariables::Package;

#[path = "wasi_sdk.rs"]
mod wasi_sdk;

/// C flags of every C file in slidesim (reproducible floating point on every target).
const C_FLAGS: [&str; 2] = ["-ffp-contract=off", "-fno-fast-math"];

fn run(cmd: &str, args: &[&str], envs: &[(&str, String)]) -> String {
    let out = Command::new(cmd)
        .args(args)
        .envs(envs.iter().map(|(k, v)| (k, v)))
        .output()
        .unwrap_or_else(|e| panic!("cannot run {cmd}: {e}"));
    assert!(
        out.status.success(),
        "{cmd} {args:?} failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).expect("utf-8 output")
}

/// Emits the `-L` and `-l` flags of a link line.
fn link_line(line: &str) {
    for tok in line.split_whitespace() {
        if let Some(dir) = tok.strip_prefix("-L") {
            println!("cargo:rustc-link-search=native={dir}");
        } else if let Some(lib) = tok.strip_prefix("-l") {
            println!("cargo:rustc-link-lib={lib}");
        }
    }
}

/// Links PETSc and MPI (shared libraries).
fn link_native(petsc_dir: &Path) {
    // pkg-config reads the `.pc` file of exactly this installation.
    let pc_path = match env::var("PKG_CONFIG_PATH") {
        Ok(p) if !p.is_empty() => format!("{}/lib/pkgconfig:{p}", petsc_dir.display()),
        _ => format!("{}/lib/pkgconfig", petsc_dir.display()),
    };
    let petsc = run(
        "pkg-config",
        &["--libs", "PETSc"],
        &[("PKG_CONFIG_PATH", pc_path)],
    );
    link_line(&petsc);
    link_line(&run("mpicc", &["-show"], &[]));
}

/// PT-Scotch, if PETSc has it: the shim calls it directly (`libpetsc.so` links it, but that
/// does not link the shim to it). It may be installed outside `PETSC_DIR`, so its include and
/// link flags come from PETSc's `petscvariables`. Panics if PETSc has PT-Scotch but its flags
/// cannot be found: the shim would lose the reset of PT-Scotch's random generator.
fn ptscotch(petsc_dir: &Path) -> Option<Package> {
    for file in petscvariables::files(petsc_dir) {
        println!("cargo:rerun-if-changed={}", file.display());
    }
    let package = petscvariables::package(petsc_dir, "PTSCOTCH")
        .unwrap_or_else(|e| panic!("petsc-sys: {e}"))?;
    for flag in package.include.ignored.iter().chain(&package.lib.ignored) {
        println!("cargo:warning=petsc-sys: ignored flag {flag:?} of PT-Scotch in petscvariables");
    }
    Some(package)
}

/// Links the libraries of `package` (an rpath is for the binaries to set, see the crate docs).
fn link_package(package: &Package) {
    for dir in &package.lib.link_dirs {
        println!("cargo:rustc-link-search=native={dir}");
    }
    for lib in &package.lib.libs {
        println!("cargo:rustc-link-lib={lib}");
    }
}

/// Include directories of PETSc and MPI.
fn include_dirs_native(petsc_dir: &Path) -> Vec<String> {
    let mut dirs = vec![format!("{}/include", petsc_dir.display())];
    let show = run("mpicc", &["-show"], &[]);
    dirs.extend(
        show.split_whitespace()
            .filter_map(|t| t.strip_prefix("-I"))
            .map(String::from),
    );
    dirs
}

/// `shim.c`: values PETSc defines as macros which bindgen cannot translate, and the calls into
/// PETSc's external packages.
fn compile_shim(include_dirs: &[String]) {
    println!("cargo:rerun-if-changed=shim.c");
    let mut build = cc::Build::new();
    build
        .file("shim.c")
        .includes(include_dirs)
        .opt_level(3)
        .define("NDEBUG", None)
        .warnings(true);
    for flag in C_FLAGS {
        build.flag(flag);
    }
    build.compile("petsc_sys_shim");
}

fn generate_bindings(target: &str, sdk: &Path, include_dirs: &[String]) {
    // SAFETY: the build script is single threaded; clang-sys reads it when loading libclang.
    unsafe { env::set_var("LIBCLANG_PATH", sdk.join("lib")) };
    let clang = sdk.join("bin/clang");
    let resource_dir = run(clang.to_str().unwrap(), &["-print-resource-dir"], &[]);
    let mut clang_args = vec![
        format!("-resource-dir={}", resource_dir.trim()),
        // wasi-sdk's libclang defaults to target wasm32-wasip1 and the wasi sysroot (bindgen only
        // passes --target when cross compiling).
        format!("--target={target}"),
        "--sysroot=/".to_string(),
    ];
    clang_args.extend(include_dirs.iter().map(|d| format!("-I{d}")));

    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_args(&clang_args)
        // The PETSc API: everything declared in PETSc's headers (`petsc*.h`), plus the MPI
        // functions used directly and the shim.
        .allowlist_file(r".*/petsc[^/]*\.h")
        .allowlist_function("MPI_(Comm_rank|Comm_size|Allreduce|Barrier|Abort)")
        .allowlist_function("petsc_sys_.*")
        .prepend_enum_name(false)
        .layout_tests(false)
        .generate_comments(false)
        .merge_extern_blocks(true)
        .rust_edition(bindgen::RustEdition::Edition2024)
        .generate()
        .expect("bindgen failed");
    let out = PathBuf::from(env::var("OUT_DIR").unwrap()).join("bindings.rs");
    bindings.write_to_file(out).expect("cannot write bindings");
}

fn main() {
    for var in ["PETSC_DIR", "PKG_CONFIG_PATH", "WASI_SDK_PATH"] {
        println!("cargo:rerun-if-env-changed={var}");
    }
    println!("cargo:rerun-if-changed=wrapper.h");

    let target = env::var("TARGET").unwrap();
    assert!(
        !target.starts_with("wasm32"),
        "petsc-sys: target {target} is not supported yet (see build.rs)"
    );
    let petsc_dir = PathBuf::from(
        env::var("PETSC_DIR").expect("PETSC_DIR unset: source deps/install/<config>/env.sh"),
    );
    let sdk = wasi_sdk::locate(env::var_os(wasi_sdk::VAR), env::var_os("HOME"))
        .unwrap_or_else(|e| panic!("{e}"));

    link_native(&petsc_dir);
    let mut include_dirs = include_dirs_native(&petsc_dir);
    if let Some(ptscotch) = ptscotch(&petsc_dir) {
        link_package(&ptscotch);
        include_dirs.extend(ptscotch.include.include_dirs);
    }
    compile_shim(&include_dirs);
    generate_bindings(&target, &sdk, &include_dirs);
}
