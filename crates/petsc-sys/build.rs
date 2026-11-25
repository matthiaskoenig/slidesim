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
//! lack its builtin headers.
//!
//! Only `-L` and `-l` of the pkg-config and `mpicc` link lines are used: anything else there is
//! a compile flag that must not reach the linker.
//!
//! wasm32-wasip1 is not supported yet. It will link the static archives of `deps/install/wasi`
//! (`pkg-config --static`) plus the wasi-libc emulation libraries listed in
//! `spike/petsc-wasm/PATCHES.md`, take the sysroot from wasi-sdk instead of MPI include flags,
//! and pass `-fvisibility=default` to bindgen (see `spike/petsc-wasm/diffusion/build.rs`).

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

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

/// `shim.c`: values PETSc defines as macros which bindgen cannot translate.
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
        .allowlist_function("MPI_(Comm_rank|Comm_size|Allreduce|Abort)")
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
    let sdk = env::var("WASI_SDK_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(env::var("HOME").expect("HOME unset")).join("opt/wasi-sdk")
        });

    link_native(&petsc_dir);
    let include_dirs = include_dirs_native(&petsc_dir);
    compile_shim(&include_dirs);
    generate_bindings(&target, &sdk, &include_dirs);
}
