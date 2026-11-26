//! Bindings and link flags for PETSc (with Mmg) from the installation in `PETSC_DIR`
//! (`deps/install/spike-mmg-<target>/env.sh` sets it and `PKG_CONFIG_PATH`). Copied from
//! `spike/petsc-wasm/diffusion/build.rs` without libCEED.
//!
//! - native: shared `libpetsc.so` with an rpath, MPI headers from `mpicc -show`;
//! - wasm32-wasip1: static archives (`pkg-config --static`), wasi-sdk sysroot from `WASI_SDK_PATH`.
//!
//! bindgen uses wasi-sdk's libclang (`$WASI_SDK_PATH/lib`) and its resource directory for both
//! targets: one clang (23) for all bindings, and the system libclang has no builtin headers.
//!
//! Only `-L` and `-l` of the pkg-config link line are used.

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn run(cmd: &str, args: &[&str]) -> String {
    let out = Command::new(cmd)
        .args(args)
        .output()
        .unwrap_or_else(|e| panic!("cannot run {cmd}: {e}"));
    assert!(
        out.status.success(),
        "{cmd} {args:?} failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).expect("utf-8 output")
}

fn main() {
    for var in ["PETSC_DIR", "PKG_CONFIG_PATH", "WASI_SDK_PATH"] {
        println!("cargo:rerun-if-env-changed={var}");
    }
    println!("cargo:rerun-if-changed=wrapper.h");

    let target = env::var("TARGET").unwrap();
    let wasm = target == "wasm32-wasip1";
    let petsc_dir =
        env::var("PETSC_DIR").expect("PETSC_DIR unset: source deps/install/<config>/env.sh");

    // Link line.
    let mut pc_args = vec!["--libs", "PETSc"];
    if wasm {
        pc_args.insert(0, "--static");
    }
    for tok in run("pkg-config", &pc_args).split_whitespace() {
        if let Some(dir) = tok.strip_prefix("-L") {
            println!("cargo:rustc-link-search=native={dir}");
            if !wasm {
                println!("cargo:rustc-link-arg=-Wl,-rpath,{dir}");
            }
        } else if let Some(lib) = tok.strip_prefix("-l") {
            println!("cargo:rustc-link-lib={lib}");
        }
    }

    // Bindings.
    let sdk = env::var("WASI_SDK_PATH").expect("WASI_SDK_PATH unset");
    // SAFETY: the build script is single threaded; clang-sys reads it when loading libclang.
    unsafe { env::set_var("LIBCLANG_PATH", format!("{sdk}/lib")) };
    let resource_dir = run(&format!("{sdk}/bin/clang"), &["-print-resource-dir"]);
    let mut clang_args = vec![
        format!("-resource-dir={}", resource_dir.trim()),
        format!("-I{petsc_dir}/include"),
    ];
    if wasm {
        clang_args.extend([
            "--target=wasm32-wasip1".to_string(),
            format!("--sysroot={sdk}/share/wasi-sysroot"),
            "-D_WASI_EMULATED_SIGNAL".to_string(),
            "-D_WASI_EMULATED_PROCESS_CLOCKS".to_string(),
            "-D_WASI_EMULATED_GETPID".to_string(),
            // clang defaults to hidden visibility for wasm and bindgen skips hidden functions;
            // PETSc's wasm headers carry no visibility attribute.
            "-fvisibility=default".to_string(),
        ]);
    } else {
        // wasi-sdk's libclang defaults to target wasm32-wasip1 and the wasi sysroot (bindgen only
        // passes --target when cross compiling); PETSc's headers include mpi.h.
        clang_args.extend([format!("--target={target}"), "--sysroot=/".to_string()]);
        let show = run("mpicc", &["-show"]);
        clang_args.extend(
            show.split_whitespace()
                .filter(|t| t.starts_with("-I"))
                .map(String::from),
        );
    }
    let api = "(Petsc|PETSC|DM|Vec|IS|ISLocalToGlobal).*";
    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_args(&clang_args)
        .allowlist_function(api)
        .allowlist_type(api)
        .allowlist_var(api)
        .prepend_enum_name(false)
        .layout_tests(false)
        .rust_edition(bindgen::RustEdition::Edition2024)
        .generate()
        .expect("bindgen failed");
    let out = PathBuf::from(env::var("OUT_DIR").unwrap()).join("bindings.rs");
    bindings.write_to_file(out).expect("cannot write bindings");
}
