//! Bindings and link flags for the libCEED installation in `CEED_DIR`
//! (`deps/install/<config>/env.sh` sets it).
//!
//! Native targets link the shared `libceed.so`, found through pkg-config (`ceed.pc` of
//! `CEED_DIR`). At run time `LD_LIBRARY_PATH` (set by `env.sh`) finds it for tests, and the
//! rpath of the `slidesim` binary for it, as for PETSc (see `crates/petsc-sys/build.rs`).
//!
//! bindgen uses wasi-sdk's libclang (`$WASI_SDK_PATH/lib`, default `~/opt/wasi-sdk`) and its
//! resource directory: one clang for the bindings of every target, and the system libclang may
//! lack its builtin headers. Without wasi-sdk the build fails, with where it looked
//! (`crates/petsc-sys/wasi_sdk.rs`).
//!
//! Only `-L` and `-l` of the pkg-config link line are used: `ceed.pc` carries the compile flag
//! `-fopenmp-simd` among its link flags (`Libs.private`), which must not reach the linker.
//!
//! wasm32-wasip1 (`deps/install/wasi/env.sh`) links the static archive of libCEED
//! (`pkg-config --static`), and bindgen parses the headers for that target with wasi-sdk's
//! sysroot ([`wasi_sdk::wasi_c_flags`]) and `-fvisibility=default`, as petsc-sys does (libCEED's
//! headers carry visibility attributes, so it is not needed, but harmless).

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

#[path = "../petsc-sys/wasi_sdk.rs"]
mod wasi_sdk;

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

/// `pkg-config <args> ceed`, reading the `.pc` file of exactly this installation.
fn pkg_config(ceed_dir: &Path, args: &[&str]) -> String {
    let pc_path = match env::var("PKG_CONFIG_PATH") {
        Ok(p) if !p.is_empty() => format!("{}/lib/pkgconfig:{p}", ceed_dir.display()),
        _ => format!("{}/lib/pkgconfig", ceed_dir.display()),
    };
    let mut all = args.to_vec();
    all.push("ceed");
    run("pkg-config", &all, &[("PKG_CONFIG_PATH", pc_path)])
}

/// Links libCEED: the `-L` and `-l` flags of its link line, of the static archive (and what it
/// needs) for wasm, else of the shared library.
fn link(ceed_dir: &Path, wasm: bool) {
    let args: &[&str] = if wasm {
        &["--static", "--libs"]
    } else {
        &["--libs"]
    };
    for tok in pkg_config(ceed_dir, args).split_whitespace() {
        if let Some(dir) = tok.strip_prefix("-L") {
            println!("cargo:rustc-link-search=native={dir}");
        } else if let Some(lib) = tok.strip_prefix("-l") {
            println!("cargo:rustc-link-lib={lib}");
        }
    }
}

/// Include directories of libCEED.
fn include_dirs_native(ceed_dir: &Path) -> Vec<String> {
    pkg_config(ceed_dir, &["--cflags-only-I"])
        .split_whitespace()
        .filter_map(|t| t.strip_prefix("-I"))
        .map(String::from)
        .collect()
}

fn generate_bindings(target: &str, sdk: &Path, include_dirs: &[String], wasm: bool) {
    // SAFETY: the build script is single threaded; clang-sys reads it when loading libclang.
    unsafe { env::set_var("LIBCLANG_PATH", sdk.join("lib")) };
    let clang = sdk.join("bin/clang");
    let resource_dir = run(clang.to_str().unwrap(), &["-print-resource-dir"], &[]);
    let mut clang_args = vec![format!("-resource-dir={}", resource_dir.trim())];
    if wasm {
        clang_args.extend(wasi_sdk::wasi_c_flags(sdk));
        clang_args.push("-fvisibility=default".to_string());
    } else {
        // wasi-sdk's libclang defaults to target wasm32-wasip1 and the wasi sysroot (bindgen
        // only passes --target when cross compiling).
        clang_args.extend([format!("--target={target}"), "--sysroot=/".to_string()]);
    }
    clang_args.extend(include_dirs.iter().map(|d| format!("-I{d}")));

    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_args(&clang_args)
        // The libCEED user API (`ceed.h` and the headers it includes), and free().
        .allowlist_file(r".*/ceed(/[^/]*)?\.h")
        .allowlist_function("free")
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
    for var in ["CEED_DIR", "PKG_CONFIG_PATH", "WASI_SDK_PATH"] {
        println!("cargo:rerun-if-env-changed={var}");
    }
    println!("cargo:rerun-if-changed=wrapper.h");

    let target = env::var("TARGET").unwrap();
    let wasm = target == wasi_sdk::WASI_TARGET;
    assert!(
        wasm || !target.starts_with("wasm32"),
        "ceed-sys: target {target} is not supported; of wasm targets {} is (see build.rs)",
        wasi_sdk::WASI_TARGET
    );
    let ceed_dir = PathBuf::from(
        env::var("CEED_DIR").expect("CEED_DIR unset: source deps/install/<config>/env.sh"),
    );
    let sdk = wasi_sdk::locate(env::var_os(wasi_sdk::VAR), env::var_os("HOME"))
        .unwrap_or_else(|e| panic!("{e}"));

    link(&ceed_dir, wasm);
    let include_dirs = include_dirs_native(&ceed_dir);
    // `DEP_CEED_INCLUDE` for the build scripts of dependents compiling C against libCEED (for
    // example QFunction sources).
    let joined = env::join_paths(&include_dirs).expect("include directories joinable");
    println!(
        "cargo:include={}",
        joined.to_str().expect("utf-8 include directories")
    );
    generate_bindings(&target, &sdk, &include_dirs, wasm);
}
