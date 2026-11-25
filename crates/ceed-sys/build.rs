//! Bindings and link flags for the libCEED installation in `CEED_DIR`
//! (`deps/install/<config>/env.sh` sets it).
//!
//! Native targets link the shared `libceed.so`, found through pkg-config (`ceed.pc` of
//! `CEED_DIR`). At run time `LD_LIBRARY_PATH` (set by `env.sh`) finds it for tests, and the
//! rpath of the `slidesim` binary for it, as for PETSc (see `crates/petsc-sys/build.rs`).
//!
//! bindgen uses wasi-sdk's libclang (`$WASI_SDK_PATH/lib`, default `~/opt/wasi-sdk`) and its
//! resource directory: one clang for the bindings of every target, and the system libclang may
//! lack its builtin headers.
//!
//! Only `-L` and `-l` of the pkg-config link line are used: `ceed.pc` carries the compile flag
//! `-fopenmp-simd` among its link flags (`Libs.private`), which must not reach the linker.
//!
//! wasm32-wasip1 is not supported yet. It will link the static archive of `deps/install/wasi`
//! (`pkg-config --static`), take the sysroot from wasi-sdk, and needs no `-fvisibility=default`
//! (libCEED's headers carry visibility attributes; see `spike/petsc-wasm/diffusion/build.rs`).

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

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

/// Links libCEED (shared library): the `-L` and `-l` flags of its link line.
fn link_native(ceed_dir: &Path) {
    for tok in pkg_config(ceed_dir, &["--libs"]).split_whitespace() {
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
    assert!(
        !target.starts_with("wasm32"),
        "ceed-sys: target {target} is not supported yet (see build.rs)"
    );
    let ceed_dir = PathBuf::from(
        env::var("CEED_DIR").expect("CEED_DIR unset: source deps/install/<config>/env.sh"),
    );
    let sdk = env::var("WASI_SDK_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(env::var("HOME").expect("HOME unset")).join("opt/wasi-sdk")
        });

    link_native(&ceed_dir);
    let include_dirs = include_dirs_native(&ceed_dir);
    // `DEP_CEED_INCLUDE` for the build scripts of dependents compiling C against libCEED (for
    // example QFunction sources).
    let joined = env::join_paths(&include_dirs).expect("include directories joinable");
    println!(
        "cargo:include={}",
        joined.to_str().expect("utf-8 include directories")
    );
    generate_bindings(&target, &sdk, &include_dirs);
}
