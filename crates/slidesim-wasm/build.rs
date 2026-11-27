//! The memory of slidesim.wasm is imported (`env.memory`): the host creates it, with the maximum
//! it allows (a cap of `wasm/host` or `CreateOptions.memory` of the JavaScript API), and gives the
//! same memory to the kernel of the cells. scripts/wasm/build.sh records its minimum in the custom
//! section `slidesim.memory` (scripts/wasm/memory_section.mjs). The module exports the memory it
//! imports too (`memory`), as WASI preview 1 hosts (wasmtime-wasi) look for it there.

fn main() {
    if std::env::var("CARGO_CFG_TARGET_ARCH").as_deref() == Ok("wasm32") {
        println!("cargo:rustc-cdylib-link-arg=--import-memory");
        println!("cargo:rustc-cdylib-link-arg=--export-memory");
    }
}
