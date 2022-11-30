// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::env;

// This code is adapted from pkg-config-rs
// (https://github.com/rust-lang/pkg-config-rs).
#[cfg(feature = "cuda")]
#[allow(clippy::if_same_then_else, clippy::needless_bool)]
fn infer_static(name: &str) -> bool {
    if std::env::var(format!("{}_STATIC", name.to_uppercase())).is_ok() {
        true
    } else if std::env::var(format!("{}_DYNAMIC", name.to_uppercase())).is_ok() {
        false
    } else if std::env::var("PKG_CONFIG_ALL_STATIC").is_ok() {
        true
    } else if std::env::var("PKG_CONFIG_ALL_DYNAMIC").is_ok() {
        false
    } else {
        false
    }
}

#[cfg(feature = "cuda")]
fn parse_and_validate_compute(c: &str, var: &str) -> Vec<u16> {
    let mut out = vec![];
    for compute in c.trim().split(',') {
        // Check that there's only two numeric characters.
        if compute.len() != 2 {
            panic!("When parsing {var}, found '{compute}', which is not a two-digit number!")
        }

        match compute.parse() {
            Ok(p) => out.push(p),
            Err(_) => panic!("'{compute}', part of {var}, couldn't be parsed into a number!"),
        }
    }
    out
}

/// Search for any C/C++/CUDA files and have rerun-if-changed on all of them.
#[cfg(feature = "cuda")]
fn rerun_if_changed_c_cpp_cuda_files<P: AsRef<std::path::Path>>(dir: P) {
    for path in std::fs::read_dir(dir).expect("dir exists") {
        let path = path.expect("is readable").path();
        if path.is_dir() {
            rerun_if_changed_c_cpp_cuda_files(&path)
        }

        if let Some("cu" | "h") = path.extension().and_then(|os_str| os_str.to_str()) {
            println!("cargo:rerun-if-changed={}", path.display());
        }
    }
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    #[cfg(feature = "cuda")]
    {
        const DEFAULT_CUDA_ARCHES: &[u16] = &[60, 70, 80];
        const DEFAULT_CUDA_SMS: &[u16] = &[60, 61, 70, 75, 80, 86];

        // Attempt to read HYPERBEAM_CUDA_COMPUTE. HYPERDRIVE_CUDA_COMPUTE can
        // be used instead, too.
        println!("cargo:rerun-if-env-changed=HYPERBEAM_CUDA_COMPUTE");
        println!("cargo:rerun-if-env-changed=HYPERDRIVE_CUDA_COMPUTE");
        let (arches, sms): (Vec<u16>, Vec<u16>) = match (
            env::var("HYPERBEAM_CUDA_COMPUTE"),
            env::var("HYPERDRIVE_CUDA_COMPUTE"),
        ) {
            // When a user-supplied variable exists, use it as the CUDA arch and
            // compute level.
            (Ok(c), _) | (Err(_), Ok(c)) => {
                let compute = parse_and_validate_compute(&c, "HYPERBEAM_CUDA_COMPUTE");
                let sms = compute.clone();
                (compute, sms)
            }
            (Err(_), Err(_)) => {
                // Print out all of the default arches and computes as a
                // warning.
                println!("cargo:warning=No HYPERBEAM_CUDA_COMPUTE; Passing arch=compute_{DEFAULT_CUDA_ARCHES:?} and code=sm_{DEFAULT_CUDA_SMS:?} to nvcc");
                (DEFAULT_CUDA_ARCHES.to_vec(), DEFAULT_CUDA_SMS.to_vec())
            }
        };

        rerun_if_changed_c_cpp_cuda_files("src/");

        let mut cuda_target = cc::Build::new();
        cuda_target
            .cuda(true)
            .cudart("shared") // We handle linking cudart statically
            .include("src/fee/cuda/")
            .file("src/fee/cuda/fee.cu")
            .define(
                // The DEBUG env. variable is set by cargo. If running "cargo
                // build --release", DEBUG is "false", otherwise "true".
                // C/C++/CUDA like the compile option "NDEBUG" to be defined
                // when using assert.h, so if appropriate, define that here. We
                // also define "DEBUG" so that can be used.
                match env::var("DEBUG").as_deref() {
                    Ok("false") => "NDEBUG",
                    _ => "DEBUG",
                },
                None,
            );

        // Loop over each arch and sm
        for arch in arches {
            for &sm in &sms {
                if sm < arch {
                    continue;
                }

                cuda_target.flag("-gencode");
                cuda_target.flag(&format!("arch=compute_{arch},code=sm_{sm}"));
            }
        }

        // If we're told to, use single-precision floats. The default in the
        // CUDA code is to use double-precision.
        #[cfg(feature = "cuda-single")]
        cuda_target.define("SINGLE", None);

        cuda_target.compile("hyperbeam_cu");

        // Link CUDA. If the library path manually specified, search there.
        if let Ok(lib_dir) = std::env::var("CUDA_LIB") {
            println!("cargo:rustc-link-search=native={lib_dir}");
        }

        if infer_static("cuda") {
            // CUDA ships its static library as cudart_static.a, not cudart.a
            println!("cargo:rustc-link-lib=static=cudart_static");
        } else {
            println!("cargo:rustc-link-lib=cudart");
        }

        #[cfg(feature = "cuda-static")]
        println!("cargo:rustc-link-lib=static=cudart_static");
    }

    // Generate a C header for hyperbeam and write it to the include
    // directory. This routine only need to be done if the ffi module has
    // changed.
    let crate_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    println!("cargo:rerun-if-changed=src/fee/ffi/mod.rs");
    // Only do this if we're not on docs.rs (doesn't like writing files outside
    // of OUT_DIR).
    match env::var("DOCS_RS").as_deref() {
        Ok("1") => (),
        _ => {
            // Exclude CUDA things if CUDA isn't enabled.
            cfg_if::cfg_if! {
                if #[cfg(feature = "cuda")] {
                    let export = cbindgen::ExportConfig::default();
                } else {
                    let export = cbindgen::ExportConfig {
                        exclude: vec![
                            "FEEBeamCUDA".to_string(),
                            "new_cuda_fee_beam".to_string(),
                            "calc_jones_cuda".to_string(),
                            "calc_jones_cuda_device".to_string(),
                            "calc_jones_cuda_device_inner".to_string(),
                            "get_tile_map".to_string(),
                            "get_freq_map".to_string(),
                            "get_num_unique_tiles".to_string(),
                            "get_num_unique_fee_freqs".to_string(),
                            "free_cuda_fee_beam".to_string(),
                        ],
                        ..Default::default()
                    };
                }
            }

            // Rename an internal-only name depending on the CUDA precision.
            #[cfg(feature = "cuda-single")]
            let c_type = "float";
            #[cfg(not(feature = "cuda-single"))]
            let c_type = "double";

            cbindgen::Builder::new()
                .with_config({
                    let mut config = cbindgen::Config::default();
                    config.cpp_compat = true;
                    config.pragma_once = true;
                    config.export = export;
                    config
                })
                .with_crate(crate_dir)
                .with_language(cbindgen::Language::C)
                .rename_item("CudaFloat", c_type)
                .generate()
                .expect("Unable to generate bindings")
                .write_to_file("include/mwa_hyperbeam.h");
        }
    }
}
