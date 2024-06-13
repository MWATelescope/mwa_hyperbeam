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

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    #[cfg(all(feature = "cuda", feature = "hip"))]
    compile_error!("Both 'cuda' and 'hip' features are enabled; only one can be used.");
    #[cfg(all(not(feature = "cuda"), not(feature = "hip"), feature = "gpu-single"))]
    compile_error!(
        "The 'gpu-single' feature must be used with either of the 'cuda' or 'hip' features."
    );

    #[cfg(any(feature = "cuda", feature = "hip"))]
    gpu::build_and_link();

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
            // Exclude GPU things if GPU features aren't enabled.
            cfg_if::cfg_if! {
                if #[cfg(any(feature = "cuda", feature = "hip"))] {
                    let export = cbindgen::ExportConfig::default();
                } else {
                    let export = cbindgen::ExportConfig {
                        exclude: vec![
                            "FEEBeamGpu".to_string(),
                            "new_gpu_fee_beam".to_string(),
                            "fee_calc_jones_gpu".to_string(),
                            "fee_calc_jones_gpu_device".to_string(),
                            "fee_calc_jones_gpu_device_inner".to_string(),
                            "get_fee_tile_map".to_string(),
                            "get_fee_device_tile_map".to_string(),
                            "get_fee_freq_map".to_string(),
                            "get_fee_device_freq_map".to_string(),
                            "get_num_unique_fee_tiles".to_string(),
                            "get_num_unique_fee_freqs".to_string(),
                            "free_gpu_fee_beam".to_string(),
                            "AnalyticBeamGpu".to_string(),
                            "new_gpu_analytic_beam".to_string(),
                            "analytic_calc_jones_gpu".to_string(),
                            "analytic_calc_jones_gpu_device".to_string(),
                            "analytic_calc_jones_gpu_device_inner".to_string(),
                            "get_analytic_tile_map".to_string(),
                            "get_analytic_device_tile_map".to_string(),
                            "get_num_unique_analytic_tiles".to_string(),
                            "free_gpu_analytic_beam".to_string(),
                        ],
                        ..Default::default()
                    };
                }
            }

            // Rename an internal-only name depending on the GPU precision.
            #[cfg(feature = "gpu-single")]
            let c_type = "float";
            #[cfg(not(feature = "gpu-single"))]
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
                .rename_item("GpuFloat", c_type)
                .generate()
                .expect("Unable to generate bindings")
                .write_to_file("include/mwa_hyperbeam.h");
        }
    }
}

#[cfg(any(feature = "cuda", feature = "hip"))]
mod gpu {
    /// Search for any C/C++/CUDA files and have rerun-if-changed on all of them.
    #[cfg(any(feature = "cuda", feature = "hip"))]
    fn rerun_if_changed_c_cpp_gpu_files<P: AsRef<std::path::Path>>(dir: P) {
        for path in std::fs::read_dir(dir).expect("dir exists") {
            let path = path.expect("is readable").path();
            if path.is_dir() {
                rerun_if_changed_c_cpp_gpu_files(&path)
            }

            if let Some("c" | "cpp" | "cu" | "h" | "cuh") =
                path.extension().and_then(|os_str| os_str.to_str())
            {
                println!("cargo:rerun-if-changed={}", path.display());
            }
        }
    }

    pub(super) fn build_and_link() {
        use std::env;

        rerun_if_changed_c_cpp_gpu_files("src/");

        #[cfg(feature = "cuda")]
        let mut gpu_target = {
            const DEFAULT_CUDA_ARCHES: &[u16] = &[60, 70, 80];
            const DEFAULT_CUDA_SMS: &[u16] = &[60, 61, 70, 75, 80, 86];

            fn parse_and_validate_compute(c: &str, var: &str) -> Vec<u16> {
                let mut out = vec![];
                for compute in c.trim().split(',') {
                    // Check that there's only two numeric characters.
                    if compute.len() != 2 {
                        panic!("When parsing {var}, found '{compute}', which is not a two-digit number!")
                    }

                    match compute.parse() {
                        Ok(p) => out.push(p),
                        Err(_) => {
                            panic!("'{compute}', part of {var}, couldn't be parsed into a number!")
                        }
                    }
                }
                out
            }

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

            let mut cuda_target = cc::Build::new();
            cuda_target
                .cuda(true)
                .cudart("shared") // We handle linking cudart statically
                .include("src/gpu_common/")
                .include("src/fee/gpu/")
                .file("src/fee/gpu/fee.cu")
                .include("src/analytic/gpu/")
                .file("src/analytic/gpu/analytic.cu");
            // If $CXX is not set but $CUDA_PATH is, search for
            // $CUDA_PATH/bin/g++ and if it exists, set that as $CXX.
            if env::var_os("CXX").is_none() {
                // Unlike above, we care about $CUDA_PATH being unicode.
                if let Ok(cuda_path) = env::var("CUDA_PATH") {
                    // Look for the g++ that CUDA wants.
                    let compiler = std::path::PathBuf::from(cuda_path).join("bin/g++");
                    if compiler.exists() {
                        println!("cargo:warning=Setting $CXX to {}", compiler.display());
                        env::set_var("CXX", compiler.into_os_string());
                    }
                }
            }
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

            if crate::infer_static("cuda") {
                // CUDA ships its static library as cudart_static.a, not cudart.a
                println!("cargo:rustc-link-lib=static=cudart_static");
            }

            #[cfg(feature = "cuda-static")]
            println!("cargo:rustc-link-lib=static=cudart_static");

            match env::var("DEBUG").as_deref() {
                Ok("false") => (),
                _ => {
                    cuda_target.flag("-G");
                }
            };

            cuda_target
        };

        #[cfg(feature = "hip")]
        let mut gpu_target = {
            println!("cargo:rerun-if-env-changed=HIP_PATH");
            let mut hip_path = match env::var_os("HIP_PATH") {
                Some(p) => {
                    println!(
                        "cargo:warning=HIP_PATH set from env {}",
                        p.to_string_lossy()
                    );
                    std::path::PathBuf::from(p)
                }
                None => {
                    let hip_path = hip_sys::hiprt::get_hip_path();
                    println!(
                        "cargo:warning=HIP_PATH set from hip_sys {}",
                        hip_path.display()
                    );
                    hip_path
                }
            };

            // It seems that various ROCm releases change where hipcc is...
            let mut compiler = hip_path.join("bin/hipcc");
            if !compiler.exists() {
                // Try the dir above, which might be the ROCm dir.
                hip_path = hip_path.parent().unwrap().into();
                compiler = hip_path.join("bin/hipcc");
                if !compiler.exists() {
                    panic!(
                        "Couldn't find hipcc in either {} or {}",
                        hip_sys::hiprt::get_hip_path().display(),
                        hip_path.parent().unwrap().display()
                    );
                }
            }
            if !hip_path.join("include/hip/hip_runtime_api.h").exists() {
                panic!(
                    "Couldn't find include/hip/hip_runtime_api.h in {}",
                    hip_path.display()
                );
            }
            // TODO: this
            // if !hip_path.join("llvm/lib/clang/17.0.0/include/cuda_wrappers/cmath").exists() {
            //     panic!("Seriously AMD? What are you doing? {}", hip_path.display());
            // }
            println!("cargo:warning=install libstdc++-12-dev if you get cmath errors");
            // TODO: set the env LIBCLANG_PATH=/opt/rocm/llvm/lib to fix clang errors
            println!(
                "cargo:warning=If you get clang errors, set LIBCLANG_PATH={}",
                hip_path.join("llvm/lib").display()
            );

            let mut hip_target = cc::Build::new();
            hip_target
                .compiler(compiler)
                .include(hip_path.join("include/hip"))
                .include("src/gpu_common/")
                .file("src/fee/gpu/fee.cu")
                .file("src/analytic/gpu/analytic.cu");

            println!("cargo:rerun-if-env-changed=HIP_FLAGS");
            if let Some(p) = env::var_os("HIP_FLAGS") {
                println!(
                    "cargo:warning=HIP_FLAGS set from env {}",
                    p.to_string_lossy()
                );
                hip_target.flag(&p.to_string_lossy());
            }

            println!("cargo:rerun-if-env-changed=ROCM_VER");
            println!("cargo:rerun-if-env-changed=ROCM_PATH");
            println!("cargo:rerun-if-env-changed=HYPERBEAM_HIP_ARCH");
            println!("cargo:rerun-if-env-changed=HYPERDRIVE_HIP_ARCH");
            let arches: Vec<String> = match (
                env::var("HYPERBEAM_HIP_ARCH"),
                env::var("HYPERDRIVE_HIP_ARCH"),
            ) {
                // When a user-supplied variable exists, use it as the CUDA arch and
                // compute level.
                (Ok(c), _) | (Err(_), Ok(c)) => {
                    vec![c]
                }
                _ => {
                    // Print out all of the default arches and computes as a
                    // warning.
                    println!("cargo:warning=No offload arch found, try HYPERBEAM_HIP_ARCH");
                    vec![]
                }
            };

            for arch in arches {
                hip_target.flag(&format!("--offload-arch={arch}"));
            }

            match env::var("DEBUG").as_deref() {
                Ok("false") => (),
                _ => {
                    hip_target
                        .flag("-ggdb")
                        .flag("-O1") // <- don't use -O0 https://github.com/ROCm/HIP/issues/3183
                        .flag("-gmodules");
                }
            };

            hip_target
        };

        // The DEBUG env. variable is set by cargo. If running "cargo build
        // --release", DEBUG is "false", otherwise "true". C/C++/CUDA like
        // the compile option "NDEBUG" to be defined when using assert.h, so
        // if appropriate, define that here. We also define "DEBUG" so that
        // can be used.
        match env::var("DEBUG").as_deref() {
            Ok("false") => {
                gpu_target.define("NDEBUG", "");
            }
            _ => {
                gpu_target.define("DEBUG", "").flag("-v");
            }
        };

        // Break in case of emergency.
        // gpu_target.debug(true);
        // println!("cargo:warning={gpu_target:?}");

        // If we're told to, use single-precision floats. The default in the GPU
        // code is to use double-precision.
        #[cfg(feature = "gpu-single")]
        gpu_target.define("SINGLE", None);

        gpu_target.compile("hyperbeam_gpu");
    }
}
