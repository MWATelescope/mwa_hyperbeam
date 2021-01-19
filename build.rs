// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::env;

#[cfg(feature = "cuda")]
fn parse_and_validate_compute(c: &str, var: &str) -> u16 {
    // Check that there's only two numeric characters.
    if c.len() != 2 {
        panic!("{} is not a two-digit number!", var)
    }

    match c.parse::<u16>() {
        Ok(p) => p,
        Err(_) => panic!("{} couldn't be parsed into a number!", var),
    }
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    #[cfg(feature = "cuda")]
    {
        const DEFAULT_CUDA_ARCHES: &[u16] = &[60, 70, 80];
        const DEFAULT_CUDA_SMS: &[u16] = &[60, 70, 75, 86];

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
            (Ok(c), _) => {
                let compute = parse_and_validate_compute(&c, "HYPERBEAM_CUDA_COMPUTE");
                (vec![compute], vec![compute])
            }
            (Err(_), Ok(c)) => {
                let compute = parse_and_validate_compute(&c, "HYPERDRIVE_CUDA_COMPUTE");
                (vec![compute], vec![compute])
            }
            (Err(_), Err(_)) => {
                // Print out all of the default arches and computes as a
                // warning.
                let mut warn_str = String::new();
                warn_str.push_str("cargo:warning=No HYPERBEAM_CUDA_COMPUTE; Passing ");
                warn_str.push_str(&format!("arch=compute_{:?}", DEFAULT_CUDA_ARCHES));
                warn_str.push_str(" and ");
                warn_str.push_str(&format!("code=sm_{:?}", DEFAULT_CUDA_SMS));
                warn_str.push_str(" to nvcc");
                println!("{}", warn_str);
                (DEFAULT_CUDA_ARCHES.to_vec(), DEFAULT_CUDA_SMS.to_vec())
            }
        };

        // TODO: Search for any C/C++/CUDA files and have rerun-if-changed on
        // all of them.
        println!("cargo:rerun-if-changed=src/fee/cuda/fee.h");
        println!("cargo:rerun-if-changed=src/fee/cuda/fee.cu");

        let mut cuda_target = cc::Build::new();
        cuda_target
            .cuda(true)
            .flag("-cudart=static")
            .include("src/fee/cuda/")
            .file("src/fee/cuda/fee.cu");
        // Loop over each arch and sm
        for arch in arches {
            for &sm in &sms {
                if sm < arch {
                    continue;
                }

                let mut flag = String::new();
                cuda_target.flag("-gencode");
                flag.push_str(&format!("arch=compute_{},", arch));
                flag.push_str(&format!("code=sm_{}", sm));
                cuda_target.flag(&flag);
            }
        }

        // If we're told to, use single-precision floats. The default in the
        // CUDA code is to use double-precision.
        #[cfg(feature = "cuda-single")]
        cuda_target.define("SINGLE", None);

        cuda_target.compile("hyperbeam_cu");
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
            // Rename an internal-only name depending on the CUDA precision.
            #[cfg(feature = "cuda-single")]
            let c_type = "float";
            #[cfg(not(feature = "cuda-single"))]
            let c_type = "double";

            cbindgen::Builder::new()
                .with_config(cbindgen::Config {
                    cpp_compat: true,
                    pragma_once: true,
                    ..Default::default()
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
