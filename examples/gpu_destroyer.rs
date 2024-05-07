// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Example code using hyperbeam's GPU code with Rust.
//!
//! Build and run with something like:
//! `cargo run --release --features=hip --example gpu_destroyer -- 1000000 mwa_full_embedded_element_pattern.h5`
//! `cargo run --release --features=hip,gpu-single --example gpu_destroyer -- 1000000 mwa_full_embedded_element_pattern.h5`
//!
//! If the "gpu-single" feature is given, then single-precision floats are used
//! on the GPU. This trades precision for speed. The speed gain is considerable if
//! using a desktop GPU.
//!
//! If you want to use hyperbeam in your own Rust crate, then check out the latest
//! version on crates.io:
//!
//! https://crates.io/crates/mwa_hyperbeam

use std::f64::consts::{FRAC_PI_2, PI};

use mwa_hyperbeam::{fee::FEEBeam, AzEl, GpuFloat, Jones};
use ndarray::prelude::*;
use rayon::prelude::*;

// #[inline(never)]
pub fn paniq_() -> () {}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let num_directions: usize = args
        .next()
        .expect("number of directions supplied")
        .parse()
        .expect("number of directions is a number");

    println!(
        "GPU float precision is {} bits",
        std::mem::size_of::<GpuFloat>() * 8
    );

    let beam_file = args.next();
    // If we were given a file, use it. Otherwise, fall back on MWA_BEAM_FILE.
    let beam = match beam_file {
        Some(f) => FEEBeam::new(f)?,
        None => FEEBeam::new_from_env()?,
    };

    // Set up our "GPU beam".
    let num_freqs = 1;
    let freqs_hz = (0..num_freqs)
        .map(|i| (150_000 + 1000 * i) as u32)
        .collect::<Vec<_>>();
    let latitude_rad = Some(-0.4660608448386394); // MWA
    let iau_order = true;

    // Delays and amps correspond to dipoles in the "M&C order". See
    // https://wiki.mwatelescope.org/pages/viewpage.action?pageId=48005139) for
    // more info. Here, each row of the 2D array corresponds to a tile. In this
    // example, all delays and amps are the same, but they are allowed to vary
    // between tiles.
    let num_tiles = 1;
    let delays = Array2::zeros((num_tiles, 16));
    let amps = Array2::ones((num_tiles, 16));
    let norm_to_zenith = true;

    // Set up the directions to test. The type depends on the GPU precision.
    let (az, za): (Vec<_>, Vec<_>) = (0..num_directions)
        .map(|i| {
            (
                0.4 + 0.3 * PI * (i as f64 / num_directions as f64),
                0.3 + 0.4 * FRAC_PI_2 * (i as f64 / num_directions as f64),
            )
        })
        .unzip();

    // for ((az, za), i) in az.iter().zip(za.iter()).zip(0..) {
    //     println!("i {:3} az={:.3} za={:.3}", i, az, za);
    // }

    // compute on CPU for comparison.
    #[cfg(feature = "gpu-single")]
    let elem: Jones<f32> = Jones::default();
    #[cfg(not(feature = "gpu-single"))]
    let elem: Jones<f64> = Jones::default();
    let mut cpu_jones = Array3::from_elem((delays.dim().0, freqs_hz.len(), az.len()), elem);

    for ((mut out, delays), amps) in cpu_jones
        .outer_iter_mut()
        .zip(delays.outer_iter())
        .zip(amps.outer_iter())
    {
        for (mut out, &freq) in out.outer_iter_mut().zip(freqs_hz.iter()) {
            let cpu_results = beam
                .calc_jones_array_pair(
                    &az,
                    &za,
                    freq,
                    delays.as_slice().unwrap(),
                    amps.as_slice().unwrap(),
                    norm_to_zenith,
                    latitude_rad,
                    iau_order,
                )
                .unwrap();

            // Demote the CPU results if we have to.
            #[cfg(feature = "gpu-single")]
            let cpu_results: Vec<Jones<f32>> = cpu_results.into_iter().map(|j| j.into()).collect();

            out.assign(&Array1::from(cpu_results));
        }
    }

    // let gpu_az = az.iter().map(|&a| a as GpuFloat).collect::<Vec<_>>();
    // let gpu_za = za.iter().map(|&a| a as GpuFloat).collect::<Vec<_>>();
    let azels: Vec<_> = az
        .iter()
        .zip(za.iter())
        .map(|(&az, &za)| AzEl {
            az,
            el: FRAC_PI_2 - za,
        })
        .collect();

    let num_attempts = 9999;
    (0..num_attempts).into_par_iter().for_each(|i| {
        let gpu_beam = unsafe {
            beam.gpu_prepare(
                freqs_hz.as_slice(),
                delays.view(),
                amps.view(),
                norm_to_zenith,
            )
            .expect("beam.gpu_prepare")
        };

        // Call hyperbeam GPU code.
        let gpu_jones = gpu_beam
            .calc_jones(azels.as_slice(), latitude_rad, iau_order)
            .expect("gpu_beam.calc_jones");

        // assert_eq!(gpu_jones.dim(), cpu_jones.dim());

        // Compare the differences with the CPU-generated Jones matrices
        let mut pass = true;
        for (&cpu, &gpu) in cpu_jones.iter().zip(gpu_jones.iter()) {
            let norm = (cpu - gpu).norm_sqr();
            #[cfg(feature = "gpu-single")]
            if norm.iter().sum::<f32>() > 1e-6_f32 {
                pass = false;
                break;
            }
            #[cfg(not(feature = "gpu-single"))]
            if norm.iter().sum::<f64>() > 1e-12_f64 {
                pass = false;
                paniq_();
                break;
            }
        }
        #[cfg(feature = "gpu-single")]
        let init: (f32, f32) = (f32::MAX, f32::MIN);
        #[cfg(not(feature = "gpu-single"))]
        let init: (f64, f64) = (f64::MAX, f64::MIN);
        let (min_norm, max_norm) = cpu_jones
            .iter()
            .zip(gpu_jones.iter())
            .map(|(&cpu, &gpu)| (cpu - gpu).norm_sqr())
            .fold(init, |(min, max), norm| {
                #[cfg(feature = "gpu-single")]
                let s: f32 = norm.iter().sum();
                #[cfg(not(feature = "gpu-single"))]
                let s: f64 = norm.iter().sum();
                (min.min(s), max.max(s))
            });
        if pass {
            eprintln!(
                "     attempt {:4} passed, min_norm={:?} max_norm={:?}",
                i, min_norm, max_norm
            );
        } else {
            eprintln!(
                " !!! attempt {:4} failed, min_norm={:?} max_norm={:?}",
                i, min_norm, max_norm
            );
        }
    });

    Ok(())
}

/*
DEBUG=1 cargo build --example gpu_destroyer --features=hip,hdf5-static --profile dev
RAYON_NUM_THREADS=3 target/debug/examples/gpu_destroyer 9999
cat > rocgdbinit <<EOF
set auto-load safe-path /
dir $PYTHONPATH
set amdgpu precise-memory on
set breakpoint pending on
set disassemble-next-line on

break gpu_destroyer::paniq_
commands
    up
    info locals
    thread apply all backtrace 5
    quit 1
end
break mwa_hyperbeam::fee::FEEBeam::gpu_prepare
commands
    info args
    continue
end
break mwa_hyperbeam::fee::gpu::FEEBeamGpu::calc_jones
commands
    info args
    continue
end

run
EOF
# info reg
# thread apply all backtrace
# info threads
# quit 1
# EOF
export MWA_BEAM_FILE="mwa_full_embedded_element_pattern.h5"
[ -f $MWA_BEAM_FILE ] || wget -O "$MWA_BEAM_FILE" $'http://ws.mwatelescope.org/static/'$MWA_BEAM_FILE
export ROCM_VER=5.4.6 RUST_VER=stable
for ROCM_VER in system 5.4.6 5.5.3 5.6.1 5.7.0 5.7.1; do
    load_rocm $ROCM_VER
    cargo +$RUST_VER build --features=hdf5-static,hip --example gpu_destroyer && \
    RAYON_NUM_THREADS=2 rocgdb -x rocgdbinit --args target/debug/examples/gpu_destroyer 1 \
     | tee gpu_destroyer-$ROCM_VER.log 2>&1
done


load_rocm $ROCM_VER
cargo +$RUST_VER build --features=hdf5-static,hip --example gpu_destroyer
RAYON_NUM_THREADS=3 target/debug/examples/gpu_destroyer 9999
RAYON_NUM_THREADS=4 rocgdb -x rocgdbinit --args target/debug/examples/gpu_destroyer 9999
*/

/*
attempt 676 passed, min_norm=[0.0, 0.0, 0.0, 0.0]                                                                    [17/1978]
attempt 2524 passed, min_norm=[0.0, 0.0, 0.0, 0.0]
attempt 343 passed, min_norm=[0.0, 0.0, 0.0, 0.0]
thread '<unnamed>' panicked at examples/gpu_destroyer.rs:146:13:
attempt 1341 failed az=0.400 za=0.300 norm=[1.7982786046614562e-6, 0.0022572476886556668, 0.0023788106642785314, 9.57502419389
7969e-8]
attempt 5026 passed, min_norm=[0.0, 0.0, 0.0, 0.0]
attempt 3769 passed, min_norm=[0.0, 0.0, 0.0, 0.0]
attempt 677 passed, min_norm=[0.0, 0.0, 0.0, 0.0]
attempt 5027 passed, min_norm=[0.0, 0.0, 0.0, 0.0]
attempt 1346 passed, min_norm=[0.0, 0.0, 0.0, 0.0]
attempt 2525 passed, min_norm=[0.0, 0.0, 0.0, 0.0]
attempt 344 passed, min_norm=[0.0, 0.0, 0.0, 0.0]
attempt 3770 passed, min_norm=[0.0, 0.0, 0.0, 0.0]
attempt 678 passed, min_norm=[0.0, 0.0, 0.0, 0.0]
thread '<unnamed>' panicked at examples/gpu_destroyer.rs:146:13:
attempt 345 failed az=0.400 za=0.300 norm=[1.7982786046614562e-6, 0.0022572476886556668, 0.0023788106642785314, 9.575024193897
969e-8]
attempt 1347 passed, min_norm=[0.0, 0.0, 0.0, 0.0]
attempt 3771 passed, min_norm=[0.0, 0.0, 0.0, 0.0]
attempt 2526 passed, min_norm=[0.0, 0.0, 0.0, 0.0]
attempt 1348 passed, min_norm=[0.0, 0.0, 0.0, 0.0]
attempt 5028 passed, min_norm=[1.4095292761800317e-18, 4.554040319507578e-16, 4.808375260085222e-16, 2.1365382960150389e-19]
attempt 679 passed, min_norm=[0.0, 0.0, 0.0, 0.0]
attempt 346 passed, min_norm=[0.0, 0.0, 0.0, 0.0]
attempt 2527 passed, min_norm=[0.0, 0.0, 0.0, 0.0]

*/
