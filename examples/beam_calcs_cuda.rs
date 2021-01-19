// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Example code using hyperbeam's CUDA with Rust.
//!
//! Build and run with something like:
//! `cargo run --release --features=cuda-single --example beam_calcs_cuda -- mwa_full_embedded_element_pattern.h5 1000000`
//!
//! If you want to use hyperbeam in your own Rust crate, then check out the latest
//! version on crates.io:
//!
//! https://crates.io/crates/mwa_hyperbeam

use std::f64::consts::PI;

use marlu::{ndarray, Jones};
use ndarray::prelude::*;
use structopt::*;

use mwa_hyperbeam::fee::FEEBeam;

#[cfg(feature = "cuda-single")]
type CudaFloat = f32;
#[cfg(not(feature = "cuda-single"))]
type CudaFloat = f64;

#[derive(StructOpt, Debug)]
struct Opts {
    /// Path to the HDF5 file.
    #[structopt(short, long, parse(from_os_str))]
    hdf5_file: Option<std::path::PathBuf>,

    /// The number of directions to run.
    #[structopt()]
    num_directions: usize,
}

fn main() -> Result<(), anyhow::Error> {
    let opts = Opts::from_args();
    if opts.num_directions == 0 {
        eprintln!("num_directions cannot be 0.");
        std::process::exit(1);
    }

    // If we were given a file, use it. Otherwise, fall back on MWA_BEAM_FILE.
    let beam = match opts.hdf5_file {
        Some(f) => FEEBeam::new(f)?,
        None => FEEBeam::new_from_env()?,
    };
    // Set up our "CUDA beam".
    let freqs_hz = [150e6 as u32, 200e6 as _];
    // Delays and amps correspond to dipoles in the "M&C order". See
    // https://wiki.mwatelescope.org/pages/viewpage.action?pageId=48005139) for
    // more info. Here, each row of the 2D array corresponds to a tile. In this
    // example, all delays and amps are the same, but they are allowed to vary
    // between tiles.
    let num_tiles = 1;
    let delays = Array2::zeros((num_tiles, 16));
    let amps = Array2::ones((num_tiles, 16));
    let norm_to_zenith = false;
    let cuda_beam =
        unsafe { beam.cuda_prepare(&freqs_hz, delays.view(), amps.view(), norm_to_zenith)? };

    // Set up the directions to test. The type depends on the CUDA precision.
    let mut azs = Vec::with_capacity(opts.num_directions);
    let mut zas = Vec::with_capacity(opts.num_directions);
    for i in 0..opts.num_directions {
        azs.push(0.4 + 0.3 * PI as CudaFloat * i as CudaFloat / opts.num_directions as CudaFloat);
        zas.push(
            0.3 + 0.4 * PI as CudaFloat / 2.0 * i as CudaFloat / opts.num_directions as CudaFloat,
        );
    }
    let parallactic_correction = true;

    // Call hyperbeam CUDA code.
    let jones = cuda_beam.calc_jones(&azs, &zas, parallactic_correction)?;
    println!("The first Jones matrix:");
    // This works, but the formatting for this isn't very pretty.
    // println!("{}", jones[(0, 0, 0)]);

    // For demonstrations' sake, this gives easier-to-read output.
    let j = jones[(0, 0, 0)];
    println!(
        "[[{:+.8}, {:+.8}]\n [{:+.8}, {:+.8}]]",
        j[0], j[1], j[2], j[3],
    );

    // Compare the differences with the CPU-generated Jones matrices, just for
    // fun. `beam.calc_jones` does the parallactic correction,
    // `beam.calc_jones_eng` does not. Regenerate the direction at double
    // precision for CPU usage.
    let az = azs[0] as f64;
    let za = zas[0] as f64;

    let jones_cpu = beam.calc_jones(
        az,
        za,
        freqs_hz[0],
        delays.slice(s![0, ..]).as_slice().unwrap(),
        amps.slice(s![0, ..]).as_slice().unwrap(),
        norm_to_zenith,
    )?;

    #[cfg(not(feature = "cuda-single"))]
    let diff: Jones<f64> = jones[[0, 0, 0]] - jones_cpu;
    #[cfg(feature = "cuda-single")]
    let diff: Jones<f32> = jones[[0, 0, 0]] - Jones::<f32>::from(jones_cpu);

    println!("Difference between first GPU and CPU Jones matrices");
    println!(
        "[[{:e}, {:e}]\n [{:e}, {:e}]]",
        diff[0], diff[1], diff[2], diff[3]
    );

    Ok(())
}
