// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Example code using hyperbeam's CUDA with Rust.
//!
//! Build and run with something like:
//! `cargo run --release --features=cuda-single --example fee_cuda -- mwa_full_embedded_element_pattern.h5 1000000`
//!
//! If you want to use hyperbeam in your own Rust crate, then check out the latest
//! version on crates.io:
//!
//! https://crates.io/crates/mwa_hyperbeam

use std::f64::consts::{FRAC_PI_2, PI};

use clap::Parser;
use mwa_hyperbeam::{fee::FEEBeam, AzEl, CudaFloat, Jones};
use ndarray::prelude::*;

#[derive(Parser, Debug)]
#[clap(allow_negative_numbers = true)]
struct Args {
    /// Path to the HDF5 file.
    #[clap(short, long, parse(from_os_str))]
    hdf5_file: Option<std::path::PathBuf>,

    /// If provided, use this latitude for the parallactic-angle correction.
    #[clap(short, long)]
    latitude_rad: Option<f64>,

    /// The number of directions to run.
    #[clap()]
    num_directions: usize,
}

fn main() -> Result<(), anyhow::Error> {
    let args = Args::parse();
    if args.num_directions == 0 {
        eprintln!("num_directions cannot be 0.");
        std::process::exit(1);
    }

    println!(
        "CUDA float precision is {} bits",
        std::mem::size_of::<CudaFloat>() * 8
    );

    // If we were given a file, use it. Otherwise, fall back on MWA_BEAM_FILE.
    let beam = match args.hdf5_file {
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
    let mut azels = Vec::with_capacity(args.num_directions);
    for i in 0..args.num_directions {
        let az = 0.4 + 0.3 * PI * (i / args.num_directions) as f64;
        let za = 0.3 + 0.4 * FRAC_PI_2 * (i / args.num_directions) as f64;
        azels.push(AzEl::new(az, FRAC_PI_2 - za));
    }
    let iau_order = true;

    // Call hyperbeam CUDA code.
    let jones = cuda_beam.calc_jones(&azels, args.latitude_rad, iau_order)?;
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
    let azel = AzEl::new(azels[0].az as f64, azels[0].el as f64);

    let jones_cpu = beam.calc_jones(
        azel,
        freqs_hz[0],
        delays.slice(s![0, ..]).as_slice().unwrap(),
        amps.slice(s![0, ..]).as_slice().unwrap(),
        norm_to_zenith,
        args.latitude_rad,
        iau_order,
    )?;

    let diff = jones[(0, 0, 0)] - Jones::<CudaFloat>::from(jones_cpu);

    println!("Difference between first GPU and CPU Jones matrices");
    println!(
        "[[{:e}, {:e}]\n [{:e}, {:e}]]",
        diff[0], diff[1], diff[2], diff[3]
    );

    Ok(())
}
