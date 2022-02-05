// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Example code using hyperbeam's GPU code with Rust.
//!
//! Build and run with something like:
//! `cargo run --release --features=hip --example analytic_hip -- 1000000`
//! `cargo run --release --features=hip,gpu-single --example analytic_hip -- 1000000`
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

use mwa_hyperbeam::{analytic::AnalyticBeam, AzEl, GpuFloat, Jones};
use ndarray::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let num_directions: usize = args
        .next()
        .expect("number of directions supplied")
        .parse()
        .expect("number of directions is a number");
    let beam = AnalyticBeam::new();
    // Could also use a different flavour:
    // let beam = AnalyticBeam::new_rts();

    println!(
        "GPU float precision is {} bits",
        std::mem::size_of::<GpuFloat>() * 8
    );

    // Set up our "GPU beam".
    let freqs_hz = [150e6 as u32, 200e6 as _];
    // Delays and amps correspond to dipoles in the "M&C order". See
    // https://wiki.mwatelescope.org/pages/viewpage.action?pageId=48005139) for
    // more info. Here, each row of the 2D array corresponds to a tile. In this
    // example, all delays and amps are the same, but they are allowed to vary
    // between tiles.
    let num_tiles = 1;
    let delays = Array2::zeros((num_tiles, 16));
    let amps = Array2::ones((num_tiles, 16));
    let norm_to_zenith = true;

    let gpu_beam = unsafe { beam.gpu_prepare(delays.view(), amps.view())? };

    // Set up the directions to test. The type depends on the GPU precision.
    let mut azels = Vec::with_capacity(num_directions);
    for i in 0..num_directions {
        let az = 0.4 + 0.3 * PI * (i / num_directions) as f64;
        let za = 0.3 + 0.4 * FRAC_PI_2 * (i / num_directions) as f64;
        azels.push(AzEl::from_radians(az, FRAC_PI_2 - za));
    }

    // Call hyperbeam GPU code.
    let latitude_rad = -0.4660608448386394; // MWA
    let jones = gpu_beam.calc_jones(&azels, &freqs_hz, latitude_rad, norm_to_zenith)?;
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
    let azel = AzEl::from_radians(azels[0].az, azels[0].el);

    let jones_cpu = beam.calc_jones(
        azel,
        freqs_hz[0],
        delays.slice(s![0, ..]).as_slice().unwrap(),
        amps.slice(s![0, ..]).as_slice().unwrap(),
        latitude_rad,
        norm_to_zenith,
    )?;

    let diff = jones[(0, 0, 0)] - Jones::<GpuFloat>::from(jones_cpu);

    println!("Difference between first GPU and CPU Jones matrices");
    println!(
        "[[{:e}, {:e}]\n [{:e}, {:e}]]",
        diff[0], diff[1], diff[2], diff[3]
    );

    Ok(())
}
