// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Example code using hyperbeam with Rust.
//!
//! Build and run with something like:
//! `cargo run --release --example fee -- mwa_full_embedded_element_pattern.h5 10000`
//!
//! If you want to use hyperbeam in your own Rust crate, then check out the latest
//! version on crates.io:
//!
//! https://crates.io/crates/mwa_hyperbeam

use std::f64::consts::{FRAC_PI_2, PI};

use clap::Parser;
use mwa_hyperbeam::{fee::FEEBeam, AzEl};

#[derive(Parser, Debug)]
#[clap(allow_negative_numbers = true)]
struct Args {
    /// Path to the HDF5 file.
    #[clap(short, long, parse(from_os_str))]
    hdf5_file: Option<std::path::PathBuf>,

    /// The number of directions to run.
    #[clap()]
    num_directions: usize,

    /// Use these delays when calculating the beam response. There must be 16
    /// values.
    #[clap(short, long, multiple_values(true))]
    delays: Option<Vec<u32>>,

    /// Use these dipole gains when calculating the beam response. There must be
    /// 16 or 32 values.
    #[clap(short, long, multiple_values(true))]
    gains: Option<Vec<f64>>,

    /// Calculate the Jones matrices in parallel.
    #[clap(short, long)]
    parallel: bool,

    /// If provided, use this latitude for the parallactic-angle correction.
    #[clap(short, long)]
    latitude_rad: Option<f64>,
}

fn main() -> Result<(), anyhow::Error> {
    let args = Args::parse();
    // If we were given a file, use it. Otherwise, fall back on MWA_BEAM_FILE.
    let beam = match args.hdf5_file {
        Some(f) => FEEBeam::new(f)?,
        None => FEEBeam::new_from_env()?,
    };

    // Set up the directions to test.
    let mut azels = vec![];
    for i in 0..args.num_directions {
        let az = 0.9 * PI * i as f64 / args.num_directions as f64;
        let za = 0.1 + 0.9 * PI / 2.0 * i as f64 / args.num_directions as f64;
        azels.push(AzEl::new(az, FRAC_PI_2 - za));
    }
    let freq_hz = 51200000;
    // Delays and amps correspond to dipoles in the "M&C order". See
    // https://wiki.mwatelescope.org/pages/viewpage.action?pageId=48005139) for
    // more info.
    let delays = args.delays.unwrap_or_else(|| vec![0; 16]);
    assert_eq!(delays.len(), 16);
    let amps = args.gains.unwrap_or_else(|| vec![1.0; 16]);
    assert!(amps.len() == 16 || amps.len() == 32);
    let norm_to_zenith = false;
    let iau_order = true;

    // Call hyperbeam.
    let jones = if args.parallel {
        println!("Running in parallel");
        beam.calc_jones_array(
            &azels,
            freq_hz,
            &delays,
            &amps,
            norm_to_zenith,
            args.latitude_rad,
            iau_order,
        )
        .unwrap()
    } else {
        println!("Not running in parallel");
        azels
            .into_iter()
            .map(|azel| {
                beam.calc_jones(
                    azel,
                    freq_hz,
                    &delays,
                    &amps,
                    norm_to_zenith,
                    args.latitude_rad,
                    iau_order,
                )
                .unwrap()
            })
            .collect()
    };
    println!("The first Jones matrix:");
    // This works, but the formatting for this isn't very pretty.
    // println!("{}", jones[0]);

    // For demonstrations' sake, this gives easier-to-read output.
    let j = jones[0];
    println!(
        "[[{:+.8}, {:+.8}]\n [{:+.8}, {:+.8}]]",
        j[0], j[1], j[2], j[3]
    );

    Ok(())
}
