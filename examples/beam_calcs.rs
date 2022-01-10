// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Example code using hyperbeam with Rust.
//!
//! Build and run with something like:
//! `cargo run --release --example beam_calcs -- mwa_full_embedded_element_pattern.h5 10000`
//!
//! If you want to use hyperbeam in your own Rust crate, then check out the latest
//! version on crates.io:
//!
//! https://crates.io/crates/mwa_hyperbeam

use std::f64::consts::PI;

use clap::Parser;
use mwa_hyperbeam::fee::FEEBeam;

#[derive(Parser, Debug)]
#[clap()]
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

    /// Don't apply parallactic-angle correction.
    #[clap(short, long)]
    no_parallactic: bool,
}

fn main() -> Result<(), anyhow::Error> {
    let args = Args::parse();
    // If we were given a file, use it. Otherwise, fall back on MWA_BEAM_FILE.
    let beam = match args.hdf5_file {
        Some(f) => FEEBeam::new(f)?,
        None => FEEBeam::new_from_env()?,
    };

    // Set up the directions to test.
    let mut azs = vec![];
    let mut zas = vec![];
    for i in 0..args.num_directions {
        azs.push(0.9 * PI * i as f64 / args.num_directions as f64);
        zas.push(0.1 + 0.9 * PI / 2.0 * i as f64 / args.num_directions as f64);
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

    // Call hyperbeam.
    let jones = if args.parallel {
        if args.no_parallactic {
            beam.calc_jones_eng_array(&azs, &zas, freq_hz, &delays, &amps, norm_to_zenith)
                .unwrap()
        } else {
            beam.calc_jones_array(&azs, &zas, freq_hz, &delays, &amps, norm_to_zenith)
                .unwrap()
        }
    } else {
        let mut results = vec![];
        for (az, za) in azs.into_iter().zip(zas.into_iter()) {
            let j = if args.no_parallactic {
                beam.calc_jones_eng(az, za, freq_hz, &delays, &amps, norm_to_zenith)
                    .unwrap()
            } else {
                beam.calc_jones(az, za, freq_hz, &delays, &amps, norm_to_zenith)
                    .unwrap()
            };
            results.push(j);
        }
        results
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
