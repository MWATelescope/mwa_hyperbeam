// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

/*!
Example code using hyperbeam with Rust.

Build and run with something like:
cargo run --release --example beam_calcs -- mwa_full_embedded_element_pattern.h5 10000
 */

use mwa_hyperbeam::*;
use structopt::*;

#[derive(StructOpt, Debug)]
struct Opts {
    /// Path to the HDF5 file.
    #[structopt(short, long, parse(from_os_str))]
    hdf5_file: Option<std::path::PathBuf>,

    /// The number of pointings to run.
    #[structopt()]
    num_pointings: usize,
}

fn main() -> Result<(), anyhow::Error> {
    let opts = Opts::from_args();
    // If we were given a file, use it. Otherwise, fall back on MWA_BEAM_FILE.
    let mut beam = match opts.hdf5_file {
        Some(f) => FEEBeam::new(f)?,
        None => FEEBeam::new_from_env()?,
    };

    // Set up the pointings to test.
    let mut az = Vec::with_capacity(opts.num_pointings);
    let mut za = Vec::with_capacity(opts.num_pointings);
    for i in 0..opts.num_pointings {
        az.push(0.9 * std::f64::consts::PI * i as f64 / opts.num_pointings as f64);
        za.push(0.1 + 0.9 * std::f64::consts::PI / 2.0 * i as f64 / opts.num_pointings as f64);
    }
    let freq_hz = 51200000;
    let delays = vec![0; 16];
    let amps = vec![1.0; 16];
    let norm_to_zenith = false;

    // Call hyperbeam.
    let jones = beam
        .calc_jones_array(&az, &za, freq_hz, &delays, &amps, norm_to_zenith)
        .unwrap();
    println!("The first Jones matrix:");
    for j in jones.outer_iter() {
        // This works, but the formatting for this isn't very pretty.
        // println!("{}", j);

        // For demonstrations' sake, this gives easier-to-read output.
        println!(
            "[[{:+.8}, {:+.8}]\n [{:+.8}, {:+.8}]]",
            j[0], j[1], j[2], j[3]
        );
        break;
    }

    Ok(())
}
