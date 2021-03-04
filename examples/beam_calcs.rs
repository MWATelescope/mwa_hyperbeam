// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

/*!
Example code using hyperbeam with Rust.

Build and run with something like:
cargo run --release --example beam_calcs -- mwa_full_embedded_element_pattern.h5 10000

If you want to use hyperbeam in your own Rust crate, then check out the latest
version on crates.io:

https://crates.io/crates/mwa_hyperbeam
 */

use mwa_hyperbeam::fee::FEEBeam;
use structopt::*;

#[derive(StructOpt, Debug)]
struct Opts {
    /// Path to the HDF5 file.
    #[structopt(short, long, parse(from_os_str))]
    hdf5_file: Option<std::path::PathBuf>,

    /// The number of directions to run.
    #[structopt()]
    num_directions: usize,

    /// Calculate the Jones matrices in parallel.
    #[structopt(short, long)]
    parallel: bool,
}

fn main() -> Result<(), anyhow::Error> {
    let opts = Opts::from_args();
    // If we were given a file, use it. Otherwise, fall back on MWA_BEAM_FILE.
    let mut beam = match opts.hdf5_file {
        Some(f) => FEEBeam::new(f)?,
        None => FEEBeam::new_from_env()?,
    };

    // Set up the directions to test.
    let mut az = Vec::with_capacity(opts.num_directions);
    let mut za = Vec::with_capacity(opts.num_directions);
    for i in 0..opts.num_directions {
        az.push(0.9 * std::f64::consts::PI * i as f64 / opts.num_directions as f64);
        za.push(0.1 + 0.9 * std::f64::consts::PI / 2.0 * i as f64 / opts.num_directions as f64);
    }
    let freq_hz = 51200000;
    let delays = vec![0; 16];
    let amps = vec![1.0; 16];
    let norm_to_zenith = false;

    // Call hyperbeam.
    let jones = if opts.parallel {
        beam.calc_jones_array(&az, &za, freq_hz, &delays, &amps, norm_to_zenith)
            .unwrap()
    } else {
        az.into_iter()
            .zip(za.into_iter())
            .map(|(a, z)| {
                beam.calc_jones(a, z, freq_hz, &delays, &amps, norm_to_zenith)
                    .unwrap()
            })
            .collect::<Vec<_>>()
            .into()
    };
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
