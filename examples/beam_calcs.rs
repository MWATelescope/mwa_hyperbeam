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
    #[structopt(parse(from_os_str))]
    hdf5_file: std::path::PathBuf,

    /// The number of pointings to run.
    #[structopt()]
    num_pointings: u32,
}

fn main() -> Result<(), anyhow::Error> {
    let opts = Opts::from_args();
    let mut beam = FEEBeam::new(opts.hdf5_file).unwrap();

    // Set up the pointings to test.
    let mut az = vec![];
    let mut za = vec![];
    for i in 0..opts.num_pointings {
        let coord_deg = 5.0 + i as f64 * 80.0 / opts.num_pointings as f64;
        let coord_rad = coord_deg.to_radians();
        az.push(coord_rad);
        za.push(coord_rad);
    }
    let freq_hz = 51200000;
    let delays = vec![3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0];
    let amps = vec![
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0,
    ];
    let zenith_norm = false;

    // Call hyperbeam.
    let jones = beam
        .calc_jones_array(&az, &za, freq_hz, &delays, &amps, zenith_norm)
        .unwrap();
    println!("The first Jones matrix:");
    println!("{:#?}", jones[0]);

    Ok(())
}
