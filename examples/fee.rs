// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Example code using hyperbeam with Rust.
//!
//! Build and run with something like:
//! `cargo run --release --example fee -- 10000 mwa_full_embedded_element_pattern.h5`
//!
//! If you want to use hyperbeam in your own Rust crate, then check out the latest
//! version on crates.io:
//!
//! https://crates.io/crates/mwa_hyperbeam

use std::f64::consts::{FRAC_PI_2, PI};

use mwa_hyperbeam::{fee::FEEBeam, AzEl};

fn main() {
    if let Err(e) = try_main() {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}

fn try_main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let num_directions: usize = args
        .next()
        .expect("number of directions supplied")
        .parse()
        .expect("number of directions is a number");
    let beam_file = args.next();
    // If we were given a file, use it. Otherwise, fall back on MWA_BEAM_FILE.
    let beam = match beam_file {
        Some(f) => FEEBeam::new(f)?,
        None => FEEBeam::new_from_env()?,
    };

    // Set up the directions to test.
    let mut azels = vec![];
    for i in 0..num_directions {
        let az = 0.9 * PI * i as f64 / num_directions as f64;
        let za = 0.1 + 0.9 * PI / 2.0 * i as f64 / num_directions as f64;
        azels.push(AzEl::from_radians(az, FRAC_PI_2 - za));
    }
    let freq_hz = 51200000;
    // Delays and amps correspond to dipoles in the "M&C order". See
    // https://wiki.mwatelescope.org/pages/viewpage.action?pageId=48005139) for
    // more info.
    let delays = [0; 16];
    assert_eq!(delays.len(), 16);
    let amps = [1.0; 16];
    assert!(amps.len() == 16 || amps.len() == 32);
    let norm_to_zenith = true;
    let array_latitude_rad = Some(-0.4660608448386394); // MWA
    let iau_order = true;

    // Call hyperbeam.
    let jones = beam.calc_jones_array(
        &azels,
        freq_hz,
        &delays,
        &amps,
        norm_to_zenith,
        array_latitude_rad,
        iau_order,
    )?;
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
