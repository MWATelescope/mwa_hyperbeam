// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! This program calculates the "m_accum" and "n_accum" for all frequencies
//! specified in the given HDF5 file to verify the file's data. hyperbeam used
//! to have runtime checks for this, but it's much easier on the computer to not
//! check something that likely never happens.

use mwa_hyperbeam::fee::{FEEBeam, InitFEEBeamError};

fn main() {
    // Test each input file.
    for beam_file in std::env::args().skip(1) {
        // If this threw an error, it was during initialisation.
        if let Err(e) = test_file(&beam_file) {
            println!("File '{}' failed to create an FEEBeam: {}", &beam_file, e);
        }
    }
}

fn test_file(beam_file: &str) -> Result<(), InitFEEBeamError> {
    println!("Testing file '{}'", beam_file);
    let beam = FEEBeam::new(&beam_file)?;
    // It does not matter what the direction, delays or amps are, the m_accum
    // and n_accum that we're interested in testing depend only on the
    // frequency. (Frequency determined the number of dipole coeffs, which
    // determines which m_accum and n_accum values are used.)
    for &file_freq in beam.get_freqs() {
        println!("Testing freq {}", file_freq);
        // If this blows up, we know there's a problem...
        beam.calc_jones_pair(
            0.0, 0.0, file_freq, &[0; 16], &[1.0; 16], false, None, false,
        )
        .unwrap();
    }

    println!("File '{}' is all good!", beam_file);
    Ok(())
}
