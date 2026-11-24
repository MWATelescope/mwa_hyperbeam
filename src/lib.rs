// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Primary beam code for the Murchison Widefield Array.
//!
//! Two kinds of beam are provided:
//!
//! - [`fee::FEEBeam`], the Full Embedded Element beam. It needs the MWA FEE
//!   HDF5 file, given either as a path ([`fee::FEEBeam::new`]) or with the
//!   `MWA_BEAM_FILE` environment variable ([`fee::FEEBeam::new_from_env`]).
//! - [`analytic::AnalyticBeam`], the analytic beam, in the flavours used by
//!   `mwa_pb` and the RTS. It needs no file.
//!
//! Both give beam-response Jones matrices for one direction (`calc_jones`) or
//! many (`calc_jones_array`), given a frequency and a tile's 16 dipole delays
//! and 16 or 32 dipole gains. With the `cuda` or `hip` cargo features, each has
//! a GPU equivalent (`FEEBeam::gpu_prepare` and `AnalyticBeam::gpu_prepare`)
//! for many tiles, frequencies and directions at once.
//!
//! ```no_run
//! use mwa_hyperbeam::{fee::FEEBeam, AzEl};
//!
//! let beam = FEEBeam::new_from_env().unwrap();
//! let delays = [0; 16];
//! let gains = [1.0; 16];
//! let jones = beam
//!     .calc_jones(
//!         AzEl::from_degrees(45.0, 60.0),
//!         167_000_000,
//!         &delays,
//!         &gains,
//!         // Normalise to the zenith.
//!         true,
//!         // Apply the parallactic-angle correction for this latitude...
//!         Some(-0.4660608448386394),
//!         // ... and give the Jones matrix in the IAU order.
//!         true,
//!     )
//!     .unwrap();
//! println!("{jones:?}");
//! ```
//!
//! The C and Python interfaces are described in the
//! [README](https://github.com/MWATelescope/mwa_hyperbeam#readme) and shown in
//! the [examples
//! directory](https://github.com/MWATelescope/mwa_hyperbeam/tree/main/examples).

pub mod analytic;
mod constants;
mod factorial;
pub mod fee;
mod ffi;
mod legendre;
mod types;

#[cfg(feature = "python")]
mod python;

// Re-exports.
cfg_if::cfg_if! {
    if #[cfg(any(feature = "cuda", feature = "hip"))] {
        mod gpu;
        /// The float type use in GPU code. This depends on how `hyperbeam` was
        /// compiled (used cargo feature "gpu-single" or not).
        pub use gpu::{GpuFloat, GpuComplex};
    }
}

pub use marlu::{AzEl, Jones}; // So that callers can have a different version of Marlu.

use ndarray::ArrayView1;

/// Ensure that any delays of 32 have an amplitude (dipole gain) of 0.
///
/// The results are bad otherwise!
/// Also ensure that we have 32 dipole gains (amps) here.
/// Also return a Rust array of delays for convenience.
pub fn fix_amps_ndarray(amps: ArrayView1<f64>, delays: ArrayView1<u32>) -> ([f64; 32], [u32; 16]) {
    let mut full_amps: [f64; 32] = [1.0; 32];
    full_amps
        .iter_mut()
        .zip(amps.iter().cycle())
        .zip(delays.iter().cycle())
        .for_each(|((out_amp, &in_amp), &delay)| {
            if delay == 32 {
                *out_amp = 0.0;
            } else {
                *out_amp = in_amp;
            }
        });

    // So that we don't have to do .as_slice().unwrap() on our ndarrays outside
    // of this function, return a Rust array of delays here.
    let mut delays_a: [u32; 16] = [0; 16];
    delays_a.iter_mut().zip(delays).for_each(|(da, d)| *da = *d);

    (full_amps, delays_a)
}
