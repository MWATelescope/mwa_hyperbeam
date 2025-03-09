// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Primary beam code for the Murchison Widefield Array.

pub mod analytic;
pub mod constants;
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
        /// The float type used in GPU code. This depends on how `hyperbeam` was
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
