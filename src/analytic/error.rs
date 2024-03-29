// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Errors associated with the analytic beam.

use thiserror::Error;

#[derive(Error, Debug)]
pub enum AnalyticBeamError {
    #[error("The number of amps wasn't {expected1} or {expected2} (got {got}); these must either correspond to bowties or X dipoles then Y dipoles in the M&C order")]
    IncorrectAmpsLength {
        got: usize,
        expected1: usize,
        expected2: usize,
    },

    #[error("The number of delays wasn't {expected} (got {got}); these must either correspond to bowties in the M&C order")]
    IncorrectDelaysLength { got: usize, expected: usize },

    #[error("Got a zenith angle ({za} radians), but this is below the horizon")]
    BelowHorizon { za: f64 },

    #[cfg(any(feature = "cuda", feature = "hip"))]
    #[error("The number of delays wasn't {expected} (got {rows} tiles with {num_delays} each); each tile's {expected} delays these must correspond to bowties in the M&C order")]
    IncorrectDelaysArrayColLength {
        rows: usize,
        num_delays: usize,
        expected: usize,
    },

    #[cfg(any(feature = "cuda", feature = "hip"))]
    #[error(transparent)]
    Gpu(#[from] crate::gpu::GpuError),
}
