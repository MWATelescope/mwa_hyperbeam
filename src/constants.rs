// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Useful constants.

use num_complex::Complex64 as c64;

/// Beamformer delay step \[seconds\]
pub(crate) const DELAY_STEP: f64 = 435.0e-12;
/// The number of dipoles per MWA tile.
pub(crate) const NUM_DIPOLES: u8 = 16;

pub(crate) const J_POWER_TABLE: [c64; 4] = [
    c64::new(1.0, 0.0),
    c64::new(0.0, 1.0),
    c64::new(-1.0, 0.0),
    c64::new(0.0, -1.0),
];

/// MWA dipole separation \[metres\]
pub(crate) const MWA_DPL_SEP: f64 = 1.100;
