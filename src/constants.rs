// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Useful constants.

use num_complex::Complex64 as c64;

/// Beamformer delay step \[seconds\]
pub(crate) const DELAY_STEP: f64 = 435.0e-12;

pub(crate) const J_POWER_TABLE: [c64; 4] = [
    c64::new(1.0, 0.0),
    c64::new(0.0, 1.0),
    c64::new(-1.0, 0.0),
    c64::new(0.0, -1.0),
];

/// MWA dipole separation \[metres\]
pub const MWA_DPL_SEP: f64 = 1.100;

/// MWA dipole height (according to mwa_pb) \[metres\]
pub const MWA_DPL_HGT: f64 = 0.278;

/// MWA dipole height (according to the RTS) \[metres\]
pub const MWA_DPL_HGT_RTS: f64 = 0.30;
