// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

/*!
Useful constants.
 */

use crate::types::{c64, Jones};

/// Beamformer delay step [seconds]
pub(crate) const DELAY_STEP: f64 = 435.0e-12;
/// The number of dipoles per MWA tile.
pub(crate) const NUM_DIPOLES: u8 = 16;

pub(crate) const J_POWER_TABLE: Jones = [
    c64::new(1.0, 0.0),
    c64::new(0.0, 1.0),
    c64::new(-1.0, 0.0),
    c64::new(0.0, -1.0),
];
