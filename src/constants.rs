// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

/*!
Useful constants.
 */

pub use std::f64::consts::PI;

use lazy_static::lazy_static;
use num::complex::Complex64;

lazy_static! {
/// 2 * PI
pub(crate) static ref D2PI: f64 = 2.0 * PI;
/// PI / 2
pub(crate) static ref DPIBY2: f64 = PI / 2.0;

/// Beamformer delay step [seconds]
pub(crate) static ref DELAY_STEP: f64 = 435.0e-12;
/// The number of dipoles per MWA tile.
pub(crate) static ref NUM_DIPOLES: u8 = 16;

pub(crate) static ref J_POWER_TABLE: Vec<Complex64> = vec![
    Complex64::new(1.0, 0.0),
    Complex64::new(0.0, 1.0),
    Complex64::new(-1.0, 0.0),
    Complex64::new(0.0, -1.0),
];
}
