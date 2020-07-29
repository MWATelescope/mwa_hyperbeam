// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

/*!
Interface code for hyperbeam.
 */

use num::Complex;

use crate::*;

struct Beam;

pub fn beam_2016_implementation(delays: &[f64; 16], amps: &[f64; 16], hdf5_file: &std::path::Path) {
    let j_power_table: Jones = [
        Complex::new(1.0, 0.0),
        Complex::new(0.0, 1.0),
        Complex::new(-1.0, 0.0),
        Complex::new(0.0, -1.0),
    ];
}
