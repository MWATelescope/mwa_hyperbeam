// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

/*!
Finite embedded element (FEE) primary beam code for the Murchison Widefield
Array.
 */

mod constants;
pub mod errors;
pub mod read_hdf5;
pub(crate) mod types;

use constants::*;
pub use errors::*;
pub use read_hdf5::*;
use types::*;

use num::complex::Complex64;

pub(crate) type Jones = [Complex64; 4];
