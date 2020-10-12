// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

/*!
Primary beam code for the Murchison Widefield Array.
 */

mod constants;
pub mod errors;
pub(crate) mod factorial;
pub mod fee;
mod ffi;
pub(crate) mod legendre;
pub(crate) mod types;

use constants::*;
pub use errors::*;
pub use fee::*;
pub(crate) use types::*;

// Re-exports.
use num::complex::Complex64 as c64;
