// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Primary beam code for the Murchison Widefield Array.

mod constants;
mod factorial;
pub mod fee;
mod legendre;
mod types;

#[cfg(feature = "python")]
mod python;

// Re-exports.
#[cfg(feature = "cuda")]
/// The float type use in CUDA code. This depends on how `hyperbeam` was
/// compiled (used cargo feature "cuda-single" or "cuda").
pub use fee::CudaFloat;
pub use marlu::{AzEl, Jones}; // So that callers can have a different version of Marlu.
