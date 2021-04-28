// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Helper types for the FEE beam.

use dashmap::DashMap;

use crate::types::*;

/// Coefficients for X and Y.
// TODO: Improve docs.
pub(super) struct PolCoefficients {
    pub(super) q1_accum: Vec<c64>,
    pub(super) q2_accum: Vec<c64>,
    pub(super) m_accum: Vec<i8>,
    pub(super) n_accum: Vec<i8>,
    /// The sign of M coefficients (i.e. -1 or 1).
    pub(super) m_signs: Vec<i8>,
    /// The biggest N coefficient.
    pub(super) n_max: usize,
}

pub(super) struct DipoleCoefficients {
    pub(super) x: PolCoefficients,
    pub(super) y: PolCoefficients,
}

/// `CoeffCache` is mostly just a `RwLock` around a `HashMap` (which is handled
/// by `DashMap`). This allows multiple concurrent readers with the ability to
/// halt all reading when writing.
///
/// A `CacheHash` is used as the key. This is a wrapper around Rust's own
/// hashing code so that we get something specific to FEE beam settings.
#[derive(Default)]
pub(super) struct CoeffCache(DashMap<CacheHash, DipoleCoefficients>);

impl std::ops::Deref for CoeffCache {
    type Target = DashMap<CacheHash, DipoleCoefficients>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// `NormCache` is very similar to `CoeffCache`. It stores Jones matrices used
/// to normalise beam responses at various frequencies (i.e. frequency is the
/// key of the `HashMap`).
#[derive(Default)]
pub(super) struct NormCache(DashMap<u32, Jones>);

impl std::ops::Deref for NormCache {
    type Target = DashMap<u32, Jones>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
