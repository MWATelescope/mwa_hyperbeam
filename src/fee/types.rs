// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Helper types for the FEE beam.

use std::collections::HashMap;

use marlu::{c64, Jones};
use parking_lot::RwLock;

use crate::types::CacheKey;

/// Coefficients for the X or Y dipole on an MWA bowtie. When combined with an
/// (az, za) direction, this is everything that's needed to calculate a beam
/// response.
// TODO: Improve docs. What does these values actually do?
pub(super) struct DipoleCoefficients {
    pub(super) q1_accum: Vec<c64>,
    pub(super) q2_accum: Vec<c64>,
    pub(super) m_accum: Vec<i8>,
    pub(super) n_accum: Vec<i8>,
    /// The sign of M coefficients (i.e. -1 or 1).
    pub(super) m_signs: Vec<i8>,
    /// The biggest N coefficient.
    pub(super) n_max: usize,
}

pub(super) struct BowtieCoefficients {
    pub(super) x: DipoleCoefficients,
    pub(super) y: DipoleCoefficients,
}

/// [CoeffCache] is just a `RwLock` around a `HashMap`. This allows multiple
/// concurrent readers with the ability to halt all reading when writing.
#[derive(Default)]
pub(super) struct CoeffCache(RwLock<HashMap<CacheKey, BowtieCoefficients>>);

impl std::ops::Deref for CoeffCache {
    type Target = RwLock<HashMap<CacheKey, BowtieCoefficients>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// [NormCache] is very similar to [CoeffCache]. It stores Jones matrices used
/// to normalise beam responses at various frequencies (i.e. frequency is the
/// key of the cache).
#[derive(Default)]
pub(super) struct NormCache(RwLock<HashMap<u32, Jones<f64>>>);

impl std::ops::Deref for NormCache {
    type Target = RwLock<HashMap<u32, Jones<f64>>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
