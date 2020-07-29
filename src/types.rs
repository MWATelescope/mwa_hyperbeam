// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

/*!
Types that can be pulled away from the main `Hyperbeam` struct.
 */

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

#[derive(Clone, Debug)]
pub(crate) enum Sign {
    Positive,
    Negative,
}

#[derive(Debug)]
pub(crate) enum Pol {
    X,
    Y,
}

impl std::fmt::Display for Pol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Pol::X => "X",
                Pol::Y => "Y",
            }
        )
    }
}

/// A special hash used to determine what's in our coefficients cache.
#[derive(Hash, Debug, Eq, PartialEq)]
pub(crate) struct CacheHash(u64);

impl CacheHash {
    /// Create a new `CacheHash`.
    ///
    /// It hashes the input parameters for a unique hash. If these parameters
    /// are re-used, the same hash will be generated, and we can use the cache
    /// that these `CacheHash`es guard.
    pub(crate) fn new(freq: u32, delays: &[u32], amps: &[f64]) -> Self {
        let mut hasher = DefaultHasher::new();
        freq.hash(&mut hasher);
        delays.hash(&mut hasher);
        // We can't hash f64 values, so convert the amps to ints. Multiply by a
        // big number to get away from integer rounding.
        let amps_ints: Vec<u32> = amps.iter().map(|a| (a * 1e6) as u32).collect();
        &amps_ints.hash(&mut hasher);
        Self(hasher.finish())
    }
}
