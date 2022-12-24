// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Generic types.

use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
};

#[derive(Debug, Clone, Copy)]
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

/// A special key used to access our own coefficients cache.
#[derive(Hash, Debug, Clone, Copy, Eq, PartialEq, Default)]
pub(crate) struct CacheKey(u64);

impl CacheKey {
    /// Create a new [`CacheKey`].
    ///
    /// It hashes the input parameters for a unique hash. If these parameters
    /// are re-used, the same hash will be generated, and we can use the cache
    /// that these [`CacheKey`]s guard.
    pub(crate) fn new(freq: u32, delays: &[u32; 16], amps: &[f64; 32]) -> Self {
        let mut hasher = DefaultHasher::new();
        freq.hash(&mut hasher);
        delays.hash(&mut hasher);
        // We can't hash f64 values, but we can hash their bits.
        for amp in amps {
            amp.to_bits().hash(&mut hasher);
        }
        Self(hasher.finish())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn settings_1() -> (u32, [u32; 16], [f64; 32]) {
        (51200000, [0; 16], [1.0; 32])
    }

    fn settings_2() -> (u32, [u32; 16], [f64; 32]) {
        (
            51200000,
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1.0; 32],
        )
    }

    fn settings_3() -> (u32, [u32; 16], [f64; 32]) {
        (
            51200000,
            [0; 16],
            [
                1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            ],
        )
    }

    fn settings_4() -> (u32, [u32; 16], [f64; 32]) {
        (
            51200000,
            [0; 16],
            [
                1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                // One amp difference from settings_3
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            ],
        )
    }

    #[test]
    fn same() {
        let s1 = settings_1();
        let hash1 = CacheKey::new(s1.0, &s1.1, &s1.2).0;

        let s2 = settings_1();
        let hash2 = CacheKey::new(s2.0, &s2.1, &s2.2).0;
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn different1() {
        let s1 = settings_1();
        let hash1 = CacheKey::new(s1.0, &s1.1, &s1.2).0;

        let s2 = settings_2();
        let hash2 = CacheKey::new(s2.0, &s2.1, &s2.2).0;
        assert_ne!(hash1, hash2);
    }

    #[test]
    fn different2() {
        let s1 = settings_1();
        let hash1 = CacheKey::new(s1.0, &s1.1, &s1.2).0;

        let s2 = settings_3();
        let hash2 = CacheKey::new(s2.0, &s2.1, &s2.2).0;
        assert_ne!(hash1, hash2);
    }

    #[test]
    fn different3() {
        let s1 = settings_2();
        let hash1 = CacheKey::new(s1.0, &s1.1, &s1.2).0;

        let s2 = settings_3();
        let hash2 = CacheKey::new(s2.0, &s2.1, &s2.2).0;
        assert_ne!(hash1, hash2);
    }

    #[test]
    fn different4() {
        let s1 = settings_1();
        let hash1 = CacheKey::new(s1.0, &s1.1, &s1.2).0;

        // This situation is a little unrealistic; when the settings are being
        // used for the FEE beam, the frequency will be "rounded" to the nearest
        // defined frequency in the HDF5 file. Such a small change here would
        // actually give the same cache, because the same frequency is
        // used. But, if we compute the hash before swapping out the frequency
        // (which is what happens in the real code), we expect a difference.
        let mut s2 = settings_1();
        s2.0 += 1;
        let hash2 = CacheKey::new(s2.0, &s2.1, &s2.2).0;
        assert_ne!(hash1, hash2);
    }

    #[test]
    fn different5() {
        let s1 = settings_3();
        let hash1 = CacheKey::new(s1.0, &s1.1, &s1.2).0;

        let s2 = settings_4();
        let hash2 = CacheKey::new(s2.0, &s2.1, &s2.2).0;
        assert_ne!(hash1, hash2);
    }
}
