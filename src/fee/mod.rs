// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code to implement the MWA Fully Embedded Element (FEE) beam, a.k.a. "the
//! 2016 beam".

mod error;
mod ffi;
#[cfg(any(feature = "cuda", feature = "hip"))]
mod gpu;
#[cfg(test)]
mod tests;
mod types;

pub use error::{FEEBeamError, InitFEEBeamError};
use types::*;

#[cfg(any(feature = "cuda", feature = "hip"))]
pub use gpu::FEEBeamGpu;

use std::{
    f64::consts::{FRAC_PI_2, TAU},
    sync::Mutex,
};

use marlu::{AzEl, Jones};
use ndarray::prelude::*;
use num_complex::Complex64 as c64;
use parking_lot::{MappedRwLockReadGuard, RwLockReadGuard};
use rayon::prelude::*;

use crate::{
    constants::*,
    factorial::FACTORIAL,
    legendre::p1sin,
    types::{CacheKey, Pol},
};

/// The main struct to be used for calculating Jones matrices.
#[allow(clippy::upper_case_acronyms)]
pub struct FEEBeam {
    /// The [`hdf5::File`] struct associated with the opened HDF5 file. It is
    /// behind a [`Mutex`] to prevent parallel usage of the file.
    hdf5_file: Mutex<hdf5::File>,
    /// An ascendingly-sorted vector of frequencies available in the HDF5 file.
    /// Not allowed to be empty.
    freqs: Vec<u32>,
    /// Values used in calculating coefficients for X and Y.
    /// Row 0: Type
    /// Row 1: M
    /// Row 2: N
    modes: Array2<i8>,
    /// A cache of X and Y coefficients.
    coeff_cache: CoeffCache,
    /// A cache of normalisation Jones matrices.
    norm_cache: NormCache,
}

impl FEEBeam {
    /// Given the path to an FEE beam file, create a new [`FEEBeam`] struct.
    pub fn new<T: AsRef<std::path::Path>>(file: T) -> Result<Self, InitFEEBeamError> {
        // so that libhdf5 doesn't print errors to stdout
        hdf5::silence_errors(true);

        // If the file doesn't exist, hdf5::File::open will handle it, but the
        // error message is horrendous.
        if !file.as_ref().exists() {
            return Err(InitFEEBeamError::BeamFileDoesntExist(
                file.as_ref().display().to_string(),
            ));
        }
        let h5 = hdf5::File::open(file)?;
        // We want all of the available frequencies and the biggest antenna index.
        let mut freqs: Vec<u32> = vec![];
        let mut biggest_dip_index: Option<u8> = None;
        // Iterate over all of the h5 dataset names.
        for d in h5.member_names()? {
            if d.starts_with('X') {
                // This is the part between 'X' and '_';
                let dipole_index_str = d.strip_prefix('X').unwrap().split('_').next();
                let dipole_index = match dipole_index_str {
                    Some(s) => match s.parse() {
                        Ok(i) => i,
                        Err(_) => return Err(InitFEEBeamError::Parse(s.to_string())),
                    },
                    None => return Err(InitFEEBeamError::MissingDipole),
                };
                match biggest_dip_index {
                    None => biggest_dip_index = Some(dipole_index),
                    Some(b) => {
                        if dipole_index > b {
                            biggest_dip_index = Some(dipole_index);
                        }
                    }
                }
            } else {
                continue;
            }

            // Get all the frequencies from the datasets with names starting "X1_".
            if d.starts_with("X1_") {
                let freq_str = d.strip_prefix("X1_").unwrap();
                let freq: u32 = match freq_str.parse() {
                    Ok(f) => f,
                    Err(_) => return Err(InitFEEBeamError::Parse(freq_str.to_string())),
                };
                freqs.push(freq);
            }
        }

        // Sanity checks.
        match biggest_dip_index {
            None => return Err(InitFEEBeamError::NoDipoles),
            Some(NUM_DIPOLES) => (),
            Some(i) => {
                return Err(InitFEEBeamError::DipoleCountMismatch {
                    expected: NUM_DIPOLES,
                    got: i,
                });
            }
        }
        if freqs.is_empty() {
            return Err(InitFEEBeamError::NoFreqs);
        }

        freqs.sort_unstable();

        let modes = {
            let h5_modes = h5.dataset("modes")?.read_raw()?;
            // The modes dataset is a 2D array with three rows. If 3 doesn't
            // divide evenly into the data length, then something is wrong.
            if h5_modes.len() % 3 == 0 {
                Array2::from_shape_vec((3, h5_modes.len() / 3), h5_modes).unwrap()
            } else {
                return Err(InitFEEBeamError::ModesShape);
            }
        };

        Ok(Self {
            hdf5_file: Mutex::new(h5),
            freqs,
            modes,
            coeff_cache: CoeffCache::default(),
            norm_cache: NormCache::default(),
        })
    }

    /// Create a new [`FEEBeam`] struct from the `MWA_BEAM_FILE` environment
    /// variable.
    pub fn new_from_env() -> Result<Self, InitFEEBeamError> {
        match std::env::var("MWA_BEAM_FILE") {
            Ok(f) => Self::new(f),
            Err(e) => Err(InitFEEBeamError::MwaBeamFileVarError(e)),
        }
    }

    /// Get the frequencies defined in the HDF5 file that was used to create
    /// this [`FEEBeam`]. They are ascendingly sorted.
    pub fn get_freqs(&self) -> &[u32] {
        &self.freqs
    }

    /// Given a frequency in Hz, find the closest frequency that is defined in
    /// the HDF5 file.
    pub fn find_closest_freq(&self, desired_freq_hz: u32) -> u32 {
        let mut best_freq = None;
        let mut best_diff = u32::MAX;
        for &freq in self.freqs.iter() {
            if let Some(best) = best_freq {
                let this_diff = desired_freq_hz.abs_diff(freq);
                if this_diff < best_diff {
                    best_diff = this_diff;
                    best_freq = Some(freq);
                } else {
                    // Because the frequencies are always ascendingly
                    // sorted, if the frequency difference is getting
                    // bigger, we can return early.
                    return best;
                }
            } else {
                best_freq = Some(freq);
            }
        }

        best_freq.expect("self.freqs is not allowed to be empty so this can't fail")
    }

    /// Given a key, get a dataset from the HDF5 file.
    ///
    /// This function is expected to only receive keys like X16_51200000
    fn get_dataset(&self, key: &str) -> Result<Array2<f64>, FEEBeamError> {
        let h5 = self.hdf5_file.lock().unwrap();
        let h5_data = h5.dataset(key)?.read_raw()?;
        // The aforementioned expected keys are 2D arrays with two rows. If 2
        // doesn't divide evenly into the data length, then something is wrong.
        if h5_data.len() % 2 == 0 {
            let arr = Array2::from_shape_vec((2, h5_data.len() / 2), h5_data).unwrap();
            Ok(arr)
        } else {
            Err(FEEBeamError::DatasetShape {
                key: key.to_string(),
                exp: 2,
            })
        }
    }

    /// Get [`DipoleCoefficients`] for the input parameters.
    ///
    /// This function is deliberately private; it uses a cache on [`FEEBeam`] as
    /// calculating [`DipoleCoefficients`] is expensive, and it's easy to
    /// accidentally stall the cache with locks. This function automatically
    /// populates the cache with [`DipoleCoefficients`] and returns a reference
    /// to them.
    ///
    /// Note that specified frequencies are "rounded" to frequencies that are
    /// defined the HDF5 file.
    fn get_modes(
        &self,
        desired_freq_hz: u32,
        delays: &[u32; 16],
        amps: &[f64; 32],
    ) -> Result<MappedRwLockReadGuard<'_, BowtieCoefficients>, FEEBeamError> {
        let fee_freq = self.find_closest_freq(desired_freq_hz);

        // Are the input settings already cached? Hash them to check.
        let hash = CacheKey::new(fee_freq, delays, amps);

        // If the cache for this hash is already populated, we can return the reference.
        {
            let cache = self.coeff_cache.read();
            if cache.contains_key(&hash) {
                return Ok(RwLockReadGuard::map(cache, |hm| &hm[&hash]));
            }
        }

        // If we hit this part of the code, we need to populate the cache.
        let m = self.calc_modes(fee_freq, delays, amps)?;
        {
            let mut locked_cache = self.coeff_cache.write();
            locked_cache.insert(hash, m);
        }
        Ok(RwLockReadGuard::map(self.coeff_cache.read(), |hm| {
            &hm[&hash]
        }))
    }

    /// Get a [`Jones`] matrix for beam normalisation.
    ///
    /// This function is deliberately private and is intertwined with
    /// `get_modes`; this function should always be called before `get_modes` to
    /// prevent a deadlock. Beam normalisation Jones matrices are cached but
    /// because [`Jones`] is [`Copy`], an owned copy is returned from the cache.
    fn get_norm_jones(&self, desired_freq_hz: u32) -> Result<Jones<f64>, FEEBeamError> {
        // Are the input settings already cached? Hash them to check.
        let fee_freq = self.find_closest_freq(desired_freq_hz);

        // If the cache for this hash is already populated, we can return the
        // reference.
        {
            let cache = self.norm_cache.read();
            if cache.contains_key(&fee_freq) {
                return Ok(cache[&fee_freq]);
            }
        }

        // If we hit this part of the code, we need to populate the modes cache.
        let n = {
            let norm_coeffs = self.get_modes(fee_freq, &[0; 16], &[1.0; 32])?;
            calc_zenith_norm_jones(&norm_coeffs)
        };
        {
            let mut locked_cache = self.norm_cache.write();
            locked_cache.insert(fee_freq, n);
        }
        Ok(n)
    }

    /// Given the input parameters, calculate and return the X and Y
    /// coefficients ("modes"). As this function is relatively expensive, it
    /// should only be called by `Self::get_modes` to cache the outputs.
    fn calc_modes(
        &self,
        freq: u32,
        delays: &[u32; 16],
        amps: &[f64; 32],
    ) -> Result<BowtieCoefficients, FEEBeamError> {
        let mut x = self.calc_mode(freq, delays, amps, Pol::X)?;
        let mut y = self.calc_mode(freq, delays, amps, Pol::Y)?;

        // Shave off any excess capacity before we store the results in the
        // cache.
        for dipole_coeffs in [&mut x, &mut y] {
            dipole_coeffs.q1_accum.shrink_to_fit();
            dipole_coeffs.q2_accum.shrink_to_fit();
            dipole_coeffs.m_accum.shrink_to_fit();
            dipole_coeffs.n_accum.shrink_to_fit();
            dipole_coeffs.m_signs.shrink_to_fit();
        }

        Ok(BowtieCoefficients { x, y })
    }

    /// Given the input parameters, calculate and return the coefficients for a
    /// single polarisation (X or Y). This function should only be called by
    /// `Self::calc_modes`.
    ///
    /// This function can only produce sensible results if the HDF5 file
    /// supplying the modes is also sensible. To check that the file is OK, run
    /// verify-beam-file.
    fn calc_mode(
        &self,
        freq_hz: u32,
        delays: &[u32; 16],
        amps: &[f64; 32],
        pol: Pol,
    ) -> Result<DipoleCoefficients, FEEBeamError> {
        let mut q1_accum: Vec<c64> = vec![c64::default(); self.modes.dim().1];
        let mut q2_accum: Vec<c64> = vec![c64::default(); self.modes.dim().1];
        let mut m_accum = vec![];
        let mut n_accum = vec![];
        // Biggest N coefficient.
        let mut n_max = 0;

        // Indices of S=1 coefficients.
        let mut s1_list: Vec<usize> = vec![];
        // Indices of S=2 coefficients.
        let mut s2_list: Vec<usize> = vec![];
        let mut ms1 = vec![];
        let mut ns1 = vec![];
        let mut ms2 = vec![];
        let mut ns2 = vec![];

        // Use the X or Y dipole gains based off how many elements to skip in
        // `amps`.
        let skip = match pol {
            Pol::X => 0,
            Pol::Y => 16,
        };

        for (dipole_num, (&amp, &delay)) in amps.iter().skip(skip).zip(delays.iter()).enumerate() {
            s1_list.clear();
            s2_list.clear();
            ms1.clear();
            ns1.clear();
            ms2.clear();
            ns2.clear();

            // Get the relevant HDF5 data.
            let q_all: Array2<f64> = {
                let key = format!("{}{}_{}", pol, dipole_num + 1, freq_hz);
                self.get_dataset(&key)?
            };
            let n_dip_coeffs: usize = q_all.dim().1;

            // Complex excitation voltage.
            let v: c64 = {
                let phase = TAU * freq_hz as f64 * (-(delay as f64)) * DELAY_STEP;
                let phase_factor = c64::cis(phase);
                amp * phase_factor
            };

            // What does this do???
            let mut b_update_n_accum = false;
            for i in 0..n_dip_coeffs {
                let mode_type = self.modes[[0, i]];
                let mode_m = self.modes[[1, i]];
                let mode_n = self.modes[[2, i]];

                if mode_type <= 1 {
                    s1_list.push(i);
                    ms1.push(mode_m);
                    ns1.push(mode_n);

                    if mode_n > n_max {
                        n_max = mode_n;
                        b_update_n_accum = true;
                    }
                } else {
                    s2_list.push(i);
                    ms2.push(mode_m);
                    ns2.push(mode_n);
                }
            }

            if b_update_n_accum {
                std::mem::swap(&mut m_accum, &mut ms1);
                std::mem::swap(&mut n_accum, &mut ns1);
            };

            if s1_list.len() != s2_list.len() || s2_list.len() != n_dip_coeffs / 2 {
                return Err(FEEBeamError::S1S2CountMismatch {
                    expected: n_dip_coeffs / 2,
                    got: s2_list.len(),
                });
            }

            for i in 0..n_dip_coeffs / 2 {
                // Calculate Q1.
                let s1_idx = s1_list[i];
                let s10_coeff = q_all[[0, s1_idx]];
                let s11_coeff = q_all[[1, s1_idx]];
                let arg = s11_coeff.to_radians();
                let q1_val = s10_coeff * c64::cis(arg);
                q1_accum[i] += q1_val * v;

                // Calculate Q2.
                let s2_idx = s2_list[i];
                let s20_coeff = q_all[[0, s2_idx]];
                let s21_coeff = q_all[[1, s2_idx]];
                let arg = s21_coeff.to_radians();
                let q2_val = s20_coeff * c64::cis(arg);
                q2_accum[i] += q2_val * v;
            }
        }

        let mut m_signs = Vec::with_capacity(m_accum.len());
        for m in &m_accum {
            let sign = if *m > 0 && *m % 2 != 0 { -1 } else { 1 };
            m_signs.push(sign)
        }

        if m_accum.len() != n_accum.len() {
            return Err(FEEBeamError::CoeffCountMismatch {
                ctype: "n_accum",
                got: q1_accum.len(),
                expected: m_accum.len(),
            });
        }
        if m_accum.len() != m_signs.len() {
            return Err(FEEBeamError::CoeffCountMismatch {
                ctype: "m_signs",
                got: m_signs.len(),
                expected: m_accum.len(),
            });
        }

        Ok(DipoleCoefficients {
            q1_accum,
            q2_accum,
            m_accum,
            n_accum,
            m_signs,
            n_max: n_max.try_into().expect("n_max is always positive"),
        })
    }

    /// Calculate the beam-response Jones matrix for a given direction and
    /// pointing. If `latitude_rad` is *not* supplied, the result will match
    /// the original specification of the FEE beam code (possibly more useful
    /// for engineers).
    ///
    /// Astronomers are more likely to want to specify `latitude_rad` (which
    /// will apply the parallactic-angle correction using the Earth latitude
    /// provided for the telescope) and `iau_order`. If `latitude_rad` is
    /// `None`, then `iau_reorder` does nothing. See this document for more
    /// information:
    /// <https://github.com/MWATelescope/mwa_hyperbeam/blob/main/fee_pols.pdf>
    ///
    /// `delays` and `amps` apply to each dipole in an MWA tile in the M&C
    /// order; see
    /// <https://wiki.mwatelescope.org/pages/viewpage.action?pageId=48005139>.
    /// `delays` *must* have 16 elements, whereas `amps` can have 16 or 32
    /// elements; if 16 are given, then these map 1:1 with dipoles, otherwise
    /// the first 16 are for X dipole elements, and the next 16 are for Y.
    #[allow(clippy::too_many_arguments)]
    pub fn calc_jones(
        &self,
        azel: AzEl,
        freq_hz: u32,
        delays: &[u32],
        amps: &[f64],
        norm_to_zenith: bool,
        latitude_rad: Option<f64>,
        iau_order: bool,
    ) -> Result<Jones<f64>, FEEBeamError> {
        self.calc_jones_pair(
            azel.az,
            azel.za(),
            freq_hz,
            delays,
            amps,
            norm_to_zenith,
            latitude_rad,
            iau_order,
        )
    }

    /// Calculate the beam-response Jones matrix for a given direction and
    /// pointing. If `latitude_rad` is *not* supplied, the result will match
    /// the original specification of the FEE beam code (possibly more useful
    /// for engineers).
    ///
    /// Astronomers are more likely to want to specify `latitude_rad` (which
    /// will apply the parallactic-angle correction using the Earth latitude
    /// provided for the telescope) and `iau_order`. If `latitude_rad` is
    /// `None`, then `iau_reorder` does nothing. See this document for more
    /// information:
    /// <https://github.com/MWATelescope/mwa_hyperbeam/blob/main/fee_pols.pdf>
    ///
    /// `delays` and `amps` apply to each dipole in an MWA tile in the M&C
    /// order; see
    /// <https://wiki.mwatelescope.org/pages/viewpage.action?pageId=48005139>.
    /// `delays` *must* have 16 elements, whereas `amps` can have 16 or 32
    /// elements; if 16 are given, then these map 1:1 with dipoles, otherwise
    /// the first 16 are for X dipole elements, and the next 16 are for Y.
    #[allow(clippy::too_many_arguments)]
    pub fn calc_jones_pair(
        &self,
        az_rad: f64,
        za_rad: f64,
        freq_hz: u32,
        delays: &[u32],
        amps: &[f64],
        norm_to_zenith: bool,
        latitude_rad: Option<f64>,
        iau_order: bool,
    ) -> Result<Jones<f64>, FEEBeamError> {
        if delays.len() != 16 {
            return Err(FEEBeamError::IncorrectDelaysLength(delays.len()));
        }
        let delays: &[u32; 16] = delays.try_into().unwrap();
        if !(amps.len() == 16 || amps.len() == 32) {
            return Err(FEEBeamError::IncorrectAmpsLength(amps.len()));
        }
        let full_amps = fix_amps(amps, delays);

        // If we're normalising the beam, get the normalisation Jones matrix here.
        let norm_jones = match norm_to_zenith {
            true => Some(self.get_norm_jones(freq_hz)?),
            false => None,
        };

        // Populate the coefficients cache if it isn't already populated.
        let coeffs = self.get_modes(freq_hz, delays, &full_amps)?;

        let mut jones = calc_jones_direct(az_rad, za_rad, &coeffs, norm_jones);
        if let Some(latitude_rad) = latitude_rad {
            apply_parallactic_correction(az_rad, za_rad, latitude_rad, iau_order, &mut jones);
        }

        Ok(jones)
    }

    /// Calculate the beam-response Jones matrices for many directions given a
    /// pointing. This is basically a wrapper around `calc_jones` that
    /// efficiently calculates the Jones matrices in parallel. The number of
    /// parallel threads used can be controlled by setting `RAYON_NUM_THREADS`.
    ///
    /// `delays` and `amps` apply to each dipole in an MWA tile in the M&C
    /// order; see
    /// <https://wiki.mwatelescope.org/pages/viewpage.action?pageId=48005139>.
    /// `delays` *must* have 16 elements, whereas `amps` can have 16 or 32
    /// elements; if 16 are given, then these map 1:1 with dipoles, otherwise
    /// the first 16 are for X dipole elements, and the next 16 are for Y.
    #[allow(clippy::too_many_arguments)]
    pub fn calc_jones_array(
        &self,
        azels: &[AzEl],
        freq_hz: u32,
        delays: &[u32],
        amps: &[f64],
        norm_to_zenith: bool,
        latitude_rad: Option<f64>,
        iau_order: bool,
    ) -> Result<Vec<Jones<f64>>, FEEBeamError> {
        let mut results = vec![Jones::default(); azels.len()];
        self.calc_jones_array_inner(
            azels,
            freq_hz,
            delays,
            amps,
            norm_to_zenith,
            latitude_rad,
            iau_order,
            &mut results,
        )?;
        Ok(results)
    }

    /// Calculate the Jones matrices for many directions given a pointing. This
    /// is the same as `calc_jones_array` but uses pre-allocated memory.
    ///
    /// `delays` and `amps` apply to each dipole in an MWA tile in the M&C
    /// order; see
    /// <https://wiki.mwatelescope.org/pages/viewpage.action?pageId=48005139>.
    /// `delays` *must* have 16 elements, whereas `amps` can have 16 or 32
    /// elements; if 16 are given, then these map 1:1 with dipoles, otherwise
    /// the first 16 are for X dipole elements, and the next 16 are for Y.
    #[allow(clippy::too_many_arguments)]
    pub fn calc_jones_array_inner(
        &self,
        azels: &[AzEl],
        freq_hz: u32,
        delays: &[u32],
        amps: &[f64],
        norm_to_zenith: bool,
        latitude_rad: Option<f64>,
        iau_order: bool,
        results: &mut [Jones<f64>],
    ) -> Result<(), FEEBeamError> {
        if delays.len() != 16 {
            return Err(FEEBeamError::IncorrectDelaysLength(delays.len()));
        }
        let delays: &[u32; 16] = delays.try_into().unwrap();
        if !(amps.len() == 16 || amps.len() == 32) {
            return Err(FEEBeamError::IncorrectAmpsLength(amps.len()));
        }
        let full_amps = fix_amps(amps, delays);

        // If we're normalising the beam, get the normalisation Jones matrix here.
        let norm_jones = match norm_to_zenith {
            true => Some(self.get_norm_jones(freq_hz)?),
            false => None,
        };

        // Populate the coefficients cache if it isn't already populated.
        let coeffs = self.get_modes(freq_hz, delays, &full_amps)?;

        azels
            .par_iter()
            .zip(results.par_iter_mut())
            .for_each(|(&azel, result)| {
                let az = azel.az;
                let za = azel.za();
                let mut jones = calc_jones_direct(az, za, &coeffs, norm_jones);
                if let Some(latitude_rad) = latitude_rad {
                    apply_parallactic_correction(az, za, latitude_rad, iau_order, &mut jones);
                }
                *result = jones;
            });
        Ok(())
    }

    /// Calculate the Jones matrices for many directions given a pointing. This
    /// is basically a wrapper around `calc_jones` that efficiently calculates
    /// the Jones matrices in parallel. The number of parallel threads used can
    /// be controlled by setting `RAYON_NUM_THREADS`.
    ///
    /// `delays` and `amps` apply to each dipole in an MWA tile in the M&C
    /// order; see
    /// <https://wiki.mwatelescope.org/pages/viewpage.action?pageId=48005139>.
    /// `delays` *must* have 16 elements, whereas `amps` can have 16 or 32
    /// elements; if 16 are given, then these map 1:1 with dipoles, otherwise
    /// the first 16 are for X dipole elements, and the next 16 are for Y.
    #[allow(clippy::too_many_arguments)]
    pub fn calc_jones_array_pair(
        &self,
        az_rad: &[f64],
        za_rad: &[f64],
        freq_hz: u32,
        delays: &[u32],
        amps: &[f64],
        norm_to_zenith: bool,
        latitude_rad: Option<f64>,
        iau_order: bool,
    ) -> Result<Vec<Jones<f64>>, FEEBeamError> {
        let mut results = vec![Jones::default(); az_rad.len()];
        self.calc_jones_array_pair_inner(
            az_rad,
            za_rad,
            freq_hz,
            delays,
            amps,
            norm_to_zenith,
            latitude_rad,
            iau_order,
            &mut results,
        )?;
        Ok(results)
    }

    /// Calculate the Jones matrices for many directions given a pointing. This
    /// is the same as `calc_jones_array_pair` but uses pre-allocated memory.
    ///
    /// `delays` and `amps` apply to each dipole in an MWA tile in the M&C
    /// order; see
    /// <https://wiki.mwatelescope.org/pages/viewpage.action?pageId=48005139>.
    /// `delays` *must* have 16 elements, whereas `amps` can have 16 or 32
    /// elements; if 16 are given, then these map 1:1 with dipoles, otherwise
    /// the first 16 are for X dipole elements, and the next 16 are for Y.
    #[allow(clippy::too_many_arguments)]
    pub fn calc_jones_array_pair_inner(
        &self,
        az_rad: &[f64],
        za_rad: &[f64],
        freq_hz: u32,
        delays: &[u32],
        amps: &[f64],
        norm_to_zenith: bool,
        latitude_rad: Option<f64>,
        iau_order: bool,
        results: &mut [Jones<f64>],
    ) -> Result<(), FEEBeamError> {
        if delays.len() != 16 {
            return Err(FEEBeamError::IncorrectDelaysLength(delays.len()));
        }
        let delays: &[u32; 16] = delays.try_into().unwrap();
        if !(amps.len() == 16 || amps.len() == 32) {
            return Err(FEEBeamError::IncorrectAmpsLength(amps.len()));
        }
        let full_amps = fix_amps(amps, delays);

        // If we're normalising the beam, get the normalisation Jones matrix here.
        let norm_jones = match norm_to_zenith {
            true => Some(self.get_norm_jones(freq_hz)?),
            false => None,
        };

        // Populate the coefficients cache if it isn't already populated.
        let coeffs = self.get_modes(freq_hz, delays, &full_amps)?;

        az_rad
            .par_iter()
            .zip(za_rad.par_iter())
            .zip(results.par_iter_mut())
            .for_each(|((&az, &za), result)| {
                let mut jones = calc_jones_direct(az, za, &coeffs, norm_jones);
                if let Some(latitude_rad) = latitude_rad {
                    apply_parallactic_correction(az, za, latitude_rad, iau_order, &mut jones);
                }
                *result = jones;
            });
        Ok(())
    }

    /// Empty the cached dipole coefficients and normalisation Jones matrices to
    /// recover memory.
    pub fn empty_cache(&self) {
        self.coeff_cache.write().clear();
        self.norm_cache.write().clear();
    }

    /// Prepare a compute-capable GPU device for beam-response computations
    /// given the frequencies, delays and amps to be used. The resulting object
    /// takes directions and computes the beam responses on the device.
    ///
    /// `delays_array` and `amps_array` must have the same number of rows; these
    /// correspond to tile configurations (i.e. each tile is allowed to have
    /// distinct delays and amps). `delays_array` must have 16 elements per row,
    /// but `amps_array` can have 16 or 32 elements per row (see `calc_jones`
    /// for an explanation).
    ///
    /// The code will automatically de-duplicate tile configurations so that no
    /// redundant calculations are done.
    ///
    /// # Safety
    ///
    /// This function interfaces directly with the CUDA/HIP API. Rust errors
    /// attempt to catch problems but there are no guarantees.
    #[cfg(any(feature = "cuda", feature = "hip"))]
    pub unsafe fn gpu_prepare(
        &self,
        freqs_hz: &[u32],
        delays_array: ArrayView2<u32>,
        amps_array: ArrayView2<f64>,
        norm_to_zenith: bool,
    ) -> Result<gpu::FEEBeamGpu, FEEBeamError> {
        // This function is deliberately kept thin to keep the focus of this
        // module on the CPU code.
        gpu::FEEBeamGpu::new(self, freqs_hz, delays_array, amps_array, norm_to_zenith)
    }
}

/// Calculate the Jones matrix components given a pointing and coefficients
/// associated with a single dipole polarisation.
fn calc_sigmas(
    phi: f64,
    theta: f64,
    coeffs: &DipoleCoefficients,
    p1sin_arr: &[f64],
    p1_arr: &[f64],
) -> (c64, c64) {
    let u = theta.cos();
    let mut sigma_p = c64::default();
    let mut sigma_t = c64::default();
    // Use an iterator for maximum performance.
    coeffs
        .m_accum
        .iter()
        .zip(coeffs.n_accum.iter())
        .zip(coeffs.m_signs.iter())
        .zip(coeffs.q1_accum.iter())
        .zip(coeffs.q2_accum.iter())
        .zip(p1sin_arr.iter())
        .zip(p1_arr.iter())
        .for_each(|((((((m, n), sign), q1), q2), p1sin), p1)| {
            let mf = *m as f64;
            let nf = *n as f64;
            let signf = *sign as f64;

            unsafe {
                let c_mn = ((0.5 * (2 * n + 1) as f64)
                // TODO: Is using get_unchecked going to help here?
                * FACTORIAL[(n - m.abs()) as usize]
                    / FACTORIAL[(n + m.abs()) as usize])
                    .sqrt();
                let ejm_phi = c64::cis(mf * phi);
                let phi_comp = (ejm_phi * c_mn) / (nf * (nf + 1.0)).sqrt() * signf;
                let j_power_n = J_POWER_TABLE.get_unchecked((*n % 4) as usize);
                let e_theta_mn = j_power_n * ((p1sin * (mf.abs() * q2 * u - mf * q1)) + q2 * p1);
                let j_power_np1 = J_POWER_TABLE.get_unchecked(((*n + 1) % 4) as usize);
                let e_phi_mn = j_power_np1 * ((p1sin * (mf * q2 - mf.abs() * q1 * u)) - q1 * p1);
                sigma_p += phi_comp * e_phi_mn;
                sigma_t += phi_comp * e_theta_mn;
            }
        });

    // The C++ code currently doesn't distinguish between the polarisations.
    (sigma_t, -sigma_p)
}

/// Actually calculate a Jones matrix. All other "calc" functions use this
/// function.
fn calc_jones_direct(
    az_rad: f64,
    za_rad: f64,
    coeffs: &BowtieCoefficients,
    norm_matrix: Option<Jones<f64>>,
) -> Jones<f64> {
    // Convert azimuth to FEKO phi (East through North).
    let phi_rad = FRAC_PI_2 - az_rad;
    let (p1sin_arr, p1_arr) = p1sin(coeffs.x.n_max.max(coeffs.y.n_max), za_rad);

    let (j00, j01) = calc_sigmas(phi_rad, za_rad, &coeffs.x, &p1sin_arr, &p1_arr);
    let (j10, j11) = calc_sigmas(phi_rad, za_rad, &coeffs.y, &p1sin_arr, &p1_arr);
    let mut jones = [j00, j01, j10, j11];
    if let Some(norm) = norm_matrix {
        jones.iter_mut().zip(norm.iter()).for_each(|(j, n)| *j /= n);
    }
    Jones::from(jones)
}

fn calc_zenith_norm_jones(coeffs: &BowtieCoefficients) -> Jones<f64> {
    // Azimuth angles at which Jones components are maximum.
    let max_phi = [0.0, -FRAC_PI_2, FRAC_PI_2, 0.0];
    let (p1sin_arr, p1_arr) = p1sin(coeffs.x.n_max.max(coeffs.y.n_max), 0.0);

    let (j00, _) = calc_sigmas(max_phi[0], 0.0, &coeffs.x, &p1sin_arr, &p1_arr);
    let (_, j01) = calc_sigmas(max_phi[1], 0.0, &coeffs.x, &p1sin_arr, &p1_arr);
    let (j10, _) = calc_sigmas(max_phi[2], 0.0, &coeffs.y, &p1sin_arr, &p1_arr);
    let (_, j11) = calc_sigmas(max_phi[3], 0.0, &coeffs.y, &p1sin_arr, &p1_arr);
    // C++ uses abs(c) here, where abs is the magnitude of the complex number
    // vector. The result of this function should be a complex Jones matrix,
    // but, confusingly, the returned "Jones matrix" is all real in the C++.
    // This less ambiguous in Rust.
    let abs = |c: c64| c64::new(c.norm(), 0.0);
    Jones::from([abs(j00), abs(j01), abs(j10), abs(j11)])
}

/// Ensure that any delays of 32 have an amplitude (dipole gain) of 0. The
/// results are bad otherwise! Also ensure that we have 32 dipole gains (amps)
/// here.
fn fix_amps(amps: &[f64], delays: &[u32; 16]) -> [f64; 32] {
    let mut full_amps: [f64; 32] = [1.0; 32];
    full_amps
        .iter_mut()
        .zip(amps.iter().cycle())
        .zip(delays.iter().cycle())
        .for_each(|((out_amp, &in_amp), &delay)| {
            if delay == 32 {
                *out_amp = 0.0;
            } else {
                *out_amp = in_amp;
            }
        });
    full_amps
}

/// Apply the parallactic angle correction to a beam-response Jones matrix (when
/// also given its corresponding direction). Also re-arrange the Jones matrix if
/// we need to; when `iau_order` is `true`, then the beam response is [NS-NS
/// NS-EW EW-NS EW-EW](). Otherwise it is [EW-EW EW-NS NS-EW NS-NS]();
///
/// See for how/why this is done:
/// <https://github.com/MWATelescope/mwa_hyperbeam/blob/main/fee_pols.pdf>
fn apply_parallactic_correction(
    az_rad: f64,
    za_rad: f64,
    latitude_rad: f64,
    iau_order: bool,
    jones: &mut Jones<f64>,
) {
    // Get the parallactic-angle and find its sine and cosine.
    let para_angle = AzEl::from_radians(az_rad, FRAC_PI_2 - za_rad)
        .to_hadec(latitude_rad)
        .get_parallactic_angle(latitude_rad);
    let (s_rot, c_rot) = para_angle.sin_cos();
    *jones = if iau_order {
        Jones::from([
            jones[2] * -c_rot + jones[3] * s_rot,
            jones[2] * -s_rot + jones[3] * -c_rot,
            jones[0] * -c_rot + jones[1] * s_rot,
            jones[0] * -s_rot + jones[1] * -c_rot,
        ])
    } else {
        Jones::from([
            jones[0] * -s_rot + jones[1] * -c_rot,
            jones[0] * -c_rot + jones[1] * s_rot,
            jones[2] * -s_rot + jones[3] * -c_rot,
            jones[2] * -c_rot + jones[3] * s_rot,
        ])
    };
}
