// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

/*!
Code to implement the MWA Fully Embedded Element (FEE) beam, a.k.a. "the 2016
beam".
 */

pub mod error;
mod types;

use std::f64::consts::{FRAC_PI_2, TAU};
use std::sync::Mutex;

use ndarray::Array2;
use rayon::prelude::*;

use crate::constants::*;
use crate::factorial::FACTORIAL;
use crate::legendre::p1sin;
use crate::types::*;
pub use error::{FEEBeamError, InitFEEBeamError};
use types::*;

/// The main struct to be used for calculating FEE pointings.
pub struct FEEBeam {
    /// The `hdf5::File` struct associated with the opened HDF5 file. It is
    /// behind a `Mutex` to prevent parallel usage of the file.
    hdf5_file: Mutex<hdf5::File>,
    /// An ascendingly-sorted vector of frequencies available in the HDF5 file.
    pub freqs: Vec<u32>,
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
    /// Given the path to the FEE beam file, create a new `FEEBeam` struct.
    pub fn new<T: AsRef<std::path::Path>>(file: T) -> Result<Self, InitFEEBeamError> {
        // so that libhdf5 doesn't print errors to stdout
        let _e = hdf5::silence_errors();

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
        if biggest_dip_index.is_none() {
            return Err(InitFEEBeamError::NoDipoles);
        }
        if freqs.is_empty() {
            return Err(InitFEEBeamError::NoFreqs);
        }
        if biggest_dip_index.unwrap() != NUM_DIPOLES {
            return Err(InitFEEBeamError::DipoleCountMismatch {
                expected: NUM_DIPOLES,
                got: biggest_dip_index.unwrap(),
            });
        }

        freqs.sort_unstable();

        let modes = h5.dataset("modes")?.read_2d()?;

        Ok(Self {
            hdf5_file: Mutex::new(h5),
            freqs,
            modes,
            coeff_cache: CoeffCache::default(),
            norm_cache: NormCache::default(),
        })
    }

    /// Create a new `FEEBeam` struct from the `MWA_BEAM_FILE` environment
    /// variable.
    pub fn new_from_env() -> Result<Self, InitFEEBeamError> {
        match std::env::var("MWA_BEAM_FILE") {
            Ok(f) => Self::new(f),
            Err(e) => Err(InitFEEBeamError::MwaBeamFileVarError(e)),
        }
    }

    /// Given a frequency in Hz, find the closest frequency that is defined in
    /// the HDF5 file.
    pub fn find_closest_freq(&self, desired_freq_hz: u32) -> u32 {
        let mut best_freq_diff: Option<i64> = None;
        let mut best_index: Option<usize> = None;
        for (i, &freq) in self.freqs.iter().enumerate() {
            let this_diff = (desired_freq_hz as i64 - freq as i64).abs();
            match best_freq_diff {
                None => {
                    best_freq_diff = Some(this_diff);
                    best_index = Some(i);
                }
                Some(best) => {
                    if this_diff < best {
                        best_freq_diff = Some(this_diff);
                        best_index = Some(i);
                    } else {
                        // Because the frequencies are always ascendingly
                        // sorted, if the frequency difference is getting
                        // bigger, we can break early.
                        break;
                    }
                }
            }
        }

        // TODO: Error handling.
        self.freqs[best_index.unwrap()]
    }

    /// Given a key, get a dataset from the HDF5 file.
    ///
    /// This function is expected to only receive keys like X16_51200000
    fn get_dataset(&self, key: &str) -> Result<Array2<f64>, FEEBeamError> {
        let h5 = self.hdf5_file.lock().unwrap();
        let data = h5.dataset(key)?.read_2d()?;
        Ok(data)
    }

    /// Check that `DipoleCoefficients` are cached for the input parameters. If
    /// they aren't, populate the cache. The returned hash is used to access the
    /// populated cache.
    ///
    /// This function is intended to be used every time the cache is to be
    /// accessed. By ensuring that the right coefficients are available at the
    /// end of this function, the caller can then directly access the cache. The
    /// only way to make Rust return the coefficients would be by keeping the
    /// whole cache locked, which ruins concurrent performance.
    ///
    /// Note that specified frequencies are "rounded" to frequencies that are
    /// defined the HDF5 file.
    fn populate_modes(
        &self,
        desired_freq: u32,
        delays: &[u32],
        amps: &[f64],
    ) -> Result<CacheHash, FEEBeamError> {
        let freq = self.find_closest_freq(desired_freq);

        // Are the input settings already cached? Hash them to check.
        let hash = CacheHash::new(freq, delays, amps);

        // If the cache for this hash exists, we can return the hash.
        if self.coeff_cache.contains_key(&hash) {
            return Ok(hash);
        }

        // If we hit this part of the code, the coefficients were not in the
        // cache.
        let modes = self.calc_modes(freq, delays, amps)?;
        self.coeff_cache.insert(hash.clone(), modes);
        Ok(hash)
    }

    /// Given the input parameters, calculate and return the X and Y
    /// coefficients ("modes"). As this function is relatively expensive, it
    /// should only be called by `Self::get_modes` to cache the outputs.
    fn calc_modes(
        &self,
        freq: u32,
        delays: &[u32],
        amps: &[f64],
    ) -> Result<DipoleCoefficients, FEEBeamError> {
        let x = self.calc_mode(freq, delays, amps, Pol::X)?;
        let y = self.calc_mode(freq, delays, amps, Pol::Y)?;
        Ok(DipoleCoefficients { x, y })
    }

    /// Given the input parameters, calculate and return the coefficients for a
    /// single polarisation (X or Y). This function should only be called by
    /// `Self::calc_modes`.
    fn calc_mode(
        &self,
        freq: u32,
        delays: &[u32],
        amps: &[f64],
        pol: Pol,
    ) -> Result<PolCoefficients, FEEBeamError> {
        let mut q1: Vec<c64> = vec![];
        let mut q2: Vec<c64> = vec![];
        let mut q1_accum: Vec<c64> = vec![c64::new(0.0, 0.0); self.modes.shape()[1]];
        let mut q2_accum: Vec<c64> = vec![c64::new(0.0, 0.0); self.modes.shape()[1]];
        let mut m_accum = vec![];
        let mut n_accum = vec![];
        // Biggest N coefficient.
        let mut n_max = 0;

        for (dipole_num, (&amp, &delay)) in amps.iter().zip(delays.iter()).enumerate() {
            // Get the relevant HDF5 data.
            let q_all: Array2<f64> = {
                let key = format!("{}{}_{}", pol, dipole_num + 1, freq);
                self.get_dataset(&key)?
            };
            let n_dip_coeffs: usize = q_all.shape()[1];

            // Complex excitation voltage.
            let v: c64 = {
                let phase = TAU * freq as f64 * (-(delay as f64)) * DELAY_STEP;
                let (s_phase, c_phase) = phase.sin_cos();
                let phase_factor = c64::new(c_phase, s_phase);
                amp * phase_factor
            };

            // Indices of S=1 coefficients.
            let mut s1_list: Vec<usize> = Vec::with_capacity(n_dip_coeffs / 2);
            // Indices of S=2 coefficients.
            let mut s2_list: Vec<usize> = Vec::with_capacity(n_dip_coeffs / 2);
            let mut ms1 = Vec::with_capacity(n_dip_coeffs / 2);
            let mut ns1 = Vec::with_capacity(n_dip_coeffs / 2);
            let mut ms2 = Vec::with_capacity(n_dip_coeffs / 2);
            let mut ns2 = Vec::with_capacity(n_dip_coeffs / 2);

            // ???
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
                m_accum = ms1;
                n_accum = ns1;
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
                let (s_arg, c_arg) = arg.sin_cos();
                let q1_val = s10_coeff * c64::new(c_arg, s_arg);
                q1.push(q1_val);
                q1_accum[i] += q1_val * v;

                // Calculate Q2.
                let s2_idx = s2_list[i];
                let s20_coeff = q_all[[0, s2_idx]];
                let s21_coeff = q_all[[1, s2_idx]];
                let arg = s21_coeff.to_radians();
                let (s_arg, c_arg) = arg.sin_cos();
                let q2_val = s20_coeff * c64::new(c_arg, s_arg);
                q2.push(q2_val);
                q2_accum[i] += q2_val * v;
            }
        }

        let mut m_signs = Vec::with_capacity(m_accum.len());
        for m in &m_accum {
            let sign = if *m > 0 && *m % 2 != 0 { -1 } else { 1 };
            m_signs.push(sign)
        }

        Ok(PolCoefficients {
            q1_accum,
            q2_accum,
            m_accum,
            n_accum,
            m_signs,
            n_max: n_max as usize,
        })
    }

    fn populate_norm_jones(&self, desired_freq: u32) -> Result<u32, FEEBeamError> {
        let freq = self.find_closest_freq(desired_freq);

        // If the cache for this freq exists, we can return it.
        if self.norm_cache.contains_key(&freq) {
            return Ok(freq);
        }

        // If we hit this part of the code, the normalisation Jones matrix was
        // not in the cache.
        let hash = self.populate_modes(freq, &[0; 16], &[1.0; 16])?;
        let coeffs = self.coeff_cache.get(&hash).unwrap();
        let jones = calc_zenith_norm_jones(&coeffs);
        self.norm_cache.insert(freq, jones);

        Ok(freq)
    }

    /// Calculate the Jones matrix for a pointing.
    ///
    /// `delays` and `amps` apply to each dipole in a given MWA tile (in the M&C
    /// order), and *must* have 16 elements. `amps` being dipole gains (usually
    /// 1 or 0), not digital gains.
    pub fn calc_jones(
        &self,
        az_rad: f64,
        za_rad: f64,
        freq_hz: u32,
        delays: &[u32],
        amps: &[f64],
        norm_to_zenith: bool,
    ) -> Result<Jones, FEEBeamError> {
        debug_assert_eq!(delays.len(), 16);
        debug_assert_eq!(amps.len(), 16);

        // Ensure that any delays of 32 have an amplitude (dipole gain) of 0.
        // The results are bad otherwise!
        let amps = delays
            .iter()
            .zip(amps.iter())
            .map(|(&d, &a)| if d == 32 { 0.0 } else { a })
            .collect::<Vec<_>>();

        // Ensure the dipole coefficients for the provided parameters exist.
        let hash = self.populate_modes(freq_hz, delays, &amps)?;

        // If we're normalising the beam, get the normalisation frequency here.
        let norm_freq = if norm_to_zenith {
            Some(self.populate_norm_jones(freq_hz)?)
        } else {
            None
        };

        let coeffs = self.coeff_cache.get(&hash).unwrap();
        let norm_jones = norm_freq.and_then(|f| self.norm_cache.get(&f));

        let jones = calc_jones_direct(az_rad, za_rad, &coeffs, norm_jones.as_deref());
        Ok(jones)
    }

    /// Calculate the Jones matrices for many pointings.
    ///
    /// This is basically a wrapper around `calc_jones`; this function
    /// calculates the Jones matrices in parallel. `delays` and `amps` *must*
    /// have 16 elements.
    pub fn calc_jones_array(
        &mut self,
        az_rad: &[f64],
        za_rad: &[f64],
        freq_hz: u32,
        delays: &[u32],
        amps: &[f64],
        norm_to_zenith: bool,
    ) -> Result<Array2<c64>, FEEBeamError> {
        debug_assert_eq!(delays.len(), 16);
        debug_assert_eq!(amps.len(), 16);

        // Ensure that any delays of 32 have an amplitude (dipole gain) of 0.
        // The results are bad otherwise!
        let amps = delays
            .iter()
            .zip(amps.iter())
            .map(|(&d, &a)| if d == 32 { 0.0 } else { a })
            .collect::<Vec<_>>();

        // Ensure the dipole coefficients for the provided parameters exist.
        let hash = self.populate_modes(freq_hz, delays, &amps)?;

        // If we're normalising the beam, get the normalisation Jones matrix
        // here.
        let norm_freq = if norm_to_zenith {
            Some(self.populate_norm_jones(freq_hz)?)
        } else {
            None
        };

        let coeffs = self.coeff_cache.get(&hash).unwrap();
        let norm = norm_freq.and_then(|f| self.norm_cache.get(&f));

        let mut out = Vec::with_capacity(az_rad.len());
        az_rad
            .par_iter()
            .zip(za_rad.par_iter())
            .map(|(&az, &za)| calc_jones_direct(az, za, &coeffs, norm.as_deref()))
            .collect_into_vec(&mut out);
        Ok(Array2::from(out))
    }

    /// Empty the cached dipole coefficients and normalisation Jones matrices to
    /// recover memory.
    pub fn empty_cache(&self) {
        self.coeff_cache.clear();
        self.norm_cache.clear();
    }
}

/// Calculate the Jones matrix for a pointing for a single dipole polarisation.
fn calc_sigmas(phi: f64, theta: f64, coeffs: &PolCoefficients) -> (c64, c64) {
    let u = theta.cos();
    let (p1sin_arr, p1_arr) = p1sin(coeffs.n_max, theta);

    // TODO: Check that the sizes of N_accum and M_accum agree. This check
    // should actually be in the PolCoefficients generation.

    let mut sigma_p = c64::new(0.0, 0.0);
    let mut sigma_t = c64::new(0.0, 0.0);
    // Use an iterator for maximum performance.
    for ((((((m, n), sign), q1), q2), p1sin), p1) in coeffs
        .m_accum
        .iter()
        .zip(coeffs.n_accum.iter())
        .zip(coeffs.m_signs.iter())
        .zip(coeffs.q1_accum.iter())
        .zip(coeffs.q2_accum.iter())
        .zip(p1sin_arr.iter())
        .zip(p1_arr.iter())
    {
        let mf = *m as f64;
        let nf = *n as f64;
        let signf = *sign as f64;

        let c_mn = ((0.5 * (2 * n + 1) as f64) * FACTORIAL[(n - m.abs()) as usize]
            / FACTORIAL[(n + m.abs()) as usize])
            .sqrt();
        let (s_m_phi, c_m_phi) = (mf * phi).sin_cos();
        let ejm_phi = c64::new(c_m_phi, s_m_phi);
        let phi_comp = (ejm_phi * c_mn) / (nf * (nf + 1.0)).sqrt() * signf;
        let j_power_n = J_POWER_TABLE[(*n % 4) as usize];
        let e_theta_mn = j_power_n * ((p1sin * (mf.abs() * q2 * u - mf * q1)) + q2 * p1);
        let j_power_np1 = J_POWER_TABLE[((*n + 1) % 4) as usize];
        let e_phi_mn = j_power_np1 * ((p1sin * (mf * q2 - mf.abs() * q1 * u)) - q1 * p1);
        sigma_p += phi_comp * e_phi_mn;
        sigma_t += phi_comp * e_theta_mn;
    }

    // The C++ code currently doesn't distinguish between the polarisations.
    (sigma_t, -sigma_p)
}

/// Calculate the Jones matrix for a pointing for both dipole polarisations.
fn calc_jones_direct(
    az_rad: f64,
    za_rad: f64,
    coeffs: &DipoleCoefficients,
    norm_matrix: Option<&Jones>,
) -> Jones {
    // Convert azimuth to FEKO phi (East through North).
    let phi_rad = FRAC_PI_2 - az_rad;
    let (mut j00, mut j01) = calc_sigmas(phi_rad, za_rad, &coeffs.x);
    let (mut j10, mut j11) = calc_sigmas(phi_rad, za_rad, &coeffs.y);
    if let Some(norm) = norm_matrix {
        j00 /= norm[0];
        j01 /= norm[1];
        j10 /= norm[2];
        j11 /= norm[3];
    }
    [j00, j01, j10, j11]
}

fn calc_zenith_norm_jones(coeffs: &DipoleCoefficients) -> Jones {
    // Azimuth angles at which Jones components are maximum.
    let max_phi = [0.0, -FRAC_PI_2, FRAC_PI_2, 0.0];
    let (j00, _) = calc_sigmas(max_phi[0], 0.0, &coeffs.x);
    let (_, j01) = calc_sigmas(max_phi[1], 0.0, &coeffs.x);
    let (j10, _) = calc_sigmas(max_phi[2], 0.0, &coeffs.y);
    let (_, j11) = calc_sigmas(max_phi[3], 0.0, &coeffs.y);
    // C++ uses abs(c) here, where abs is the magnitude of the complex number
    // vector. Confusingly, it looks like the returned Jones matrix is all real,
    // but it should be complex. This is more explicit in Rust.
    let abs = |c: c64| c64::new(c.norm(), 0.0);
    [abs(j00), abs(j01), abs(j10), abs(j11)]
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::*;
    use ndarray::prelude::*;
    use serial_test::serial;

    #[test]
    #[serial]
    fn new() {
        let beam = FEEBeam::new("mwa_full_embedded_element_pattern.h5");
        assert!(beam.is_ok());
    }

    #[test]
    #[serial]
    fn new_from_env() {
        std::env::set_var("MWA_BEAM_FILE", "mwa_full_embedded_element_pattern.h5");
        let beam = FEEBeam::new_from_env();
        assert!(beam.is_ok());
    }

    #[test]
    #[serial]
    fn test_find_nearest_freq() {
        let beam = FEEBeam::new("mwa_full_embedded_element_pattern.h5").unwrap();
        // Dancing around an available freq.
        assert_eq!(beam.find_closest_freq(51199999), 51200000);
        assert_eq!(beam.find_closest_freq(51200000), 51200000);
        assert_eq!(beam.find_closest_freq(51200001), 51200000);
        // On the precipice of choosing between two freqs: 51200000 and
        // 52480000. When searching with 51840000, we will get the same
        // difference in frequency for both nearby, defined freqs. Because we
        // compare with "less than", the first freq. will be selected. This
        // should be consistent with the C++ code.
        assert_eq!(beam.find_closest_freq(51840000), 51200000);
        assert_eq!(beam.find_closest_freq(51840001), 52480000);
    }

    #[test]
    #[serial]
    /// Check that we can open the dataset "X16_51200000".
    fn test_get_dataset() {
        let beam = FEEBeam::new("mwa_full_embedded_element_pattern.h5").unwrap();
        assert!(beam.get_dataset("X16_51200000").is_ok());
    }

    #[test]
    #[serial]
    fn test_get_modes() {
        let beam = FEEBeam::new("mwa_full_embedded_element_pattern.h5").unwrap();
        let hash = match beam.populate_modes(51200000, &[0; 16], &[1.0; 16]) {
            Ok(h) => h,
            Err(e) => panic!("{}", e),
        };
        let coeffs = beam.coeff_cache.get(&hash).unwrap();

        // Values taken from the C++ code.
        // m_accum and n_accum are floats in the C++ code, but these appear to
        // always be small integers. I've converted the C++ output to ints here.
        let x_m_expected = array![
            -1, 0, 1, -2, -1, 0, 1, 2, -3, -2, -1, 0, 1, 2, 3, -4, -3, -2, -1, 0, 1, 2, 3, 4, -5,
            -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, -7, -6,
            -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3,
            4, 5, 6, 7, 8, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, -10,
            -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, -11, -10, -9, -8,
            -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, -12, -11, -10, -9,
            -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, -13, -12,
            -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
            13, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6,
            7, 8, 9, 10, 11, 12, 13, 14, -15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3,
            -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, -16, -15, -14, -13, -12,
            -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
            13, 14, 15, 16
        ];
        let y_m_expected = array![
            -1, 0, 1, -2, -1, 0, 1, 2, -3, -2, -1, 0, 1, 2, 3, -4, -3, -2, -1, 0, 1, 2, 3, 4, -5,
            -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, -7, -6,
            -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3,
            4, 5, 6, 7, 8, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, -10,
            -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, -11, -10, -9, -8,
            -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, -12, -11, -10, -9,
            -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, -13, -12,
            -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
            13, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6,
            7, 8, 9, 10, 11, 12, 13, 14, -15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3,
            -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, -16, -15, -14, -13, -12,
            -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
            13, 14, 15, 16
        ];
        let x_n_expected = array![
            1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5,
            5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
            7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9,
            9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
            10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11,
            11, 11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
            12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13,
            13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14,
            14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
            14, 14, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
            15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
            16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16
        ];
        let y_n_expected = array![
            1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5,
            5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
            7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9,
            9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
            10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11,
            11, 11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
            12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13,
            13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14,
            14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
            14, 14, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
            15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
            16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16
        ];

        let x_q1_expected_first = array![
            c64::new(-0.024744, 0.009424),
            c64::new(0.000000, 0.000000),
            c64::new(-0.024734, 0.009348),
            c64::new(0.000000, -0.000000),
            c64::new(0.005766, 0.015469),
        ];
        let x_q1_expected_last = array![
            c64::new(0.0, 0.0),
            c64::new(0.0, 0.0),
            c64::new(0.0, 0.0),
            c64::new(0.0, 0.0),
            c64::new(0.0, 0.0),
        ];

        let x_q2_expected_first = array![
            c64::new(-0.026122, 0.009724),
            c64::new(-0.000000, -0.000000),
            c64::new(0.026116, -0.009643),
            c64::new(0.000000, -0.000000),
            c64::new(0.006586, 0.018925),
        ];
        let x_q2_expected_last = array![
            c64::new(0.0, 0.0),
            c64::new(0.0, 0.0),
            c64::new(0.0, 0.0),
            c64::new(0.0, 0.0),
            c64::new(0.0, 0.0),
        ];

        let y_q1_expected_first = array![
            c64::new(-0.009398, -0.024807),
            c64::new(0.000000, -0.000000),
            c64::new(0.009473, 0.024817),
            c64::new(0.000000, 0.000000),
            c64::new(-0.015501, 0.005755),
        ];
        let y_q1_expected_last = array![
            c64::new(0.0, 0.0),
            c64::new(0.0, 0.0),
            c64::new(0.0, 0.0),
            c64::new(0.0, 0.0),
            c64::new(0.0, 0.0),
        ];

        let y_q2_expected_first = array![
            c64::new(-0.009692, -0.026191),
            c64::new(0.000000, 0.000000),
            c64::new(-0.009773, -0.026196),
            c64::new(0.000000, 0.000000),
            c64::new(-0.018968, 0.006566),
        ];
        let y_q2_expected_last = array![
            c64::new(0.0, 0.0),
            c64::new(0.0, 0.0),
            c64::new(0.0, 0.0),
            c64::new(0.0, 0.0),
            c64::new(0.0, 0.0),
        ];

        assert_eq!(Array1::from(coeffs.x.m_accum.clone()), x_m_expected);
        assert_eq!(Array1::from(coeffs.y.m_accum.clone()), y_m_expected);
        assert_eq!(Array1::from(coeffs.x.n_accum.clone()), x_n_expected);
        assert_eq!(Array1::from(coeffs.y.n_accum.clone()), y_n_expected);

        let x_q1_accum_arr = Array1::from(coeffs.x.q1_accum.clone());
        assert_abs_diff_eq!(
            x_q1_accum_arr.slice(s![0..5]),
            x_q1_expected_first,
            epsilon = 1e-6
        );
        assert_abs_diff_eq!(
            x_q1_accum_arr.slice(s![-5..]),
            x_q1_expected_last,
            epsilon = 1e-6
        );

        let x_q2_accum_arr = Array1::from(coeffs.x.q2_accum.clone());
        assert_abs_diff_eq!(
            x_q2_accum_arr.slice(s![0..5]),
            x_q2_expected_first,
            epsilon = 1e-6
        );
        assert_abs_diff_eq!(
            x_q2_accum_arr.slice(s![-5..]),
            x_q2_expected_last,
            epsilon = 1e-6
        );

        let y_q1_accum_arr = Array1::from(coeffs.y.q1_accum.clone());
        assert_abs_diff_eq!(
            y_q1_accum_arr.slice(s![0..5]),
            y_q1_expected_first,
            epsilon = 1e-6
        );
        assert_abs_diff_eq!(
            y_q1_accum_arr.slice(s![-5..]),
            y_q1_expected_last,
            epsilon = 1e-6
        );

        let y_q2_accum_arr = Array1::from(coeffs.y.q2_accum.clone());
        assert_abs_diff_eq!(
            y_q2_accum_arr.slice(s![0..5]),
            y_q2_expected_first,
            epsilon = 1e-6
        );
        assert_abs_diff_eq!(
            y_q2_accum_arr.slice(s![-5..]),
            y_q2_expected_last,
            epsilon = 1e-6
        );
    }

    #[test]
    #[serial]
    fn test_get_modes2() {
        let beam = FEEBeam::new("mwa_full_embedded_element_pattern.h5").unwrap();
        let hash = match beam.populate_modes(
            51200000,
            &[3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0],
            &[
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0,
            ],
        ) {
            Ok(h) => h,
            Err(e) => panic!("{}", e),
        };
        let coeffs = beam.coeff_cache.get(&hash).unwrap();

        // Values taken from the C++ code.
        let x_m_expected = array![
            -1, 0, 1, -2, -1, 0, 1, 2, -3, -2, -1, 0, 1, 2, 3, -4, -3, -2, -1, 0, 1, 2, 3, 4, -5,
            -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, -7, -6,
            -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3,
            4, 5, 6, 7, 8, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, -10,
            -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, -11, -10, -9, -8,
            -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, -12, -11, -10, -9,
            -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, -13, -12,
            -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
            13, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6,
            7, 8, 9, 10, 11, 12, 13, 14, -15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3,
            -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, -16, -15, -14, -13, -12,
            -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
            13, 14, 15, 16
        ];
        let y_m_expected = array![
            -1, 0, 1, -2, -1, 0, 1, 2, -3, -2, -1, 0, 1, 2, 3, -4, -3, -2, -1, 0, 1, 2, 3, 4, -5,
            -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, -7, -6,
            -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3,
            4, 5, 6, 7, 8, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, -10,
            -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, -11, -10, -9, -8,
            -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, -12, -11, -10, -9,
            -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, -13, -12,
            -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
            13, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6,
            7, 8, 9, 10, 11, 12, 13, 14, -15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3,
            -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, -16, -15, -14, -13, -12,
            -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
            13, 14, 15, 16
        ];
        let x_n_expected = array![
            1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5,
            5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
            7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9,
            9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
            10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11,
            11, 11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
            12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13,
            13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14,
            14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
            14, 14, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
            15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
            16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16
        ];
        let y_n_expected = array![
            1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5,
            5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
            7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9,
            9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
            10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11,
            11, 11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
            12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13,
            13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14,
            14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
            14, 14, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
            15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
            16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16
        ];

        let x_q1_expected_first = array![
            c64::new(-0.020504, 0.013376),
            c64::new(-0.001349, 0.000842),
            c64::new(-0.020561, 0.013291),
            c64::new(0.001013, 0.001776),
            c64::new(0.008222, 0.012569),
        ];
        let x_q1_expected_last = array![
            c64::new(0.0, 0.0),
            c64::new(0.0, 0.0),
            c64::new(0.0, 0.0),
            c64::new(0.0, 0.0),
            c64::new(0.0, 0.0),
        ];

        let x_q2_expected_first = array![
            c64::new(-0.021903, 0.013940),
            c64::new(0.001295, -0.000767),
            c64::new(0.021802, -0.014047),
            c64::new(0.001070, 0.002039),
            c64::new(0.009688, 0.016040),
        ];
        let x_q2_expected_last = array![
            c64::new(0.0, 0.0),
            c64::new(0.0, 0.0),
            c64::new(0.0, 0.0),
            c64::new(0.0, 0.0),
            c64::new(0.0, 0.0),
        ];

        let y_q1_expected_first = array![
            c64::new(-0.013471, -0.020753),
            c64::new(0.001130, 0.002400),
            c64::new(0.013576, 0.020683),
            c64::new(-0.001751, 0.001023),
            c64::new(-0.013183, 0.008283),
        ];
        let y_q1_expected_last = array![
            c64::new(0.0, 0.0),
            c64::new(0.0, 0.0),
            c64::new(0.0, 0.0),
            c64::new(0.0, 0.0),
            c64::new(0.0, 0.0),
        ];

        let y_q2_expected_first = array![
            c64::new(-0.014001, -0.021763),
            c64::new(-0.000562, -0.000699),
            c64::new(-0.013927, -0.021840),
            c64::new(-0.002247, 0.001152),
            c64::new(-0.015716, 0.009685),
        ];
        let y_q2_expected_last = array![
            c64::new(0.0, 0.0),
            c64::new(0.0, 0.0),
            c64::new(0.0, 0.0),
            c64::new(0.0, 0.0),
            c64::new(0.0, 0.0),
        ];

        assert_eq!(Array1::from(coeffs.x.m_accum.clone()), x_m_expected);
        assert_eq!(Array1::from(coeffs.y.m_accum.clone()), y_m_expected);
        assert_eq!(Array1::from(coeffs.x.n_accum.clone()), x_n_expected);
        assert_eq!(Array1::from(coeffs.y.n_accum.clone()), y_n_expected);

        let x_q1_accum_arr = Array1::from(coeffs.x.q1_accum.clone());
        assert_abs_diff_eq!(
            x_q1_accum_arr.slice(s![0..5]),
            x_q1_expected_first,
            epsilon = 1e-6
        );
        assert_abs_diff_eq!(
            x_q1_accum_arr.slice(s![-5..]),
            x_q1_expected_last,
            epsilon = 1e-6
        );

        let x_q2_accum_arr = Array1::from(coeffs.x.q2_accum.clone());
        assert_abs_diff_eq!(
            x_q2_accum_arr.slice(s![0..5]),
            x_q2_expected_first,
            epsilon = 1e-6
        );
        assert_abs_diff_eq!(
            x_q2_accum_arr.slice(s![-5..]),
            x_q2_expected_last,
            epsilon = 1e-6
        );

        let y_q1_accum_arr = Array1::from(coeffs.y.q1_accum.clone());
        assert_abs_diff_eq!(
            y_q1_accum_arr.slice(s![0..5]),
            y_q1_expected_first,
            epsilon = 1e-6
        );
        assert_abs_diff_eq!(
            y_q1_accum_arr.slice(s![-5..]),
            y_q1_expected_last,
            epsilon = 1e-6
        );

        let y_q2_accum_arr = Array1::from(coeffs.y.q2_accum.clone());
        assert_abs_diff_eq!(
            y_q2_accum_arr.slice(s![0..5]),
            y_q2_expected_first,
            epsilon = 1e-6
        );
        assert_abs_diff_eq!(
            y_q2_accum_arr.slice(s![-5..]),
            y_q2_expected_last,
            epsilon = 1e-6
        );
    }

    #[test]
    #[serial]
    fn test_calc_jones() {
        let beam = FEEBeam::new("mwa_full_embedded_element_pattern.h5").unwrap();
        let jones = match beam.calc_jones(
            45.0_f64.to_radians(),
            10.0_f64.to_radians(),
            51200000,
            &[0; 16],
            &[1.0; 16],
            false,
        ) {
            Ok(j) => Array1::from(j.to_vec()),
            Err(e) => panic!("{}", e),
        };

        let expected = array![
            c64::new(0.036179, 0.103586),
            c64::new(0.036651, 0.105508),
            c64::new(0.036362, 0.103868),
            c64::new(-0.036836, -0.105791),
        ];
        assert_abs_diff_eq!(jones, expected, epsilon = 1e-6);
    }

    #[test]
    #[serial]
    fn test_calc_jones2() {
        let beam = FEEBeam::new("mwa_full_embedded_element_pattern.h5").unwrap();
        let jones = match beam.calc_jones(
            70.0_f64.to_radians(),
            10.0_f64.to_radians(),
            51200000,
            &[3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0],
            &[
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0,
            ],
            false,
        ) {
            Ok(j) => Array1::from(j.to_vec()),
            Err(e) => panic!("{}", e),
        };

        let expected = array![
            c64::new(0.068028, 0.111395),
            c64::new(0.025212, 0.041493),
            c64::new(0.024792, 0.040577),
            c64::new(-0.069501, -0.113706),
        ];
        assert_abs_diff_eq!(jones, expected, epsilon = 1e-6);
    }

    #[test]
    #[serial]
    fn test_calc_jones_norm() {
        let beam = FEEBeam::new("mwa_full_embedded_element_pattern.h5").unwrap();
        let jones = match beam.calc_jones(0.1_f64, 0.1_f64, 150000000, &[0; 16], &[1.0; 16], true) {
            Ok(j) => Array1::from(j.to_vec()),
            Err(e) => panic!("{}", e),
        };

        let expected = array![
            c64::new(0.0887949, 0.0220569),
            c64::new(0.891024, 0.2211),
            c64::new(0.887146, 0.216103),
            c64::new(-0.0896141, -0.021803),
        ];
        assert_abs_diff_eq!(jones, expected, epsilon = 1e-6);
    }

    #[test]
    #[serial]
    fn test_calc_jones_norm2() {
        let beam = FEEBeam::new("mwa_full_embedded_element_pattern.h5").unwrap();
        let jones = match beam.calc_jones(
            0.1_f64,
            0.1_f64,
            150000000,
            &[3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0],
            &[
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0,
            ],
            true,
        ) {
            Ok(j) => Array1::from(j.to_vec()),
            Err(e) => panic!("{}", e),
        };

        let expected = array![
            c64::new(0.0704266, -0.0251082),
            c64::new(0.705241, -0.254518),
            c64::new(0.697787, -0.257219),
            c64::new(-0.0711516, 0.0264293),
        ];
        assert_abs_diff_eq!(jones, expected, epsilon = 1e-6);
    }
}
