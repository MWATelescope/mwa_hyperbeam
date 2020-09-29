// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

/*!
Code to read the spherical harmonic coefficients from the supplied HDF5 file.
 */

use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};

use ndarray::Array2;
use rayon::prelude::*;

use crate::factorial::FACTORIAL;
use crate::legendre::p1sin;
use crate::*;

/// Coefficients for X and Y.
// TODO: Improve docs.
pub(crate) struct PolCoefficients {
    q1_accum: Vec<Complex64>,
    q2_accum: Vec<Complex64>,
    m_accum: Vec<i8>,
    n_accum: Vec<i8>,
    /// The sign of M coefficients (i.e. -1 or 1).
    m_signs: Vec<i8>,
    /// The biggest N coefficient.
    n_max: usize,
}

pub(crate) struct DipoleCoefficients {
    x: PolCoefficients,
    y: PolCoefficients,
}

/// `CoeffCache` is mostly just a `RwLock` around a `HashMap`. This allows
/// multiple concurrent readers with the ability to halt all reading when
/// writing.
///
/// A `CacheHash` is used as the key. This is a wrapper around Rust's own
/// hashing code so that we get something specific to FEE beam settings.
///
/// `Rc` allows us to access the data without copying it. Benchmarks suggest
/// that this is important; if `Rc` isn't used, the compiler really does do
/// memory copies instead of just handing out a pointer.
struct CoeffCache(RwLock<HashMap<CacheHash, Arc<DipoleCoefficients>>>);

/// `NormCache` is very similar to `CoeffCache`. It stores Jones matrices used
/// to normalise beam responses at various frequencies (i.e. frequency is the
/// key of the `HashMap`).
struct NormCache(RwLock<HashMap<u32, Arc<Jones>>>);

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
    modes: Array2<f64>,
    /// A cache of X and Y coefficients.
    coeff_cache: CoeffCache,
    /// A cache of normalisation Jones matrices.
    norm_cache: NormCache,
}

impl FEEBeam {
    /// Given the path to the FEE beam file, create a new `FEEBeam` struct.
    pub fn new<T: AsRef<std::path::Path>>(file: T) -> Result<Self, FEEBeamError> {
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
                        Err(_) => return Err(FEEBeamError::Parse(s.to_string())),
                    },
                    None => return Err(FEEBeamError::MissingDipole),
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
                    Err(_) => return Err(FEEBeamError::Parse(freq_str.to_string())),
                };
                freqs.push(freq);
            }
        }

        // Sanity checks.
        if biggest_dip_index.is_none() {
            return Err(FEEBeamError::NoDipoles);
        }
        if freqs.is_empty() {
            return Err(FEEBeamError::NoFreqs);
        }
        if biggest_dip_index.unwrap() != *NUM_DIPOLES {
            return Err(FEEBeamError::DipoleCountMismatch {
                expected: *NUM_DIPOLES,
                got: biggest_dip_index.unwrap(),
            });
        }

        freqs.sort_unstable();

        let modes = h5.dataset("modes")?.read_2d()?;

        Ok(Self {
            hdf5_file: Mutex::new(h5),
            freqs,
            modes,
            coeff_cache: CoeffCache(RwLock::new(HashMap::new())),
            norm_cache: NormCache(RwLock::new(HashMap::new())),
        })
    }

    /// Create a new `FEEBeam` struct from the MWA_BEAM_FILE environment
    /// variable.
    pub fn new_from_env() -> Result<Self, FEEBeamError> {
        match std::env::var("MWA_BEAM_FILE") {
            Ok(f) => Self::new(f),
            Err(e) => Err(FEEBeamError::MwaBeamFileVarError(e)),
        }
    }

    /// Given a frequency [Hz], find the closest frequency that is defined in
    /// the HDF5 file.
    pub fn find_closest_freq(&self, desired_freq: u32) -> u32 {
        let mut best_freq_diff: Option<i64> = None;
        let mut best_index: Option<usize> = None;
        for (i, &freq) in self.freqs.iter().enumerate() {
            let this_diff = (desired_freq as i64 - freq as i64).abs();
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
                        // Because the frequencies are always ascendingly sorted, if
                        // the frequency difference is getting worse, we can break
                        // early.
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

    /// Get the dipole coefficients for the provided parameters. Note that
    /// specified frequencies are "rounded" to frequencies that are defined the
    /// HDF5 file. The results of this function are cached; if the input
    /// parameters match previously supplied parameters, then the cache is
    /// utilised.
    fn get_modes(
        &mut self,
        desired_freq: u32,
        delays: &[u32],
        amps: &[f64],
    ) -> Result<Arc<DipoleCoefficients>, FEEBeamError> {
        let freq = self.find_closest_freq(desired_freq);
        // Are the input settings already cached? Hash them to check.
        let hash = CacheHash::new(freq, delays, amps);
        {
            let cache = &*self.coeff_cache.0.read().unwrap();
            // If the cache for this hash exists, we can return a copy.
            if let Some(c) = cache.get(&hash) {
                return Ok(Arc::clone(&c));
            }
        }
        // If we hit this part of the code, the coefficients were not in the
        // cache. Lock the cache, populate it, then return the coefficients we
        // just calculated.
        let modes = {
            let mut cache = self.coeff_cache.0.write().unwrap();
            let modes = Arc::new(self.calc_modes(freq, delays, amps)?);
            cache.insert(hash.clone(), modes);
            Arc::clone(&cache.get(&hash).unwrap())
        };
        Ok(modes)
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
        let mut q1: Vec<Complex64> = vec![];
        let mut q2: Vec<Complex64> = vec![];
        let mut q1_accum: Vec<Complex64> = vec![Complex64::new(0.0, 0.0); self.modes.shape()[1]];
        let mut q2_accum: Vec<Complex64> = vec![Complex64::new(0.0, 0.0); self.modes.shape()[1]];
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
            let v: Complex64 = {
                let phase = *D2PI * freq as f64 * (-(delay as f64)) * *DELAY_STEP;
                let (s_phase, c_phase) = phase.sin_cos();
                let phase_factor = Complex64::new(c_phase, s_phase);
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
                // TODO: Convert the modes before getting here.
                let mode_type = self.modes[[0, i]];
                let mode_m = self.modes[[1, i]] as i8;
                let mode_n = self.modes[[2, i]] as i8;

                if mode_type <= 1.0 {
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
                let q1_val = s10_coeff * Complex64::new(c_arg, s_arg);
                q1.push(q1_val);
                q1_accum[i] += q1_val * v;

                // Calculate Q2.
                let s2_idx = s2_list[i];
                let s20_coeff = q_all[[0, s2_idx]];
                let s21_coeff = q_all[[1, s2_idx]];
                let arg = s21_coeff.to_radians();
                let (s_arg, c_arg) = arg.sin_cos();
                let q2_val = s20_coeff * Complex64::new(c_arg, s_arg);
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

    fn get_norm_jones(&mut self, desired_freq: u32) -> Result<Arc<Jones>, FEEBeamError> {
        let freq = self.find_closest_freq(desired_freq);
        {
            let cache = &*self.norm_cache.0.read().unwrap();
            // If the cache for this freq exists, we can return a copy.
            if let Some(c) = cache.get(&freq) {
                return Ok(Arc::clone(&c));
            }
        }
        // If we hit this part of the code, the normalisation Jones matrix was
        // not in the cache. Lock the cache, populate it, then return the
        // matrix we just calculated.
        let coeffs = self.get_modes(freq, &[0; 16], &[1.0; 16])?;
        let mut cache = self.norm_cache.0.write().unwrap();
        let jones = Arc::new(calc_zenith_norm_jones(&coeffs));
        cache.insert(freq, jones);
        Ok(Arc::clone(&cache.get(&freq).unwrap()))
    }

    /// Calculate the Jones matrix for a pointing.
    ///
    /// `delays` and `amps` *must* have 16 elements.
    pub fn calc_jones(
        &mut self,
        az_rad: f64,
        za_rad: f64,
        freq_hz: u32,
        delays: &[u32],
        amps: &[f64],
        norm_to_zenith: bool,
    ) -> Result<Jones, FEEBeamError> {
        debug_assert_eq!(delays.len(), 16);
        debug_assert_eq!(amps.len(), 16);

        let coeffs = self.get_modes(freq_hz, delays, amps)?;
        // If we're normalising the beam, get the normalisation Jones matrix
        // here.
        let norm = if norm_to_zenith {
            Some(*self.get_norm_jones(freq_hz)?)
        } else {
            None
        };
        let jones = calc_jones_direct(az_rad, za_rad, &coeffs, norm);
        Ok(jones)
    }

    /// Calculate the Jones matrix for many pointings.
    ///
    /// This is basically a wrapper around `calc_jones`. `delays` and `amps`
    /// *must* have 16 elements.
    pub fn calc_jones_array(
        &mut self,
        az_rad: &[f64],
        za_rad: &[f64],
        freq_hz: u32,
        delays: &[u32],
        amps: &[f64],
        norm_to_zenith: bool,
    ) -> Result<Array2<Complex64>, FEEBeamError> {
        debug_assert_eq!(delays.len(), 16);
        debug_assert_eq!(amps.len(), 16);

        let coeffs = self.get_modes(freq_hz, delays, amps)?;
        // If we're normalising the beam, get the normalisation Jones matrix
        // here.
        let norm = if norm_to_zenith {
            Some(*self.get_norm_jones(freq_hz)?)
        } else {
            None
        };

        let mut out = Vec::with_capacity(az_rad.len());
        az_rad
            .par_iter()
            .zip(za_rad.par_iter())
            .map(|(&az, &za)| calc_jones_direct(az, za, &coeffs, norm))
            .collect_into_vec(&mut out);
        Ok(Array2::from(out))
    }
}

/// Calculate the Jones matrix for a pointing for a single dipole polarisation.
fn calc_sigmas(phi: f64, theta: f64, coeffs: &PolCoefficients) -> (Complex64, Complex64) {
    let u = theta.cos();
    let (p1sin_arr, p1_arr) = p1sin(coeffs.n_max, theta);

    // TODO: Check that the sizes of N_accum and M_accum agree. This check
    // should actually be in the PolCoefficients generation.

    let mut sigma_p = Complex64::new(0.0, 0.0);
    let mut sigma_t = Complex64::new(0.0, 0.0);
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
        let ejm_phi = Complex64::new(c_m_phi, s_m_phi);
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
    norm_matrix: Option<Jones>,
) -> Jones {
    // Convert azimuth to FEKO phi (East through North).
    let phi_rad = *DPIBY2 - az_rad;
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
    let max_phi = [0.0, -*DPIBY2, *DPIBY2, 0.0];
    let (j00, _) = calc_sigmas(max_phi[0], 0.0, &coeffs.x);
    let (_, j01) = calc_sigmas(max_phi[1], 0.0, &coeffs.x);
    let (j10, _) = calc_sigmas(max_phi[2], 0.0, &coeffs.y);
    let (_, j11) = calc_sigmas(max_phi[3], 0.0, &coeffs.y);
    // C++ uses abs(c) here, where abs is the magnitude of the complex number
    // vector. Confusingly, it looks like the returned Jones matrix is all real,
    // but it should be complex. This is more explicit in Rust.
    let abs = |c: Complex64| Complex64::new(c.norm(), 0.0);
    [abs(j00), abs(j01), abs(j10), abs(j11)]
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::*;
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
        let mut beam = FEEBeam::new("mwa_full_embedded_element_pattern.h5").unwrap();
        let result = beam.get_modes(51200000, &[0; 16], &[1.0; 16]);
        assert!(result.is_ok());
        let coeffs = result.unwrap();
        // Values taken from the C++ code.
        // m_accum and n_accum are floats in the C++ code, but these appear to
        // always be small integers. I've converted the C++ output to ints here.
        let x_m_expected = vec![-1, 0, 1, -2, -1, 0, 1, 2, -3, -2, -1, 0, 1, 2, 3];
        let y_m_expected = vec![-1, 0, 1, -2, -1, 0, 1, 2, -3, -2, -1, 0, 1, 2, 3];
        let x_n_expected = vec![1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3];
        let y_n_expected = vec![1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3];
        let x_q1_expected = vec![
            Complex64::new(-0.024744, 0.009424),
            Complex64::new(0.000000, 0.000000),
            Complex64::new(-0.024734, 0.009348),
            Complex64::new(0.000000, -0.000000),
            Complex64::new(0.005766, 0.015469),
        ];
        let x_q2_expected = vec![
            Complex64::new(-0.026122, 0.009724),
            Complex64::new(-0.000000, -0.000000),
            Complex64::new(0.026116, -0.009643),
            Complex64::new(0.000000, -0.000000),
            Complex64::new(0.006586, 0.018925),
        ];
        let y_q1_expected = vec![
            Complex64::new(-0.009398, -0.024807),
            Complex64::new(0.000000, -0.000000),
            Complex64::new(0.009473, 0.024817),
            Complex64::new(0.000000, 0.000000),
            Complex64::new(-0.015501, 0.005755),
        ];
        let y_q2_expected = vec![
            Complex64::new(-0.009692, -0.026191),
            Complex64::new(0.000000, 0.000000),
            Complex64::new(-0.009773, -0.026196),
            Complex64::new(0.000000, 0.000000),
            Complex64::new(-0.018968, 0.006566),
        ];

        for (&r, e) in coeffs.x.m_accum.iter().zip(x_m_expected) {
            assert_eq!(r, e);
        }
        for (&r, e) in coeffs.y.m_accum.iter().zip(y_m_expected) {
            assert_eq!(r, e);
        }
        for (&r, e) in coeffs.x.n_accum.iter().zip(x_n_expected) {
            assert_eq!(r, e);
        }
        for (&r, e) in coeffs.y.n_accum.iter().zip(y_n_expected) {
            assert_eq!(r, e);
        }
        for (r, e) in coeffs.x.q1_accum.iter().zip(x_q1_expected) {
            assert_abs_diff_eq!(r.re, e.re, epsilon = 1e-6);
            assert_abs_diff_eq!(r.im, e.im, epsilon = 1e-6);
        }
        for (r, e) in coeffs.x.q2_accum.iter().zip(x_q2_expected) {
            assert_abs_diff_eq!(r.re, e.re, epsilon = 1e-6);
            assert_abs_diff_eq!(r.im, e.im, epsilon = 1e-6);
        }
        for (r, e) in coeffs.y.q1_accum.iter().zip(y_q1_expected) {
            assert_abs_diff_eq!(r.re, e.re, epsilon = 1e-6);
            assert_abs_diff_eq!(r.im, e.im, epsilon = 1e-6);
        }
        for (r, e) in coeffs.y.q2_accum.iter().zip(y_q2_expected) {
            assert_abs_diff_eq!(r.re, e.re, epsilon = 1e-6);
            assert_abs_diff_eq!(r.im, e.im, epsilon = 1e-6);
        }
    }

    #[test]
    #[serial]
    fn test_get_modes2() {
        let mut beam = FEEBeam::new("mwa_full_embedded_element_pattern.h5").unwrap();
        let result = beam.get_modes(
            51200000,
            &[3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0],
            &[
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0,
            ],
        );
        assert!(result.is_ok());
        let coeffs = result.unwrap();

        // Values taken from the C++ code.
        let x_m_expected = vec![-1, 0, 1, -2, -1, 0, 1, 2, -3, -2, -1, 0, 1, 2, 3];
        let y_m_expected = vec![-1, 0, 1, -2, -1, 0, 1, 2, -3, -2, -1, 0, 1, 2, 3];
        let x_n_expected = vec![1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3];
        let y_n_expected = vec![1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3];
        let x_q1_expected = vec![
            Complex64::new(-0.020504, 0.013376),
            Complex64::new(-0.001349, 0.000842),
            Complex64::new(-0.020561, 0.013291),
            Complex64::new(0.001013, 0.001776),
            Complex64::new(0.008222, 0.012569),
        ];
        let x_q2_expected = vec![
            Complex64::new(-0.021903, 0.013940),
            Complex64::new(0.001295, -0.000767),
            Complex64::new(0.021802, -0.014047),
            Complex64::new(0.001070, 0.002039),
            Complex64::new(0.009688, 0.016040),
        ];
        let y_q1_expected = vec![
            Complex64::new(-0.013471, -0.020753),
            Complex64::new(0.001130, 0.002400),
            Complex64::new(0.013576, 0.020683),
            Complex64::new(-0.001751, 0.001023),
            Complex64::new(-0.013183, 0.008283),
        ];
        let y_q2_expected = vec![
            Complex64::new(-0.014001, -0.021763),
            Complex64::new(-0.000562, -0.000699),
            Complex64::new(-0.013927, -0.021840),
            Complex64::new(-0.002247, 0.001152),
            Complex64::new(-0.015716, 0.009685),
        ];

        for (&r, e) in coeffs.x.m_accum.iter().zip(x_m_expected) {
            assert_eq!(r, e);
        }
        for (&r, e) in coeffs.y.m_accum.iter().zip(y_m_expected) {
            assert_eq!(r, e);
        }
        for (&r, e) in coeffs.x.n_accum.iter().zip(x_n_expected) {
            assert_eq!(r, e);
        }
        for (&r, e) in coeffs.y.n_accum.iter().zip(y_n_expected) {
            assert_eq!(r, e);
        }
        for (r, e) in coeffs.x.q1_accum.iter().zip(x_q1_expected) {
            assert_abs_diff_eq!(r.re, e.re, epsilon = 1e-6);
            assert_abs_diff_eq!(r.im, e.im, epsilon = 1e-6);
        }
        for (r, e) in coeffs.x.q2_accum.iter().zip(x_q2_expected) {
            assert_abs_diff_eq!(r.re, e.re, epsilon = 1e-6);
            assert_abs_diff_eq!(r.im, e.im, epsilon = 1e-6);
        }
        for (r, e) in coeffs.y.q1_accum.iter().zip(y_q1_expected) {
            assert_abs_diff_eq!(r.re, e.re, epsilon = 1e-6);
            assert_abs_diff_eq!(r.im, e.im, epsilon = 1e-6);
        }
        for (r, e) in coeffs.y.q2_accum.iter().zip(y_q2_expected) {
            assert_abs_diff_eq!(r.re, e.re, epsilon = 1e-6);
            assert_abs_diff_eq!(r.im, e.im, epsilon = 1e-6);
        }
    }

    #[test]
    #[serial]
    fn test_calc_jones() {
        let mut beam = FEEBeam::new("mwa_full_embedded_element_pattern.h5").unwrap();
        let result = beam.calc_jones(
            45.0_f64.to_radians(),
            10.0_f64.to_radians(),
            51200000,
            &[0; 16],
            &[1.0; 16],
            false,
        );
        assert!(result.is_ok());
        let jones = result.unwrap();

        let expected = [
            Complex64::new(0.036179, 0.103586),
            Complex64::new(0.036651, 0.105508),
            Complex64::new(0.036362, 0.103868),
            Complex64::new(-0.036836, -0.105791),
        ];
        for (&r, e) in jones.iter().zip(&expected) {
            assert_abs_diff_eq!(r.re, e.re, epsilon = 1e-6);
            assert_abs_diff_eq!(r.im, e.im, epsilon = 1e-6);
        }
    }

    #[test]
    #[serial]
    fn test_calc_jones2() {
        let mut beam = FEEBeam::new("mwa_full_embedded_element_pattern.h5").unwrap();
        let result = beam.calc_jones(
            70.0_f64.to_radians(),
            10.0_f64.to_radians(),
            51200000,
            &[3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0],
            &[
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0,
            ],
            false,
        );
        assert!(result.is_ok());
        let jones = result.unwrap();
        let expected = [
            Complex64::new(0.068028, 0.111395),
            Complex64::new(0.025212, 0.041493),
            Complex64::new(0.024792, 0.040577),
            Complex64::new(-0.069501, -0.113706),
        ];
        for (&r, e) in jones.iter().zip(&expected) {
            assert_abs_diff_eq!(r.re, e.re, epsilon = 1e-6);
            assert_abs_diff_eq!(r.im, e.im, epsilon = 1e-6);
        }
    }

    #[test]
    #[serial]
    fn test_calc_jones_norm() {
        let mut beam = FEEBeam::new("mwa_full_embedded_element_pattern.h5").unwrap();
        let result = beam.calc_jones(0.1_f64, 0.1_f64, 150000000, &[0; 16], &[1.0; 16], true);
        assert!(result.is_ok());
        let jones = result.unwrap();
        let expected = [
            Complex64::new(0.0887949, 0.0220569),
            Complex64::new(0.891024, 0.2211),
            Complex64::new(0.887146, 0.216103),
            Complex64::new(-0.0896141, -0.021803),
        ];
        for (&r, e) in jones.iter().zip(&expected) {
            assert_abs_diff_eq!(r.re, e.re, epsilon = 1e-6);
            assert_abs_diff_eq!(r.im, e.im, epsilon = 1e-6);
        }
    }

    #[test]
    #[serial]
    fn test_calc_jones_norm2() {
        let mut beam = FEEBeam::new("mwa_full_embedded_element_pattern.h5").unwrap();
        let result = beam.calc_jones(
            0.1_f64,
            0.1_f64,
            150000000,
            &[3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0],
            &[
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0,
            ],
            true,
        );
        assert!(result.is_ok());
        let jones = result.unwrap();
        let expected = [
            Complex64::new(0.0704266, -0.0251082),
            Complex64::new(0.705241, -0.254518),
            Complex64::new(0.697787, -0.257219),
            Complex64::new(-0.0711516, 0.0264293),
        ];
        for (&r, e) in jones.iter().zip(&expected) {
            assert_abs_diff_eq!(r.re, e.re, epsilon = 1e-6);
            assert_abs_diff_eq!(r.im, e.im, epsilon = 1e-6);
        }
    }
}
