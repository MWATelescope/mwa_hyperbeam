// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

/*!
Code to read the spherical harmonic coefficients from the supplied HDF5 file.
 */

use std::cmp::Ordering;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::{Mutex, RwLock};

use ndarray::Array2;

use crate::legendre::*;
use crate::*;

/// Coefficients for X and Y.
// TODO: Make the docs better.
pub(crate) struct PolCoefficients {
    q1_accum: Vec<Complex64>,
    q2_accum: Vec<Complex64>,
    m_accum: Vec<f64>,
    n_accum: Vec<f64>,
    /// precalculated m/abs(m) to make it once for all pointings
    m_signs: Vec<Sign>,
    /// maximum N coefficient for Y (=max(N_accum_X)) - to avoid relaculations
    n_max: f64,
    // /// coefficient under sumation in equation 3 for X pol.
    // cMN: Vec<f64>,
}

pub(crate) struct DipoleCoefficients {
    x: PolCoefficients,
    y: PolCoefficients,
}

/// `Cache` is mostly just a `RwLock` around a `HashMap`. This allows multiple
/// concurrent readers with the ability to halt all reading when writing.
///
/// A `CacheHash` is used as the key. This is a wrapper around Rust's own
/// hashing code so that we get something specific to FEE beam settings.
///
/// `Rc` allows us to access the data without copying it. Benchmarks suggest
/// that this is important; if `Rc` isn't used, the compiler really does do
/// memory copies instead of just handing out a pointer.
struct Cache(RwLock<HashMap<CacheHash, Rc<DipoleCoefficients>>>);

pub struct FEEBeam {
    /// The `hdf5::File` struct associated with the opened HDF5 file. It is
    /// behind a `Mutex` to prevent parallel usage of the file.
    hdf5_file: Mutex<hdf5::File>,
    /// An ascendingly-sorted vector of frequencies available in the HDF5 file.
    freqs: Vec<u32>,
    /// Values used in calculating coefficients for X and Y.
    /// Row 0: Type
    /// Row 1: M
    /// Row 2: N
    modes: Array2<f64>,
    /// A cache of X and Y coefficients.
    cache: Cache,
}

impl FEEBeam {
    pub fn new<T: AsRef<std::path::Path>>(file: T) -> Result<Self, FEEBeamError> {
        // so that libhdf5 doesn't print errors to stdout
        let _e = hdf5::silence_errors();

        let h5 = hdf5::File::open(file)?;
        // We want all of the available frequencies and the biggest antenna index.
        let mut freqs: Vec<u32> = vec![];
        let mut biggest_dip_index: Option<u8> = None;
        // Iterate over all of the h5 dataset names.
        for d in h5.member_names()? {
            if d.starts_with("X") {
                // This is the part between 'X' and '_';
                let dipole_index_str = d.strip_prefix("X").unwrap().split("_").next();
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
            cache: Cache(RwLock::new(HashMap::new())),
        })
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
        delays: &[u32; 16],
        amps: &[f64; 16],
    ) -> Result<Rc<DipoleCoefficients>, FEEBeamError> {
        let freq = self.find_closest_freq(desired_freq);
        // Are the input settings already cached? Hash them to check.
        let hash = CacheHash::new(freq, delays, amps);
        {
            let cache = &*self.cache.0.read().unwrap();
            match cache.get(&hash) {
                // If the cache for this hash exists, we can return a copy.
                Some(c) => return Ok(Rc::clone(&c)),
                // Some(c) => {
                //     eprintln!("Cache hit!");
                //     return Ok(c.clone());
                // }
                None => (),
            }
        }
        // If we hit this part of the code, the coefficients were not in the
        // cache. Lock the cache, populate it, then return the coefficients we
        // just calculated.
        let modes = {
            // eprintln!(
            //     "Have to calc coeffs for\n\tfreq: {}\n\tdelays: {:?}\n\tamps: {:?}",
            //     freq, delays, amps
            // );
            let mut cache = self.cache.0.write().unwrap();
            let modes = Rc::new(self.calc_modes(freq, delays, amps)?);
            cache.insert(hash.clone(), modes);
            Rc::clone(&cache.get(&hash).unwrap())
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
        let mut m_accum: Vec<f64> = vec![];
        let mut n_accum: Vec<f64> = vec![];
        // Biggest N coefficient.
        let mut n_max: f64 = 0.0;

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
            let mut ms1: Vec<f64> = Vec::with_capacity(n_dip_coeffs / 2);
            let mut ns1: Vec<f64> = Vec::with_capacity(n_dip_coeffs / 2);
            let mut ms2: Vec<f64> = Vec::with_capacity(n_dip_coeffs / 2);
            let mut ns2: Vec<f64> = Vec::with_capacity(n_dip_coeffs / 2);

            // ???
            let mut b_update_n_accum = false;
            for i in 0..n_dip_coeffs {
                let mode_type = self.modes[[0, i]];
                let mode_m = self.modes[[1, i]];
                let mode_n = self.modes[[2, i]];

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

        let mut m_signs: Vec<Sign> = vec![];
        for m in &m_accum {
            let mut sign = Sign::Positive;
            if *m > 0.0 && (*m as i32) % 2 != 0 {
                sign = Sign::Negative;
            }
            m_signs.push(sign)
        }

        Ok(PolCoefficients {
            q1_accum,
            q2_accum,
            m_accum,
            n_accum,
            m_signs,
            n_max,
        })
    }
}

// fn calc_sigmas(phi: f64, theta: f64, coeffs: &DipoleCoefficients) -> Jones {
//     // The n_max is the same in both pols.
//     // TODO: Remove.
//     debug_assert_eq!(coeffs.x.n_max, coeffs.y.n_max);
//     let n_max = coeffs.x.n_max;
//     let u = theta.cos();

//     // TODO: Is this the only place that n_max is actually used? If so, make the
//     // struct type a usize.
//     let (p1sin_arr, p1_arr) = p1sin(coeffs.x.n_max as usize, theta);

//     // TODO: Check that the sizes of N_accum and M_accum agree. This check
//     // should actually be in the PolCoefficients generation.

//     let sigma_p = Complex64::new(0.0, 0.0);
//     let sigma_t = Complex64::new(0.0, 0.0);
//     for i in 0..coeffs.x.n_accum.len() {
//     }
// }

#[cfg(test)]
mod tests {
    use super::*;
    use approx::*;

    #[test]
    fn new() {
        let beam = FEEBeam::new("mwa_full_embedded_element_pattern.h5");
        assert!(beam.is_ok());
    }

    #[test]
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
    /// Check that we can open the dataset "X16_51200000".
    fn test_get_dataset() {
        let beam = FEEBeam::new("mwa_full_embedded_element_pattern.h5").unwrap();
        assert!(beam.get_dataset("X16_51200000").is_ok());
    }

    #[test]
    fn test_get_modes() {
        let mut beam = FEEBeam::new("mwa_full_embedded_element_pattern.h5").unwrap();
        let result = beam.get_modes(51200000, &[0; 16], &[1.0; 16]);
        assert!(result.is_ok());
        let coeffs = result.unwrap();

        // Values taken from the C++ code.
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
}
