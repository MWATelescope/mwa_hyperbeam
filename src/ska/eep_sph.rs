// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// Portions of this file are derived from ska-sdp-func
// (Copyright 2022 SKA Organisation; Copyright 2022 Fred Dulwich, Andrew
// Ensor, Anthony Griffin, Seth Hall, Adam Campbell; Copyright 2022 University
// of Oxford), used under the BSD-3-Clause licence; see LICENSE-ska-sdp-func in
// the root of this repository.

//! The SKA-Low spherical-wave embedded-element-pattern model
//! ("ska-low-eep-sph"), equivalent to EveryBeam's `skalow_feko` element
//! response model.
//!
//! The spherical-wave evaluation follows
//! `sdp_element_beam_spherical_wave_feko` and `sdp_legendre_norm` from
//! [ska-sdp-func](https://gitlab.com/ska-telescope/sdp/ska-sdp-func)
//! (BSD-3-Clause), which EveryBeam calls. That method is in turn based on
//! MATLAB code by David Davidson and Adrian Sutinjo.

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

use marlu::{c64, Jones};
use ndarray::prelude::*;
use parking_lot::RwLock;

use super::{InitSkaBeamError, SkaBeamError, SkaElementResponse};

/// The environment variable that [`SkaLowEepSph::new_from_env`] reads.
pub const EEP_SPH_PATH_VAR: &str = "SKA_LOW_EEP_SPH_PATH";

/// Free-space impedance (post-2019 definition) \[Ohm\].
const ETA_0: f64 = 376.730313668;

/// Metadata on a single FEKO coefficients file (one frequency).
struct FekoFile {
    path: PathBuf,
    freq_hz: f64,
    l_max: usize,
    num_elements: usize,
}

/// The SKA-Low spherical-wave embedded-element-pattern model
/// ("ska-low-eep-sph").
///
/// This is equivalent to EveryBeam's `skalow_feko` element response model
/// (`kSkaLowFeko`): per-element embedded element patterns simulated with FEKO
/// and expanded in spherical waves.
/// Each element of an SKA-Low station has its own embedded element pattern,
/// described by spherical-wave (TE/TM) coefficients; the element is selected
/// with the `element_id` argument of the [`SkaElementResponse`] methods.
///
/// The coefficients are read from a directory of HDF5 files, one per
/// frequency (the format produced by EveryBeam's
/// `scripts/coeff_scripts/prepare_feko_coefficients.py`), each containing:
///
/// - `coefficients`: complex, `num_elements x l_max * (l_max + 2) x 4`, with
///   the last dimension being `[TE_X, TM_X, TE_Y, TM_Y]`;
/// - `modes`: integer, `3 x l_max * (l_max + 2)`; `l_max` is `modes[2, -1]`.
///
/// The frequency of each file is taken from its `frequency_in_Hz` attribute
/// or, failing that, from the 7th underscore-separated token of the file name
/// \[MHz\] (as EveryBeam does), e.g.
/// `SKA-Low_Perturbed_Vogel_256_elem_FEKO_100.00000_MHz.h5`.
///
/// As in EveryBeam, no frequency interpolation is done; the coefficients of
/// the file with the closest frequency are used. Coefficients are read lazily
/// (one frequency at a time) and cached.
pub struct SkaLowEepSph {
    /// Ascendingly sorted by frequency.
    files: Vec<FekoFile>,
    /// The maximum degree to use.
    l_max: usize,
    /// Prevents parallel usage of HDF5.
    hdf5_lock: Mutex<()>,
    /// Cache of coefficients, keyed by file index. Each array has dimensions
    /// `num_elements x num_coeffs x 4`.
    cache: RwLock<HashMap<usize, Arc<Array3<c64>>>>,
}

impl SkaLowEepSph {
    /// Load all FEKO coefficient files (`*.h5`) in `dir`, using all available
    /// spherical-wave degrees.
    pub fn new<P: AsRef<Path>>(dir: P) -> Result<SkaLowEepSph, InitSkaBeamError> {
        Self::new_with_l_max(dir, None)
    }

    /// Load all FEKO coefficient files (`*.h5`) in `dir`. If `l_max` is given
    /// (and non-zero), the spherical-wave expansion is truncated at this
    /// degree (or the degree available in each file, whichever is smaller).
    pub fn new_with_l_max<P: AsRef<Path>>(
        dir: P,
        l_max: Option<usize>,
    ) -> Result<SkaLowEepSph, InitSkaBeamError> {
        let dir = dir.as_ref();
        hdf5_metno::silence_errors(true);
        if !dir.exists() {
            return Err(InitSkaBeamError::PathDoesntExist(dir.display().to_string()));
        }
        let mut paths = vec![];
        for entry in std::fs::read_dir(dir)
            .map_err(|_| InitSkaBeamError::PathDoesntExist(dir.display().to_string()))?
        {
            let Ok(entry) = entry else { continue };
            let path = entry.path();
            if path.is_file() && path.extension().is_some_and(|e| e == "h5") {
                paths.push(path);
            }
        }
        if paths.is_empty() {
            return Err(InitSkaBeamError::NoFekoFiles(dir.display().to_string()));
        }

        let mut files = Vec::with_capacity(paths.len());
        for path in paths {
            let file_str = path.display().to_string();
            let h5 = hdf5_metno::File::open(&path)?;

            let freq_hz = match h5
                .attr("frequency_in_Hz")
                .and_then(|a| a.read_scalar::<f64>())
            {
                Ok(f) => f,
                Err(_) => freq_from_file_name(&path)
                    .ok_or_else(|| InitSkaBeamError::FekoFrequency(file_str.clone()))?,
            };

            let modes = h5.dataset("modes")?;
            let modes_shape = modes.shape();
            if modes_shape.len() != 2 || modes_shape[0] < 3 || modes_shape[1] == 0 {
                return Err(InitSkaBeamError::BadShape {
                    file: file_str,
                    dataset: "modes".to_string(),
                    got: modes_shape,
                    expected: "expected 3 x N".to_string(),
                });
            }
            // Read raw buffers (row major) rather than with hdf5's ndarray
            // support, so that downstream crates don't need hdf5's ndarray to
            // be the same version as ours.
            let modes: Vec<i64> = modes.read_raw()?;
            let file_l_max = modes[2 * modes_shape[1] + modes_shape[1] - 1].max(0) as usize;

            let coeffs = h5.dataset("coefficients")?;
            let shape = coeffs.shape();
            if shape.len() != 3 || shape[2] != 4 || shape[1] < num_coeffs(file_l_max) {
                return Err(InitSkaBeamError::BadShape {
                    file: file_str,
                    dataset: "coefficients".to_string(),
                    got: shape,
                    expected: format!(
                        "expected num_elements x {} x 4 (l_max = {file_l_max})",
                        num_coeffs(file_l_max)
                    ),
                });
            }

            files.push(FekoFile {
                path,
                freq_hz,
                l_max: file_l_max,
                num_elements: shape[0],
            });
        }
        files.sort_unstable_by(|a, b| a.freq_hz.total_cmp(&b.freq_hz));
        if let Some(w) = files.windows(2).find(|w| w[0].freq_hz == w[1].freq_hz) {
            return Err(InitSkaBeamError::DuplicateFekoFrequency {
                freq_hz: w[0].freq_hz,
                file1: w[0].path.display().to_string(),
                file2: w[1].path.display().to_string(),
            });
        }

        let max_file_l_max = files.iter().map(|f| f.l_max).max().unwrap_or(0);
        let l_max = match l_max {
            Some(l) if l > 0 => l,
            _ => max_file_l_max,
        };

        Ok(SkaLowEepSph {
            files,
            l_max,
            hdf5_lock: Mutex::new(()),
            cache: RwLock::new(HashMap::new()),
        })
    }

    /// Load the FEKO coefficients in the directory named by the
    /// `SKA_LOW_EEP_SPH_PATH` environment variable or, if that isn't set,
    /// `$EVERYBEAM_DATADIR/skalow` (EveryBeam's default).
    pub fn new_from_env() -> Result<SkaLowEepSph, InitSkaBeamError> {
        if let Ok(d) = std::env::var(EEP_SPH_PATH_VAR) {
            return Self::new(d);
        }
        match std::env::var(super::eep_grid::EVERYBEAM_DATADIR_VAR) {
            Ok(d) => Self::new(Path::new(&d).join("skalow")),
            Err(_) => Err(InitSkaBeamError::EnvVar {
                var: EEP_SPH_PATH_VAR,
            }),
        }
    }

    /// The frequencies of the available coefficient files \[Hz\], ascending.
    pub fn get_freqs(&self) -> Vec<f64> {
        self.files.iter().map(|f| f.freq_hz).collect()
    }

    /// The maximum spherical-wave degree used.
    pub fn get_l_max(&self) -> usize {
        self.l_max
    }

    /// Find the index of the file with the frequency closest to `freq_hz`
    /// (ties go to the lower frequency, as in EveryBeam).
    pub(crate) fn closest_file(&self, freq_hz: f64) -> usize {
        let i = self.files.partition_point(|f| f.freq_hz < freq_hz);
        if i == 0 {
            0
        } else if i == self.files.len() {
            i - 1
        } else {
            let right = self.files[i].freq_hz - freq_hz;
            let left = freq_hz - self.files[i - 1].freq_hz;
            if right >= left {
                i - 1
            } else {
                i
            }
        }
    }

    /// The frequency of the coefficients that will be used for `freq_hz`.
    pub fn find_closest_freq(&self, freq_hz: f64) -> f64 {
        self.files[self.closest_file(freq_hz)].freq_hz
    }

    /// Get (reading if necessary) the coefficients of a file, truncated to
    /// this object's `l_max`.
    pub(crate) fn get_coeffs(&self, file_index: usize) -> Result<Arc<Array3<c64>>, SkaBeamError> {
        if let Some(c) = self.cache.read().get(&file_index) {
            return Ok(Arc::clone(c));
        }
        let file = &self.files[file_index];
        let l_max = self.l_max.min(file.l_max);
        let coeffs = {
            let _guard = self.hdf5_lock.lock().unwrap_or_else(|e| e.into_inner());
            let h5 = hdf5_metno::File::open(&file.path)?;
            let ds = h5.dataset("coefficients")?;
            let shape = ds.shape();
            let raw: Vec<c64> = ds.read_raw()?;
            let all = ArrayView3::from_shape((shape[0], shape[1], shape[2]), &raw)
                .expect("the dataset's shape was checked when the file was opened");
            // EveryBeam's preparation script zeroes NaNs; do the same in case
            // files were made some other way.
            all.slice(s![.., ..num_coeffs(l_max), ..]).mapv(|c| {
                if c.is_nan() {
                    c64::default()
                } else {
                    c
                }
            })
        };
        let coeffs = Arc::new(coeffs);
        self.cache.write().insert(file_index, Arc::clone(&coeffs));
        Ok(coeffs)
    }

    /// Remove all cached coefficients.
    pub fn empty_cache(&self) {
        self.cache.write().clear();
    }

    /// Evaluate the response of element `element_id` for many directions,
    /// with separate phi values for the X and Y dipoles (as EveryBeam's
    /// `SkaLowFekoElementResponse::Response` overload does). The results are
    /// in EveryBeam's convention.
    pub fn element_responses(
        &self,
        element_id: usize,
        freq_hz: f64,
        thetas: &[f64],
        phis_x: &[f64],
        phis_y: &[f64],
    ) -> Result<Vec<Jones<f64>>, SkaBeamError> {
        let file_index = self.closest_file(freq_hz);
        let coeffs = self.get_coeffs(file_index)?;
        let num_elements = coeffs.len_of(Axis(0));
        if element_id >= num_elements {
            return Err(SkaBeamError::ElementOutOfRange {
                got: element_id,
                num_elements,
            });
        }
        let l_max = self.l_max.min(self.files[file_index].l_max);
        let coeffs = coeffs.slice(s![element_id, .., ..]);
        let mut workspace = vec![0.0; l_max + 2];
        Ok(thetas
            .iter()
            .zip(phis_x)
            .zip(phis_y)
            .map(|((&theta, &phi_x), &phi_y)| {
                let p = spherical_wave_feko(theta, phi_x, phi_y, l_max, coeffs, &mut workspace);
                // The SKA-Low antenna design requires a conjugation of both
                // dipoles and a negation of the Y dipole (as in EveryBeam).
                Jones::from([p[0].conj(), p[1].conj(), -p[2].conj(), -p[3].conj()])
            })
            .collect())
    }
}

impl SkaElementResponse for SkaLowEepSph {
    fn element_response(
        &self,
        element_id: usize,
        freq_hz: f64,
        theta_rad: f64,
        phi_rad: f64,
    ) -> Result<Jones<f64>, SkaBeamError> {
        let mut r =
            self.element_responses(element_id, freq_hz, &[theta_rad], &[phi_rad], &[phi_rad])?;
        Ok(r.remove(0))
    }

    fn num_elements(&self) -> Option<usize> {
        Some(self.files.iter().map(|f| f.num_elements).min().unwrap_or(0))
    }
}

/// The number of spherical-wave coefficients for degrees 1 to `l_max`.
fn num_coeffs(l_max: usize) -> usize {
    l_max * (l_max + 2)
}

/// Parse a frequency from a FEKO coefficient file name, as EveryBeam does: the
/// 7th underscore-separated token of the file stem is the frequency in MHz
/// (only its leading numeric part is used).
fn freq_from_file_name(path: &Path) -> Option<f64> {
    let stem = path.file_stem()?.to_str()?;
    let token = stem.split('_').nth(6)?;
    let end = token
        .find(|c: char| {
            !(c.is_ascii_digit() || c == '.' || c == '-' || c == '+' || c == 'e' || c == 'E')
        })
        .unwrap_or(token.len());
    token[..end].parse::<f64>().ok().map(|f| f * 1e6)
}

/// `i^n`.
#[inline]
fn ipow(n: usize) -> c64 {
    match n & 3 {
        0 => c64::new(1.0, 0.0),
        1 => c64::new(0.0, 1.0),
        2 => c64::new(-1.0, 0.0),
        _ => c64::new(0.0, -1.0),
    }
}

/// Evaluate the spherical-wave pattern of one element for one direction. The
/// result is `[E_theta_X, E_phi_X, E_theta_Y, E_phi_Y]` (before the SKA-Low
/// sign convention is applied). `coeffs` has dimensions `num_coeffs x 4`, the
/// last dimension being `[TE_X, TM_X, TE_Y, TM_Y]`. `workspace` must have at
/// least `l_max + 2` elements.
fn spherical_wave_feko(
    theta: f64,
    phi_x: f64,
    phi_y: f64,
    l_max: usize,
    coeffs: ArrayView2<c64>,
    workspace: &mut [f64],
) -> [c64; 4] {
    let sqrt_fac = (ETA_0 / std::f64::consts::TAU).sqrt();
    let mut e_theta = [c64::default(); 2];
    let mut e_phi = [c64::default(); 2];

    // Avoid divide-by-zero.
    let theta = if theta < 1e-5 { 1e-5 } else { theta };
    if phi_x.is_nan() {
        let nan = c64::new(phi_x, phi_x);
        return [nan; 4];
    }

    let cos_phi = [phi_x.cos(), phi_y.cos()];
    let sin_phi = [phi_x.sin(), phi_y.sin()];
    let (sin_theta, cos_theta) = theta.sin_cos();

    for n in 1..=l_max {
        legendre_norm(n, cos_theta, sin_theta, workspace);
        workspace[n + 1] = 0.0;
        let nf = n as f64;
        let sqrt_n = (nf * (nf + 1.0)).sqrt();
        let i_n = ipow(n);
        let i_np1 = ipow(n + 1);

        // cos(m phi) and sin(m phi) for m = 0, for each dipole.
        let mut cos_m_phi = [1.0, 1.0];
        let mut sin_m_phi = [0.0, 0.0];

        for abs_m in 0..=n {
            let abs_mf = abs_m as f64;
            let cmn = (((n + abs_m + 1) * (n - abs_m)) as f64).sqrt();
            let np_1 = cmn * workspace[abs_m + 1];
            let np_ds = workspace[abs_m] / sin_theta;
            let np_diff = np_ds * abs_mf * cos_theta - np_1;

            let ms: &[i64] = if abs_m == 0 {
                &[0]
            } else {
                &[abs_m as i64, -(abs_m as i64)]
            };
            for &m in ms {
                let sign = if m >= 0 || m & 1 == 0 { 1.0 } else { -1.0 };
                let np_ds_m = np_ds * m as f64;
                let row = (n * (n + 1)) as i64 + m - 1;
                let row = row as usize;

                for a in 0..2 {
                    let te = coeffs[(row, 2 * a)];
                    let tm = coeffs[(row, 2 * a + 1)];
                    let sin_m_phi_m = if m >= 0 { sin_m_phi[a] } else { -sin_m_phi[a] };
                    let phi_const = c64::new(cos_m_phi[a], sin_m_phi_m) * (sign / sqrt_n);
                    e_theta[a] += phi_const * i_n * (-te * np_ds_m + tm * np_diff);
                    e_phi[a] += phi_const * i_np1 * (tm * np_ds_m - te * np_diff);
                }
            }

            // cos((m + 1) phi) and sin((m + 1) phi).
            for a in 0..2 {
                let (c, s) = (cos_m_phi[a], sin_m_phi[a]);
                cos_m_phi[a] = cos_phi[a] * c - sin_phi[a] * s;
                sin_m_phi[a] = sin_phi[a] * c + cos_phi[a] * s;
            }
        }
    }

    [
        e_theta[0] * sqrt_fac,
        e_phi[0] * sqrt_fac,
        e_theta[1] * sqrt_fac,
        e_phi[1] * sqrt_fac,
    ]
}

/// Fully-normalised associated Legendre functions of degree `n` and orders
/// `0..=n`, written to `p[0..=n]` (following ska-sdp-func's
/// `sdp_legendre_norm`, which in turn follows MATLAB's `legendre(n, x,
/// 'norm')`, including its underflow handling).
pub(super) fn legendre_norm(n: usize, cos_theta: f64, sin_theta: f64, p: &mut [f64]) {
    let nf = n as f64;
    let tol = (f32::MIN_POSITIVE as f64).sqrt();
    let root_n_plus_half = (nf + 0.5).sqrt();
    let twocot = if sin_theta > 0.0 {
        -2.0 * cos_theta / sin_theta
    } else {
        0.0
    };
    let sn = if sin_theta == 0.0 {
        0.0
    } else if n.is_multiple_of(2) {
        sin_theta.powi(n as i32)
    } else {
        -sin_theta.powi(n as i32)
    };
    let root = |i: usize| (i as f64).sqrt();

    p[..=n].iter_mut().for_each(|v| *v = 0.0);

    // Underflow branch: used only when cos_theta is very close to 1.
    if sin_theta > 0.0 && sn.abs() <= tol {
        let tstart = f32::EPSILON as f64;
        let v = 9.2 - tol.ln() / (nf * sin_theta);
        let w = 1.0 / v.ln();
        let m1d = 1.0 + nf * sin_theta * v * w * (1.0058 + w * (3.819 - w * 12.173));
        let m1 = (m1d.floor() as i64).clamp(1, n as i64) as usize;
        let sgn_m1 = if m1 % 2 == 1 { 1.0 } else { -1.0 };
        let sgn_n1 = if (n + 1) % 2 == 1 { 1.0 } else { -1.0 };
        p[m1 - 1] = if cos_theta < 0.0 {
            sgn_n1 * tstart
        } else {
            sgn_m1 * tstart
        };
        let mut sumsq = tol;
        for m in (0..m1.saturating_sub(1)).rev() {
            let a = (m + 1) as f64 * twocot * p[m + 1];
            let b = root(n + m + 2) * root(n - m - 1) * p[m + 2];
            p[m] = (a - b) / (root(n + m + 1) * root(n - m));
            sumsq += p[m] * p[m];
        }
        let scale = 1.0 / (2.0 * sumsq - p[0] * p[0]).sqrt();
        p[..=m1].iter_mut().for_each(|v| *v *= scale);
    }

    // Non-underflow branch.
    if cos_theta != 1.0 && sn.abs() >= tol {
        let mut c = 1.0;
        for k in 1..=n {
            c *= 1.0 - 1.0 / (2.0 * k as f64);
        }
        p[n] = c.sqrt() * sn;
        p[n - 1] = p[n] * twocot * nf / root(2 * n);
        for m in (0..n.saturating_sub(1)).rev() {
            let a = (m + 1) as f64 * twocot * p[m + 1];
            let b = root(n + m + 2) * root(n - m - 1) * p[m + 2];
            p[m] = (a - b) / (root(n + m + 1) * root(n - m));
        }
    }

    // Polar case.
    if sin_theta == 0.0 {
        p[0] = cos_theta.powi(n as i32);
    }

    // Normalisation.
    for (m, v) in p[..=n].iter_mut().enumerate() {
        let sgn = if m % 2 == 0 { 1.0 } else { -1.0 };
        *v *= root_n_plus_half * sgn;
    }
}
