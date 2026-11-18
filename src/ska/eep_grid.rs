// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! The SKA-Low gridded embedded-element-pattern model ("ska-low-eep-grid"),
//! equivalent to EveryBeam's `gridded_eep` element response model.

use std::{f64::consts::TAU, path::Path, sync::Arc};

use marlu::{c64, AzEl, Jones};
use parking_lot::Mutex;
use rayon::prelude::*;

use super::{
    az_to_phi, everybeam_to_hyperbeam, par_map_jones, InitSkaBeamError, SkaBeamError,
    SkaElementResponse,
};

/// The environment variable that [`SkaLowEepGrid::new_from_env`] reads first.
pub const EEP_GRID_FILE_VAR: &str = "SKA_LOW_EEP_GRID_FILE";
/// The environment variable [`SkaLowEepGrid::new_from_env`] falls back to; the
/// file `oskar_eep_grid.h5` in this directory is used (as EveryBeam does for
/// OSKAR telescopes).
pub const EVERYBEAM_DATADIR_VAR: &str = "EVERYBEAM_DATADIR";

/// A grid interpolated to a single frequency (phi x theta, flattened).
pub(crate) type Plane = Arc<Vec<[c64; 4]>>;

/// The SKA-Low gridded, station-averaged embedded-element-pattern (EEP) model
/// ("ska-low-eep-grid").
///
/// This is equivalent to EveryBeam's `gridded_eep` element response model
/// (`kGriddedEep`), as of EveryBeam master (November 2026, after 0.9.1). The HDF5 file must contain a group
/// `averageEEP` with the datasets:
///
/// - `Freq`: frequencies \[Hz\], ascending;
/// - `Phi`: phi values \[radians\] in `[0, 2pi]`, ascending;
/// - `Theta`: theta values \[radians\] in `[0, pi]`, ascending;
/// - `averageEEP`: a `Phi x Theta x Freq x 4 x 2` dataset of doubles, where
///   the last dimension holds real and imaginary parts.
///
/// (The one-dimensional datasets may also be `N x 1` or `1 x N`, as written
/// by MATLAB.)
///
/// As in EveryBeam, the four correlations are stored in the order `[J01, J00,
/// J11, J10]` relative to the returned Jones matrix, and values are
/// tri-linearly interpolated in phi, theta and frequency. Values outside of
/// the grid are an error (no extrapolation is done). EveryBeam notes that this
/// file format "is not final and will be adjusted in the future".
///
/// By default the SKA-Low (OSKAR) response convention is applied, i.e. all
/// four values are conjugated and the Y-dipole values negated, as EveryBeam
/// does for OSKAR telescopes. Use [`SkaLowEepGrid::new_with_convention`] to
/// disable this.
pub struct SkaLowEepGrid {
    phis: Axis,
    thetas: Axis,
    freqs: Axis,
    /// Dimensions: freq x phi x theta, flattened (frequency-major, unlike the
    /// file, so that a single frequency plane is contiguous). Each element is
    /// `[J00, J01, J10, J11]` (already re-ordered from the file order).
    grid: Vec<[c64; 4]>,
    ska_low_convention: bool,
    /// The most recently used frequency plane (see `freq_plane`).
    plane_cache: Mutex<Option<(f64, Plane)>>,
}

impl SkaLowEepGrid {
    /// Load an EEP grid file, applying the SKA-Low (OSKAR) response
    /// convention, as EveryBeam does for OSKAR telescopes.
    pub fn new<P: AsRef<Path>>(file: P) -> Result<SkaLowEepGrid, InitSkaBeamError> {
        Self::new_with_convention(file, true)
    }

    /// Load an EEP grid file. If `ska_low_convention` is `true`, values are
    /// conjugated and the Y-dipole (second row) values negated after
    /// interpolation (EveryBeam's behaviour for OSKAR telescopes); otherwise
    /// the interpolated values are returned as they are (EveryBeam's
    /// behaviour for other telescopes).
    pub fn new_with_convention<P: AsRef<Path>>(
        file: P,
        ska_low_convention: bool,
    ) -> Result<SkaLowEepGrid, InitSkaBeamError> {
        let file = file.as_ref();
        hdf5_metno::silence_errors(true);
        if !file.exists() {
            return Err(InitSkaBeamError::PathDoesntExist(
                file.display().to_string(),
            ));
        }
        let file_str = file.display().to_string();
        let h5 = hdf5_metno::File::open(file)?;
        let group = h5.group("averageEEP")?;

        let read_vector = |name: &str| -> Result<Vec<f64>, InitSkaBeamError> {
            let dataset = group.dataset(name)?;
            let shape = dataset.shape();
            let ok = match shape.as_slice() {
                [n] => *n >= 2,
                [1, n] | [n, 1] => *n >= 2,
                _ => false,
            };
            if !ok {
                return Err(InitSkaBeamError::BadShape {
                    file: file_str.clone(),
                    dataset: name.to_string(),
                    got: shape,
                    expected: "expected a vector with at least two values".to_string(),
                });
            }
            let values: Vec<f64> = dataset.read_raw()?;
            if values.windows(2).any(|w| w[0] > w[1]) {
                return Err(InitSkaBeamError::NotSorted {
                    file: file_str.clone(),
                    dataset: name.to_string(),
                });
            }
            Ok(values)
        };
        let phis = read_vector("Phi")?;
        let thetas = read_vector("Theta")?;
        let freqs = read_vector("Freq")?;

        let dataset = group.dataset("averageEEP")?;
        let shape = dataset.shape();
        let expected_shape = [phis.len(), thetas.len(), freqs.len(), 4, 2];
        if shape.as_slice() != expected_shape {
            return Err(InitSkaBeamError::BadShape {
                file: file_str,
                dataset: "averageEEP".to_string(),
                got: shape,
                expected: format!("expected {expected_shape:?} (Phi x Theta x Freq x 4 x 2)"),
            });
        }
        let buffer: Vec<f64> = dataset.read_raw()?;
        let (np, nt, nf) = (phis.len(), thetas.len(), freqs.len());
        let mut grid = vec![[c64::default(); 4]; np * nt * nf];
        for (i_file, b) in buffer.as_chunks::<8>().0.iter().enumerate() {
            let (p, rest) = (i_file / (nt * nf), i_file % (nt * nf));
            let (t, f) = (rest / nf, rest % nf);
            // The file stores [J01, J00, J11, J10] (EveryBeam's phi/theta
            // swap when reading the grid).
            grid[(f * np + p) * nt + t] = [
                c64::new(b[2], b[3]),
                c64::new(b[0], b[1]),
                c64::new(b[6], b[7]),
                c64::new(b[4], b[5]),
            ];
        }

        Ok(SkaLowEepGrid {
            phis: Axis::new(phis),
            thetas: Axis::new(thetas),
            freqs: Axis::new(freqs),
            grid,
            ska_low_convention,
            plane_cache: Mutex::new(None),
        })
    }

    /// Load the EEP grid file named by the `SKA_LOW_EEP_GRID_FILE` environment
    /// variable or, if that isn't set, `$EVERYBEAM_DATADIR/oskar_eep_grid.h5`
    /// (EveryBeam's default for OSKAR telescopes). The SKA-Low convention is
    /// applied.
    pub fn new_from_env() -> Result<SkaLowEepGrid, InitSkaBeamError> {
        if let Ok(f) = std::env::var(EEP_GRID_FILE_VAR) {
            return Self::new(f);
        }
        match std::env::var(EVERYBEAM_DATADIR_VAR) {
            Ok(d) => Self::new(Path::new(&d).join("oskar_eep_grid.h5")),
            Err(_) => Err(InitSkaBeamError::EnvVar {
                var: EEP_GRID_FILE_VAR,
            }),
        }
    }

    /// The phi values of the grid \[radians\].
    pub fn get_phis(&self) -> &[f64] {
        &self.phis.values
    }

    /// The theta values of the grid \[radians\].
    pub fn get_thetas(&self) -> &[f64] {
        &self.thetas.values
    }

    /// The frequencies of the grid \[Hz\].
    pub fn get_freqs(&self) -> &[f64] {
        &self.freqs.values
    }

    /// Interpolate the grid in frequency, giving a phi x theta plane (with the
    /// sign convention applied).
    ///
    /// The most recently used plane is cached, so repeated calls at the same
    /// frequency are cheap.
    pub(crate) fn freq_plane(&self, freq_hz: f64) -> Result<Plane, SkaBeamError> {
        if let Some((f, plane)) = &*self.plane_cache.lock() {
            if *f == freq_hz {
                return Ok(Arc::clone(plane));
            }
        }
        let (i_f, wf) = self.freqs.locate("frequency", freq_hz)?;
        let plane_size = self.phis.values.len() * self.thetas.values.len();
        let lo = &self.grid[i_f * plane_size..(i_f + 1) * plane_size];
        let hi = &self.grid[(i_f + 1) * plane_size..(i_f + 2) * plane_size];
        let plane: Vec<[c64; 4]> = lo
            .par_iter()
            .zip(hi.par_iter())
            .map(|(lo, hi)| {
                let mut out = [c64::default(); 4];
                for k in 0..4 {
                    out[k] = lo[k] * (1.0 - wf) + hi[k] * wf;
                }
                self.apply_convention(out)
            })
            .collect();
        let plane = Arc::new(plane);
        *self.plane_cache.lock() = Some((freq_hz, Arc::clone(&plane)));
        Ok(plane)
    }

    #[inline]
    fn apply_convention(&self, j: [c64; 4]) -> [c64; 4] {
        if self.ska_low_convention {
            [j[0].conj(), j[1].conj(), -j[2].conj(), -j[3].conj()]
        } else {
            j
        }
    }

    /// Bi-linearly interpolate a frequency plane made by `freq_plane`.
    #[inline]
    fn interpolate_plane(
        &self,
        plane: &[[c64; 4]],
        phi_rad: f64,
        theta_rad: f64,
    ) -> Result<Jones<f64>, SkaBeamError> {
        let (ip, wp) = self.phis.locate("phi", wrap_phi(phi_rad))?;
        let (it, wt) = self.thetas.locate("theta", theta_rad)?;
        let nt = self.thetas.values.len();
        let row0 = &plane[ip * nt + it..ip * nt + it + 2];
        let row1 = &plane[(ip + 1) * nt + it..(ip + 1) * nt + it + 2];
        let w = [
            (1.0 - wp) * (1.0 - wt),
            (1.0 - wp) * wt,
            wp * (1.0 - wt),
            wp * wt,
        ];
        let mut out = [c64::default(); 4];
        for k in 0..4 {
            out[k] = row0[0][k] * w[0] + row0[1][k] * w[1] + row1[0][k] * w[2] + row1[1][k] * w[3];
        }
        Ok(Jones::from(out))
    }

    /// Evaluate the element response (EveryBeam convention) for many
    /// directions at a single frequency. This is much faster than calling
    /// [`SkaElementResponse::element_response`] repeatedly, as the grid is
    /// interpolated in frequency only once.
    pub fn element_responses(
        &self,
        freq_hz: f64,
        thetas: &[f64],
        phis: &[f64],
    ) -> Result<Vec<Jones<f64>>, SkaBeamError> {
        let plane = self.freq_plane(freq_hz)?;
        par_map_jones(thetas.len().min(phis.len()), |i| {
            self.interpolate_plane(&plane, phis[i], thetas[i])
        })
    }

    /// Tri-linearly interpolate the grid (without applying any sign
    /// convention).
    pub fn interpolate(
        &self,
        phi_rad: f64,
        theta_rad: f64,
        freq_hz: f64,
    ) -> Result<Jones<f64>, SkaBeamError> {
        let (ip, wp) = self.phis.locate("phi", wrap_phi(phi_rad))?;
        let (it, wt) = self.thetas.locate("theta", theta_rad)?;
        let (i_f, wf) = self.freqs.locate("frequency", freq_hz)?;

        let np = self.phis.values.len();
        let nt = self.thetas.values.len();
        let mut out = [c64::default(); 4];
        for (dp, w_p) in [(0, 1.0 - wp), (1, wp)] {
            for (dt, w_t) in [(0, 1.0 - wt), (1, wt)] {
                for (df, w_f) in [(0, 1.0 - wf), (1, wf)] {
                    let w = w_p * w_t * w_f;
                    let g = &self.grid[((i_f + df) * np + (ip + dp)) * nt + (it + dt)];
                    out.iter_mut().zip(g).for_each(|(o, g)| *o += g * w);
                }
            }
        }
        Ok(Jones::from(out))
    }
}

/// A grid axis: ascending values, plus the step if they are (to within
/// rounding) uniformly spaced, allowing O(1) lookups.
struct Axis {
    values: Vec<f64>,
    /// 1 / step, if the values are uniformly spaced.
    uniform_inv_step: Option<f64>,
}

impl Axis {
    fn new(values: Vec<f64>) -> Axis {
        let n = values.len();
        let step = (values[n - 1] - values[0]) / (n - 1) as f64;
        let uniform = step > 0.0
            && values
                .iter()
                .enumerate()
                .all(|(i, &v)| (v - (values[0] + i as f64 * step)).abs() <= 1e-6 * step);
        Axis {
            values,
            uniform_inv_step: uniform.then_some(1.0 / step),
        }
    }

    /// Find the lower index `i` of the grid interval containing `value`, and
    /// the weight of the upper value (`values[i + 1]`). The interval chosen is
    /// exactly the one EveryBeam chooses (`std::lower_bound` followed by a
    /// decrement).
    #[inline]
    fn locate(&self, what: &'static str, value: f64) -> Result<(usize, f64), SkaBeamError> {
        let values = &self.values;
        let n = values.len();
        let (min, max) = (values[0], values[n - 1]);
        // `!(a <= b)` rather than `a > b` so that NaN is caught.
        #[allow(clippy::neg_cmp_op_on_partial_ord)]
        if !(value >= min && value <= max) {
            return Err(SkaBeamError::OutOfGrid {
                what,
                value,
                min,
                max,
            });
        }
        let i = match self.uniform_inv_step {
            Some(inv_step) => {
                // Guess, then correct the guess to match lower_bound semantics.
                let mut i = (((value - min) * inv_step) as usize).min(n - 2);
                while i > 0 && values[i] >= value {
                    i -= 1;
                }
                while i < n - 2 && values[i + 1] < value {
                    i += 1;
                }
                i
            }
            None => values
                .partition_point(|&v| v < value)
                .saturating_sub(1)
                .min(n - 2),
        };
        let (lo, hi) = (values[i], values[i + 1]);
        Ok((i, (value - lo) / (hi - lo)))
    }
}

/// Map phi into [0, 2pi]. Unlike EveryBeam, which only adds 2pi once to
/// negative values, wrap fully, so that e.g. phi = pi/2 - az works for any az.
#[inline]
fn wrap_phi(phi_rad: f64) -> f64 {
    if (0.0..=TAU).contains(&phi_rad) {
        phi_rad
    } else if (-TAU..0.0).contains(&phi_rad) {
        phi_rad + TAU
    } else {
        phi_rad.rem_euclid(TAU)
    }
}

impl SkaElementResponse for SkaLowEepGrid {
    fn element_response(
        &self,
        _element_id: usize,
        freq_hz: f64,
        theta_rad: f64,
        phi_rad: f64,
    ) -> Result<Jones<f64>, SkaBeamError> {
        let j = self.interpolate(phi_rad, theta_rad, freq_hz)?;
        Ok(Jones::from(self.apply_convention(*j)))
    }

    fn calc_jones_array_pair(
        &self,
        az_rad: &[f64],
        za_rad: &[f64],
        freq_hz: f64,
        _element_id: usize,
        latitude_rad: Option<f64>,
        iau_order: bool,
    ) -> Result<Vec<Jones<f64>>, SkaBeamError> {
        let plane = self.freq_plane(freq_hz)?;
        par_map_jones(az_rad.len().min(za_rad.len()), |i| {
            let (az, za) = (az_rad[i], za_rad[i]);
            let j = self.interpolate_plane(&plane, az_to_phi(az), za)?;
            Ok(everybeam_to_hyperbeam(j, az, za, latitude_rad, iau_order))
        })
    }

    fn calc_jones_array(
        &self,
        azels: &[AzEl],
        freq_hz: f64,
        element_id: usize,
        latitude_rad: Option<f64>,
        iau_order: bool,
    ) -> Result<Vec<Jones<f64>>, SkaBeamError> {
        let (az, za): (Vec<f64>, Vec<f64>) = azels.iter().map(|a| (a.az, a.za())).unzip();
        self.calc_jones_array_pair(&az, &za, freq_hz, element_id, latitude_rad, iau_order)
    }
}
