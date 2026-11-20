// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Python interface to hyperbeam's SKA-Low beam models.

use std::{borrow::Cow, path::PathBuf};

use self::ndarray::prelude::*;
use marlu::Jones;
use num_complex::Complex64 as c64;
use numpy::*;
use pyo3::prelude::*;

use crate::ska::{beam::SkaLowBeamInner, SkaElementResponse, SkaLowBeam, SkaStation};
#[cfg(any(feature = "cuda", feature = "hip"))]
use crate::{
    ska::{SkaGpuTiles, SkaLowBeamGpu},
    GpuComplex, GpuFloat,
};

/// Borrow a 1D array as a slice, copying only if it isn't contiguous.
fn contiguous<'a>(a: &'a ArrayView1<'_, f64>) -> Cow<'a, [f64]> {
    match a.as_slice() {
        Some(s) => Cow::Borrowed(s),
        None => Cow::Owned(a.to_vec()),
    }
}

fn jones_to_py<'py>(py: Python<'py>, j: Jones<f64>) -> Bound<'py, PyArray1<c64>> {
    PyArray1::from_vec(py, j.to_vec())
}

fn jones_vec_to_py<'py>(py: Python<'py>, js: Vec<Jones<f64>>) -> Bound<'py, PyArray2<c64>> {
    let arr = Array2::from_shape_fn((js.len(), 4), |(i, j)| js[i][j]);
    arr.into_pyarray(py)
}

/// Generate a Python class for an SKA element model. All classes share the
/// same calculation methods; the constructor is supplied per class.
macro_rules! ska_py_class {
    ($(#[$meta:meta])* $name:ident, { $($ctor:tt)* }) => {
        $(#[$meta])*
        #[pyclass]
        pub(super) struct $name {
            beam: SkaLowBeam,
        }

        #[pymethods]
        impl $name {
            $($ctor)*

            /// Calculate the element-response Jones matrix for a direction
            /// (azimuth North through East, zenith angle; radians). The
            /// result has 4 elements: [X_theta, X_az, Y_theta, Y_az], unless
            /// `latitude_rad` is given, in which case the parallactic-angle
            /// correction is applied (with `iau_order` controlling the order),
            /// exactly as for `FEEBeam`. `element_id` selects the element
            /// for models with per-element patterns (ignored otherwise).
            #[pyo3(signature = (az_rad, za_rad, freq_hz, element_id=0, latitude_rad=None, iau_order=None))]
            #[allow(clippy::too_many_arguments)]
            fn calc_jones<'py>(
                &self,
                py: Python<'py>,
                az_rad: f64,
                za_rad: f64,
                freq_hz: f64,
                element_id: usize,
                latitude_rad: Option<f64>,
                iau_order: Option<bool>,
            ) -> PyResult<Bound<'py, PyArray1<c64>>> {
                let j = self.beam.calc_jones_pair(
                    az_rad,
                    za_rad,
                    freq_hz,
                    element_id,
                    latitude_rad,
                    iau_order.unwrap_or(false),
                )?;
                Ok(jones_to_py(py, j))
            }

            /// Calculate element-response Jones matrices for many directions
            /// in parallel. Returns an array with shape (N, 4). See
            /// `calc_jones`.
            #[pyo3(signature = (az_rad, za_rad, freq_hz, element_id=0, latitude_rad=None, iau_order=None))]
            #[allow(clippy::too_many_arguments)]
            fn calc_jones_array<'py>(
                &self,
                py: Python<'py>,
                az_rad: PyArrayLike1<'py, f64, AllowTypeChange>,
                za_rad: PyArrayLike1<'py, f64, AllowTypeChange>,
                freq_hz: f64,
                element_id: usize,
                latitude_rad: Option<f64>,
                iau_order: Option<bool>,
            ) -> PyResult<Bound<'py, PyArray2<c64>>> {
                let (az_rad, za_rad) = (az_rad.as_array(), za_rad.as_array());
                let (az_rad, za_rad) = (contiguous(&az_rad), contiguous(&za_rad));
                if az_rad.len() != za_rad.len() {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "az_rad and za_rad must have the same length",
                    ));
                }
                let js = py.detach(|| {
                    self.beam.calc_jones_array_pair(
                        &az_rad,
                        &za_rad,
                        freq_hz,
                        element_id,
                        latitude_rad,
                        iau_order.unwrap_or(false),
                    )
                })?;
                Ok(jones_vec_to_py(py, js))
            }

            /// The raw element response in EveryBeam's convention, i.e.
            /// exactly what EveryBeam's `ElementResponse.response(element_id,
            /// freq, theta, phi)` gives: [X_theta, X_phi, Y_theta, Y_phi],
            /// with `theta_rad` the zenith angle and `phi_rad` measured from
            /// East through North.
            #[pyo3(signature = (theta_rad, phi_rad, freq_hz, element_id=0))]
            fn element_response<'py>(
                &self,
                py: Python<'py>,
                theta_rad: f64,
                phi_rad: f64,
                freq_hz: f64,
                element_id: usize,
            ) -> PyResult<Bound<'py, PyArray1<c64>>> {
                let j = self.beam.element_response(element_id, freq_hz, theta_rad, phi_rad)?;
                Ok(jones_to_py(py, j))
            }

            /// Calculate station-beam Jones matrices for many directions,
            /// given a pointing and the station's element positions
            /// (`positions_enu`, shape (N_elements, 3), East/North/Up in
            /// metres). Optional `weights` (one per element; 0 flags an
            /// element) default to 1. `rotation_rad` is the station's
            /// rotation about the local vertical: the angle of its X dipoles
            /// from East towards North (SKA-Low stations are rigidly rotated;
            /// see the COORDINATE_AXES of the MS PHASED_ARRAY table). For
            /// models with per-element patterns, element i uses element_id i.
            /// Returns shape (N, 4); see `calc_jones` for the output
            /// convention.
            #[pyo3(signature = (az_rad, za_rad, freq_hz, pointing_az_rad, pointing_za_rad, positions_enu, weights=None, rotation_rad=0.0, latitude_rad=None, iau_order=None))]
            #[allow(clippy::too_many_arguments)]
            fn calc_station_jones_array<'py>(
                &self,
                py: Python<'py>,
                az_rad: PyArrayLike1<'py, f64, AllowTypeChange>,
                za_rad: PyArrayLike1<'py, f64, AllowTypeChange>,
                freq_hz: f64,
                pointing_az_rad: f64,
                pointing_za_rad: f64,
                positions_enu: PyReadonlyArray2<f64>,
                weights: Option<Vec<f64>>,
                rotation_rad: f64,
                latitude_rad: Option<f64>,
                iau_order: Option<bool>,
            ) -> PyResult<Bound<'py, PyArray2<c64>>> {
                let (az_rad, za_rad) = (az_rad.as_array(), za_rad.as_array());
                let (az_rad, za_rad) = (contiguous(&az_rad), contiguous(&za_rad));
                if az_rad.len() != za_rad.len() {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "az_rad and za_rad must have the same length",
                    ));
                }
                let positions = positions_enu.as_array();
                if positions.ncols() != 3 {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "positions_enu must have shape (N, 3)",
                    ));
                }
                let positions: Vec<[f64; 3]> = positions
                    .outer_iter()
                    .map(|r| [r[0], r[1], r[2]])
                    .collect();
                let station = SkaStation::new(&self.beam, positions, weights)?.with_rotation(rotation_rad);
                let js = py.detach(|| {
                    station.calc_jones_array_pair(
                        &az_rad,
                        &za_rad,
                        freq_hz,
                        pointing_az_rad,
                        pointing_za_rad,
                        latitude_rad,
                        iau_order.unwrap_or(false),
                    )
                })?;
                Ok(jones_vec_to_py(py, js))
            }

            /// Prepare the GPU for beam-response calculations at the given
            /// frequencies [Hz], returning an `SkaLowGpu` object. Each "tile"
            /// is either a single element (the default; `element_ids` gives
            /// the element of each tile for models with per-element
            /// patterns, else one tile of element 0), or, if `positions_enu`
            /// is given (shape (N_stations, N_elements, 3), East/North/Up in
            /// metres), a station. Station `weights` (shape (N_stations,
            /// N_elements)) default to 1, and `rotations_rad` (one per
            /// station) to 0.
            #[cfg(any(feature = "cuda", feature = "hip"))]
            #[pyo3(signature = (freqs_hz, element_ids=None, positions_enu=None, weights=None, rotations_rad=None))]
            fn gpu_prepare(
                &self,
                freqs_hz: Vec<f64>,
                element_ids: Option<Vec<usize>>,
                positions_enu: Option<PyReadonlyArray3<f64>>,
                weights: Option<PyReadonlyArray2<f64>>,
                rotations_rad: Option<Vec<f64>>,
            ) -> PyResult<SkaLowGpu> {
                let ids = element_ids.unwrap_or_else(|| vec![0]);
                let positions = positions_enu.as_ref().map(|p| p.as_array());
                let weights = weights.as_ref().map(|w| w.as_array());
                let tiles = match positions {
                    Some(p) => {
                        if p.len_of(Axis(2)) != 3 {
                            return Err(pyo3::exceptions::PyValueError::new_err(
                                "positions_enu must have shape (N_stations, N_elements, 3)",
                            ));
                        }
                        SkaGpuTiles::Stations {
                            positions_enu: p,
                            weights,
                            rotations_rad: rotations_rad.as_deref(),
                            weights_y: None,
                            local_frames: None,
                        }
                    }
                    None => SkaGpuTiles::Elements(&ids),
                };
                let gpu = unsafe { self.beam.gpu_prepare(&freqs_hz, tiles) }?;
                Ok(SkaLowGpu { gpu })
            }
        }
    };
}

/// A GPU SKA-Low beam, prepared with `gpu_prepare` on one of the SKA-Low
/// beam classes.
#[cfg(any(feature = "cuda", feature = "hip"))]
#[pyclass]
pub(super) struct SkaLowGpu {
    gpu: SkaLowBeamGpu,
}

#[cfg(any(feature = "cuda", feature = "hip"))]
#[pymethods]
impl SkaLowGpu {
    /// Calculate beam-response Jones matrices on the GPU for many directions
    /// (azimuth North through East, zenith angle; radians). Returns an array
    /// with shape (N_tiles, N_freqs, N_directions, 4), in the same convention
    /// as `calc_jones`. The pointing is only used for stations. The complex
    /// type is single or double precision, depending on how hyperbeam was
    /// compiled.
    #[pyo3(signature = (az_rad, za_rad, pointing_az_rad=0.0, pointing_za_rad=0.0, latitude_rad=None, iau_order=None))]
    #[allow(clippy::too_many_arguments)]
    fn calc_jones<'py>(
        &self,
        py: Python<'py>,
        az_rad: PyArrayLike1<'py, f64, AllowTypeChange>,
        za_rad: PyArrayLike1<'py, f64, AllowTypeChange>,
        pointing_az_rad: f64,
        pointing_za_rad: f64,
        latitude_rad: Option<f64>,
        iau_order: Option<bool>,
    ) -> PyResult<Bound<'py, PyArray4<GpuComplex>>> {
        let az: Vec<GpuFloat> = az_rad.as_array().iter().map(|&x| x as GpuFloat).collect();
        let za: Vec<GpuFloat> = za_rad.as_array().iter().map(|&x| x as GpuFloat).collect();
        if az.len() != za.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "az_rad and za_rad must have the same length",
            ));
        }
        let pointing = marlu::AzEl::from_radians(
            pointing_az_rad,
            std::f64::consts::FRAC_PI_2 - pointing_za_rad,
        );
        let results = py.detach(|| {
            self.gpu
                .calc_jones_pair(&az, &za, pointing, latitude_rad, iau_order.unwrap_or(false))
        })?;
        let (nt, nf, nd) = results.dim();
        let flat: Vec<GpuComplex> = results.iter().flat_map(|j| j.to_complex_array()).collect();
        let arr = Array4::from_shape_vec((nt, nf, nd, 4), flat).expect("correct shape");
        Ok(arr.into_pyarray(py))
    }

    /// The number of tiles (elements or stations).
    fn get_num_tiles(&self) -> i32 {
        self.gpu.get_num_tiles()
    }

    /// The number of frequencies.
    fn get_num_freqs(&self) -> i32 {
        self.gpu.get_num_freqs()
    }
}

ska_py_class!(
    /// The SKA-Low analytic element model ("ska-low-analytic"): a
    /// half-wavelength dipole with a cos(theta) taper. Equivalent to
    /// EveryBeam's `oskar_dipole_cos` model (or `oskar_dipole` with
    /// `cos_taper=False`).
    SkaLowAnalytic,
    {
        #[new]
        #[pyo3(signature = (cos_taper=true))]
        fn new(cos_taper: bool) -> Self {
            Self {
                beam: SkaLowBeam::new_analytic(cos_taper),
            }
        }
    }
);

ska_py_class!(
    /// The SKA-Low gridded, station-averaged embedded-element-pattern model
    /// ("ska-low-eep-grid"). Equivalent to EveryBeam's `gridded_eep` model. If
    /// `hdf5_file` is not given, the `SKA_LOW_EEP_GRID_FILE` environment
    /// variable is used, or failing that `$EVERYBEAM_DATADIR/oskar_eep_grid.h5`.
    /// `ska_low_convention` (default True) applies EveryBeam's OSKAR/SKA-Low
    /// conjugation and negation.
    SkaLowEepGrid,
    {
        #[new]
        #[pyo3(signature = (hdf5_file=None, ska_low_convention=true))]
        fn new(hdf5_file: Option<PathBuf>, ska_low_convention: bool) -> PyResult<Self> {
            if hdf5_file.is_none() && !ska_low_convention {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "ska_low_convention=False requires an explicit hdf5_file",
                ));
            }
            Ok(Self {
                beam: SkaLowBeam::new_eep_grid(hdf5_file, ska_low_convention)?,
            })
        }

        /// The frequencies of the grid [Hz].
        fn get_freqs(&self) -> Vec<f64> {
            match &self.beam.inner {
                SkaLowBeamInner::EepGrid(b) => b.get_freqs().to_vec(),
                _ => unreachable!(),
            }
        }
    }
);

ska_py_class!(
    /// The SKA-Low spherical-wave embedded-element-pattern model
    /// ("ska-low-eep-sph"), with a distinct pattern per element. Equivalent to
    /// EveryBeam's `skalow_feko` model. `coeff_dir` is a directory of FEKO
    /// coefficient HDF5 files (one per frequency); if not given, the
    /// `SKA_LOW_EEP_SPH_PATH` environment variable is used, or failing that
    /// `$EVERYBEAM_DATADIR/skalow`. `l_max` optionally truncates the
    /// spherical-wave expansion.
    SkaLowEepSph,
    {
        #[new]
        #[pyo3(signature = (coeff_dir=None, l_max=None))]
        fn new(coeff_dir: Option<PathBuf>, l_max: Option<usize>) -> PyResult<Self> {
            Ok(Self {
                beam: SkaLowBeam::new_eep_sph(coeff_dir, l_max)?,
            })
        }

        /// The frequencies of the available coefficient files [Hz].
        fn get_freqs(&self) -> Vec<f64> {
            match &self.beam.inner {
                SkaLowBeamInner::EepSph(b) => b.get_freqs(),
                _ => unreachable!(),
            }
        }

        /// The number of elements (with distinct patterns).
        fn get_num_elements(&self) -> usize {
            self.beam.num_elements().unwrap_or(0)
        }

        /// The frequency of the coefficients used for `freq_hz` [Hz].
        fn find_closest_freq(&self, freq_hz: f64) -> f64 {
            match &self.beam.inner {
                SkaLowBeamInner::EepSph(b) => b.find_closest_freq(freq_hz),
                _ => unreachable!(),
            }
        }
    }
);
