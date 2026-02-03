// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Python interface to hyperbeam FEE beam code.

use std::path::PathBuf;

use self::ndarray::prelude::*;
use num_complex::Complex64 as c64;
use numpy::*;
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::fee::FEEBeam as FEEBeamRust;
#[cfg(any(feature = "cuda", feature = "hip"))]
use crate::GpuComplex;

/// A Python class interfacing with the hyperbeam FEE beam code written in Rust.
#[pyclass]
#[allow(clippy::upper_case_acronyms)]
pub(super) struct FEEBeam {
    beam: FEEBeamRust,
}

#[pymethods]
impl FEEBeam {
    /// Create a new `FEEBeam` object. This object is used for all beam
    /// calculations. If the path to the beam HDF5 file is not given, then the
    /// `MWA_BEAM_FILE` environment variable is used.
    #[new]
    #[pyo3(signature = (hdf5_file))]
    fn new(hdf5_file: Option<PyObject>) -> PyResult<Self> {
        let strct = match hdf5_file {
            Some(f) => {
                let result = Python::with_gil(|py| {
                    let f: PathBuf = f
                        .extract(py)
                        .expect("can convert python string to rust path");
                    FEEBeamRust::new(f)
                });
                result?
            }
            None => FEEBeamRust::new_from_env()?,
        };
        Ok(FEEBeam { beam: strct })
    }

    /// Calculate the beam-response Jones matrix for a given direction and
    /// pointing. If `latitude_rad` is *not* supplied, the result will match
    /// the original specification of the FEE beam code (possibly more useful
    /// for engineers).
    ///
    /// Astronomers are more likely to want to specify `latitude_rad` (which
    /// will apply the parallactic-angle correction using the Earth latitude
    /// provided for the telescope) and `iau_order`. If `latitude_rad` is not
    /// given, then `iau_reorder` does nothing. See this document for more
    /// information:
    /// <https://github.com/MWATelescope/mwa_hyperbeam/blob/main/fee_pols.pdf>
    ///
    /// `delays` and `amps` apply to each dipole in an MWA tile in the M&C
    /// order; see
    /// <https://wiki.mwatelescope.org/pages/viewpage.action?pageId=48005139>.
    /// `delays` *must* have 16 elements, whereas `amps` can have 16 or 32
    /// elements; if 16 are given, then these map 1:1 with dipoles, otherwise
    /// the first 16 are for X dipole elements, and the next 16 are for Y.
    #[pyo3(
        signature = (az_rad, za_rad, freq_hz, delays, amps, norm_to_zenith, latitude_rad=None, iau_order=None)
    )]
    #[allow(clippy::too_many_arguments)]
    fn calc_jones<'py>(
        &self,
        py: Python<'py>,
        az_rad: f64,
        za_rad: f64,
        freq_hz: f64,
        delays: [u32; 16],
        amps: Vec<f64>,
        norm_to_zenith: bool,
        latitude_rad: Option<f64>,
        iau_order: Option<bool>,
    ) -> PyResult<Bound<'py, PyArray1<c64>>> {
        let jones = self.beam.calc_jones(
            (az_rad, za_rad),
            // hyperbeam expects an int for the frequency. By specifying that
            // Python should pass in a float, it also allows an int to be passed
            // in (!?). Convert the float here in Rust for usage in hyperbeam.
            freq_hz.round() as _,
            &delays,
            &amps,
            norm_to_zenith,
            latitude_rad,
            iau_order.unwrap_or(false),
        )?;
        let jones_py: Vec<c64> = jones.iter().map(|c| c64::new(c.re, c.im)).collect();
        let np_array = PyArray1::from_vec_bound(py, jones_py);
        Ok(np_array)
    }

    /// Calculate the beam-response Jones matrices for many directions given a
    /// pointing. This is basically a wrapper around `calc_jones` that
    /// efficiently calculates the Jones matrices in parallel. The number of
    /// parallel threads used can be controlled by setting `RAYON_NUM_THREADS`
    ///
    /// `delays` and `amps` apply to each dipole in an MWA tile in the M&C
    /// order; see
    /// <https://wiki.mwatelescope.org/pages/viewpage.action?pageId=48005139>.
    /// `delays` *must* have 16 elements, whereas `amps` can have 16 or 32
    /// elements; if 16 are given, then these map 1:1 with dipoles, otherwise
    /// the first 16 are for X dipole elements, and the next 16 are for Y.
    #[pyo3(
        signature = (az_rad, za_rad, freq_hz, delays, amps, norm_to_zenith, latitude_rad=None, iau_order=None)
    )]
    #[allow(clippy::too_many_arguments)]
    fn calc_jones_array<'py>(
        &self,
        py: Python<'py>,
        az_rad: Vec<f64>,
        za_rad: Vec<f64>,
        freq_hz: f64,
        delays: [u32; 16],
        amps: Vec<f64>,
        norm_to_zenith: bool,
        latitude_rad: Option<f64>,
        iau_order: Option<bool>,
    ) -> PyResult<Bound<'py, PyArray2<c64>>> {
        let jones = self.beam.calc_jones_array(
            (&az_rad, &za_rad),
            freq_hz.round() as _,
            &delays,
            &amps,
            norm_to_zenith,
            latitude_rad,
            iau_order.unwrap_or(false),
        )?;

        // Convert to a 2D array of c64 from Jones (one row per beam response).
        // Use unsafe code to ensure that no useless copying is done!
        // https://users.rust-lang.org/t/sound-conversion-from-vec-num-complex-complex64-4-to-ndarray-array2-num-complex-complex64-without-copying/78973/2
        let mut jones = std::mem::ManuallyDrop::new(jones);

        let old_len = jones.len();
        let new_len = old_len * 4;
        let new_cap = jones.capacity() * 4;
        let new_ptr = jones.as_mut_ptr() as *mut c64;
        // SAFETY: new_cap == old_cap * N, align_of::<C64>() == align_of::<Jones>()
        let flat = unsafe { Vec::from_raw_parts(new_ptr, new_len, new_cap) };
        let a2 = Array2::from_shape_vec((old_len, 4), flat).unwrap();
        Ok(a2.into_pyarray_bound(py))
    }

    /// Get the available frequencies inside the HDF5 file.
    fn get_fee_beam_freqs<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        self.beam.get_freqs().to_vec().into_pyarray_bound(py)
    }

    /// Given a frequency in Hz, get the closest available frequency inside the
    /// HDF5 file.
    #[pyo3(text_signature = "(freq_hz)")]
    fn closest_freq(&self, freq_hz: f64) -> u32 {
        self.beam.find_closest_freq(freq_hz.round() as _)
    }

    /// Calculate the Jones matrices for multiple directions given a pointing
    /// and multiple frequencies on a GPU.
    ///
    /// `delays_array` and `amps_array` must have the same number of rows; these
    /// correspond to tile configurations (i.e. each tile is allowed to have
    /// distinct delays and amps). `delays_array` must have 16 elements per row,
    /// but `amps_array` can have 16 or 32 elements per row (see `calc_jones`
    /// for an explanation).
    #[cfg(any(feature = "cuda", feature = "hip"))]
    #[pyo3(
        signature = (az_rad, za_rad, freqs_hz, delays_array, amps_array, norm_to_zenith, latitude_rad=None, iau_order=None)
    )]
    #[allow(clippy::too_many_arguments)]
    fn calc_jones_gpu<'py>(
        &self,
        py: Python<'py>,
        az_rad: Vec<f64>,
        za_rad: Vec<f64>,
        freqs_hz: Vec<f64>,
        delays_array: Vec<u32>,
        amps_array: Vec<f64>,
        norm_to_zenith: bool,
        latitude_rad: Option<f64>,
        iau_order: Option<bool>,
    ) -> PyResult<Bound<'py, PyArray4<GpuComplex>>> {
        // hyperbeam expects ints for the frequencies. Convert them to make sure
        // everything's OK.
        let freqs: Vec<u32> = freqs_hz.iter().map(|&f| f.round() as _).collect();
        // We assume that there are 16 delays per row of delays, so we can get
        // the number of tiles.
        let num_tiles = delays_array.len() / 16;
        let delays = Array2::from_shape_vec((num_tiles, 16), delays_array).unwrap();
        // We then know how many amps per tile are provided.
        let amps =
            Array2::from_shape_vec((num_tiles, amps_array.len() / num_tiles), amps_array).unwrap();
        // Convert the direction type to match the GPU precision.
        let azs: Vec<_> = az_rad.into_iter().map(|f| f as _).collect();
        let zas: Vec<_> = za_rad.into_iter().map(|f| f as _).collect();

        let gpu_beam = unsafe {
            self.beam
                .gpu_prepare(&freqs, delays.view(), amps.view(), norm_to_zenith)?
        };
        let jones =
            gpu_beam.calc_jones_pair(&azs, &zas, latitude_rad, iau_order.unwrap_or(false))?;

        // Convert to a 4D array of Complex from Jones.
        // Use unsafe code to ensure that no useless copying is done!
        // https://users.rust-lang.org/t/sound-conversion-from-vec-num-complex-complex64-4-to-ndarray-array2-num-complex-complex64-without-copying/78973/2
        let old_dim = jones.dim();
        let mut jones = std::mem::ManuallyDrop::new(jones.into_raw_vec_and_offset().0);

        let new_len = jones.len() * 4;
        let new_cap = jones.capacity() * 4;
        let new_ptr = jones.as_mut_ptr() as *mut GpuComplex;
        // SAFETY: new_cap == old_cap * N, align_of::<Complex>() == align_of::<Jones>()
        let flat = unsafe { Vec::from_raw_parts(new_ptr, new_len, new_cap) };
        let a4 = Array4::from_shape_vec((old_dim.0, old_dim.1, old_dim.2, 4), flat).unwrap();
        Ok(a4.into_pyarray_bound(py))
    }
}
