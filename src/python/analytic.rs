// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Python interface to hyperbeam analytic beam code.

use marlu::c64;
use ndarray::prelude::*;
use numpy::*;
use pyo3::prelude::*;

use crate::analytic::{AnalyticBeam as AnalyticBeamRust, AnalyticType};
#[cfg(any(feature = "cuda", feature = "hip"))]
use crate::{GpuComplex, GpuFloat};

/// A Python class interfacing with the hyperbeam analytic beam code written in
/// Rust.
#[pyclass]
pub(super) struct AnalyticBeam {
    beam: AnalyticBeamRust,
}

#[pymethods]
impl AnalyticBeam {
    /// Create a new `AnalyticBeam` object. This object is used for all beam
    /// calculations. Here, one can opt into RTS behaviour and/or control the
    /// dipole height (which also differs between mwa_pb and the RTS).
    #[new]
    #[pyo3(text_signature = "(rts_behaviour, dipole_height)")]
    fn new(rts_behaviour: Option<bool>, dipole_height: Option<f64>) -> AnalyticBeam {
        let beam_type = if let Some(true) = rts_behaviour {
            AnalyticType::Rts
        } else {
            AnalyticType::MwaPb
        };
        AnalyticBeam {
            beam: AnalyticBeamRust::new_custom(
                beam_type,
                dipole_height.unwrap_or(beam_type.get_default_dipole_height()),
            ),
        }
    }

    /// Calculate the Jones matrix for a single direction given a pointing.
    /// `delays` must have 16 ints, and `amps` must have either 16 or 32 floats.
    /// If there are 16, then the dipole gains apply to both X and Y elements of
    /// dipoles. If there are 32, the first 16 amps are for the X elements, the
    /// next 16 the Y elements.
    #[pyo3(
        text_signature = "(az_rad, za_rad, freq_hz, delays, amps, latitude_rad, norm_to_zenith)"
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
        latitude_rad: f64,
        norm_to_zenith: Option<bool>,
    ) -> PyResult<&'py PyArray1<c64>> {
        let jones = self.beam.calc_jones_pair(
            az_rad,
            za_rad,
            // hyperbeam expects an int for the frequency. By specifying that
            // Python should pass in a float, it also allows an int to be passed
            // in (!?). Convert the float here in Rust for usage in hyperbeam.
            freq_hz.round() as _,
            &delays,
            &amps,
            latitude_rad,
            norm_to_zenith.unwrap_or(false),
        )?;
        let jones_py: Vec<c64> = jones.iter().map(|c| c64::new(c.re, c.im)).collect();
        let np_array = PyArray1::from_vec(py, jones_py);
        Ok(np_array)
    }

    /// Calculate the Jones matrices for multiple directions given a pointing.
    /// Each direction is calculated in parallel by Rust. The number of parallel
    /// threads used can be controlled by setting RAYON_NUM_THREADS. `delays`
    /// must have 16 ints, and `amps` must have 16 or 32 floats.
    #[pyo3(
        text_signature = "(az_rad, za_rad, freq_hz, delays, amps, latitude_rad, norm_to_zenith)"
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
        latitude_rad: f64,
        norm_to_zenith: Option<bool>,
    ) -> PyResult<&'py PyArray2<c64>> {
        let jones = self.beam.calc_jones_array_pair(
            &az_rad,
            &za_rad,
            freq_hz.round() as _,
            &delays,
            &amps,
            latitude_rad,
            norm_to_zenith.unwrap_or(false),
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
        Ok(a2.into_pyarray(py))
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
        text_signature = "(az_rad, za_rad, freqs_hz, delays_array, amps_array, latitude_rad, norm_to_zenith)"
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
        latitude_rad: f64,
        norm_to_zenith: Option<bool>,
    ) -> PyResult<&'py PyArray4<GpuComplex>> {
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

        let gpu_beam = unsafe { self.beam.gpu_prepare(delays.view(), amps.view())? };
        let jones = gpu_beam.calc_jones_pair(
            &azs,
            &zas,
            &freqs,
            latitude_rad as GpuFloat,
            norm_to_zenith.unwrap_or(false),
        )?;

        // Convert to a 4D array of Complex from Jones.
        // Use unsafe code to ensure that no useless copying is done!
        // https://users.rust-lang.org/t/sound-conversion-from-vec-num-complex-complex64-4-to-ndarray-array2-num-complex-complex64-without-copying/78973/2
        let old_dim = jones.dim();
        let mut jones = std::mem::ManuallyDrop::new(jones.into_raw_vec());

        let new_len = jones.len() * 4;
        let new_cap = jones.capacity() * 4;
        let new_ptr = jones.as_mut_ptr() as *mut GpuComplex;
        // SAFETY: new_cap == old_cap * N, align_of::<Complex>() == align_of::<Jones>()
        let flat = unsafe { Vec::from_raw_parts(new_ptr, new_len, new_cap) };
        let a4 = Array4::from_shape_vec((old_dim.0, old_dim.1, old_dim.2, 4), flat).unwrap();
        Ok(a4.into_pyarray(py))
    }
}
