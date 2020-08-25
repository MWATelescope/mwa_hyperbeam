// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

/*!
Python interface to hyperbeam via pyo3.
 */

use ndarray::Array2;
use numpy::*;
use pyo3::create_exception;
use pyo3::prelude::*;

use crate::FEEBeam as FEEBeamRust;
use crate::FEEBeamError;

// Add a python exception for hyperbeam.
create_exception!(mwa_hyperbeam, HyperbeamError, pyo3::exceptions::PyException);
impl std::convert::From<FEEBeamError> for PyErr {
    fn from(err: FEEBeamError) -> PyErr {
        HyperbeamError::new_err(err.to_string())
    }
}

/// A Python class interfacing with the hyperbeam code written in Rust.
#[pyclass]
#[text_signature = "(hdf5_file)"]
struct FEEBeam {
    beam: FEEBeamRust,
}

#[pymethods]
impl FEEBeam {
    /// Create a new `FEEBeam` object. This object is used for all beam
    /// calculations. If the path to the beam HDF5 file is not given, then the
    /// MWA_BEAM_FILE environment variable is used.
    #[new]
    fn new(hdf5_file: Option<PyObject>) -> PyResult<Self> {
        let strct = match hdf5_file {
            Some(f) => FEEBeamRust::new(f.to_string())?,
            None => FEEBeamRust::new_from_env()?,
        };
        Ok(FEEBeam { beam: strct })
    }

    /// Calculate the Jones matrix for a single pointing. `delays` must have 16
    /// ints, and `amps` must have 16 floats.
    #[text_signature = "(az_rad, za_rad, freq_hz, delays, amps, norm_to_zenith)"]
    fn calc_jones(
        &mut self,
        az_rad: f64,
        za_rad: f64,
        freq_hz: u32,
        delays: [u32; 16],
        amps: [f64; 16],
        norm_to_zenith: bool,
    ) -> PyResult<Py<PyArray1<numpy::c64>>> {
        let jones = self
            .beam
            .calc_jones(az_rad, za_rad, freq_hz, &delays, &amps, norm_to_zenith)?
            .to_vec();
        let numpy_jones =
            unsafe { std::mem::transmute::<Vec<num::complex::Complex64>, Vec<numpy::c64>>(jones) };

        let gil = pyo3::Python::acquire_gil();
        let np_array = PyArray1::from_vec(gil.python(), numpy_jones).to_owned();
        Ok(np_array)
    }

    /// Calculate the Jones matrices for multiple pointings. Each pointing is
    /// calculated in parallel by Rust. The number of parallel threads used can
    /// be controlled by setting RAYON_NUM_THREADS. `delays` must have 16 ints,
    /// and `amps` must have 16 floats.
    #[text_signature = "(az_rad, za_rad, freq_hz, delays, amps, norm_to_zenith)"]
    fn calc_jones_array(
        &mut self,
        az_rad: Vec<f64>,
        za_rad: Vec<f64>,
        freq_hz: u32,
        delays: [u32; 16],
        amps: [f64; 16],
        norm_to_zenith: bool,
    ) -> PyResult<Py<PyArray2<numpy::c64>>> {
        let jones = self.beam.calc_jones_array(
            &az_rad,
            &za_rad,
            freq_hz,
            &delays,
            &amps,
            norm_to_zenith,
        )?;
        let numpy_jones = unsafe {
            std::mem::transmute::<Array2<num::complex::Complex64>, Array2<numpy::c64>>(jones)
        };

        let gil = pyo3::Python::acquire_gil();
        let np_array = PyArray2::from_owned_array(gil.python(), numpy_jones).to_owned();
        Ok(np_array)
    }

    /// Get the available frequencies inside the HDF5 file.
    fn get_fee_beam_freqs(&self) -> Py<PyArray1<u32>> {
        let gil = pyo3::Python::acquire_gil();
        self.beam.freqs.to_pyarray(gil.python()).to_owned()
    }

    /// Given a frequency in Hz, get the closest available frequency inside the
    /// HDF5 file.
    #[text_signature = "(freq_hz)"]
    fn closest_freq(&self, freq: u32) -> u32 {
        self.beam.find_closest_freq(freq)
    }
}

/// A Python module interfacing with the hyperbeam code written in Rust. This
/// module depends on and will import numpy.
#[pymodule]
fn mwa_hyperbeam(py: Python, m: &PyModule) -> PyResult<()> {
    py.import("numpy")?;
    m.add_class::<FEEBeam>()?;
    m.add("HyperbeamError", py.get_type::<HyperbeamError>())?;

    Ok(())
}
