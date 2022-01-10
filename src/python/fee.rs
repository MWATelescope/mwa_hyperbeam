// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Python interface to hyperbeam FEE code.

use numpy::*;
use pyo3::create_exception;
use pyo3::prelude::*;

use crate::fee::{FEEBeam as FEEBeamRust, FEEBeamError, InitFEEBeamError};

#[cfg(feature = "cuda")]
use marlu::ndarray::prelude::*;

// Add a python exception for hyperbeam.
create_exception!(mwa_hyperbeam, HyperbeamError, pyo3::exceptions::PyException);
impl std::convert::From<FEEBeamError> for PyErr {
    fn from(err: FEEBeamError) -> PyErr {
        HyperbeamError::new_err(err.to_string())
    }
}
impl std::convert::From<InitFEEBeamError> for PyErr {
    fn from(err: InitFEEBeamError) -> PyErr {
        HyperbeamError::new_err(err.to_string())
    }
}

/// A Python class interfacing with the hyperbeam code written in Rust.
#[pyclass]
#[pyo3(text_signature = "(hdf5_file)")]
#[allow(clippy::upper_case_acronyms)]
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

    /// Calculate the Jones matrix for a single direction given a pointing.
    /// `delays` must have 16 ints, and `amps` must have either 16 or 32 floats.
    /// If there are 16, then the dipole gains apply to both X and Y elements of
    /// dipoles. If there are 32, the first 16 amps are for the X elements, the
    /// next 16 the Y elements.
    #[pyo3(text_signature = "(az_rad, za_rad, freq_hz, delays, amps, norm_to_zenith, parallactic)")]
    #[allow(clippy::too_many_arguments)]
    fn calc_jones(
        &mut self,
        az_rad: f64,
        za_rad: f64,
        freq_hz: f64,
        delays: [u32; 16],
        amps: Vec<f64>,
        norm_to_zenith: bool,
        parallactic: bool,
    ) -> PyResult<Py<PyArray1<numpy::c64>>> {
        let jones = if parallactic {
            self.beam.calc_jones(
                az_rad,
                za_rad,
                // hyperbeam expects an int for the frequency. By specifying
                // that Python should pass in a float, it also allows an int to
                // be passed in (!?). Convert the float here in Rust for usage
                // in hyperbeam.
                freq_hz.round() as _,
                &delays,
                &amps,
                norm_to_zenith,
            )
        } else {
            self.beam.calc_jones_eng(
                az_rad,
                za_rad,
                freq_hz.round() as _,
                &delays,
                &amps,
                norm_to_zenith,
            )
        }?;
        // Ensure that the numpy crate's c64 is being used.
        let jones_py: Vec<numpy::c64> = jones.iter().map(|c| numpy::c64::new(c.re, c.im)).collect();

        let gil = pyo3::Python::acquire_gil();
        let np_array = PyArray1::from_vec(gil.python(), jones_py).to_owned();
        Ok(np_array)
    }

    /// Calculate the Jones matrices for multiple directions given a pointing.
    /// Each direction is calculated in parallel by Rust. The number of parallel
    /// threads used can be controlled by setting RAYON_NUM_THREADS. `delays`
    /// must have 16 ints, and `amps` must have 16 or 32 floats.
    #[pyo3(text_signature = "(az_rad, za_rad, freq_hz, delays, amps, norm_to_zenith, parallactic)")]
    #[allow(clippy::too_many_arguments)]
    fn calc_jones_array(
        &mut self,
        az_rad: Vec<f64>,
        za_rad: Vec<f64>,
        freq_hz: f64,
        delays: [u32; 16],
        amps: Vec<f64>,
        norm_to_zenith: bool,
        parallactic: bool,
    ) -> PyResult<Py<PyArray2<numpy::c64>>> {
        let jones = if parallactic {
            self.beam.calc_jones_array(
                &az_rad,
                &za_rad,
                freq_hz.round() as _,
                &delays,
                &amps,
                norm_to_zenith,
            )
        } else {
            self.beam.calc_jones_eng_array(
                &az_rad,
                &za_rad,
                freq_hz.round() as _,
                &delays,
                &amps,
                norm_to_zenith,
            )
        }?;
        // Flatten the four-element arrays into a single vector.
        let jones: Vec<numpy::c64> = jones
            .into_iter()
            .flat_map(|j| {
                [
                    numpy::c64::new(j[0].re, j[0].im),
                    numpy::c64::new(j[1].re, j[1].im),
                    numpy::c64::new(j[2].re, j[2].im),
                    numpy::c64::new(j[3].re, j[3].im),
                ]
            })
            .collect();
        // Now populate a numpy array.
        let gil = pyo3::Python::acquire_gil();
        let np_array1 = PyArray1::from_vec(gil.python(), jones);
        // Reshape with the second dimension being each Jones matrix (as a
        // 4-element sub-array).
        let np_array2 = np_array1.reshape([np_array1.len() / 4, 4]).unwrap();

        Ok(np_array2.to_owned())
    }

    /// Get the available frequencies inside the HDF5 file.
    fn get_fee_beam_freqs(&self) -> Py<PyArray1<u32>> {
        let gil = pyo3::Python::acquire_gil();
        self.beam.get_freqs().to_pyarray(gil.python()).to_owned()
    }

    /// Given a frequency in Hz, get the closest available frequency inside the
    /// HDF5 file.
    #[pyo3(text_signature = "(freq_hz)")]
    fn closest_freq(&self, freq_hz: f64) -> u32 {
        self.beam.find_closest_freq(freq_hz.round() as _)
    }

    /// Calculate the Jones matrices for multiple directions given a pointing on
    /// a CUDA-capable device.
    ///
    /// `delays_array` and `amps_array` must have the same number of rows; these
    /// correspond to tile configurations (i.e. each tile is allowed to have
    /// distinct delays and amps). `delays_array` must have 16 elements per row,
    /// but `amps_array` can have 16 or 32 elements per row (see `calc_jones`
    /// for an explanation).
    #[cfg(feature = "cuda")]
    #[pyo3(
        text_signature = "(az_rad, za_rad, freq_hz, delays_array, amps_array, norm_to_zenith, parallactic)"
    )]
    #[allow(clippy::too_many_arguments)]
    fn calc_jones_cuda(
        &mut self,
        az_rad: Vec<f64>,
        za_rad: Vec<f64>,
        freqs_hz: Vec<f64>,
        delays_array: Vec<u32>,
        amps_array: Vec<f64>,
        norm_to_zenith: bool,
        parallactic: bool,
    ) -> PyResult<Py<PyArray4<numpy::c64>>> {
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
        // Convert the direction type to match the CUDA precision.
        let azs: Vec<_> = az_rad.into_iter().map(|f| f as _).collect();
        let zas: Vec<_> = za_rad.into_iter().map(|f| f as _).collect();

        let cuda_beam = unsafe {
            self.beam
                .cuda_prepare(&freqs, delays.view(), amps.view(), norm_to_zenith)?
        };
        let jones = cuda_beam.calc_jones(&azs, &zas, parallactic)?;

        // Convert the Rust Jones type into numpy::c64.
        let d = jones.dim();
        let jones: Vec<numpy::c64> = jones
            .into_iter()
            .flat_map(|j| {
                [
                    numpy::c64::new(j[0].re as _, j[0].im as _),
                    numpy::c64::new(j[1].re as _, j[1].im as _),
                    numpy::c64::new(j[2].re as _, j[2].im as _),
                    numpy::c64::new(j[3].re as _, j[3].im as _),
                ]
            })
            .collect();
        // Now populate a numpy array.
        let gil = pyo3::Python::acquire_gil();
        let np_array1 = PyArray1::from_vec(gil.python(), jones);
        // Reshape with the fourth dimension being each Jones matrix (as a
        // 4-element sub-array).
        let np_array4 = np_array1
            .reshape([
                np_array1.len() / (d.1 * d.2 * 4),
                np_array1.len() / (d.0 * d.2 * 4),
                np_array1.len() / (d.0 * d.1 * 4),
                4,
            ])
            .unwrap();

        Ok(np_array4.to_owned())
    }
}

/// A Python module interfacing with the hyperbeam code written in Rust. This
/// module depends on and will import numpy.
#[pymodule]
fn mwa_hyperbeam(py: Python, m: &PyModule) -> PyResult<()> {
    py.import("numpy")?;
    m.add_class::<FEEBeam>()?;
    m.add("HyperbeamError", py.get_type::<HyperbeamError>())?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    Ok(())
}
