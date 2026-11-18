// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Errors associated with the SKA element and station beams.

use thiserror::Error;

/// Errors that can occur when creating an SKA beam object (e.g. while reading
/// data files).
#[derive(Error, Debug)]
pub enum InitSkaBeamError {
    #[error("Specified file or directory '{0}' doesn't exist")]
    PathDoesntExist(String),

    #[error("No FEKO coefficient files ('*.h5') were found in '{0}'")]
    NoFekoFiles(String),

    #[error("Couldn't determine the frequency of FEKO coefficient file '{0}' (no 'frequency_in_Hz' attribute, and the file name doesn't encode one)")]
    FekoFrequency(String),

    #[error("Two FEKO coefficient files have the same frequency ({freq_hz} Hz): '{file1}' and '{file2}'")]
    DuplicateFekoFrequency {
        freq_hz: f64,
        file1: String,
        file2: String,
    },

    #[error("HDF5 dataset '{dataset}' in '{file}' has an unexpected shape {got:?}; {expected}")]
    BadShape {
        file: String,
        dataset: String,
        got: Vec<usize>,
        expected: String,
    },

    #[error("Values in HDF5 dataset '{dataset}' of '{file}' are not sorted in ascending order")]
    NotSorted { file: String, dataset: String },

    #[error("No EEP grid file was given, and the {var} environment variable is not set")]
    EnvVar { var: &'static str },

    #[error("HDF5 error: {0}")]
    Hdf5(#[from] hdf5_metno::Error),
}

/// Errors that can occur when calculating SKA beam responses.
#[derive(Error, Debug)]
pub enum SkaBeamError {
    #[error(
        "Requested {what} value {value} is outside the range of the EEP grid ({min} to {max})"
    )]
    OutOfGrid {
        what: &'static str,
        value: f64,
        min: f64,
        max: f64,
    },

    #[error("Requested element {got}, but only {num_elements} elements are available")]
    ElementOutOfRange { got: usize, num_elements: usize },

    #[error("This beam model has a different response for each element, so a station beam requires exactly one position per element ({expected}); got {got} positions")]
    PositionsLength { got: usize, expected: usize },

    #[error("The number of element weights ({got}) doesn't match the number of element positions ({expected})")]
    WeightsLength { got: usize, expected: usize },

    #[error("All station elements are flagged (have zero weight)")]
    AllElementsFlagged,

    #[error("HDF5 error: {0}")]
    Hdf5(#[from] hdf5_metno::Error),

    #[cfg(any(feature = "cuda", feature = "hip"))]
    #[error(
        "The number of station rotations ({got}) doesn't match the number of stations ({expected})"
    )]
    RotationsLength { got: usize, expected: usize },

    #[cfg(any(feature = "cuda", feature = "hip"))]
    #[error("The GPU code supports spherical-wave degrees up to {max}, but {got} is needed; use a smaller l_max")]
    GpuLMax { got: usize, max: usize },

    #[cfg(any(feature = "cuda", feature = "hip"))]
    #[error("The number of elements differs between the coefficient files used for the requested frequencies")]
    GpuInconsistentElements,

    #[cfg(any(feature = "cuda", feature = "hip"))]
    #[error(transparent)]
    Gpu(#[from] crate::gpu::GpuError),
}
