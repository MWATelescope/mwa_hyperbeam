// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Errors associated with the FEE beam.

use thiserror::Error;

#[derive(Error, Debug)]
#[allow(clippy::upper_case_acronyms)]
pub enum InitFEEBeamError {
    #[error("Specified beam file '{0}' doesn't exist")]
    BeamFileDoesntExist(String),

    #[error("One of the HDF5 datasets started with 'X_'; what's wrong with your file?")]
    MissingDipole,

    #[error("No HDF5 datasets started with a 'X'; is there any data in the file?")]
    NoDipoles,

    #[error("No frequency information was gathered from the HDF5 datasets; is there any data in the file?")]
    NoFreqs,

    /// Incorrect number of dipoles in the HDF5 file.
    #[error("Got information on {got} dipoles from the HDF5 file, but expected {expected}")]
    DipoleCountMismatch { expected: u8, got: u8 },

    /// An error associated with parsing a string into another type.
    #[error("Couldn't parse '{0}' to a number")]
    Parse(String),

    /// An error associated with the MWA_BEAM_FILE environment variable.
    #[error("Problem with the MWA_BEAM_FILE variable: {0}")]
    MwaBeamFileVarError(#[from] std::env::VarError),

    #[error("Unexpected array shape when reading HDF5 dataset 'modes': expected 3 rows")]
    ModesShape,

    /// An error associated with the hdf5 crate.
    #[error("HDF5 error: {0}")]
    Hdf5(#[from] hdf5::Error),
}

#[derive(Error, Debug)]
#[allow(clippy::upper_case_acronyms)]
pub enum FEEBeamError {
    #[error("Expected {expected} dipole coefficients, but got {got}")]
    S1S2CountMismatch { expected: usize, got: usize },

    #[error("The number of {ctype} coefficients did not match m_accum - got {got} when we expected {expected}")]
    CoeffCountMismatch {
        ctype: &'static str,
        got: usize,
        expected: usize,
    },

    #[error("Unexpected array shape when reading HDF5 dataset '{key}': expected {exp} rows")]
    DatasetShape { key: String, exp: usize },

    /// An error associated with the hdf5 crate.
    #[error("HDF5 error: {0}")]
    Hdf5Error(#[from] hdf5::Error),

    #[cfg(feature = "cuda")]
    #[error("{0}")]
    Cuda(#[from] marlu::cuda::CudaError),
}
