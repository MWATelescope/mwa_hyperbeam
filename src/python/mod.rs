// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Python interface to hyperbeam.

mod analytic;
mod fee;

use pyo3::create_exception;
use pyo3::prelude::*;

use crate::analytic::AnalyticBeamError;
use crate::fee::{FEEBeamError, InitFEEBeamError};

/// A Python module interfacing with the hyperbeam code written in Rust. This
/// module depends on and will import numpy.
#[pymodule]
fn mwa_hyperbeam(py: Python, m: &PyModule) -> PyResult<()> {
    py.import("numpy")?;
    m.add_class::<fee::FEEBeam>()?;
    m.add_class::<analytic::AnalyticBeam>()?;
    m.add("HyperbeamError", py.get_type::<HyperbeamError>())?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    Ok(())
}

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
impl std::convert::From<AnalyticBeamError> for PyErr {
    fn from(err: AnalyticBeamError) -> PyErr {
        HyperbeamError::new_err(err.to_string())
    }
}
