// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! A runtime-selected SKA-Low element model.

use std::path::Path;

use marlu::Jones;

use super::{
    InitSkaBeamError, SkaBeamError, SkaElementResponse, SkaLowAnalytic, SkaLowEepGrid, SkaLowEepSph,
};

/// Which SKA-Low element model a [`SkaLowBeam`] uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SkaLowModel {
    /// `ska-low-analytic` (EveryBeam's `oskar_dipole_cos`, or `oskar_dipole`
    /// without the taper).
    Analytic,
    /// `ska-low-eep-grid` (EveryBeam's `gridded_eep`).
    EepGrid,
    /// `ska-low-eep-sph` (EveryBeam's `skalow_feko`).
    EepSph,
}

impl SkaLowModel {
    /// The hyperbeam name of the model, e.g. "ska-low-eep-grid".
    pub fn name(self) -> &'static str {
        match self {
            SkaLowModel::Analytic => "ska-low-analytic",
            SkaLowModel::EepGrid => "ska-low-eep-grid",
            SkaLowModel::EepSph => "ska-low-eep-sph",
        }
    }

    /// The equivalent EveryBeam `element_response_model` name.
    pub fn everybeam_name(self) -> &'static str {
        match self {
            SkaLowModel::Analytic => "oskar_dipole_cos",
            SkaLowModel::EepGrid => "gridded_eep",
            SkaLowModel::EepSph => "skalow_feko",
        }
    }
}

impl std::str::FromStr for SkaLowModel {
    type Err = String;

    /// Parse a hyperbeam model name or the EveryBeam equivalent.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "ska-low-analytic" | "oskar_dipole_cos" => Ok(SkaLowModel::Analytic),
            "ska-low-eep-grid" | "gridded_eep" => Ok(SkaLowModel::EepGrid),
            "ska-low-eep-sph" | "skalow_feko" => Ok(SkaLowModel::EepSph),
            _ => Err(format!(
                "Unknown SKA-Low model '{s}'; expected one of ska-low-analytic, ska-low-eep-grid, ska-low-eep-sph"
            )),
        }
    }
}

/// Any one of the SKA-Low element models, selected at runtime. This is useful
/// when the model is a user choice (and is what the C FFI uses).
///
/// `SkaLowBeam` implements [`SkaElementResponse`], so all of the `calc_jones*`
/// methods and [`SkaStation`](super::SkaStation) can be used with it.
pub struct SkaLowBeam {
    pub(crate) inner: SkaLowBeamInner,
}

/// The concrete model behind a [`SkaLowBeam`].
pub(crate) enum SkaLowBeamInner {
    Analytic(SkaLowAnalytic),
    EepGrid(SkaLowEepGrid),
    EepSph(SkaLowEepSph),
}

/// Dispatch a method call to the concrete model.
macro_rules! dispatch {
    ($self:expr, $m:ident => $e:expr) => {
        match &$self.inner {
            SkaLowBeamInner::Analytic($m) => $e,
            SkaLowBeamInner::EepGrid($m) => $e,
            SkaLowBeamInner::EepSph($m) => $e,
        }
    };
}

impl SkaLowBeam {
    /// `ska-low-analytic`. If `cos_taper` is false, EveryBeam's
    /// `oskar_dipole` model is used instead of `oskar_dipole_cos`.
    pub fn new_analytic(cos_taper: bool) -> SkaLowBeam {
        let beam = if cos_taper {
            SkaLowAnalytic::new()
        } else {
            SkaLowAnalytic::new_without_taper()
        };
        SkaLowBeam {
            inner: SkaLowBeamInner::Analytic(beam),
        }
    }

    /// `ska-low-eep-grid`. If `file` is `None`, the file is found from the
    /// environment (see [`SkaLowEepGrid::new_from_env`]).
    pub fn new_eep_grid<P: AsRef<Path>>(
        file: Option<P>,
        ska_low_convention: bool,
    ) -> Result<SkaLowBeam, InitSkaBeamError> {
        let beam = match file {
            Some(f) => SkaLowEepGrid::new_with_convention(f, ska_low_convention)?,
            None => SkaLowEepGrid::new_from_env()?,
        };
        Ok(SkaLowBeam {
            inner: SkaLowBeamInner::EepGrid(beam),
        })
    }

    /// `ska-low-eep-sph`. If `dir` is `None`, the directory is found from the
    /// environment (see [`SkaLowEepSph::new_from_env`]).
    pub fn new_eep_sph<P: AsRef<Path>>(
        dir: Option<P>,
        l_max: Option<usize>,
    ) -> Result<SkaLowBeam, InitSkaBeamError> {
        let beam = match dir {
            Some(d) => SkaLowEepSph::new_with_l_max(d, l_max)?,
            None => SkaLowEepSph::new_from_env()?,
        };
        Ok(SkaLowBeam {
            inner: SkaLowBeamInner::EepSph(beam),
        })
    }

    /// The frequencies at which the model is defined \[Hz\], ascending: the
    /// frequencies of an EEP grid (`ska-low-eep-grid`, which is linearly
    /// interpolated between them) or of the FEKO coefficient files
    /// (`ska-low-eep-sph`, which uses the closest one). `None` for
    /// `ska-low-analytic`, which is defined at any frequency.
    pub fn get_freqs(&self) -> Option<Vec<f64>> {
        match &self.inner {
            SkaLowBeamInner::Analytic(_) => None,
            SkaLowBeamInner::EepGrid(b) => Some(b.get_freqs().to_vec()),
            SkaLowBeamInner::EepSph(b) => Some(b.get_freqs()),
        }
    }

    /// The frequency that will actually be used for `freq_hz`: the closest
    /// coefficient-file frequency for `ska-low-eep-sph`, otherwise `freq_hz`.
    pub fn find_closest_freq(&self, freq_hz: f64) -> f64 {
        match &self.inner {
            SkaLowBeamInner::EepSph(b) => b.find_closest_freq(freq_hz),
            _ => freq_hz,
        }
    }

    /// Remove any cached data (FEKO coefficients for `ska-low-eep-sph`).
    pub fn empty_cache(&self) {
        if let SkaLowBeamInner::EepSph(b) = &self.inner {
            b.empty_cache();
        }
    }

    /// Which model this is.
    pub fn model(&self) -> SkaLowModel {
        match &self.inner {
            SkaLowBeamInner::Analytic(_) => SkaLowModel::Analytic,
            SkaLowBeamInner::EepGrid(_) => SkaLowModel::EepGrid,
            SkaLowBeamInner::EepSph(_) => SkaLowModel::EepSph,
        }
    }
}

impl SkaElementResponse for SkaLowBeam {
    fn element_response(
        &self,
        element_id: usize,
        freq_hz: f64,
        theta_rad: f64,
        phi_rad: f64,
    ) -> Result<Jones<f64>, SkaBeamError> {
        dispatch!(self, b => b.element_response(element_id, freq_hz, theta_rad, phi_rad))
    }

    fn num_elements(&self) -> Option<usize> {
        dispatch!(self, b => b.num_elements())
    }

    // Forward the array methods so that model-specific fast paths are used.
    fn calc_jones_array(
        &self,
        azels: &[marlu::AzEl],
        freq_hz: f64,
        element_id: usize,
        latitude_rad: Option<f64>,
        iau_order: bool,
    ) -> Result<Vec<Jones<f64>>, SkaBeamError> {
        dispatch!(self, b => b.calc_jones_array(azels, freq_hz, element_id, latitude_rad, iau_order))
    }

    fn calc_jones_array_pair(
        &self,
        az_rad: &[f64],
        za_rad: &[f64],
        freq_hz: f64,
        element_id: usize,
        latitude_rad: Option<f64>,
        iau_order: bool,
    ) -> Result<Vec<Jones<f64>>, SkaBeamError> {
        dispatch!(self, b => b.calc_jones_array_pair(az_rad, za_rad, freq_hz, element_id, latitude_rad, iau_order))
    }
}
