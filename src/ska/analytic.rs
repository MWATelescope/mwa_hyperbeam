// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// Portions of this file are derived from ska-sdp-func
// (Copyright 2022 SKA Organisation; Copyright 2022 Fred Dulwich, Andrew
// Ensor, Anthony Griffin, Seth Hall, Adam Campbell; Copyright 2022 University
// of Oxford), used under the BSD-3-Clause licence; see LICENSE-ska-sdp-func in
// the root of this repository.

//! The SKA-Low analytic element model ("ska-low-analytic"), equivalent to
//! EveryBeam's `oskar_dipole_cos` element response model (and, without the
//! taper, `oskar_dipole`).
//!
//! The dipole pattern follows `sdp_element_beam_dipole` from
//! [ska-sdp-func](https://gitlab.com/ska-telescope/sdp/ska-sdp-func)
//! (BSD-3-Clause), which EveryBeam calls.

use std::f64::consts::{FRAC_PI_2, PI};

use marlu::{c64, constants::VEL_C, Jones};

use super::{SkaBeamError, SkaElementResponse};

/// The SKA-Low analytic element model ("ska-low-analytic"): a half-wavelength
/// dipole with a `cos(theta)` taper.
///
/// | hyperbeam                              | EveryBeam `element_response_model` |
/// |----------------------------------------|------------------------------------|
/// | [`SkaLowAnalytic::new`]                | `oskar_dipole_cos` (`kOSKARDipoleCos`; the default for OSKAR telescopes since EveryBeam 0.9.0) |
/// | [`SkaLowAnalytic::new_without_taper`]  | `oskar_dipole` (`kOSKARDipole`)    |
///
/// The dipole length is set to half the wavelength at each frequency (as in
/// EveryBeam). X dipoles are aligned along phi = 0 (East-West), Y dipoles
/// along phi = pi/2 (North-South). EveryBeam describes the `cos(theta)` taper
/// as "a quick phenomenological model, that will likely be deprecated when
/// 'official' beam models for SKA-LOW are implemented".
#[derive(Clone, Copy, Debug)]
pub struct SkaLowAnalytic {
    cos_taper: bool,
}

impl Default for SkaLowAnalytic {
    fn default() -> Self {
        Self::new()
    }
}

impl SkaLowAnalytic {
    /// The dipole model with a `cos(theta)` taper (EveryBeam's
    /// `oskar_dipole_cos`).
    pub fn new() -> SkaLowAnalytic {
        SkaLowAnalytic { cos_taper: true }
    }

    /// The dipole model without a taper (EveryBeam's `oskar_dipole`).
    pub fn new_without_taper() -> SkaLowAnalytic {
        SkaLowAnalytic { cos_taper: false }
    }

    /// Is the `cos(theta)` taper applied?
    pub fn has_cos_taper(&self) -> bool {
        self.cos_taper
    }
}

/// The far-field pattern (E_theta, E_phi) of a dipole of electrical
/// half-length `kl` (i.e. `k * L / 2`) aligned along phi = 0.
#[inline]
fn dipole(kl: f64, cos_kl: f64, phi: f64, sin_theta: f64, cos_theta: f64) -> (f64, f64) {
    let (sin_phi, cos_phi) = phi.sin_cos();
    let denominator = 1.0 + cos_phi * cos_phi * (cos_theta * cos_theta - 1.0);
    if denominator == 0.0 {
        (0.0, 0.0)
    } else {
        let temp = ((kl * cos_phi * sin_theta).cos() - cos_kl) / denominator;
        (-cos_phi * cos_theta * temp, sin_phi * temp)
    }
}

impl SkaElementResponse for SkaLowAnalytic {
    fn element_response(
        &self,
        _element_id: usize,
        freq_hz: f64,
        theta_rad: f64,
        phi_rad: f64,
    ) -> Result<Jones<f64>, SkaBeamError> {
        let dipole_length_m = 0.5 * VEL_C / freq_hz;
        let kl = dipole_length_m * (PI * freq_hz / VEL_C);
        let cos_kl = kl.cos();
        let (sin_theta, cos_theta) = theta_rad.sin_cos();
        let (x_theta, x_phi) = dipole(kl, cos_kl, phi_rad, sin_theta, cos_theta);
        let (y_theta, y_phi) = dipole(kl, cos_kl, phi_rad + FRAC_PI_2, sin_theta, cos_theta);
        let taper = if self.cos_taper { cos_theta } else { 1.0 };
        Ok(Jones::from([
            c64::new(x_theta * taper, 0.0),
            c64::new(x_phi * taper, 0.0),
            c64::new(y_theta * taper, 0.0),
            c64::new(y_phi * taper, 0.0),
        ]))
    }
}
