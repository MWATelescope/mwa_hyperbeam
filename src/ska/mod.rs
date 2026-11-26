// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! SKA-Low (OSKAR) element-response models, mirroring those available in
//! [EveryBeam](https://git.astron.nl/RD/EveryBeam):
//!
//! | hyperbeam model    | hyperbeam type      | EveryBeam `element_response_model` |
//! |--------------------|---------------------|------------------------------------|
//! | `ska-low-analytic` | [`SkaLowAnalytic`]  | `oskar_dipole_cos` (`oskar_dipole` with [`SkaLowAnalytic::new_without_taper`]) |
//! | `ska-low-eep-grid` | [`SkaLowEepGrid`]   | `gridded_eep`                      |
//! | `ska-low-eep-sph`  | [`SkaLowEepSph`]    | `skalow_feko`                      |
//!
//! All models implement the [`SkaElementResponse`] trait. Its
//! [`element_response`](SkaElementResponse::element_response) method returns
//! exactly what EveryBeam's `ElementResponse::Response(element_id, freq,
//! theta, phi)` returns, i.e. a Jones matrix `[[X_theta, X_phi], [Y_theta,
//! Y_phi]]` (in EveryBeam's SKA-Low sign convention), where `theta` is the
//! zenith angle and `phi` is measured from East through North. X dipoles are
//! aligned East-West and Y dipoles North-South.
//!
//! The `calc_jones*` methods instead take an azimuth (North through East) and
//! zenith angle, and return Jones matrices in the same convention as the rest
//! of hyperbeam (see [`FEEBeam::calc_jones`](crate::fee::FEEBeam::calc_jones));
//! this means `[X_theta, X_az, Y_theta, Y_az]` (with `X_az = -X_phi`) unless a
//! latitude is given, in which case the parallactic-angle correction is
//! applied (with optional IAU ordering).
//!
//! [`SkaStation`] combines any of these element models with an array factor
//! to produce a station (tile) beam, in the same way as EveryBeam's OSKAR
//! telescope does, including each station's rigid rotation about the local
//! vertical ([`SkaStation::with_rotation`]; EveryBeam takes this from the
//! `COORDINATE_AXES` of the MS `PHASED_ARRAY` table).

mod analytic;
pub(crate) mod beam;
mod eep_grid;
mod eep_sph;
mod error;
mod fastmath;
mod ffi;
#[cfg(any(feature = "cuda", feature = "hip"))]
mod gpu;
mod station;
#[cfg(test)]
pub(crate) mod tests;

pub use analytic::SkaLowAnalytic;
pub use beam::{SkaLowBeam, SkaLowModel};
pub use eep_grid::SkaLowEepGrid;
pub use eep_sph::SkaLowEepSph;
pub use error::{InitSkaBeamError, SkaBeamError};
#[cfg(any(feature = "cuda", feature = "hip"))]
pub use gpu::{SkaGpuTiles, SkaLowBeamGpu, SKA_GPU_MAX_L_MAX};
pub use station::{rotation_from_p_axis, SkaStation};

use std::f64::consts::FRAC_PI_2;

use marlu::{AzEl, Jones};
use rayon::prelude::*;

use crate::fee::apply_parallactic_rotation;

/// Convert an azimuth (North through East) to an EveryBeam/OSKAR phi (East
/// through North).
#[inline]
pub(crate) fn az_to_phi(az_rad: f64) -> f64 {
    FRAC_PI_2 - az_rad
}

/// Convert a Jones matrix in the EveryBeam convention (`[[X_theta, X_phi],
/// [Y_theta, Y_phi]]`) to the hyperbeam convention (`[X_theta, X_az, Y_theta,
/// Y_az]`), optionally applying the parallactic-angle correction.
#[inline]
pub(crate) fn everybeam_to_hyperbeam(
    j: Jones<f64>,
    az_rad: f64,
    za_rad: f64,
    latitude_rad: Option<f64>,
    iau_order: bool,
) -> Jones<f64> {
    everybeam_to_hyperbeam_with_pa(
        j,
        latitude_rad.map(|lat| parallactic_angle(az_rad, za_rad, lat)),
        iau_order,
    )
}

/// Like [`everybeam_to_hyperbeam`], but with the parallactic angle (if any)
/// already computed.
#[inline]
pub(crate) fn everybeam_to_hyperbeam_with_pa(
    j: Jones<f64>,
    parallactic_angle_rad: Option<f64>,
    iau_order: bool,
) -> Jones<f64> {
    let mut jones = Jones::from([j[0], -j[1], j[2], -j[3]]);
    if let Some(pa) = parallactic_angle_rad {
        apply_parallactic_rotation(pa, iau_order, &mut jones);
    }
    jones
}

/// The parallactic angle of a direction (azimuth North through East, zenith
/// angle) at a latitude.
///
/// This is the same angle as ERFA's `hd2pa` (as used by the FEE beam), but it
/// is computed directly from the azimuth and zenith angle rather than via
/// (HA, Dec), which is well conditioned near the zenith (important for the
/// single-precision GPU code; the GPU code uses the same formula).
#[inline]
pub(crate) fn parallactic_angle(az_rad: f64, za_rad: f64, latitude_rad: f64) -> f64 {
    let (s_az, c_az) = az_rad.sin_cos();
    let (s_za, c_za) = za_rad.sin_cos();
    let (s_lat, c_lat) = latitude_rad.sin_cos();
    (-s_az * c_lat).atan2(s_lat * s_za - c_lat * c_za * c_az)
}

/// Evaluate `f(i)` for `i` in `0..n` in parallel into a pre-allocated vector,
/// stopping at the first error. (This is considerably faster than collecting
/// an iterator of `Result`s with rayon.) Work is split into chunks of at least
/// 16 items: small enough that a few hundred expensive items (e.g. station
/// responses) still use all threads, large enough to keep the overhead for
/// cheap items small.
pub(crate) fn par_map_jones<F>(n: usize, f: F) -> Result<Vec<Jones<f64>>, SkaBeamError>
where
    F: Fn(usize) -> Result<Jones<f64>, SkaBeamError> + Sync + Send,
{
    let mut out = vec![Jones::default(); n];
    out.par_iter_mut()
        .enumerate()
        .with_min_len(16)
        .try_for_each(|(i, o)| {
            *o = f(i)?;
            Ok::<(), SkaBeamError>(())
        })?;
    Ok(out)
}

/// An SKA-Low element-response model.
///
/// Only [`element_response`](Self::element_response) needs to be implemented;
/// everything else is derived from it.
pub trait SkaElementResponse: Sync {
    /// The response of a single element in EveryBeam's convention.
    ///
    /// `theta_rad` is the zenith angle and `phi_rad` the angle from East
    /// through North (both in the element's local frame). The result is
    /// `[[X_theta, X_phi], [Y_theta, Y_phi]]`. `element_id` is ignored by
    /// models where all elements have the same response.
    fn element_response(
        &self,
        element_id: usize,
        freq_hz: f64,
        theta_rad: f64,
        phi_rad: f64,
    ) -> Result<Jones<f64>, SkaBeamError>;

    /// The number of elements with distinct responses, or `None` if all
    /// elements share the same response (in which case a station beam can be
    /// computed as an array factor times a single element response).
    fn num_elements(&self) -> Option<usize> {
        None
    }

    /// Calculate the element-response Jones matrix for a direction given as
    /// an [`AzEl`]. See [`calc_jones_pair`](Self::calc_jones_pair).
    fn calc_jones(
        &self,
        azel: AzEl,
        freq_hz: f64,
        element_id: usize,
        latitude_rad: Option<f64>,
        iau_order: bool,
    ) -> Result<Jones<f64>, SkaBeamError> {
        self.calc_jones_pair(
            azel.az,
            azel.za(),
            freq_hz,
            element_id,
            latitude_rad,
            iau_order,
        )
    }

    /// Calculate the element-response Jones matrix for a direction (azimuth
    /// North through East, zenith angle) in hyperbeam's convention.
    ///
    /// If `latitude_rad` is `None`, the result is `[X_theta, X_az, Y_theta,
    /// Y_az]`. If it is given, the parallactic-angle correction is applied,
    /// and `iau_order` controls the ordering of the output (exactly as for
    /// [`FEEBeam::calc_jones`](crate::fee::FEEBeam::calc_jones)).
    fn calc_jones_pair(
        &self,
        az_rad: f64,
        za_rad: f64,
        freq_hz: f64,
        element_id: usize,
        latitude_rad: Option<f64>,
        iau_order: bool,
    ) -> Result<Jones<f64>, SkaBeamError> {
        let j = self.element_response(element_id, freq_hz, za_rad, az_to_phi(az_rad))?;
        Ok(everybeam_to_hyperbeam(
            j,
            az_rad,
            za_rad,
            latitude_rad,
            iau_order,
        ))
    }

    /// Calculate element-response Jones matrices for many directions in
    /// parallel. See [`calc_jones_pair`](Self::calc_jones_pair).
    fn calc_jones_array(
        &self,
        azels: &[AzEl],
        freq_hz: f64,
        element_id: usize,
        latitude_rad: Option<f64>,
        iau_order: bool,
    ) -> Result<Vec<Jones<f64>>, SkaBeamError> {
        par_map_jones(azels.len(), |i| {
            self.calc_jones(azels[i], freq_hz, element_id, latitude_rad, iau_order)
        })
    }

    /// Calculate element-response Jones matrices for many directions in
    /// parallel. See [`calc_jones_pair`](Self::calc_jones_pair).
    fn calc_jones_array_pair(
        &self,
        az_rad: &[f64],
        za_rad: &[f64],
        freq_hz: f64,
        element_id: usize,
        latitude_rad: Option<f64>,
        iau_order: bool,
    ) -> Result<Vec<Jones<f64>>, SkaBeamError> {
        par_map_jones(az_rad.len().min(za_rad.len()), |i| {
            self.calc_jones_pair(
                az_rad[i],
                za_rad[i],
                freq_hz,
                element_id,
                latitude_rad,
                iau_order,
            )
        })
    }
}

impl<T: SkaElementResponse + ?Sized> SkaElementResponse for &T {
    fn element_response(
        &self,
        element_id: usize,
        freq_hz: f64,
        theta_rad: f64,
        phi_rad: f64,
    ) -> Result<Jones<f64>, SkaBeamError> {
        (**self).element_response(element_id, freq_hz, theta_rad, phi_rad)
    }

    fn num_elements(&self) -> Option<usize> {
        (**self).num_elements()
    }
}

impl<T: SkaElementResponse + ?Sized + Send> SkaElementResponse for Box<T> {
    fn element_response(
        &self,
        element_id: usize,
        freq_hz: f64,
        theta_rad: f64,
        phi_rad: f64,
    ) -> Result<Jones<f64>, SkaBeamError> {
        (**self).element_response(element_id, freq_hz, theta_rad, phi_rad)
    }

    fn num_elements(&self) -> Option<usize> {
        (**self).num_elements()
    }
}
