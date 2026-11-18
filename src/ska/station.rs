// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! SKA-Low station beams: element responses combined with an array factor.

use marlu::{c64, constants::VEL_C, AzEl, Jones};

use super::{
    az_to_phi, everybeam_to_hyperbeam_with_pa, fastmath, par_map_jones, parallactic_angle,
    SkaBeamError, SkaElementResponse,
};

/// An SKA-Low station: an element-response model, the positions of the
/// elements and their weights.
///
/// The station response is computed as EveryBeam does for OSKAR telescopes
/// (with `use_channel_frequency = true`, which EveryBeam requires for OSKAR):
///
/// `J(s) = sum_i w_i exp(-2 pi i f / c (s0 - s) . r_i) E_i(s) / sum_i w_i`
///
/// where `s` is the direction of interest, `s0` the pointing direction, `r_i`
/// the element positions and `E_i` the element responses. If the element model
/// has the same response for every element (e.g. [`SkaLowEepGrid`],
/// [`SkaLowAnalytic`]), `E_i` is only evaluated once.
///
/// Element positions are in metres in a local East, North, Up frame (relative
/// to any origin; the station centre is conventional). Only the relative
/// geometry matters, apart from an overall phase.
///
/// # Station rotation
///
/// SKA-Low stations may be rigidly rotated about the local vertical. In a
/// station rotated by `rotation_rad` (see [`SkaStation::with_rotation`]), the
/// X dipoles lie along the station's first axis, `rotation_rad` from East
/// towards North (counter-clockwise seen from above), and the Y dipoles
/// `rotation_rad` from North towards West; the element patterns (including
/// EEP grids and FEKO patterns, which are defined in the station's frame) are
/// rotated with them. This matches EveryBeam, which evaluates OSKAR element
/// responses in each station's `COORDINATE_AXES` frame (from the MS
/// `PHASED_ARRAY` table); use [`rotation_from_p_axis`] to get the angle from
/// those axes. Element positions are still given in East, North, Up.
///
/// The output Jones matrices keep hyperbeam's convention: the columns are
/// still the theta and azimuth components on the sky, and the rows are the
/// station's (rotated) X and Y dipoles.
///
/// # Local frames
///
/// The stations of a large array don't share a horizon: an SKA-Low station
/// 40 km from the array centre has a local vertical about 0.36 degrees from
/// the centre's. By default, directions, pointings and latitudes are taken to
/// be in the station's own frame. With [`SkaStation::with_local_frame`],
/// they are instead given in a common (array) frame and converted to the
/// station's frame, as EveryBeam does with each station's ITRF frame.
///
/// [`SkaLowEepGrid`]: super::SkaLowEepGrid
/// [`SkaLowAnalytic`]: super::SkaLowAnalytic
pub struct SkaStation<E: SkaElementResponse> {
    element: E,
    positions_enu: Vec<[f64; 3]>,
    weights: Vec<f64>,
    rotation_rad: f64,
    local_frame: Option<[[f64; 3]; 3]>,
    // The positions as separate East, North and Up arrays, for the
    // vectorised array-factor code.
    xs: Vec<f64>,
    ys: Vec<f64>,
    zs: Vec<f64>,
    sum_weights: f64,
}

/// Get a station's rotation angle (as used by [`SkaStation::with_rotation`])
/// from its first coordinate axis (`p`, the X-dipole direction) expressed in
/// the station's local East, North, Up frame. For an MS, `p` is the first row
/// of the station's `COORDINATE_AXES` in the `PHASED_ARRAY` table, which is in
/// ITRF; project it onto the station's local East and North unit vectors
/// first. Any vertical component of `p` is ignored.
pub fn rotation_from_p_axis(p_enu: [f64; 3]) -> f64 {
    p_enu[1].atan2(p_enu[0])
}

/// Convert a direction (azimuth, zenith angle) in an array frame to a
/// station's local frame (see [`SkaStation::with_local_frame`]).
#[inline]
pub(crate) fn to_local(m: &[[f64; 3]; 3], az_rad: f64, za_rad: f64) -> (f64, f64) {
    let mul = |v: [f64; 3]| {
        [
            m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
            m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
            m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
        ]
    };
    let l = mul(enu_unit_vector(az_rad, za_rad));
    let horizontal = l[0].hypot(l[1]);
    // At the local zenith, the azimuth is arbitrary but it still defines the
    // theta and phi polarisation basis; use the direction of the given
    // azimuth in the local frame.
    let az = if horizontal > 1e-12 {
        l[0].atan2(l[1])
    } else {
        let (s, c) = az_rad.sin_cos();
        let h = mul([s, c, 0.0]);
        h[0].atan2(h[1])
    };
    (az, horizontal.atan2(l[2]))
}

/// The parallactic angle of a direction (azimuth, zenith angle) given in a
/// station's local frame, where `latitude_rad` is the latitude of the array
/// frame. The celestial pole, at azimuth 0 and elevation `latitude_rad` in the
/// array frame, is converted to the station's frame, where it is at some
/// latitude and (in general, slightly non-zero) azimuth.
#[inline]
pub(crate) fn local_parallactic_angle(
    m: &[[f64; 3]; 3],
    az_rad: f64,
    za_rad: f64,
    latitude_rad: f64,
) -> f64 {
    let (pole_az, pole_za) = to_local(m, 0.0, std::f64::consts::FRAC_PI_2 - latitude_rad);
    parallactic_angle(
        az_rad - pole_az,
        za_rad,
        std::f64::consts::FRAC_PI_2 - pole_za,
    )
}

/// Unit vector (East, North, Up) for an azimuth and zenith angle.
#[inline]
fn enu_unit_vector(az_rad: f64, za_rad: f64) -> [f64; 3] {
    let (s_az, c_az) = az_rad.sin_cos();
    let (s_za, c_za) = za_rad.sin_cos();
    [s_za * s_az, s_za * c_az, c_za]
}

impl<E: SkaElementResponse> SkaStation<E> {
    /// Create a new station. `weights` (if given) must have one value per
    /// element; a weight of 0 flags an element. If not given, all weights are
    /// one. If the element model has distinct per-element responses, element `i`
    /// uses the response with `element_id = i`, so the number of positions
    /// must match the number of elements in the model.
    pub fn new(
        element: E,
        positions_enu: Vec<[f64; 3]>,
        weights: Option<Vec<f64>>,
    ) -> Result<SkaStation<E>, SkaBeamError> {
        if let Some(n) = element.num_elements() {
            if positions_enu.len() != n {
                return Err(SkaBeamError::PositionsLength {
                    got: positions_enu.len(),
                    expected: n,
                });
            }
        }
        let weights = match weights {
            Some(w) => {
                if w.len() != positions_enu.len() {
                    return Err(SkaBeamError::WeightsLength {
                        got: w.len(),
                        expected: positions_enu.len(),
                    });
                }
                w
            }
            None => vec![1.0; positions_enu.len()],
        };
        if weights.iter().sum::<f64>() == 0.0 {
            return Err(SkaBeamError::AllElementsFlagged);
        }
        let xs = positions_enu.iter().map(|p| p[0]).collect();
        let ys = positions_enu.iter().map(|p| p[1]).collect();
        let zs = positions_enu.iter().map(|p| p[2]).collect();
        let sum_weights = weights.iter().sum();
        Ok(SkaStation {
            element,
            positions_enu,
            weights,
            rotation_rad: 0.0,
            local_frame: None,
            xs,
            ys,
            zs,
            sum_weights,
        })
    }

    /// Set the station's rotation about the local vertical: the angle of its
    /// X dipoles from East towards North \[radians\]. See the
    /// [type-level documentation](SkaStation#station-rotation).
    pub fn with_rotation(mut self, rotation_rad: f64) -> SkaStation<E> {
        self.rotation_rad = rotation_rad;
        self
    }

    /// Give the station its own local frame (see the
    /// [type-level documentation](SkaStation#local-frames)). The rows of
    /// `enu_to_local` are the station's local East, North and Up unit vectors,
    /// expressed in the (array) East, North, Up frame in which directions,
    /// pointings and latitudes will be given; i.e. a direction `v` in the
    /// array frame is `enu_to_local * v` in the station's frame. The
    /// station's rotation is relative to its local East.
    pub fn with_local_frame(mut self, enu_to_local: [[f64; 3]; 3]) -> SkaStation<E> {
        self.local_frame = Some(enu_to_local);
        self
    }

    /// The station's local frame, if it has one (see
    /// [`with_local_frame`](Self::with_local_frame)).
    pub fn local_frame(&self) -> Option<[[f64; 3]; 3]> {
        self.local_frame
    }

    /// The station's rotation (the angle of its X dipoles from East towards
    /// North) \[radians\].
    pub fn rotation(&self) -> f64 {
        self.rotation_rad
    }

    /// The element-response model of this station.
    pub fn element(&self) -> &E {
        &self.element
    }

    /// The element positions (East, North, Up) \[metres\].
    pub fn positions_enu(&self) -> &[[f64; 3]] {
        &self.positions_enu
    }

    /// The element weights.
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// The (scalar) array factor of the station, normalised by the sum of the
    /// weights (so it is 1 in the pointing direction). The direction and
    /// pointing are in the station's own frame (any
    /// [local frame](Self::with_local_frame) is not applied).
    pub fn array_factor(
        &self,
        az_rad: f64,
        za_rad: f64,
        freq_hz: f64,
        pointing_az_rad: f64,
        pointing_za_rad: f64,
    ) -> c64 {
        let (k, ds) = self.phase_terms(az_rad, za_rad, freq_hz, pointing_az_rad, pointing_za_rad);
        let (re, im) = fastmath::phasor_sum(k, ds, &self.xs, &self.ys, &self.zs, &self.weights);
        c64::new(re, im) / self.sum_weights
    }

    /// The array-factor phase of element `r` is `k * (ds . r)`.
    #[inline]
    fn phase_terms(
        &self,
        az_rad: f64,
        za_rad: f64,
        freq_hz: f64,
        pointing_az_rad: f64,
        pointing_za_rad: f64,
    ) -> (f64, [f64; 3]) {
        let s = enu_unit_vector(az_rad, za_rad);
        let s0 = enu_unit_vector(pointing_az_rad, pointing_za_rad);
        let ds = [s0[0] - s[0], s0[1] - s[1], s0[2] - s[2]];
        (-std::f64::consts::TAU * freq_hz / VEL_C, ds)
    }

    /// The station response in EveryBeam's convention (`[[X_theta, X_phi],
    /// [Y_theta, Y_phi]]`), for a direction (azimuth North through East,
    /// zenith angle) and pointing in the station's own frame (any
    /// [local frame](Self::with_local_frame) is not applied).
    pub fn station_response(
        &self,
        az_rad: f64,
        za_rad: f64,
        freq_hz: f64,
        pointing_az_rad: f64,
        pointing_za_rad: f64,
    ) -> Result<Jones<f64>, SkaBeamError> {
        // The direction's phi in the station's (rotated) frame.
        let phi = az_to_phi(az_rad) - self.rotation_rad;
        if self.element.num_elements().is_none() {
            let af = self.array_factor(az_rad, za_rad, freq_hz, pointing_az_rad, pointing_za_rad);
            let e = self.element.element_response(0, freq_hz, za_rad, phi)?;
            return Ok(e * af);
        }

        let (k, ds) = self.phase_terms(az_rad, za_rad, freq_hz, pointing_az_rad, pointing_za_rad);
        let n = self.xs.len();
        let (mut re, mut im) = (vec![0.0; n], vec![0.0; n]);
        fastmath::phasors(
            k,
            ds,
            &self.xs,
            &self.ys,
            &self.zs,
            &self.weights,
            &mut re,
            &mut im,
        );
        let mut result = Jones::default();
        for (i, &w) in self.weights.iter().enumerate() {
            if w == 0.0 {
                continue;
            }
            let e = self.element.element_response(i, freq_hz, za_rad, phi)?;
            result += e * c64::new(re[i], im[i]);
        }
        Ok(result / self.sum_weights)
    }

    /// Calculate the station-beam Jones matrix for a direction and pointing,
    /// in hyperbeam's convention. See
    /// [`SkaElementResponse::calc_jones_pair`] for the meaning of
    /// `latitude_rad` and `iau_order`.
    #[allow(clippy::too_many_arguments)]
    pub fn calc_jones_pair(
        &self,
        az_rad: f64,
        za_rad: f64,
        freq_hz: f64,
        pointing_az_rad: f64,
        pointing_za_rad: f64,
        latitude_rad: Option<f64>,
        iau_order: bool,
    ) -> Result<Jones<f64>, SkaBeamError> {
        let (az, za, paz, pza, pa) = match self.local_frame {
            None => (
                az_rad,
                za_rad,
                pointing_az_rad,
                pointing_za_rad,
                latitude_rad.map(|lat| parallactic_angle(az_rad, za_rad, lat)),
            ),
            Some(m) => {
                let (az, za) = to_local(&m, az_rad, za_rad);
                let (paz, pza) = to_local(&m, pointing_az_rad, pointing_za_rad);
                let pa = latitude_rad.map(|lat| local_parallactic_angle(&m, az, za, lat));
                (az, za, paz, pza, pa)
            }
        };
        let j = self.station_response(az, za, freq_hz, paz, pza)?;
        Ok(everybeam_to_hyperbeam_with_pa(j, pa, iau_order))
    }

    /// Calculate the station-beam Jones matrix for a direction and pointing
    /// given as [`AzEl`]s. See [`calc_jones_pair`](Self::calc_jones_pair).
    pub fn calc_jones(
        &self,
        azel: AzEl,
        freq_hz: f64,
        pointing: AzEl,
        latitude_rad: Option<f64>,
        iau_order: bool,
    ) -> Result<Jones<f64>, SkaBeamError> {
        self.calc_jones_pair(
            azel.az,
            azel.za(),
            freq_hz,
            pointing.az,
            pointing.za(),
            latitude_rad,
            iau_order,
        )
    }

    /// Calculate station-beam Jones matrices for many directions in parallel.
    /// See [`calc_jones_pair`](Self::calc_jones_pair).
    pub fn calc_jones_array(
        &self,
        azels: &[AzEl],
        freq_hz: f64,
        pointing: AzEl,
        latitude_rad: Option<f64>,
        iau_order: bool,
    ) -> Result<Vec<Jones<f64>>, SkaBeamError> {
        par_map_jones(azels.len(), |i| {
            self.calc_jones(azels[i], freq_hz, pointing, latitude_rad, iau_order)
        })
    }

    /// Calculate station-beam Jones matrices for many directions in parallel.
    /// See [`calc_jones_pair`](Self::calc_jones_pair).
    #[allow(clippy::too_many_arguments)]
    pub fn calc_jones_array_pair(
        &self,
        az_rad: &[f64],
        za_rad: &[f64],
        freq_hz: f64,
        pointing_az_rad: f64,
        pointing_za_rad: f64,
        latitude_rad: Option<f64>,
        iau_order: bool,
    ) -> Result<Vec<Jones<f64>>, SkaBeamError> {
        par_map_jones(az_rad.len().min(za_rad.len()), |i| {
            self.calc_jones_pair(
                az_rad[i],
                za_rad[i],
                freq_hz,
                pointing_az_rad,
                pointing_za_rad,
                latitude_rad,
                iau_order,
            )
        })
    }
}
