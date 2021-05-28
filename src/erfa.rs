// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

/*!
Simple ERFA functions.

Rather than pull ERFA in as a dependency, these couple of Rust functions do the
job.
 */

use crate::constants::MWA_LATITUDE_RADIANS;
use crate::math::*;

///  Parallactic angle for a given hour angle and declination.
///
///  Unlike ERFA, this function always assumes that the latitude is the MWA
///  latitude.
///
///  Given:
///     ha_rad   hour angle
///     dec_rad  declination
///
///  Returned (function value):
///              parallactic angle
///
/// Notes:
///
///  1)  All the arguments are angles in radians.
///
///  2)  The parallactic angle at a point in the sky is the position
///      angle of the vertical, i.e. the angle between the directions to
///      the north celestial pole and to the zenith respectively.
///
///  3)  The result is returned in the range -pi to +pi.
///
///  4)  At the pole itself a zero result is returned.
///
///  5)  The latitude phi is pi/2 minus the angle between the Earth's
///      rotation axis and the adopted zenith.  In many applications it
///      will be sufficient to use the published geodetic latitude of the
///      site.  In very precise (sub-arcsecond) applications, phi can be
///      corrected for polar motion.
///
///  6)  Should the user wish to work with respect to the astronomical
///      zenith rather than the geodetic zenith, phi will need to be
///      adjusted for deflection of the vertical (often tens of
///      arcseconds), and the zero point of the hour angle ha will also
///      be affected.
// Subject to the copyright placed at the bottom of this file.
pub(crate) fn hd2pa_mwa(ha_rad: f64, dec_rad: f64) -> f64 {
    let phi = MWA_LATITUDE_RADIANS;
    let cp = cos(phi);
    let sqsz = cp * sin(ha_rad);
    let cqsz = sin(phi) * cos(dec_rad) - cp * sin(dec_rad) * cos(ha_rad);
    if sqsz != 0.0 || cqsz != 0.0 {
        atan2(sqsz, cqsz)
    } else {
        0.0
    }
}

///  Horizon to equatorial coordinates:  transform azimuth and altitude
///  to hour angle and declination.
///
///  Unlike ERFA, this function always assumes that the latitude is the MWA
///  latitude.
///
///  Given:
///     az_rad  azimuth
///     el_rad  altitude (informally, elevation)
///
///  Returned:
///     ha      hour angle (local)
///     dec     declination
///
///  Notes:
///
///  1)  All the arguments are angles in radians.
///
///  2)  The sign convention for azimuth is north zero, east +pi/2.
///
///  3)  HA is returned in the range +/-pi.  Declination is returned in
///      the range +/-pi/2.
///
///  4)  The latitude phi is pi/2 minus the angle between the Earth's
///      rotation axis and the adopted zenith.  In many applications it
///      will be sufficient to use the published geodetic latitude of the
///      site.  In very precise (sub-arcsecond) applications, phi can be
///      corrected for polar motion.
///
///  5)  The azimuth az must be with respect to the rotational north pole,
///      as opposed to the ITRS pole, and an azimuth with respect to north
///      on a map of the Earth's surface will need to be adjusted for
///      polar motion if sub-arcsecond accuracy is required.
///
///  6)  Should the user wish to work with respect to the astronomical
///      zenith rather than the geodetic zenith, phi will need to be
///      adjusted for deflection of the vertical (often tens of
///      arcseconds), and the zero point of ha will also be affected.
///
///  7)  The transformation is the same as Ve = Ry(phi-pi/2)*Rz(pi)*Vh,
///      where Ve and Vh are lefthanded unit vectors in the (ha,dec) and
///      (az,el) systems respectively and Rz and Ry are rotations about
///      first the z-axis and then the y-axis.  (n.b. Rz(pi) simply
///      reverses the signs of the x and y components.)  For efficiency,
///      the algorithm is written out rather than calling other utility
///      functions.  For applications that require even greater
///      efficiency, additional savings are possible if constant terms
///      such as functions of latitude are computed once and for all.
///
///  8)  Again for efficiency, no range checking of arguments is carried
///      out.
// Subject to the copyright placed at the bottom of this file.
pub(crate) fn ae2hd_mwa(az_rad: f64, el_rad: f64) -> (f64, f64) {
    /* Useful trig functions. */
    let sa = sin(az_rad);
    let ca = cos(az_rad);
    let se = sin(el_rad);
    let ce = cos(el_rad);
    let sp = sin(MWA_LATITUDE_RADIANS);
    let cp = cos(MWA_LATITUDE_RADIANS);

    /* HA,Dec unit vector. */
    let x = -ca * ce * sp + se * cp;
    let y = -sa * ce;
    let z = ca * ce * cp + se * sp;

    /* To spherical. */
    let r = sqrt(x * x + y * y);
    let ha = if r != 0.0 { atan2(y, x) } else { 0.0 };
    let dec = atan2(z, r);
    (ha, dec)
}

/*----------------------------------------------------------------------
**
**
**  Copyright (C) 2013-2020, NumFOCUS Foundation.
**  All rights reserved.
**
**  This library is derived, with permission, from the International
**  Astronomical Union's "Standards of Fundamental Astronomy" library,
**  available from http://www.iausofa.org.
**
**  The ERFA version is intended to retain identical functionality to
**  the SOFA library, but made distinct through different function and
**  file names, as set out in the SOFA license conditions.  The SOFA
**  original has a role as a reference standard for the IAU and IERS,
**  and consequently redistribution is permitted only in its unaltered
**  state.  The ERFA version is not subject to this restriction and
**  therefore can be included in distributions which do not support the
**  concept of "read only" software.
**
**  Although the intent is to replicate the SOFA API (other than
**  replacement of prefix names) and results (with the exception of
**  bugs;  any that are discovered will be fixed), SOFA is not
**  responsible for any errors found in this version of the library.
**
**  If you wish to acknowledge the SOFA heritage, please acknowledge
**  that you are using a library derived from SOFA, rather than SOFA
**  itself.
**
**
**  TERMS AND CONDITIONS
**
**  Redistribution and use in source and binary forms, with or without
**  modification, are permitted provided that the following conditions
**  are met:
**
**  1 Redistributions of source code must retain the above copyright
**    notice, this list of conditions and the following disclaimer.
**
**  2 Redistributions in binary form must reproduce the above copyright
**    notice, this list of conditions and the following disclaimer in
**    the documentation and/or other materials provided with the
**    distribution.
**
**  3 Neither the name of the Standards Of Fundamental Astronomy Board,
**    the International Astronomical Union nor the names of its
**    contributors may be used to endorse or promote products derived
**    from this software without specific prior written permission.
**
**  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
**  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
**  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
**  FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE
**  COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
**  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
**  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
**  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
**  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
**  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
**  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
**  POSSIBILITY OF SUCH DAMAGE.
**
*/

#[cfg(test)]
mod tests {
    use super::*;
    use approx::*;

    #[test]
    fn test_hd2pa() {
        assert_abs_diff_eq!(hd2pa_mwa(0.5, 0.8), 2.686613147020256, epsilon = 1e-10);
    }

    #[test]
    fn test_ae2hd() {
        let (ha, dec) = ae2hd_mwa(0.5, 0.8);
        assert_abs_diff_eq!(ha, -0.3498061917273272, epsilon = 1e-10);
        assert_abs_diff_eq!(dec, 0.22576116628947418, epsilon = 1e-10);
    }
}
