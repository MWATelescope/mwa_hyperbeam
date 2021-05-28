// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

/*!
Some helper mathematics.
 */

use crate::types::c64;

#[inline]
pub(crate) fn sqrt(x: f64) -> f64 {
    x.sqrt()
}

// Make traditional trigonometry possible.
#[inline]
pub(crate) fn sin(x: f64) -> f64 {
    x.sin()
}

#[inline]
pub(crate) fn cos(x: f64) -> f64 {
    x.cos()
}

#[inline]
pub(crate) fn atan2(y: f64, x: f64) -> f64 {
    y.atan2(x)
}

/// Complex exponential. The argument is assumed to be purely imaginary.
///
/// This function doesn't actually use complex numbers; it just returns the real
/// and imag components from Euler's formula (i.e. e^{ix} = cos{x} + i sin{x}).
#[inline]
#[allow(dead_code)]
pub(crate) fn cexp(x: f64) -> c64 {
    let (s, c) = x.sin_cos();
    c64::new(c, s)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use std::f64::consts::*;

    #[test]
    fn test_sin() {
        assert_abs_diff_eq!(sin(FRAC_PI_6), 0.5);
    }

    #[test]
    fn test_cos() {
        assert_abs_diff_eq!(cos(FRAC_PI_3), 0.5);
    }

    #[test]
    fn atan2_is_correct() {
        assert_abs_diff_eq!(atan2(-2.0, 1.0), -1.1071487177940904);
        assert_abs_diff_eq!(atan2(1.0, -1.0), 3.0 * FRAC_PI_4);
    }

    #[test]
    fn test_cexp() {
        assert_abs_diff_eq!(cexp(PI), c64::new(-1.0, 0.0));
    }
}
