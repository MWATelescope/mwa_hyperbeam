// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Vectorisable sin/cos for station array factors.
//!
//! `f64::sin_cos` calls libm one value at a time, which dominates the cost of
//! a station array factor (one phasor per element per direction). This
//! module evaluates sin and cos branch-free on lanes of four values, which the
//! compiler vectorises. Accuracy is about 1 ulp: arguments are reduced by
//! multiples of pi/2 with a three-part (Cody-Waite) pi/2, and the reduced
//! argument is evaluated with fdlibm's minimax polynomials.
//!
//! The pi/2 split and polynomial coefficients are fdlibm's (`e_rem_pio2.c`,
//! `k_sin.c`, `k_cos.c`), which carry this notice:
//!
//! > Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
//! >
//! > Developed at SunPro, a Sun Microsystems, Inc. business.
//! > Permission to use, copy, modify, and distribute this
//! > software is freely granted, provided that this notice
//! > is preserved.

// The constants are fdlibm's, verbatim.
#![allow(clippy::excessive_precision)]

/// The number of values processed together.
pub(crate) const LANES: usize = 4;

const TWO_OVER_PI: f64 = std::f64::consts::FRAC_2_PI;
// pi/2 split into three parts (fdlibm's pio2_1, pio2_2, pio2_3); the first two
// have 33 significant bits, so n * part is exact for |n| < 2^20.
const PIO2_1: f64 = 1.570_796_326_734_125_614_17e+00;
const PIO2_2: f64 = 6.077_100_506_303_965_976_6e-11;
const PIO2_3: f64 = 2.022_266_248_711_166_455_8e-21;
// Adding then subtracting this rounds to the nearest integer (for |x| <
// 2^51), and leaves the integer in the low bits of the sum's representation.
const ROUNDER: f64 = 6_755_399_441_055_744.0; // 1.5 * 2^52

// fdlibm __kernel_sin and __kernel_cos coefficients.
const S1: f64 = -1.666_666_666_666_663_243_48e-01;
const S2: f64 = 8.333_333_333_322_489_461_24e-03;
const S3: f64 = -1.984_126_982_985_794_931_34e-04;
const S4: f64 = 2.755_731_370_707_006_767_89e-06;
const S5: f64 = -2.505_076_025_340_686_341_95e-08;
const S6: f64 = 1.589_690_995_211_550_102_21e-10;
const C1: f64 = 4.166_666_666_666_660_190_37e-02;
const C2: f64 = -1.388_888_888_887_410_957_49e-03;
const C3: f64 = 2.480_158_728_947_672_941_78e-05;
const C4: f64 = -2.755_731_435_139_066_330_35e-07;
const C5: f64 = 2.087_572_321_298_174_827_90e-09;
const C6: f64 = -1.135_964_755_778_819_482_65e-11;

/// sin and cos of one value, branch-free (so that loops over it vectorise).
/// Valid for |x| < 2^20 * pi/2 (about 1.6e6); array-factor phases are far
/// smaller.
#[inline(always)]
pub(crate) fn sincos(x: f64) -> (f64, f64) {
    // x = n * pi/2 + r, |r| <= pi/4.
    let t = x * TWO_OVER_PI + ROUNDER;
    let q = t.to_bits(); // the low bits hold n (two's complement)
    let n = t - ROUNDER;
    let r = ((x - n * PIO2_1) - n * PIO2_2) - n * PIO2_3;

    let z = r * r;
    let s = r + r * z * (S1 + z * (S2 + z * (S3 + z * (S4 + z * (S5 + z * S6)))));
    // fdlibm's careful form of cos for |r| <= pi/4.
    let hz = 0.5 * z;
    let w = 1.0 - hz;
    let c =
        w + (((1.0 - w) - hz) + z * z * (C1 + z * (C2 + z * (C3 + z * (C4 + z * (C5 + z * C6))))));

    // Quadrant: (sin, cos) = (s, c), (c, -s), (-s, -c), (-c, s).
    let swap = q & 1 == 1;
    let (sin, cos) = if swap { (c, s) } else { (s, c) };
    let sin = if q & 2 == 2 { -sin } else { sin };
    let cos = if q.wrapping_add(1) & 2 == 2 {
        -cos
    } else {
        cos
    };
    (sin, cos)
}
/// The weighted sum of phasors `sum_i w_i exp(i * k * (d . r_i))`, where
/// `d = (dx, dy, dz)` and the positions `r_i` are given as `xs`, `ys`, `zs`
/// (all slices the same length). Returns (real, imaginary).
pub(crate) fn phasor_sum(
    k: f64,
    d: [f64; 3],
    xs: &[f64],
    ys: &[f64],
    zs: &[f64],
    ws: &[f64],
) -> (f64, f64) {
    let n = xs.len();
    let (mut re, mut im) = ([0.0; LANES], [0.0; LANES]);
    let chunks = n / LANES;
    for c in 0..chunks {
        let i0 = c * LANES;
        let mut s = [0.0; LANES];
        let mut co = [0.0; LANES];
        for l in 0..LANES {
            let i = i0 + l;
            let phase = k * (d[0] * xs[i] + d[1] * ys[i] + d[2] * zs[i]);
            (s[l], co[l]) = sincos(phase);
        }
        for l in 0..LANES {
            re[l] += ws[i0 + l] * co[l];
            im[l] += ws[i0 + l] * s[l];
        }
    }
    let (mut re_sum, mut im_sum) = (re.iter().sum::<f64>(), im.iter().sum::<f64>());
    for i in chunks * LANES..n {
        let (s, c) = sincos(k * (d[0] * xs[i] + d[1] * ys[i] + d[2] * zs[i]));
        re_sum += ws[i] * c;
        im_sum += ws[i] * s;
    }
    (re_sum, im_sum)
}

/// Write the weighted phasors `w_i exp(i * k * (d . r_i))` to `out_re` and
/// `out_im` (see [`phasor_sum`]).
#[allow(clippy::too_many_arguments)]
pub(crate) fn phasors(
    k: f64,
    d: [f64; 3],
    xs: &[f64],
    ys: &[f64],
    zs: &[f64],
    ws: &[f64],
    out_re: &mut [f64],
    out_im: &mut [f64],
) {
    for i in 0..xs.len() {
        let (s, c) = sincos(k * (d[0] * xs[i] + d[1] * ys[i] + d[2] * zs[i]));
        out_re[i] = ws[i] * c;
        out_im[i] = ws[i] * s;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sincos_matches_libm() {
        // Array-factor phases are at most a few thousand radians; check well
        // beyond that, including exact multiples of pi/2 and tiny values.
        let mut max_err: f64 = 0.0;
        let mut x = -5.0e4_f64;
        let mut values = vec![0.0, -0.0, 1e-300, -1e-300, 1e-8];
        for k in -2000..=2000 {
            values.push(k as f64 * std::f64::consts::FRAC_PI_2);
            values.push(k as f64 * std::f64::consts::FRAC_PI_4);
        }
        while x < 5.0e4 {
            values.push(x);
            x += 0.000_731_9 * (1.0 + x.abs().sqrt());
        }
        for &v in &values {
            let (s, c) = sincos(v);
            let (s0, c0) = v.sin_cos();
            max_err = max_err.max((s - s0).abs()).max((c - c0).abs());
        }
        // ~1 ulp of 1.
        assert!(max_err < 4e-16, "max error {max_err:e}");
    }

    #[test]
    fn phasor_sum_matches_scalar() {
        let n = 257; // not a multiple of LANES
        let xs: Vec<f64> = (0..n).map(|i| (i as f64 * 0.37).sin() * 19.0).collect();
        let ys: Vec<f64> = (0..n).map(|i| (i as f64 * 0.91).cos() * 19.0).collect();
        let zs: Vec<f64> = (0..n).map(|i| (i as f64 * 0.13).sin() * 0.1).collect();
        let ws: Vec<f64> = (0..n).map(|i| 1.0 + (i % 3) as f64 * 0.25).collect();
        let (k, d) = (-7.3, [0.31, -0.52, 0.11]);
        let (re, im) = phasor_sum(k, d, &xs, &ys, &zs, &ws);
        let (mut re0, mut im0) = (0.0, 0.0);
        for i in 0..n {
            let (s, c) = (k * (d[0] * xs[i] + d[1] * ys[i] + d[2] * zs[i])).sin_cos();
            re0 += ws[i] * c;
            im0 += ws[i] * s;
        }
        assert!((re - re0).abs() < 1e-12 && (im - im0).abs() < 1e-12);

        let mut out_re = vec![0.0; n];
        let mut out_im = vec![0.0; n];
        phasors(k, d, &xs, &ys, &zs, &ws, &mut out_re, &mut out_im);
        for i in 0..n {
            let (s, c) = (k * (d[0] * xs[i] + d[1] * ys[i] + d[2] * zs[i])).sin_cos();
            assert!((out_re[i] - ws[i] * c).abs() < 1e-15 && (out_im[i] - ws[i] * s).abs() < 1e-15);
        }
    }
}
