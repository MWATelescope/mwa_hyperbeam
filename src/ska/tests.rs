// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Tests for the SKA element and station beams.

use std::{
    f64::consts::{FRAC_PI_2, FRAC_PI_4, PI, TAU},
    path::PathBuf,
};

use approx::assert_abs_diff_eq;
use marlu::{c64, Jones};
use ndarray::prelude::*;

use super::*;

/// A unique path in the system temporary directory, removed on drop.
pub(crate) struct TempPath(pub(crate) PathBuf);

impl TempPath {
    pub(crate) fn new(name: &str) -> TempPath {
        let p = std::env::temp_dir().join(format!(
            "hyperbeam_ska_test_{}_{}",
            std::process::id(),
            name
        ));
        let _ = std::fs::remove_dir_all(&p);
        let _ = std::fs::remove_file(&p);
        TempPath(p)
    }
}

impl Drop for TempPath {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
        let _ = std::fs::remove_file(&self.0);
    }
}

// ---------------------------------------------------------------------------
// Gridded EEP. These tests mirror EveryBeam's cpp/test/teepgrid.cc.
// ---------------------------------------------------------------------------

const PHIS: [f64; 2] = [0.0, 1.0];
const THETAS: [f64; 2] = [-2.0, -1.0];
const FREQS: [f64; 2] = [1.0e6, 2.0e6];
const PHI_FACTOR: f64 = 20000.0;
const THETA_FACTOR: f64 = 2000.0;
const FREQ_FACTOR: f64 = 200.0;
const CORR_FACTOR: f64 = 10.0;
/// The file stores correlations in this order relative to the Jones matrix.
const EXPECTED_ORDER: [usize; 4] = [1, 0, 3, 2];

fn simple_grid() -> Array5<f64> {
    Array5::from_shape_fn((2, 2, 2, 4, 2), |(i, j, k, l, m)| {
        (i + 1) as f64 * PHI_FACTOR
            + (j + 1) as f64 * THETA_FACTOR
            + (k + 1) as f64 * FREQ_FACTOR
            + (l + 1) as f64 * CORR_FACTOR
            + (m + 1) as f64
    })
}

fn write_simple_eep(name: &str) -> TempPath {
    let path = TempPath::new(name);
    let h5 = hdf5_metno::File::create(&path.0).unwrap();
    let group = h5.create_group("averageEEP").unwrap();
    group
        .new_dataset_builder()
        .with_data(&arr1(&PHIS))
        .create("Phi")
        .unwrap();
    // Write theta as a MATLAB-style 1 x N dataset.
    group
        .new_dataset_builder()
        .with_data(&arr2(&[THETAS]))
        .create("Theta")
        .unwrap();
    // Write freq as a MATLAB-style N x 1 dataset.
    group
        .new_dataset_builder()
        .with_data(&arr2(&[[FREQS[0]], [FREQS[1]]]))
        .create("Freq")
        .unwrap();
    group
        .new_dataset_builder()
        .with_data(&simple_grid())
        .create("averageEEP")
        .unwrap();
    path
}

#[test]
fn eep_read_simple() {
    let file = write_simple_eep("eep_read_simple.h5");
    let eep = SkaLowEepGrid::new_with_convention(&file.0, false).unwrap();
    assert_eq!(eep.get_phis(), PHIS);
    assert_eq!(eep.get_thetas(), THETAS);
    assert_eq!(eep.get_freqs(), FREQS);
}

#[test]
fn eep_interpolate_boundaries() {
    let file = write_simple_eep("eep_interpolate_boundaries.h5");
    let eep = SkaLowEepGrid::new_with_convention(&file.0, false).unwrap();
    let grid = simple_grid();
    for i in 0..2 {
        for j in 0..2 {
            for k in 0..2 {
                let r = eep.interpolate(PHIS[i], THETAS[j], FREQS[k]).unwrap();
                for l in 0..4 {
                    let o = EXPECTED_ORDER[l];
                    assert_abs_diff_eq!(r[l].re, grid[(i, j, k, o, 0)], epsilon = 1e-9);
                    assert_abs_diff_eq!(r[l].im, grid[(i, j, k, o, 1)], epsilon = 1e-9);
                }
            }
        }
    }
}

#[test]
fn eep_interpolate_center() {
    let file = write_simple_eep("eep_interpolate_center.h5");
    let eep = SkaLowEepGrid::new_with_convention(&file.0, false).unwrap();
    let phi = (PHIS[0] + PHIS[1]) / 2.0;
    let theta = (THETAS[0] + THETAS[1]) / 2.0;
    let freq = (FREQS[0] + FREQS[1]) / 2.0;
    let r1 = eep.interpolate(phi, theta, freq).unwrap();
    // Negative phi values are wrapped.
    let r2 = eep.interpolate(phi - TAU, theta, freq).unwrap();
    for i in 0..4 {
        let base = 1.5 * PHI_FACTOR
            + 1.5 * THETA_FACTOR
            + 1.5 * FREQ_FACTOR
            + (EXPECTED_ORDER[i] + 1) as f64 * CORR_FACTOR;
        for r in [r1, r2] {
            assert_abs_diff_eq!(r[i].re, base + 1.0, epsilon = 1e-9);
            assert_abs_diff_eq!(r[i].im, base + 2.0, epsilon = 1e-9);
        }
    }
}

#[test]
fn eep_interpolate_linear() {
    // Weights should be linear along each axis.
    let file = write_simple_eep("eep_interpolate_linear.h5");
    let eep = SkaLowEepGrid::new_with_convention(&file.0, false).unwrap();
    let r = eep.interpolate(0.25, -1.2, 1.9e6).unwrap();
    let expected = (1.0 + 0.25) * PHI_FACTOR
        + (1.0 + 0.8) * THETA_FACTOR
        + (1.0 + 0.9) * FREQ_FACTOR
        + (EXPECTED_ORDER[0] + 1) as f64 * CORR_FACTOR
        + 1.0;
    assert_abs_diff_eq!(r[0].re, expected, epsilon = 1e-9);
}

#[test]
fn eep_out_of_range() {
    let file = write_simple_eep("eep_out_of_range.h5");
    let eep = SkaLowEepGrid::new(&file.0).unwrap();
    assert!(matches!(
        eep.element_response(0, 1.5e6, -1.5, 1.1),
        Err(SkaBeamError::OutOfGrid { what: "phi", .. })
    ));
    assert!(matches!(
        eep.element_response(0, 1.5e6, -0.5, 0.5),
        Err(SkaBeamError::OutOfGrid { what: "theta", .. })
    ));
    assert!(matches!(
        eep.element_response(0, 3.0e6, -1.5, 0.5),
        Err(SkaBeamError::OutOfGrid {
            what: "frequency",
            ..
        })
    ));
    assert!(matches!(
        eep.element_response(0, 1.5e6, f64::NAN, 0.5),
        Err(SkaBeamError::OutOfGrid { what: "theta", .. })
    ));
}

#[test]
fn eep_ska_low_convention() {
    // Mirrors EveryBeam's element_response_oskar_convention test.
    let file = write_simple_eep("eep_ska_low_convention.h5");
    let plain = SkaLowEepGrid::new_with_convention(&file.0, false).unwrap();
    let oskar = SkaLowEepGrid::new(&file.0).unwrap();
    let r = plain.element_response(0, 1.5e6, -1.5, 0.5).unwrap();
    let o = oskar.element_response(0, 1.5e6, -1.5, 0.5).unwrap();
    for i in 0..4 {
        let expected = if i < 2 { r[i].conj() } else { -r[i].conj() };
        assert_abs_diff_eq!(o[i], expected, epsilon = 1e-9);
    }
}

#[test]
fn eep_bad_shape() {
    let path = TempPath::new("eep_bad_shape.h5");
    {
        let h5 = hdf5_metno::File::create(&path.0).unwrap();
        let group = h5.create_group("averageEEP").unwrap();
        for name in ["Phi", "Theta", "Freq"] {
            group
                .new_dataset_builder()
                .with_data(&arr1(&[0.0, 1.0]))
                .create(name)
                .unwrap();
        }
        group
            .new_dataset_builder()
            .with_data(&Array5::<f64>::zeros((2, 2, 2, 3, 2)))
            .create("averageEEP")
            .unwrap();
    }
    assert!(matches!(
        SkaLowEepGrid::new(&path.0),
        Err(InitSkaBeamError::BadShape { .. })
    ));
    assert!(matches!(
        SkaLowEepGrid::new("/does/not/exist.h5"),
        Err(InitSkaBeamError::PathDoesntExist(_))
    ));
}

#[test]
fn eep_calc_jones_convention() {
    // calc_jones_pair should be element_response with phi = pi/2 - az and the
    // phi column negated.
    let file = write_simple_eep("eep_calc_jones_convention.h5");
    let eep = SkaLowEepGrid::new(&file.0).unwrap();
    // theta must be within [-2, -1] for this toy grid, phi within [0, 1].
    let (az, za) = (FRAC_PI_2 - 0.3, -1.4);
    let raw = eep.element_response(0, 1.2e6, za, 0.3).unwrap();
    let j = eep.calc_jones_pair(az, za, 1.2e6, 0, None, false).unwrap();
    assert_abs_diff_eq!(
        j,
        Jones::from([raw[0], -raw[1], raw[2], -raw[3]]),
        epsilon = 1e-8
    );
}

// ---------------------------------------------------------------------------
// OSKAR dipoles.
// ---------------------------------------------------------------------------

#[test]
fn dipole_zenith() {
    for model in [SkaLowAnalytic::new_without_taper(), SkaLowAnalytic::new()] {
        for freq in [50e6, 150e6, 350e6] {
            let j = model.element_response(0, freq, 0.0, 0.0).unwrap();
            // X is aligned with phi = 0, so at zenith X_theta = -1, Y_phi = 1.
            let expected = Jones::from([
                c64::new(-1.0, 0.0),
                c64::new(0.0, 0.0),
                c64::new(0.0, 0.0),
                c64::new(1.0, 0.0),
            ]);
            assert_abs_diff_eq!(j, expected, epsilon = 1e-12);
        }
    }
}

#[test]
fn dipole_nulls_and_taper() {
    let plain = SkaLowAnalytic::new_without_taper();
    let cos = SkaLowAnalytic::new();
    // The X dipole has a null along its axis (horizon, phi = 0), Y along phi =
    // pi/2.
    let j = plain.element_response(0, 1e8, FRAC_PI_2, 0.0).unwrap();
    assert_abs_diff_eq!(j[0].norm() + j[1].norm(), 0.0, epsilon = 1e-12);
    let j = plain
        .element_response(0, 1e8, FRAC_PI_2, FRAC_PI_2)
        .unwrap();
    assert_abs_diff_eq!(j[2].norm() + j[3].norm(), 0.0, epsilon = 1e-12);

    // The cos model is the plain model times cos(theta), and is zero at the
    // horizon.
    for (theta, phi) in [(0.3, 0.1), (1.0, 2.0), (1.4, -0.7)] {
        let p = plain.element_response(0, 1.1e8, theta, phi).unwrap();
        let c = cos.element_response(0, 1.1e8, theta, phi).unwrap();
        assert_abs_diff_eq!(c, p * theta.cos(), epsilon = 1e-12);
    }
    let c = cos.element_response(0, 1e8, FRAC_PI_2, 0.4).unwrap();
    assert_abs_diff_eq!(c, Jones::default(), epsilon = 1e-12);
}

#[test]
fn dipole_values() {
    // Half-wave dipole far field, written independently of the implementation:
    // E = (cos(pi/2 cos(psi)) / sin^2(psi)) * (-cos(phi) cos(theta), sin(phi)),
    // where psi is the angle from the dipole axis (cos(psi) = sin(theta)
    // cos(phi)). (cos(kL) is ~6e-17 for a half-wave dipole.)
    let model = SkaLowAnalytic::new_without_taper();
    for (theta, phi) in [(0.3_f64, 0.1_f64), (1.0, 2.0), (1.4, -0.7), (0.8, PI)] {
        let j = model.element_response(0, 1.3e8, theta, phi).unwrap();
        let pattern = |phi: f64| {
            let cos_psi = theta.sin() * phi.cos();
            let t = (FRAC_PI_2 * cos_psi).cos() / (1.0 - cos_psi * cos_psi);
            (-phi.cos() * theta.cos() * t, phi.sin() * t)
        };
        let (xt, xp) = pattern(phi);
        let (yt, yp) = pattern(phi + FRAC_PI_2);
        assert_abs_diff_eq!(j[0].re, xt, epsilon = 1e-12);
        assert_abs_diff_eq!(j[1].re, xp, epsilon = 1e-12);
        assert_abs_diff_eq!(j[2].re, yt, epsilon = 1e-12);
        assert_abs_diff_eq!(j[3].re, yp, epsilon = 1e-12);
        assert!(j.iter().all(|c| c.im == 0.0));
    }
}

#[test]
fn dipole_parallactic_matches_fee_convention() {
    // With a latitude, the conversion must be the same one used by the FEE
    // beam.
    // beam (which computes the parallactic angle via (HA, Dec); the SKA code
    // computes it directly from (az, za), which agrees away from the
    // ill-conditioned zenith).
    let model = SkaLowAnalytic::new();
    for lat in [-0.4660608448386394, 0.3, 1.2, -1.4] {
        for i in 0..500 {
            let az = (i as f64 * 0.7731) % TAU;
            let za = 1e-3 + (i as f64 * 0.3117) % 1.5;
            let raw = model.calc_jones_pair(az, za, 1e8, 0, None, false).unwrap();
            for iau in [false, true] {
                let mut expected = raw;
                crate::fee::apply_parallactic_correction(az, za, lat, iau, &mut expected);
                let j = model
                    .calc_jones_pair(az, za, 1e8, 0, Some(lat), iau)
                    .unwrap();
                assert_abs_diff_eq!(j, expected, epsilon = 1e-10);
            }
        }
    }
}

#[test]
fn parallactic_angle_near_zenith() {
    // The direct formula tends smoothly to az + pi at the zenith.
    let lat = -0.4660608448386394;
    for az in [0.1_f64, 1.0, 2.5, 4.0] {
        for za in [1e-12, 1e-9, 1e-6] {
            let pa = super::parallactic_angle(az, za, lat);
            let expected = (-az.sin()).atan2(-az.cos());
            assert_abs_diff_eq!(pa, expected, epsilon = 1e-5);
        }
    }
}

#[test]
fn dipole_array_matches_single() {
    let model = SkaLowAnalytic::new();
    let az: Vec<f64> = (0..50).map(|i| i as f64 * 0.1).collect();
    let za: Vec<f64> = (0..50).map(|i| i as f64 * 0.03).collect();
    let arr = model
        .calc_jones_array_pair(&az, &za, 1.5e8, 0, Some(-0.5), true)
        .unwrap();
    for ((&a, &z), j) in az.iter().zip(&za).zip(&arr) {
        let s = model
            .calc_jones_pair(a, z, 1.5e8, 0, Some(-0.5), true)
            .unwrap();
        assert_abs_diff_eq!(*j, s);
    }
}

// ---------------------------------------------------------------------------
// SKA-Low FEKO.
// ---------------------------------------------------------------------------

/// Write a synthetic FEKO coefficients directory.
pub(crate) fn write_feko_dir(
    name: &str,
    freqs_mhz: &[f64],
    num_elements: usize,
    l_max: usize,
) -> TempPath {
    let dir = TempPath::new(name);
    std::fs::create_dir_all(&dir.0).unwrap();
    let nc = l_max * (l_max + 2);
    for (fi, f) in freqs_mhz.iter().enumerate() {
        let path = dir.0.join(format!(
            "SKA-Low_Test_Vogel_{num_elements}_elem_FEKO_{f:.5}_MHz.h5"
        ));
        let h5 = hdf5_metno::File::create(path).unwrap();
        let coeffs = Array3::from_shape_fn((num_elements, nc, 4), |(e, c, p)| {
            let x = (e * 7 + c * 3 + p + fi * 11) as f64;
            c64::new((x * 0.37).sin(), (x * 0.91).cos()) / (1.0 + c as f64)
        });
        h5.new_dataset_builder()
            .with_data(&coeffs)
            .create("coefficients")
            .unwrap();
        let mut modes = Array2::<i32>::zeros((3, nc));
        let mut i = 0;
        for n in 1..=l_max as i32 {
            for m in -n..=n {
                modes[(0, i)] = 1;
                modes[(1, i)] = m;
                modes[(2, i)] = n;
                i += 1;
            }
        }
        h5.new_dataset_builder()
            .with_data(&modes)
            .create("modes")
            .unwrap();
    }
    dir
}

#[test]
fn feko_frequencies_and_closest() {
    let dir = write_feko_dir("feko_freqs", &[103.125, 100.0, 106.25], 2, 2);
    let feko = SkaLowEepSph::new(&dir.0).unwrap();
    assert_eq!(feko.get_freqs(), vec![100e6, 103.125e6, 106.25e6]);
    assert_eq!(feko.get_l_max(), 2);
    assert_eq!(feko.num_elements(), Some(2));
    assert_eq!(feko.find_closest_freq(50e6), 100e6);
    assert_eq!(feko.find_closest_freq(100.1e6), 100e6);
    assert_eq!(feko.find_closest_freq(102e6), 103.125e6);
    // Ties go to the lower frequency.
    assert_eq!(feko.find_closest_freq(101.5625e6), 100e6);
    assert_eq!(feko.find_closest_freq(200e6), 106.25e6);
}

#[test]
fn feko_element_out_of_range() {
    let dir = write_feko_dir("feko_element_oor", &[100.0], 2, 2);
    let feko = SkaLowEepSph::new(&dir.0).unwrap();
    assert!(feko.element_response(1, 1e8, 0.3, 0.2).is_ok());
    assert!(matches!(
        feko.element_response(2, 1e8, 0.3, 0.2),
        Err(SkaBeamError::ElementOutOfRange { .. })
    ));
}

#[test]
fn feko_l_max_truncation() {
    let dir = write_feko_dir("feko_l_max", &[100.0], 1, 3);
    let full = SkaLowEepSph::new(&dir.0).unwrap();
    let trunc = SkaLowEepSph::new_with_l_max(&dir.0, Some(1)).unwrap();
    let huge = SkaLowEepSph::new_with_l_max(&dir.0, Some(500)).unwrap();
    assert_eq!(trunc.get_l_max(), 1);
    let a = full.element_response(0, 1e8, 0.5, 0.4).unwrap();
    let b = trunc.element_response(0, 1e8, 0.5, 0.4).unwrap();
    let c = huge.element_response(0, 1e8, 0.5, 0.4).unwrap();
    assert!((a - b).iter().any(|d| d.norm() > 1e-6));
    assert_abs_diff_eq!(a, c);
}

#[test]
fn feko_zenith_and_pole_are_finite() {
    let dir = write_feko_dir("feko_zenith", &[100.0], 1, 4);
    let feko = SkaLowEepSph::new(&dir.0).unwrap();
    for theta in [0.0, 1e-9, 1e-5, 1e-3, FRAC_PI_2] {
        let j = feko.element_response(0, 1e8, theta, 0.3).unwrap();
        assert!(!j.any_nan(), "theta = {theta}");
    }
    // Responses at theta = 0 and theta below the clamp are the same.
    let a = feko.element_response(0, 1e8, 0.0, 0.3).unwrap();
    let b = feko.element_response(0, 1e8, 1e-7, 0.3).unwrap();
    assert_abs_diff_eq!(a, b);
}

#[test]
fn feko_legendre_matches_closed_form() {
    // Check the fully-normalised associated Legendre functions against closed
    // forms, for n = 1 and 2. ska-sdp-func's final (-1)^m factor cancels the
    // Condon-Shortley phase, so there is none here:
    //   n = 1: P0 = sqrt(3/2) x, P1 = sqrt(3/4) s
    //   n = 2: P0 = sqrt(5/2) (3x^2 - 1)/2, P1 = sqrt(5/12) 3 x s,
    //          P2 = sqrt(5/48) 3 s^2
    let mut p = [0.0; 4];
    for theta in [0.2_f64, 1.0, 2.5] {
        let (s, x) = theta.sin_cos();
        super::eep_sph::legendre_norm(1, x, s, &mut p);
        assert_abs_diff_eq!(p[0], (1.5_f64).sqrt() * x, epsilon = 1e-12);
        assert_abs_diff_eq!(p[1], (0.75_f64).sqrt() * s, epsilon = 1e-12);
        super::eep_sph::legendre_norm(2, x, s, &mut p);
        assert_abs_diff_eq!(
            p[0],
            (2.5_f64).sqrt() * (3.0 * x * x - 1.0) / 2.0,
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(p[1], (5.0_f64 / 12.0).sqrt() * 3.0 * x * s, epsilon = 1e-12);
        assert_abs_diff_eq!(p[2], (5.0_f64 / 48.0).sqrt() * 3.0 * s * s, epsilon = 1e-12);
    }
}

/// Use EveryBeam's SKA-Low FEKO test coefficients (EveryBeam's
/// `coeffs/skalow` directory), if the `SKA_LOW_FEKO_TEST_COEFF_PATH`
/// environment variable points to them.
fn everybeam_feko_test_coeffs() -> Option<SkaLowEepSph> {
    let dir = std::env::var("SKA_LOW_FEKO_TEST_COEFF_PATH").ok()?;
    Some(SkaLowEepSph::new(dir).unwrap())
}

#[test]
#[allow(clippy::excessive_precision)]
fn feko_matches_everybeam() {
    let Some(feko) = everybeam_feko_test_coeffs() else {
        eprintln!("SKA_LOW_FEKO_TEST_COEFF_PATH not set; skipping");
        return;
    };
    assert_eq!(feko.get_freqs(), vec![100.0e6, 103.125e6, 106.25e6]);
    assert_eq!(feko.get_l_max(), 5);

    // Expected values from EveryBeam's cpp/test/tskalowfeko.cc
    // (element_response_values): element 0, 100 MHz, theta = 45 deg, phi = k *
    // 45 deg.
    #[rustfmt::skip]
    let expected: [[f64; 8]; 8] = [
        [0.1950270132940709, 0.39925054975309954, 0.03744466661624865, 0.015703932222402392,
         -0.00575107969263793, 0.012593860497960558, -0.17909226855390134, -0.71137389782626748],
        [0.26162744839215074, 0.35620732903143476, -0.016592244982825185, -0.4819820623468671,
         -0.18652367822737387, -0.27118877344460957, 0.018482511691612157, -0.4531382384344943],
        [-0.0030900770360378833, -0.07575155822977149, -0.07237621876831796, -0.8021044818805361,
         -0.23542648781866413, -0.5537214523942271, 0.0050603122260678181, -0.12075927538183927],
        [-0.11380429036365827, -0.3812025794821093, 0.03138447553080932, -0.4441061621956064,
         -0.14823395399514913, -0.31897481103626846, -0.0401198059785099, 0.49985152182108483],
        [-0.09218438611076375, -0.35264710993538845, 0.0203011709472467, -0.05123717557816926,
         -0.05808906530672036, 0.068822784656334277, 0.06376646783969632, 0.6113554205620797],
        [-0.11644327803482148, -0.3972086451207879, 0.013081463137069486, 0.42060231947274423,
         0.13280650713078945, 0.27058277151058963, -0.02241971529162428, 0.32867340222110475],
        [0.04673859724609295, 0.025738994403213172, -0.008743970036856678, 0.7846694465427109,
         0.056196165948792627, 0.45450390929411466, 0.058203602714635296, 0.059445803903843235],
        [0.06984552930386159, 0.35038323441700714, -0.03292656126968755, 0.379240392226597,
         0.20321198541945895, 0.46224879391780371, -0.032197038461156006, -0.45859502839338673],
    ];
    let thetas = [FRAC_PI_4; 8];
    let phis: Vec<f64> = (0..8).map(|k| k as f64 * FRAC_PI_4).collect();
    let batch = feko
        .element_responses(0, 100e6, &thetas, &phis, &phis)
        .unwrap();
    for (k, e) in expected.iter().enumerate() {
        let e = Jones::from(*e);
        let single = feko.element_response(0, 100e6, FRAC_PI_4, phis[k]).unwrap();
        // EveryBeam checks to a relative tolerance of 1e-5 %.
        for i in 0..4 {
            for got in [single[i], batch[k][i]] {
                assert_abs_diff_eq!(got.re, e[i].re, epsilon = 1e-7 * e[i].re.abs().max(1e-3));
                assert_abs_diff_eq!(got.im, e[i].im, epsilon = 1e-7 * e[i].im.abs().max(1e-3));
            }
        }
    }

    // Non-zero element ids work and differ from element 0.
    let e0 = feko.element_response(0, 100e6, FRAC_PI_4, 0.0).unwrap();
    let e1 = feko.element_response(1, 100e6, FRAC_PI_4, 0.0).unwrap();
    assert!((e0 - e1).iter().any(|d| d.norm() > 1e-6));
    assert_eq!(feko.num_elements(), Some(256));
}

// ---------------------------------------------------------------------------
// Stations.
// ---------------------------------------------------------------------------

fn test_positions(n: usize) -> Vec<[f64; 3]> {
    (0..n)
        .map(|i| {
            let r = 2.0 * (i as f64).sqrt();
            let a = i as f64 * 2.399963;
            [r * a.cos(), r * a.sin(), 0.01 * i as f64]
        })
        .collect()
}

#[test]
fn station_pointing_direction_is_element_response() {
    let station = SkaStation::new(SkaLowAnalytic::new(), test_positions(16), None).unwrap();
    let (az, za) = (1.1, 0.5);
    let af = station.array_factor(az, za, 1.2e8, az, za);
    assert_abs_diff_eq!(af, c64::new(1.0, 0.0), epsilon = 1e-12);
    let s = station
        .calc_jones_pair(az, za, 1.2e8, az, za, None, false)
        .unwrap();
    let e = station
        .element()
        .calc_jones_pair(az, za, 1.2e8, 0, None, false)
        .unwrap();
    assert_abs_diff_eq!(s, e, epsilon = 1e-12);
}

#[test]
fn station_array_factor() {
    // Two elements 1 wavelength apart along East: the array factor is
    // cos(pi * (s0 - s) . r / lambda) up to a phase.
    let freq = 1e8;
    let lambda = marlu::constants::VEL_C / freq;
    let station = SkaStation::new(
        SkaLowAnalytic::new_without_taper(),
        vec![[-lambda / 2.0, 0.0, 0.0], [lambda / 2.0, 0.0, 0.0]],
        None,
    )
    .unwrap();
    for za in [0.1, 0.4, 0.9] {
        let af = station.array_factor(FRAC_PI_2, za, freq, 0.0, 0.0);
        let expected = (PI * za.sin()).cos();
        assert_abs_diff_eq!(af.re, expected, epsilon = 1e-12);
        assert_abs_diff_eq!(af.im, 0.0, epsilon = 1e-12);
    }

    // Flagging an element changes the normalisation.
    let station = SkaStation::new(
        SkaLowAnalytic::new_without_taper(),
        vec![[-lambda / 2.0, 0.0, 0.0], [lambda / 2.0, 0.0, 0.0]],
        Some(vec![1.0, 0.0]),
    )
    .unwrap();
    assert_abs_diff_eq!(
        station.array_factor(FRAC_PI_2, 0.4, freq, 0.0, 0.0).norm(),
        1.0,
        epsilon = 1e-12
    );
}

#[test]
fn station_errors() {
    assert!(matches!(
        SkaStation::new(
            SkaLowAnalytic::new_without_taper(),
            test_positions(4),
            Some(vec![1.0; 3])
        ),
        Err(SkaBeamError::WeightsLength { .. })
    ));
    assert!(matches!(
        SkaStation::new(
            SkaLowAnalytic::new_without_taper(),
            test_positions(4),
            Some(vec![0.0; 4])
        ),
        Err(SkaBeamError::AllElementsFlagged)
    ));
    let dir = write_feko_dir("station_errors", &[100.0], 3, 2);
    let feko = SkaLowEepSph::new(&dir.0).unwrap();
    assert!(matches!(
        SkaStation::new(feko, test_positions(4), None),
        Err(SkaBeamError::PositionsLength { .. })
    ));
}

#[test]
fn station_per_element_sum() {
    // With per-element responses, the station response is the weighted sum of
    // phased element responses.
    let dir = write_feko_dir("station_per_element", &[100.0], 3, 2);
    let feko = SkaLowEepSph::new(&dir.0).unwrap();
    let positions = test_positions(3);
    let weights = vec![1.0, 0.5, 2.0];
    let station = SkaStation::new(feko, positions.clone(), Some(weights.clone())).unwrap();
    let (az, za, paz, pza, freq) = (0.3, 0.6, 1.0, 0.2, 1e8);
    let got = station.station_response(az, za, freq, paz, pza).unwrap();

    let k = TAU * freq / marlu::constants::VEL_C;
    let unit = |az: f64, za: f64| [za.sin() * az.sin(), za.sin() * az.cos(), za.cos()];
    let (s, s0) = (unit(az, za), unit(paz, pza));
    let mut expected = Jones::default();
    for (i, (r, w)) in positions.iter().zip(&weights).enumerate() {
        let dot: f64 = (0..3).map(|d| (s0[d] - s[d]) * r[d]).sum();
        let e = station
            .element()
            .element_response(i, freq, za, FRAC_PI_2 - az)
            .unwrap();
        expected += e * (c64::cis(-k * dot) * *w);
    }
    expected /= 3.5;
    assert_abs_diff_eq!(got, expected, epsilon = 1e-12);
}

pub(crate) fn write_eep(name: &str, phis: &[f64], thetas: &[f64], freqs: &[f64]) -> TempPath {
    let path = TempPath::new(name);
    let h5 = hdf5_metno::File::create(&path.0).unwrap();
    let group = h5.create_group("averageEEP").unwrap();
    for (n, v) in [("Phi", phis), ("Theta", thetas), ("Freq", freqs)] {
        group
            .new_dataset_builder()
            .with_data(&arr1(v))
            .create(n)
            .unwrap();
    }
    let grid = Array5::from_shape_fn(
        (phis.len(), thetas.len(), freqs.len(), 4, 2),
        |(i, j, k, l, m)| ((i * 7 + j * 13 + k * 3 + l * 5 + m) as f64 * 0.37).sin(),
    );
    group
        .new_dataset_builder()
        .with_data(&grid)
        .create("averageEEP")
        .unwrap();
    path
}

#[test]
fn eep_array_path_matches_single_path() {
    let uniform_phis: Vec<f64> = (0..=36).map(|i| i as f64 * TAU / 36.0).collect();
    let uniform_thetas: Vec<f64> = (0..=18).map(|i| i as f64 * FRAC_PI_2 / 18.0).collect();
    let mut irregular_phis = uniform_phis.clone();
    irregular_phis[5] += 0.03;
    irregular_phis[20] -= 0.05;
    let irregular_thetas: Vec<f64> = (0..=18)
        .map(|i| (i as f64 / 18.0).powf(1.3) * FRAC_PI_2)
        .collect();
    let freqs = [50e6, 100e6, 175e6, 350e6];

    for (name, phis, thetas) in [
        ("eep_array_uniform.h5", &uniform_phis, &uniform_thetas),
        ("eep_array_irregular.h5", &irregular_phis, &irregular_thetas),
    ] {
        let file = write_eep(name, phis, thetas, &freqs);
        let eep = SkaLowEepGrid::new(&file.0).unwrap();
        // Random directions, plus directions exactly on grid points and edges.
        let mut az: Vec<f64> = (0..500).map(|i| (i as f64 * 0.7731) % TAU).collect();
        let mut za: Vec<f64> = (0..500).map(|i| (i as f64 * 0.3117) % FRAC_PI_2).collect();
        for (&p, &t) in phis.iter().zip(thetas.iter().cycle()) {
            az.push(FRAC_PI_2 - p);
            za.push(t);
        }
        az.push(0.0);
        za.push(FRAC_PI_2);
        for freq in [50e6, 77.7e6, 100e6, 350e6] {
            let arr = eep
                .calc_jones_array_pair(&az, &za, freq, 0, Some(-0.47), true)
                .unwrap();
            for ((&a, &z), j) in az.iter().zip(&za).zip(&arr) {
                let s = eep
                    .calc_jones_pair(a, z, freq, 0, Some(-0.47), true)
                    .unwrap();
                assert_abs_diff_eq!(*j, s, epsilon = 1e-12);
            }
        }
        // Out-of-range frequencies are still errors.
        assert!(eep
            .calc_jones_array_pair(&az, &za, 351e6, 0, None, false)
            .is_err());
    }
}

// ---------------------------------------------------------------------------
// Station rotation.
// ---------------------------------------------------------------------------

#[test]
fn rotation_from_p_axis_values() {
    assert_abs_diff_eq!(super::rotation_from_p_axis([1.0, 0.0, 0.0]), 0.0);
    assert_abs_diff_eq!(super::rotation_from_p_axis([0.0, 1.0, 0.0]), FRAC_PI_2);
    assert_abs_diff_eq!(super::rotation_from_p_axis([-1.0, 0.0, 0.1]), PI);
    let a = 0.7_f64;
    assert_abs_diff_eq!(super::rotation_from_p_axis([a.cos(), a.sin(), 0.0]), a);
}

#[test]
fn rotated_station_matches_everybeam_projection() {
    // EveryBeam evaluates OSKAR element responses in each station's
    // (p, q, r) frame: theta/phi come from projecting the direction onto the
    // station axes. A station rotated by alpha has p = (cos a, sin a, 0) and
    // q = (-sin a, cos a, 0) in ENU.
    let dir = write_feko_dir("rotated_station", &[100.0], 3, 3);
    let feko = SkaLowEepSph::new(&dir.0).unwrap();
    let positions = vec![[1.0, 2.0, 0.0], [-3.0, 0.5, 0.1], [0.2, -1.7, 0.0]];
    let weights = vec![1.0, 0.7, 1.3];
    let (paz, pza, freq): (f64, f64, f64) = (0.4, 0.3, 100e6);
    for alpha in [0.0_f64, 0.3, 1.9, -2.5] {
        let station = SkaStation::new(&feko, positions.clone(), Some(weights.clone()))
            .unwrap()
            .with_rotation(alpha);
        let (p, q) = (
            [alpha.cos(), alpha.sin(), 0.0],
            [-alpha.sin(), alpha.cos(), 0.0],
        );
        for (az, za) in [(0.1_f64, 0.2_f64), (2.0, 0.9), (4.5, 1.3)] {
            let d = [za.sin() * az.sin(), za.sin() * az.cos(), za.cos()];
            let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
            let (x, y, z) = (dot(p, d), dot(q, d), d[2]);
            let theta_local = FRAC_PI_2 - z.atan2(x.hypot(y));
            let phi_local = y.atan2(x);

            let s0 = [pza.sin() * paz.sin(), pza.sin() * paz.cos(), pza.cos()];
            let k = TAU * freq / marlu::constants::VEL_C;
            let mut expected = Jones::default();
            for (i, (r, w)) in positions.iter().zip(&weights).enumerate() {
                let phase =
                    -k * ((s0[0] - d[0]) * r[0] + (s0[1] - d[1]) * r[1] + (s0[2] - d[2]) * r[2]);
                let e = feko
                    .element_response(i, freq, theta_local, phi_local)
                    .unwrap();
                expected += e * (c64::cis(phase) * *w);
            }
            expected /= weights.iter().sum::<f64>();

            let got = station.station_response(az, za, freq, paz, pza).unwrap();
            assert_abs_diff_eq!(got, expected, epsilon = 1e-10);
        }
    }
}

#[test]
fn rotating_by_90_degrees_swaps_dipoles() {
    // Rotating a station by +90 deg puts its X dipoles along the N-S line
    // (where the unrotated Y dipoles were) and its Y dipoles along the E-W
    // line. A dipole's sign is a convention: OSKAR/EveryBeam define the Y
    // dipole as the X pattern evaluated at phi + 90 deg, so the rotated X is
    // -(unrotated Y) and the rotated Y is +(unrotated X). (Consequently a
    // station rotated by 180 deg has an overall sign flip, as in EveryBeam.)
    // A single element at the origin has an array factor of 1.
    let model = SkaLowAnalytic::new();
    let unrotated = SkaStation::new(&model, vec![[0.0; 3]], None).unwrap();
    let rotated = SkaStation::new(&model, vec![[0.0; 3]], None)
        .unwrap()
        .with_rotation(FRAC_PI_2);
    for (az, za) in [(0.3_f64, 0.4_f64), (2.2, 1.1), (5.0, 0.05)] {
        for lat in [None, Some(-0.4660608448386394)] {
            let u = unrotated
                .calc_jones_pair(az, za, 1.2e8, 0.0, 0.0, lat, false)
                .unwrap();
            let r = rotated
                .calc_jones_pair(az, za, 1.2e8, 0.0, 0.0, lat, false)
                .unwrap();
            assert_abs_diff_eq!(r[0], -u[2], epsilon = 1e-12);
            assert_abs_diff_eq!(r[1], -u[3], epsilon = 1e-12);
            assert_abs_diff_eq!(r[2], u[0], epsilon = 1e-12);
            assert_abs_diff_eq!(r[3], u[1], epsilon = 1e-12);
        }
    }
}

#[test]
fn rotated_eep_grid_station() {
    // For identical elements, a rotated station is the array factor times the
    // element response evaluated at phi - rotation.
    let phis: Vec<f64> = (0..=72).map(|i| i as f64 * TAU / 72.0).collect();
    let thetas: Vec<f64> = (0..=18).map(|i| i as f64 * FRAC_PI_2 / 18.0).collect();
    let file = write_eep("rotated_eep_grid.h5", &phis, &thetas, &[50e6, 350e6]);
    let eep = SkaLowEepGrid::new(&file.0).unwrap();
    let positions = vec![[1.0, 2.0, 0.0], [-3.0, 0.5, 0.0]];
    let alpha = 0.83;
    let station = SkaStation::new(&eep, positions.clone(), None)
        .unwrap()
        .with_rotation(alpha);
    let (az, za, f) = (1.3, 0.7, 1.5e8);
    let af = station.array_factor(az, za, f, 0.0, 0.0);
    let e = eep
        .element_response(0, f, za, FRAC_PI_2 - az - alpha)
        .unwrap();
    assert_abs_diff_eq!(
        station.station_response(az, za, f, 0.0, 0.0).unwrap(),
        e * af,
        epsilon = 1e-12
    );
}

// ---------------------------------------------------------------------------
// Local frames.
// ---------------------------------------------------------------------------

/// The rows of a local frame whose East is `beta` from the array's East
/// towards North (a rotation about the common vertical).
fn vertical_rotation_frame(beta: f64) -> [[f64; 3]; 3] {
    let (s, c) = beta.sin_cos();
    [[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]]
}

/// The rows of the local frame of a station displaced from the array centre
/// by small angles (east `de`, north `dn`, radians) on a sphere: its vertical
/// is tilted.
fn tilted_frame(de: f64, dn: f64) -> [[f64; 3]; 3] {
    // Rotate the array's (E, N, U) about N by -de (tilting U towards E), then
    // about the new E by dn (tilting U towards N).
    let (se, ce) = de.sin_cos();
    let (sn, cn) = dn.sin_cos();
    let e = [ce, 0.0, -se];
    let u0 = [se, 0.0, ce];
    let n0 = [0.0, 1.0, 0.0];
    let n = [
        cn * n0[0] - sn * u0[0],
        cn * n0[1] - sn * u0[1],
        cn * n0[2] - sn * u0[2],
    ];
    let u = [
        sn * n0[0] + cn * u0[0],
        sn * n0[1] + cn * u0[1],
        sn * n0[2] + cn * u0[2],
    ];
    [e, n, u]
}

#[test]
fn identity_local_frame_changes_nothing() {
    let lat = -0.4660608448386394;
    let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let plain = SkaStation::new(SkaLowAnalytic::new(), test_positions(16), None)
        .unwrap()
        .with_rotation(0.4);
    let framed = SkaStation::new(SkaLowAnalytic::new(), test_positions(16), None)
        .unwrap()
        .with_rotation(0.4)
        .with_local_frame(identity);
    for (az, za) in [(0.3, 0.2), (2.0, 1.1), (4.5, 0.0), (5.9, 1.5)] {
        for lat in [None, Some(lat)] {
            let a = plain
                .calc_jones_pair(az, za, 1.2e8, 0.5, 0.3, lat, true)
                .unwrap();
            let b = framed
                .calc_jones_pair(az, za, 1.2e8, 0.5, 0.3, lat, true)
                .unwrap();
            assert_abs_diff_eq!(a, b, epsilon = 1e-13);
        }
    }
}

#[test]
fn vertical_rotation_frame_is_a_station_rotation() {
    // A station whose local frame is rotated about the vertical by beta, with
    // rotation alpha and positions p (in its frame), is physically the same as
    // a station without a local frame, rotation alpha + beta and positions
    // rotated by beta. On the sky (theta and azimuth components, and with the
    // parallactic-angle correction), their responses are the same.
    let lat = -0.4660608448386394;
    let (alpha, beta): (f64, f64) = (0.4, 0.27);
    let positions = test_positions(16);
    let (s, c) = beta.sin_cos();
    let rotated: Vec<[f64; 3]> = positions
        .iter()
        .map(|p| [c * p[0] - s * p[1], s * p[0] + c * p[1], p[2]])
        .collect();
    let framed = SkaStation::new(SkaLowAnalytic::new(), positions, None)
        .unwrap()
        .with_rotation(alpha)
        .with_local_frame(vertical_rotation_frame(beta));
    let plain = SkaStation::new(SkaLowAnalytic::new(), rotated, None)
        .unwrap()
        .with_rotation(alpha + beta);
    for (az, za) in [(0.3, 0.2), (2.0, 1.1), (4.5, 0.6), (5.9, 1.5)] {
        for lat in [None, Some(lat)] {
            for iau in [false, true] {
                let a = framed
                    .calc_jones_pair(az, za, 1.2e8, 0.5, 0.3, lat, iau)
                    .unwrap();
                let b = plain
                    .calc_jones_pair(az, za, 1.2e8, 0.5, 0.3, lat, iau)
                    .unwrap();
                assert_abs_diff_eq!(a, b, epsilon = 1e-12);
            }
        }
    }
}

#[test]
fn tilted_frame_moves_the_beam() {
    // A station whose vertical is tilted (e.g. one far from the array centre)
    // still has its peak in its pointing direction (given in the array frame),
    // and its local frame matters away from the pointing direction.
    let frame = tilted_frame(0.006, -0.004);
    let framed = SkaStation::new(SkaLowAnalytic::new(), test_positions(64), None)
        .unwrap()
        .with_local_frame(frame);
    let plain = SkaStation::new(SkaLowAnalytic::new(), test_positions(64), None).unwrap();
    let (paz, pza) = (1.1, 0.5);
    let (laz, lza) = station::to_local(&frame, paz, pza);
    // At the pointing direction, the framed station's response is the element
    // response in the local frame.
    let at_pointing = framed
        .calc_jones_pair(paz, pza, 1.5e8, paz, pza, None, false)
        .unwrap();
    let element = SkaLowAnalytic::new()
        .calc_jones_pair(laz, lza, 1.5e8, 0, None, false)
        .unwrap();
    assert_abs_diff_eq!(at_pointing, element, epsilon = 1e-12);
    // Elsewhere, the frames differ (by about the tilt).
    let a = framed
        .calc_jones_pair(1.3, 0.62, 1.5e8, paz, pza, None, false)
        .unwrap();
    let b = plain
        .calc_jones_pair(1.3, 0.62, 1.5e8, paz, pza, None, false)
        .unwrap();
    let diff = (a - b).norm_sqr().iter().sum::<f64>().sqrt();
    assert!(diff > 1e-4 && diff < 0.2, "{diff}");
}

#[test]
fn local_parallactic_angle_matches_direct_with_identity() {
    let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for (az, za, lat) in [(0.3, 0.2, -0.47), (2.0, 1.1, 0.3), (4.5, 0.01, -1.2)] {
        assert_abs_diff_eq!(
            station::local_parallactic_angle(&identity, az, za, lat),
            parallactic_angle(az, za, lat),
            epsilon = 1e-12
        );
    }
}
