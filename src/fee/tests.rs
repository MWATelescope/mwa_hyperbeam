// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Tests for FEE beam code.

use super::*;
use crate::jones_test::TestJones;
use approx::*;
use ndarray::prelude::*;
use serial_test::serial;

#[test]
fn test_fix_amps_1() {
    let amps = [1.0; 16];
    let delays = [0; 16];
    let result = Array1::from(fix_amps(&amps, &delays).to_vec());
    let expected = Array1::ones(32);
    assert_abs_diff_eq!(result, expected);
}

#[test]
fn test_fix_amps_2() {
    let mut amps = [1.0; 16];
    let delays = [3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0];
    let result = Array1::from(fix_amps(&amps, &delays).to_vec());
    let expected = Array1::ones(32);
    // No problems here.
    assert_abs_diff_eq!(result, expected);

    // But what if we turn off one of the dipoles? Then both the X and Y
    // should also be off.
    amps[1] = 0.0;
    let result = Array1::from(fix_amps(&amps, &delays).to_vec());
    assert_abs_diff_eq!(
        result,
        array![
            1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
        ]
    );
}

#[test]
fn test_fix_amps_3() {
    let mut amps = [1.0; 32];
    let delays = [3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0];
    let result = Array1::from(fix_amps(&amps, &delays).to_vec());
    let expected = Array1::ones(32);
    assert_abs_diff_eq!(result, expected);

    amps[1] = 0.0;
    let result = Array1::from(fix_amps(&amps, &delays).to_vec());
    assert_abs_diff_eq!(
        result,
        array![
            1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
        ]
    );
}

#[test]
fn test_fix_amps_4() {
    let amps = [1.0; 16];
    // Bad delay present.
    let delays = [3, 2, 1, 0, 32, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0];
    let result = Array1::from(fix_amps(&amps, &delays).to_vec());
    // Both X and Y turned off.
    assert_abs_diff_eq!(
        result,
        array![
            1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
        ]
    );

    let amps = [1.0; 32];
    let result = Array1::from(fix_amps(&amps, &delays).to_vec());
    // Again, both X and Y turned off.
    assert_abs_diff_eq!(
        result,
        array![
            1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
        ]
    );
}

#[test]
#[serial]
fn new() {
    let beam = FEEBeam::new("mwa_full_embedded_element_pattern.h5");
    assert!(beam.is_ok());
}

#[test]
#[serial]
fn new_from_env() {
    std::env::set_var("MWA_BEAM_FILE", "mwa_full_embedded_element_pattern.h5");
    let beam = FEEBeam::new_from_env();
    assert!(beam.is_ok());
}

#[test]
#[serial]
fn test_find_nearest_freq() {
    let beam = FEEBeam::new("mwa_full_embedded_element_pattern.h5").unwrap();
    // Dancing around an available freq.
    assert_eq!(beam.find_closest_freq(51199999), 51200000);
    assert_eq!(beam.find_closest_freq(51200000), 51200000);
    assert_eq!(beam.find_closest_freq(51200001), 51200000);
    // On the precipice of choosing between two freqs: 51200000 and
    // 52480000. When searching with 51840000, we will get the same
    // difference in frequency for both nearby, defined freqs. Because we
    // compare with "less than", the first freq. will be selected. This
    // should be consistent with the C++ code.
    assert_eq!(beam.find_closest_freq(51840000), 51200000);
    assert_eq!(beam.find_closest_freq(51840001), 52480000);
}

#[test]
#[serial]
/// Check that we can open the dataset "X16_51200000".
fn test_get_dataset() {
    let beam = FEEBeam::new("mwa_full_embedded_element_pattern.h5").unwrap();
    assert!(beam.get_dataset("X16_51200000").is_ok());
}

#[test]
#[serial]
fn test_get_modes() {
    let beam = FEEBeam::new("mwa_full_embedded_element_pattern.h5").unwrap();
    let result = beam.get_modes(51200000, &[0; 16], &[1.0; 32]);
    assert!(result.is_ok());
    let coeffs = result.unwrap();

    // Values taken from the C++ code.
    // m_accum and n_accum are floats in the C++ code, but these appear to
    // always be small integers. I've converted the C++ output to ints here.
    let x_m_expected = array![
        -1, 0, 1, -2, -1, 0, 1, 2, -3, -2, -1, 0, 1, 2, 3, -4, -3, -2, -1, 0, 1, 2, 3, 4, -5, -4,
        -3, -2, -1, 0, 1, 2, 3, 4, 5, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, -7, -6, -5, -4,
        -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7,
        8, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, -10, -9, -8, -7, -6,
        -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, -11, -10, -9, -8, -7, -6, -5, -4, -3,
        -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3,
        -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, -13, -12, -11, -10, -9, -8, -7, -6, -5,
        -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, -14, -13, -12, -11, -10, -9,
        -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, -15, -14,
        -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        11, 12, 13, 14, 15, -16, -15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1,
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
    ];
    let y_m_expected = array![
        -1, 0, 1, -2, -1, 0, 1, 2, -3, -2, -1, 0, 1, 2, 3, -4, -3, -2, -1, 0, 1, 2, 3, 4, -5, -4,
        -3, -2, -1, 0, 1, 2, 3, 4, 5, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, -7, -6, -5, -4,
        -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7,
        8, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, -10, -9, -8, -7, -6,
        -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, -11, -10, -9, -8, -7, -6, -5, -4, -3,
        -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3,
        -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, -13, -12, -11, -10, -9, -8, -7, -6, -5,
        -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, -14, -13, -12, -11, -10, -9,
        -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, -15, -14,
        -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        11, 12, 13, 14, 15, -16, -15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1,
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
    ];
    let x_n_expected = array![
        1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5,
        5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
        7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
        9, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
        10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11,
        11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
        12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13,
        13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
        14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15,
        15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
        15, 15, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
        16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16
    ];
    let y_n_expected = array![
        1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5,
        5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
        7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
        9, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
        10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11,
        11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
        12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13,
        13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
        14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15,
        15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
        15, 15, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
        16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16
    ];

    let x_q1_expected_first = array![
        c64::new(-0.024744, 0.009424),
        c64::new(0.000000, 0.000000),
        c64::new(-0.024734, 0.009348),
        c64::new(0.000000, -0.000000),
        c64::new(0.005766, 0.015469),
    ];
    let x_q1_expected_last = array![
        c64::new(0.0, 0.0),
        c64::new(0.0, 0.0),
        c64::new(0.0, 0.0),
        c64::new(0.0, 0.0),
        c64::new(0.0, 0.0),
    ];

    let x_q2_expected_first = array![
        c64::new(-0.026122, 0.009724),
        c64::new(-0.000000, -0.000000),
        c64::new(0.026116, -0.009643),
        c64::new(0.000000, -0.000000),
        c64::new(0.006586, 0.018925),
    ];
    let x_q2_expected_last = array![
        c64::new(0.0, 0.0),
        c64::new(0.0, 0.0),
        c64::new(0.0, 0.0),
        c64::new(0.0, 0.0),
        c64::new(0.0, 0.0),
    ];

    let y_q1_expected_first = array![
        c64::new(-0.009398, -0.024807),
        c64::new(0.000000, -0.000000),
        c64::new(0.009473, 0.024817),
        c64::new(0.000000, 0.000000),
        c64::new(-0.015501, 0.005755),
    ];
    let y_q1_expected_last = array![
        c64::new(0.0, 0.0),
        c64::new(0.0, 0.0),
        c64::new(0.0, 0.0),
        c64::new(0.0, 0.0),
        c64::new(0.0, 0.0),
    ];

    let y_q2_expected_first = array![
        c64::new(-0.009692, -0.026191),
        c64::new(0.000000, 0.000000),
        c64::new(-0.009773, -0.026196),
        c64::new(0.000000, 0.000000),
        c64::new(-0.018968, 0.006566),
    ];
    let y_q2_expected_last = array![
        c64::new(0.0, 0.0),
        c64::new(0.0, 0.0),
        c64::new(0.0, 0.0),
        c64::new(0.0, 0.0),
        c64::new(0.0, 0.0),
    ];

    assert_eq!(Array1::from(coeffs.x.m_accum.clone()), x_m_expected);
    assert_eq!(Array1::from(coeffs.y.m_accum.clone()), y_m_expected);
    assert_eq!(Array1::from(coeffs.x.n_accum.clone()), x_n_expected);
    assert_eq!(Array1::from(coeffs.y.n_accum.clone()), y_n_expected);

    let x_q1_accum_arr = Array1::from(coeffs.x.q1_accum.clone());
    assert_abs_diff_eq!(
        x_q1_accum_arr.slice(s![0..5]),
        x_q1_expected_first,
        epsilon = 1e-6
    );
    assert_abs_diff_eq!(
        x_q1_accum_arr.slice(s![-5..]),
        x_q1_expected_last,
        epsilon = 1e-6
    );

    let x_q2_accum_arr = Array1::from(coeffs.x.q2_accum.clone());
    assert_abs_diff_eq!(
        x_q2_accum_arr.slice(s![0..5]),
        x_q2_expected_first,
        epsilon = 1e-6
    );
    assert_abs_diff_eq!(
        x_q2_accum_arr.slice(s![-5..]),
        x_q2_expected_last,
        epsilon = 1e-6
    );

    let y_q1_accum_arr = Array1::from(coeffs.y.q1_accum.clone());
    assert_abs_diff_eq!(
        y_q1_accum_arr.slice(s![0..5]),
        y_q1_expected_first,
        epsilon = 1e-6
    );
    assert_abs_diff_eq!(
        y_q1_accum_arr.slice(s![-5..]),
        y_q1_expected_last,
        epsilon = 1e-6
    );

    let y_q2_accum_arr = Array1::from(coeffs.y.q2_accum.clone());
    assert_abs_diff_eq!(
        y_q2_accum_arr.slice(s![0..5]),
        y_q2_expected_first,
        epsilon = 1e-6
    );
    assert_abs_diff_eq!(
        y_q2_accum_arr.slice(s![-5..]),
        y_q2_expected_last,
        epsilon = 1e-6
    );
}

#[test]
fn test_get_modes2() {
    let beam = FEEBeam::new("mwa_full_embedded_element_pattern.h5").unwrap();
    let result = beam.get_modes(
        51200000,
        &[3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0],
        &[
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0,
        ],
    );
    assert!(result.is_ok());
    let coeffs = result.unwrap();

    // Values taken from the C++ code.
    let x_m_expected = array![
        -1, 0, 1, -2, -1, 0, 1, 2, -3, -2, -1, 0, 1, 2, 3, -4, -3, -2, -1, 0, 1, 2, 3, 4, -5, -4,
        -3, -2, -1, 0, 1, 2, 3, 4, 5, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, -7, -6, -5, -4,
        -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7,
        8, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, -10, -9, -8, -7, -6,
        -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, -11, -10, -9, -8, -7, -6, -5, -4, -3,
        -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3,
        -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, -13, -12, -11, -10, -9, -8, -7, -6, -5,
        -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, -14, -13, -12, -11, -10, -9,
        -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, -15, -14,
        -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        11, 12, 13, 14, 15, -16, -15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1,
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
    ];
    let y_m_expected = array![
        -1, 0, 1, -2, -1, 0, 1, 2, -3, -2, -1, 0, 1, 2, 3, -4, -3, -2, -1, 0, 1, 2, 3, 4, -5, -4,
        -3, -2, -1, 0, 1, 2, 3, 4, 5, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, -7, -6, -5, -4,
        -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7,
        8, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, -10, -9, -8, -7, -6,
        -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, -11, -10, -9, -8, -7, -6, -5, -4, -3,
        -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3,
        -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, -13, -12, -11, -10, -9, -8, -7, -6, -5,
        -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, -14, -13, -12, -11, -10, -9,
        -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, -15, -14,
        -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        11, 12, 13, 14, 15, -16, -15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1,
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
    ];
    let x_n_expected = array![
        1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5,
        5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
        7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
        9, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
        10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11,
        11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
        12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13,
        13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
        14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15,
        15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
        15, 15, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
        16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16
    ];
    let y_n_expected = array![
        1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5,
        5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
        7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
        9, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
        10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11,
        11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
        12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13,
        13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
        14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15,
        15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
        15, 15, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
        16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16
    ];

    let x_q1_expected_first = array![
        c64::new(-0.020504, 0.013376),
        c64::new(-0.001349, 0.000842),
        c64::new(-0.020561, 0.013291),
        c64::new(0.001013, 0.001776),
        c64::new(0.008222, 0.012569),
    ];
    let x_q1_expected_last = array![
        c64::new(0.0, 0.0),
        c64::new(0.0, 0.0),
        c64::new(0.0, 0.0),
        c64::new(0.0, 0.0),
        c64::new(0.0, 0.0),
    ];

    let x_q2_expected_first = array![
        c64::new(-0.021903, 0.013940),
        c64::new(0.001295, -0.000767),
        c64::new(0.021802, -0.014047),
        c64::new(0.001070, 0.002039),
        c64::new(0.009688, 0.016040),
    ];
    let x_q2_expected_last = array![
        c64::new(0.0, 0.0),
        c64::new(0.0, 0.0),
        c64::new(0.0, 0.0),
        c64::new(0.0, 0.0),
        c64::new(0.0, 0.0),
    ];

    let y_q1_expected_first = array![
        c64::new(-0.013471, -0.020753),
        c64::new(0.001130, 0.002400),
        c64::new(0.013576, 0.020683),
        c64::new(-0.001751, 0.001023),
        c64::new(-0.013183, 0.008283),
    ];
    let y_q1_expected_last = array![
        c64::new(0.0, 0.0),
        c64::new(0.0, 0.0),
        c64::new(0.0, 0.0),
        c64::new(0.0, 0.0),
        c64::new(0.0, 0.0),
    ];

    let y_q2_expected_first = array![
        c64::new(-0.014001, -0.021763),
        c64::new(-0.000562, -0.000699),
        c64::new(-0.013927, -0.021840),
        c64::new(-0.002247, 0.001152),
        c64::new(-0.015716, 0.009685),
    ];
    let y_q2_expected_last = array![
        c64::new(0.0, 0.0),
        c64::new(0.0, 0.0),
        c64::new(0.0, 0.0),
        c64::new(0.0, 0.0),
        c64::new(0.0, 0.0),
    ];

    assert_eq!(Array1::from(coeffs.x.m_accum.clone()), x_m_expected);
    assert_eq!(Array1::from(coeffs.y.m_accum.clone()), y_m_expected);
    assert_eq!(Array1::from(coeffs.x.n_accum.clone()), x_n_expected);
    assert_eq!(Array1::from(coeffs.y.n_accum.clone()), y_n_expected);

    let x_q1_accum_arr = Array1::from(coeffs.x.q1_accum.clone());
    assert_abs_diff_eq!(
        x_q1_accum_arr.slice(s![0..5]),
        x_q1_expected_first,
        epsilon = 1e-6
    );
    assert_abs_diff_eq!(
        x_q1_accum_arr.slice(s![-5..]),
        x_q1_expected_last,
        epsilon = 1e-6
    );

    let x_q2_accum_arr = Array1::from(coeffs.x.q2_accum.clone());
    assert_abs_diff_eq!(
        x_q2_accum_arr.slice(s![0..5]),
        x_q2_expected_first,
        epsilon = 1e-6
    );
    assert_abs_diff_eq!(
        x_q2_accum_arr.slice(s![-5..]),
        x_q2_expected_last,
        epsilon = 1e-6
    );

    let y_q1_accum_arr = Array1::from(coeffs.y.q1_accum.clone());
    assert_abs_diff_eq!(
        y_q1_accum_arr.slice(s![0..5]),
        y_q1_expected_first,
        epsilon = 1e-6
    );
    assert_abs_diff_eq!(
        y_q1_accum_arr.slice(s![-5..]),
        y_q1_expected_last,
        epsilon = 1e-6
    );

    let y_q2_accum_arr = Array1::from(coeffs.y.q2_accum.clone());
    assert_abs_diff_eq!(
        y_q2_accum_arr.slice(s![0..5]),
        y_q2_expected_first,
        epsilon = 1e-6
    );
    assert_abs_diff_eq!(
        y_q2_accum_arr.slice(s![-5..]),
        y_q2_expected_last,
        epsilon = 1e-6
    );

    // Check that if the Y dipole gains are different, they don't match the
    // earlier values.

    // We need to drop the reference to the coefficients used before, otherwise
    // we'll deadlock the cache.
    drop(coeffs);
    let result = beam.get_modes(
        51200000,
        &[3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0],
        &[
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0,
            // First value here
            10.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0,
        ],
    );
    assert!(result.is_ok());
    let coeffs = result.unwrap();

    // X values are the same.
    let x_q1_accum_arr = Array1::from(coeffs.x.q1_accum.clone());
    assert_abs_diff_eq!(
        x_q1_accum_arr.slice(s![0..5]),
        x_q1_expected_first,
        epsilon = 1e-6
    );

    let x_q2_accum_arr = Array1::from(coeffs.x.q2_accum.clone());
    assert_abs_diff_eq!(
        x_q2_accum_arr.slice(s![0..5]),
        x_q2_expected_first,
        epsilon = 1e-6
    );

    // Y values are not the same.
    let y_q1_accum_arr = Array1::from(coeffs.y.q1_accum.clone());
    assert_abs_diff_ne!(
        y_q1_accum_arr.slice(s![0..5]),
        y_q1_expected_first,
        epsilon = 1e-6
    );

    let y_q2_accum_arr = Array1::from(coeffs.y.q2_accum.clone());
    assert_abs_diff_ne!(
        y_q2_accum_arr.slice(s![0..5]),
        y_q2_expected_first,
        epsilon = 1e-6
    );

    // Test against expected Y values.
    let y_q1_expected_first = array![
        c64::new(-0.020510457596022602, -0.02719067783451879),
        c64::new(-0.005893442096591942, 0.010353674045181267),
        c64::new(0.02068524851191211, 0.028624921920498963),
        c64::new(-0.00899761240125443, 0.0017058598981518776),
        c64::new(-0.016074160827913245, 0.016292904120669145),
    ];
    let y_q2_expected_first = array![
        c64::new(-0.019651382591095376, -0.030546413628593075),
        c64::new(-0.0057627241860343775, -0.00416648736914009),
        c64::new(-0.0224990478461125, -0.02754743859133888),
        c64::new(-0.011107944651304743, 0.0002948793061857774),
        c64::new(-0.024422943755493767, 0.014316796508644876),
    ];

    assert_abs_diff_eq!(
        y_q1_accum_arr.slice(s![0..5]),
        y_q1_expected_first,
        epsilon = 1e-6
    );

    assert_abs_diff_eq!(
        y_q2_accum_arr.slice(s![0..5]),
        y_q2_expected_first,
        epsilon = 1e-6
    );
}

#[test]
#[serial]
fn test_calc_jones_eng() {
    let beam = FEEBeam::new("mwa_full_embedded_element_pattern.h5").unwrap();
    let result = beam.calc_jones_eng(
        45.0_f64.to_radians(),
        10.0_f64.to_radians(),
        51200000,
        &[0; 16],
        &[1.0; 16],
        false,
    );
    assert!(result.is_ok());
    let jones = result.unwrap();

    let expected = Jones::from([
        c64::new(0.036179, 0.103586),
        c64::new(0.036651, 0.105508),
        c64::new(0.036362, 0.103868),
        c64::new(-0.036836, -0.105791),
    ]);
    let jones = TestJones::from(jones);
    let expected = TestJones::from(expected);
    assert_abs_diff_eq!(jones, expected, epsilon = 1e-6);
}

#[test]
#[serial]
fn test_calc_jones_eng_2() {
    let beam = FEEBeam::new("mwa_full_embedded_element_pattern.h5").unwrap();
    let result = beam.calc_jones_eng(
        70.0_f64.to_radians(),
        10.0_f64.to_radians(),
        51200000,
        &[3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0],
        &[
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0,
        ],
        false,
    );
    assert!(result.is_ok());
    let jones = result.unwrap();

    let expected = Jones::from([
        c64::new(0.068028, 0.111395),
        c64::new(0.025212, 0.041493),
        c64::new(0.024792, 0.040577),
        c64::new(-0.069501, -0.113706),
    ]);
    let jones = TestJones::from(jones);
    let expected = TestJones::from(expected);
    assert_abs_diff_eq!(jones, expected, epsilon = 1e-6);
}

#[test]
#[serial]
fn test_calc_jones_eng_norm() {
    let beam = FEEBeam::new("mwa_full_embedded_element_pattern.h5").unwrap();
    let result = beam.calc_jones_eng(0.1_f64, 0.1_f64, 150000000, &[0; 16], &[1.0; 16], true);
    assert!(result.is_ok());
    let jones = result.unwrap();

    let expected = Jones::from([
        c64::new(0.0887949, 0.0220569),
        c64::new(0.891024, 0.2211),
        c64::new(0.887146, 0.216103),
        c64::new(-0.0896141, -0.021803),
    ]);
    let jones = TestJones::from(jones);
    let expected = TestJones::from(expected);
    assert_abs_diff_eq!(jones, expected, epsilon = 1e-6);
}

#[test]
#[serial]
fn test_calc_jones_eng_norm_2() {
    let beam = FEEBeam::new("mwa_full_embedded_element_pattern.h5").unwrap();
    let result = beam.calc_jones_eng(
        0.1_f64,
        0.1_f64,
        150000000,
        &[3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0],
        &[
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0,
        ],
        true,
    );
    assert!(result.is_ok());
    let jones = result.unwrap();

    let expected = Jones::from([
        c64::new(0.0704266, -0.0251082),
        c64::new(0.705241, -0.254518),
        c64::new(0.697787, -0.257219),
        c64::new(-0.0711516, 0.0264293),
    ]);
    let jones = TestJones::from(jones);
    let expected = TestJones::from(expected);
    assert_abs_diff_eq!(jones, expected, epsilon = 1e-6);
}

#[test]
#[serial]
fn test_calc_jones() {
    let beam = FEEBeam::new("mwa_full_embedded_element_pattern.h5").unwrap();
    let result = beam.calc_jones(
        45.0_f64.to_radians(),
        10.0_f64.to_radians(),
        51200000,
        &[0; 16],
        &[1.0; 16],
        false,
    );
    assert!(result.is_ok());
    let jones = result.unwrap();

    let expected = Jones::from([
        c64::new(0.051673288904250436, 0.14798615369209014),
        c64::new(-0.0029907711920181858, -0.008965331092654419),
        c64::new(0.002309524016541907, 0.006230549725189563),
        c64::new(0.05144802517335513, 0.14772685224822762),
    ]);
    let jones = TestJones::from(jones);
    let expected = TestJones::from(expected);
    assert_abs_diff_eq!(jones, expected, epsilon = 1e-6);
}

#[test]
#[serial]
fn test_calc_jones_norm() {
    let beam = FEEBeam::new("mwa_full_embedded_element_pattern.h5").unwrap();
    let result = beam.calc_jones(0.1_f64, 0.1_f64, 150000000, &[0; 16], &[1.0; 16], true);
    assert!(result.is_ok());
    let jones = result.unwrap();

    let expected = Jones::from([
        c64::new(0.8916497260404116, 0.21719761321518402),
        c64::new(-0.004453788880133702, -0.0010585985171330595),
        c64::new(0.003267814789407464, 0.0008339646338281076),
        c64::new(0.8954320133256206, 0.22219600210153623),
    ]);
    let jones = TestJones::from(jones);
    let expected = TestJones::from(expected);
    assert_abs_diff_eq!(jones, expected, epsilon = 1e-6);
}

#[test]
#[serial]
fn test_calc_jones_array() {
    let beam = FEEBeam::new("mwa_full_embedded_element_pattern.h5").unwrap();
    let result = beam.calc_jones(
        45.0_f64.to_radians(),
        10.0_f64.to_radians(),
        51200000,
        &[0; 16],
        &[1.0; 16],
        true,
    );
    assert!(result.is_ok());
    let jones = result.unwrap();

    let result = beam.calc_jones_array(
        &[45.0_f64.to_radians()],
        &[10.0_f64.to_radians()],
        51200000,
        &[0; 16],
        &[1.0; 16],
        true,
    );
    assert!(result.is_ok());
    let jones_array = result.unwrap();

    assert_eq!(jones_array.len(), 1);
    let jones = TestJones::from(jones);
    let jones_array = TestJones::from(jones_array[0]);
    assert_eq!(jones, jones_array);
}

#[test]
#[serial]
fn test_empty_cache() {
    let beam = FEEBeam::new("mwa_full_embedded_element_pattern.h5").unwrap();
    let result = beam.calc_jones(
        45.0_f64.to_radians(),
        10.0_f64.to_radians(),
        51200000,
        &[0; 16],
        &[1.0; 16],
        true,
    );
    assert!(result.is_ok());
    result.unwrap();

    assert!(!beam.coeff_cache.read().is_empty());
    assert!(!beam.norm_cache.read().is_empty());

    beam.empty_cache();
    assert!(beam.coeff_cache.read().is_empty());
    assert!(beam.norm_cache.read().is_empty());
}

// If the beam file is fine, then there should be frequencies inside it.
#[test]
#[serial]
fn test_get_freqs() {
    let beam = FEEBeam::new("mwa_full_embedded_element_pattern.h5").unwrap();
    assert!(!beam.get_freqs().is_empty());
}

// Tests for coverage follow.

#[test]
#[serial]
fn test_cache_is_used() {
    let beam = FEEBeam::new("mwa_full_embedded_element_pattern.h5").unwrap();
    let result = beam.calc_jones(
        45.0_f64.to_radians(),
        10.0_f64.to_radians(),
        51200000,
        &[0; 16],
        &[1.0; 16],
        true,
    );
    assert!(result.is_ok());
    result.unwrap();

    let result = beam.calc_jones(
        45.0_f64.to_radians(),
        10.0_f64.to_radians(),
        51200000,
        &[0; 16],
        &[1.0; 16],
        true,
    );
    assert!(result.is_ok());
    result.unwrap();
}

// Tests to expose errors follow.

#[test]
fn test_error_file_doesnt_exist() {
    let file = "/unlikely/to/exist.h5";
    let result = FEEBeam::new(file);
    assert!(result.is_err());
    match result {
        Err(e) => assert_eq!(
            e.to_string(),
            format!("Specified beam file '{}' doesn't exist", file)
        ),
        _ => unreachable!(),
    }
}

#[test]
fn test_error_env_file_doesnt_exist() {
    let file = "/unlikely/to/exist/again.h5";
    std::env::set_var("MWA_BEAM_FILE", file);
    let result = FEEBeam::new_from_env();
    assert!(result.is_err());
    match result {
        Err(e) => assert_eq!(
            e.to_string(),
            format!("Specified beam file '{}' doesn't exist", file)
        ),
        _ => unreachable!(),
    }
}
