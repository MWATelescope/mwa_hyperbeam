// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::ptr::null_mut;

use super::*;
use approx::*;
use marlu::ndarray::prelude::*;
use serial_test::serial;

#[test]
#[serial]
fn test_calc_jones_via_ffi() {
    let file = std::ffi::CString::new("mwa_full_embedded_element_pattern.h5").unwrap();
    let mut jones = Array1::zeros(8);
    unsafe {
        let mut beam = null_mut();
        let result = new_fee_beam(file.into_raw(), &mut beam, null_mut());
        assert_eq!(result, 0);
        let result = calc_jones(
            beam,
            45.0_f64.to_radians(),
            10.0_f64.to_radians(),
            51200000,
            [0; 16].as_ptr(),
            [1.0; 16].as_ptr(),
            16,
            0,
            0,
            jones.as_mut_ptr(),
            null_mut(),
        );
        assert_eq!(result, 0);
    };

    let expected =
        array![0.036179, 0.103586, 0.036651, 0.105508, 0.036362, 0.103868, -0.036836, -0.105791];
    assert_abs_diff_eq!(jones, expected, epsilon = 1e-6);
}

#[test]
#[serial]
fn test_calc_jones_32_amps_via_ffi() {
    let file = std::ffi::CString::new("mwa_full_embedded_element_pattern.h5").unwrap();
    let mut jones = Array1::zeros(8);
    unsafe {
        let mut beam = null_mut();
        let result = new_fee_beam(file.into_raw(), &mut beam, null_mut());
        assert_eq!(result, 0);
        let result = calc_jones(
            beam,
            45.0_f64.to_radians(),
            10.0_f64.to_radians(),
            51200000,
            [0; 16].as_ptr(),
            [
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0,
            ]
            .as_ptr(),
            32,
            0,
            0,
            jones.as_mut_ptr(),
            null_mut(),
        );
        assert_eq!(result, 0);
    };

    let expected = array![
        0.036179157398051796,
        0.10358641753243217,
        0.03665108136724156,
        0.1055078041087216,
        0.033872534964068265,
        0.0977782048884059,
        -0.034513724028826506,
        -0.09956978911712255
    ];
    assert_abs_diff_eq!(jones, expected);
}

#[test]
#[serial]
fn test_calc_jones_array_via_ffi() {
    let num_directions = 1000;
    let file = std::ffi::CString::new("mwa_full_embedded_element_pattern.h5").unwrap();
    let jones = unsafe {
        let mut beam = null_mut();
        let error = null_mut();
        let result = new_fee_beam(file.into_raw(), &mut beam, error);
        assert_eq!(result, 0);
        let az = vec![45.0_f64.to_radians(); num_directions];
        let za = vec![10.0_f64.to_radians(); num_directions];
        let mut jones_ptr = null_mut();
        let result = calc_jones_array(
            beam,
            num_directions as _,
            az.as_ptr(),
            za.as_ptr(),
            51200000,
            [0; 16].as_ptr(),
            [1.0; 16].as_ptr(),
            16,
            0,
            0,
            &mut jones_ptr,
            error,
        );
        assert_eq!(result, 0);
        Array1::from(Vec::from_raw_parts(
            jones_ptr,
            8 * num_directions,
            8 * num_directions,
        ))
        .into_shape((num_directions, 8))
        .unwrap()
    };

    let expected =
        array![0.036179, 0.103586, 0.036651, 0.105508, 0.036362, 0.103868, -0.036836, -0.105791];
    assert_abs_diff_eq!(jones.slice(s![0, ..]), expected, epsilon = 1e-6);
    assert_abs_diff_eq!(jones.slice(s![-1, ..]), expected, epsilon = 1e-6);
}

#[test]
#[serial]
fn test_calc_jones_array_32_amps_via_ffi() {
    let num_directions = 1000;
    let file = std::ffi::CString::new("mwa_full_embedded_element_pattern.h5").unwrap();
    let jones = unsafe {
        let mut beam = null_mut();
        let error = null_mut();
        let result = new_fee_beam(file.into_raw(), &mut beam, error);
        assert_eq!(result, 0);
        let az = vec![45.0_f64.to_radians(); num_directions];
        let za = vec![10.0_f64.to_radians(); num_directions];
        let mut jones_ptr = null_mut();
        let result = calc_jones_array(
            beam,
            num_directions as _,
            az.as_ptr(),
            za.as_ptr(),
            51200000,
            [0; 16].as_ptr(),
            [
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0,
            ]
            .as_ptr(),
            32,
            0,
            0,
            &mut jones_ptr,
            error,
        );
        assert_eq!(result, 0);
        Array1::from(Vec::from_raw_parts(
            jones_ptr,
            8 * num_directions,
            8 * num_directions,
        ))
        .into_shape((num_directions, 8))
        .unwrap()
    };

    let expected = array![
        0.036179157398051796,
        0.10358641753243217,
        0.03665108136724156,
        0.1055078041087216,
        0.033872534964068265,
        0.0977782048884059,
        -0.034513724028826506,
        -0.09956978911712255
    ];
    assert_abs_diff_eq!(jones.slice(s![0, ..]), expected);
    assert_abs_diff_eq!(jones.slice(s![-1, ..]), expected);
}
