// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::{
    ffi::CString,
    ptr::{null, null_mut},
};

use approx::*;
use marlu::constants::MWA_LAT_RAD;
use ndarray::prelude::*;

use crate::{
    analytic::tests::{
        AnalyticArgsAndExpectation, MWA_PB_1, MWA_PB_2, MWA_PB_3, MWA_PB_4, MWA_PB_5,
    },
    ffi::{hb_last_error_length, hb_last_error_message},
};

use super::*;

#[cfg(any(feature = "cuda", feature = "hip"))]
use marlu::Jones;

#[test]
fn test_ffi_analytic_new() {
    unsafe {
        let mut beam = null_mut();
        let result = new_analytic_beam(0, null(), &mut beam);
        assert_eq!(result, 0);

        free_analytic_beam(beam);
    };
    unsafe {
        let mut beam = null_mut();
        let result = new_analytic_beam(0, &1.5, &mut beam);
        assert_eq!(result, 0);

        free_analytic_beam(beam);
    };

    unsafe {
        let mut beam = null_mut();
        let result = new_analytic_beam(1, null(), &mut beam);
        assert_eq!(result, 0);

        free_analytic_beam(beam);
    };

    unsafe {
        let mut beam = null_mut();
        let result = new_analytic_beam(1, &0.2, &mut beam);
        assert_eq!(result, 0);

        free_analytic_beam(beam);
    };
}

macro_rules! new_beam {
    () => {{
        unsafe {
            let mut beam = null_mut();
            let result = new_analytic_beam(0, null(), &mut beam);
            assert_eq!(result, 0);
            beam
        }
    }};
}

macro_rules! test_analytic_calc_jones {
    ($beam:expr, $args:expr, $epsilon:expr) => {{
        let mut jones = [0.0; 8];
        let AnalyticArgsAndExpectation {
            az_rad,
            za_rad,
            freq_hz,
            delays,
            amps,
            norm_to_zenith,
            expected,
        } = $args;
        unsafe {
            let result = analytic_calc_jones(
                $beam,
                az_rad,
                za_rad,
                freq_hz,
                delays.as_ptr(),
                amps.as_ptr(),
                amps.len() as _,
                MWA_LAT_RAD,
                norm_to_zenith as _,
                jones.as_mut_ptr(),
            );
            assert_eq!(result, 0);
        }

        assert_abs_diff_eq!(jones.as_slice(), expected.as_slice(), epsilon = $epsilon);
    }};
}

#[test]
fn test_calc_jones_via_ffi() {
    let beam = new_beam!();
    test_analytic_calc_jones!(beam, MWA_PB_1, 1e-5);
    test_analytic_calc_jones!(beam, MWA_PB_2, 1e-5);
    test_analytic_calc_jones!(beam, MWA_PB_3, 1e-5);
    test_analytic_calc_jones!(beam, MWA_PB_4, 1e-4);
    test_analytic_calc_jones!(beam, MWA_PB_5, 1e-5);
    unsafe {
        free_analytic_beam(beam);
    };
}

#[test]
fn test_calc_jones_32_amps_via_ffi() {
    let beam = new_beam!();
    let mut jones = Array1::zeros(8);
    unsafe {
        let result = analytic_calc_jones(
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
            MWA_LAT_RAD,
            0,
            jones.as_mut_ptr(),
        );
        assert_eq!(result, 0);

        free_analytic_beam(beam);
    };
}

macro_rules! test_analytic_calc_jones_array {
    ($beam:expr, $args:expr) => {{
        let num_directions = 10;
        let mut jones = Array2::zeros((num_directions, 8));
        let mut jones_expected = [0.0; 8];
        let AnalyticArgsAndExpectation {
            az_rad,
            za_rad,
            freq_hz,
            delays,
            amps,
            norm_to_zenith,
            expected: _,
        } = $args;
        let az = vec![az_rad; num_directions];
        let za = vec![za_rad; num_directions];
        unsafe {
            // First, calculate the expected beam response for the single
            // direction. Then verify that all directions are the same.
            let result = analytic_calc_jones(
                $beam,
                az_rad,
                za_rad,
                freq_hz,
                delays.as_ptr(),
                amps.as_ptr(),
                amps.len() as _,
                MWA_LAT_RAD,
                norm_to_zenith as _,
                jones_expected.as_mut_ptr(),
            );
            assert_eq!(result, 0);

            let result = analytic_calc_jones_array(
                $beam,
                num_directions as _,
                az.as_ptr(),
                za.as_ptr(),
                freq_hz,
                delays.as_ptr(),
                amps.as_ptr(),
                amps.len() as _,
                MWA_LAT_RAD,
                norm_to_zenith as _,
                jones.as_mut_ptr(),
            );
            assert_eq!(result, 0);
        }

        for jones in jones.outer_iter() {
            assert_abs_diff_eq!(jones.as_slice().unwrap(), jones_expected.as_slice());
        }
    }};
}

#[test]
fn test_calc_jones_array_via_ffi() {
    let beam = new_beam!();
    test_analytic_calc_jones_array!(beam, MWA_PB_1);
    test_analytic_calc_jones_array!(beam, MWA_PB_2);
    test_analytic_calc_jones_array!(beam, MWA_PB_3);
    test_analytic_calc_jones_array!(beam, MWA_PB_4);
    test_analytic_calc_jones_array!(beam, MWA_PB_5);

    unsafe {
        free_analytic_beam(beam);
    };
}

#[test]
fn test_calc_jones_array_32_amps_via_ffi() {
    let beam = new_beam!();
    let num_directions = 1000;
    let az = vec![45.0_f64.to_radians(); num_directions];
    let za = vec![10.0_f64.to_radians(); num_directions];
    let mut jones = Array2::zeros((num_directions, 8));
    unsafe {
        let result = analytic_calc_jones_array(
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
            MWA_LAT_RAD,
            0,
            jones.as_mut_ptr(),
        );
        assert_eq!(result, 0);

        free_analytic_beam(beam);
    };
}

#[test]
#[cfg(any(feature = "cuda", feature = "hip"))]
fn test_calc_jones_gpu_via_ffi() {
    let beam = new_beam!();
    let freqs = [150e6 as u32];
    let delays = array![[3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0]];
    let amps =
        array![[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]];
    let (az, za): (Vec<_>, Vec<_>) = (0..1025)
        .map(|i| {
            (
                0.45 + i as GpuFloat / 10000.0,
                0.45 + i as GpuFloat / 10000.0,
            )
        })
        .unzip();
    let norm_to_zenith = true;

    let jones_gpu = unsafe {
        let num_freqs = freqs.len();
        let num_tiles = delays.dim().0;
        let num_amps = amps.dim().1;
        let mut gpu_beam = null_mut();

        let result = new_gpu_analytic_beam(
            beam,
            delays.as_ptr(),
            amps.as_ptr(),
            num_tiles as i32,
            num_amps as i32,
            &mut gpu_beam,
        );
        assert_eq!(result, 0);

        let num_azza = az.len() as u32;
        let mut jones: Array3<Jones<GpuFloat>> = Array3::zeros((num_tiles, num_freqs, az.len()));

        let result = analytic_calc_jones_gpu(
            gpu_beam,
            num_azza,
            az.as_ptr(),
            za.as_ptr(),
            freqs.len() as u32,
            freqs.as_ptr(),
            MWA_LAT_RAD as GpuFloat,
            norm_to_zenith as _,
            jones.as_mut_ptr().cast(),
        );
        assert_eq!(result, 0);

        free_gpu_analytic_beam(gpu_beam);

        jones
    };

    // Compare with CPU results.
    let mut jones_cpu = Array3::zeros((delays.dim().0, freqs.len(), az.len()));
    // Maybe need to regenerate the directions, depending on the GPU precision.
    let (az, za): (Vec<_>, Vec<_>) = (0..1025)
        .map(|i| (0.45 + i as f64 / 10000.0, 0.45 + i as f64 / 10000.0))
        .unzip();
    for ((mut out, delays), amps) in jones_cpu
        .outer_iter_mut()
        .zip(delays.outer_iter())
        .zip(amps.outer_iter())
    {
        for (mut out, freq) in out.outer_iter_mut().zip(freqs) {
            unsafe {
                let cpu_results = (*beam)
                    .calc_jones_array_pair(
                        &az,
                        &za,
                        freq,
                        delays.as_slice().unwrap(),
                        amps.as_slice().unwrap(),
                        MWA_LAT_RAD,
                        norm_to_zenith,
                    )
                    .unwrap();

                // Demote the CPU results if we have to.
                #[cfg(feature = "gpu-single")]
                let cpu_results: Vec<Jones<f32>> =
                    cpu_results.into_iter().map(|j| j.into()).collect();

                out.assign(&Array1::from(cpu_results));
            }
        }
    }

    unsafe {
        free_analytic_beam(beam);
    }

    #[cfg(not(feature = "gpu-single"))]
    assert_abs_diff_eq!(jones_gpu, jones_cpu, epsilon = 1e-15);

    #[cfg(feature = "gpu-single")]
    // The errors are heavily dependent on the directions.
    assert_abs_diff_eq!(jones_gpu, jones_cpu, epsilon = 1e-6);
}

// Tests to expose errors follow.

#[test]
fn test_bool_errors() {
    let beam = new_beam!();
    let mut jones = [0.0; 8];

    unsafe {
        // Bad number of amps.
        let result = analytic_calc_jones(
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
            5,
            MWA_LAT_RAD,
            2,
            jones.as_mut_ptr(),
        );
        assert_ne!(result, 0);

        let err_len = hb_last_error_length();
        let err = CString::from_vec_unchecked(vec![1; err_len as usize]);
        let err_ptr = err.into_raw();
        hb_last_error_message(err_ptr, err_len);
        let err = CString::from_raw(err_ptr);
        let err_str = err.to_str().unwrap();
        assert_eq!(err_str, "A value other than 16 or 32 was used for num_amps");

        // Bad norm_to_zenith value.
        let result = analytic_calc_jones(
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
            MWA_LAT_RAD,
            2,
            jones.as_mut_ptr(),
        );
        assert_ne!(result, 0);

        let err_len = hb_last_error_length();
        let err = CString::from_vec_unchecked(vec![1; err_len as usize]);
        let err_ptr = err.into_raw();
        hb_last_error_message(err_ptr, err_len);
        let err = CString::from_raw(err_ptr);
        let err_str = err.to_str().unwrap();
        assert_eq!(
            err_str,
            "A value other than 0 or 1 was used for norm_to_zenith"
        );

        // Do it all again for calc_jones_array.
        let az = [0.1];
        let za = [0.1];
        // Bad number of amps.
        let result = analytic_calc_jones_array(
            beam,
            az.len() as _,
            az.as_ptr(),
            za.as_ptr(),
            51200000,
            [0; 16].as_ptr(),
            [1.0; 16].as_ptr(),
            10,
            MWA_LAT_RAD,
            0,
            jones.as_mut_ptr(),
        );
        assert_ne!(result, 0);
        let err_len = hb_last_error_length();
        let err = CString::from_vec_unchecked(vec![1; err_len as usize]);
        let err_ptr = err.into_raw();
        hb_last_error_message(err_ptr, err_len);
        let err = CString::from_raw(err_ptr);
        let err_str = err.to_str().unwrap();
        assert_eq!(err_str, "A value other than 16 or 32 was used for num_amps");

        // Bad norm_to_zenith value.
        let result = analytic_calc_jones_array(
            beam,
            az.len() as _,
            az.as_ptr(),
            za.as_ptr(),
            51200000,
            [0; 16].as_ptr(),
            [1.0; 16].as_ptr(),
            16,
            MWA_LAT_RAD,
            3,
            jones.as_mut_ptr(),
        );
        assert_ne!(result, 0);
        let err_len = hb_last_error_length();
        let err = CString::from_vec_unchecked(vec![1; err_len as usize]);
        let err_ptr = err.into_raw();
        hb_last_error_message(err_ptr, err_len);
        let err = CString::from_raw(err_ptr);
        let err_str = err.to_str().unwrap();
        assert_eq!(
            err_str,
            "A value other than 0 or 1 was used for norm_to_zenith"
        );
    };
}
