// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::ffi::CString;
use std::ptr::null_mut;

use approx::*;
use marlu::ndarray::prelude::*;
use serial_test::serial;

use super::*;
use crate::jones_test::TestJones;

#[cfg(feature = "cuda")]
use marlu::Jones;

#[test]
#[serial]
fn test_calc_jones_via_ffi() {
    let file = std::ffi::CString::new("mwa_full_embedded_element_pattern.h5").unwrap();
    let mut jones = Array1::zeros(8);
    unsafe {
        let mut beam = null_mut();
        let error_str = CString::from_vec_unchecked(vec![1; 200]).into_raw();
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
            error_str,
        );
        assert_eq!(
            result,
            0,
            "{}",
            CString::from_raw(error_str).into_string().unwrap()
        );
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

#[test]
#[cfg(feature = "cuda")]
#[serial]
fn test_calc_jones_cuda_via_ffi() {
    let file = std::ffi::CString::new("mwa_full_embedded_element_pattern.h5").unwrap();
    let freqs = [150e6 as u32];
    let delays = array![[3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0]];
    let amps =
        array![[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]];
    let (az, za): (Vec<_>, Vec<_>) = (0..1025)
        .into_iter()
        .map(|i| {
            (
                0.45 + i as CudaFloat / 10000.0,
                0.45 + i as CudaFloat / 10000.0,
            )
        })
        .unzip();
    let norm_to_zenith = true;

    let mut beam = null_mut();
    let jones_gpu = unsafe {
        let error_str = CString::from_vec_unchecked(vec![1; 200]).into_raw();
        let result = new_fee_beam(file.into_raw(), &mut beam, error_str);
        assert_eq!(result, 0);

        let num_freqs = freqs.len() as u32;
        let num_tiles = delays.dim().1 as u32;
        let num_amps = amps.dim().1 as u32;
        let mut cuda_beam = null_mut();

        let result = new_cuda_fee_beam(
            beam,
            freqs.as_ptr(),
            delays.as_ptr(),
            amps.as_ptr(),
            num_freqs,
            num_tiles,
            num_amps,
            norm_to_zenith as u8,
            &mut cuda_beam,
            error_str,
        );
        assert_eq!(
            result,
            0,
            "{}",
            CString::from_raw(error_str).into_string().unwrap()
        );

        let num_azza = az.len() as u32;
        let parallactic_correction = 1;
        let mut jones_floats = null_mut();

        let result = calc_jones_cuda(
            cuda_beam,
            num_azza,
            az.as_ptr(),
            za.as_ptr(),
            parallactic_correction,
            &mut jones_floats,
            error_str,
        );
        assert_eq!(
            result,
            0,
            "{}",
            CString::from_raw(error_str).into_string().unwrap()
        );

        let jones: *mut Jones<CudaFloat> = jones_floats.cast();
        ArrayView3::from_shape_ptr((delays.dim().0, freqs.len(), az.len()), jones).into_owned()
    };

    // Compare with CPU results.
    let mut jones_cpu = Array3::zeros((delays.dim().0, freqs.len(), az.len()));
    // Maybe need to regenerate the directions, depending on the CUDA precision.
    let (az, za): (Vec<_>, Vec<_>) = (0..1025)
        .into_iter()
        .map(|i| (0.45 + i as f64 / 10000.0, 0.45 + i as f64 / 10000.0))
        .unzip();
    for ((mut out, delays), amps) in jones_cpu
        .outer_iter_mut()
        .zip(delays.outer_iter())
        .zip(amps.outer_iter())
    {
        for (mut out, freq) in out.outer_iter_mut().zip(freqs) {
            unsafe {
                let cpu_results = (&*beam)
                    .calc_jones_array(
                        &az,
                        &za,
                        freq,
                        delays.as_slice().unwrap(),
                        amps.as_slice().unwrap(),
                        norm_to_zenith,
                    )
                    .unwrap();

                // Demote the CPU results if we have to.
                #[cfg(feature = "cuda-single")]
                let cpu_results: Vec<Jones<f32>> =
                    cpu_results.into_iter().map(|j| j.into()).collect();

                out.assign(&Array1::from(cpu_results));
            }
        }
    }

    let jones_cpu = jones_cpu.mapv(TestJones::from);
    let jones_gpu = jones_gpu.mapv(TestJones::from);

    #[cfg(not(feature = "cuda-single"))]
    assert_abs_diff_eq!(jones_gpu, jones_cpu, epsilon = 1e-15);

    #[cfg(feature = "cuda-single")]
    // The errors are heavily dependent on the directions.
    assert_abs_diff_eq!(jones_gpu, jones_cpu, epsilon = 1e-6);
}
