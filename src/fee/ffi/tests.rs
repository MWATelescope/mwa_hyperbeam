// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::ffi::CString;
use std::ptr::{null, null_mut};

use approx::*;
use marlu::constants::MWA_LAT_RAD;
use ndarray::prelude::*;
use serial_test::serial;

use super::*;
use crate::ffi::{hb_last_error_length, hb_last_error_message};

#[cfg(any(feature = "cuda", feature = "hip"))]
use marlu::Jones;

#[test]
#[serial]
fn test_ffi_fee_new() {
    let file = CString::new("mwa_full_embedded_element_pattern.h5").unwrap();
    unsafe {
        let mut beam = null_mut();
        let result = new_fee_beam(file.into_raw(), &mut beam);
        assert_eq!(result, 0);

        free_fee_beam(beam);
    };
}

#[test]
#[serial]
fn test_ffi_fee_new_from_env() {
    std::env::set_var("MWA_BEAM_FILE", "mwa_full_embedded_element_pattern.h5");
    unsafe {
        let mut beam = null_mut();
        let result = new_fee_beam_from_env(&mut beam);
        assert_eq!(result, 0);

        free_fee_beam(beam);
    };
}

#[test]
#[serial]
fn test_calc_jones_via_ffi() {
    let file = std::ffi::CString::new("mwa_full_embedded_element_pattern.h5").unwrap();
    let mut jones = Array1::zeros(8);
    unsafe {
        let mut beam = null_mut();
        let result = new_fee_beam(file.into_raw(), &mut beam);
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
            &MWA_LAT_RAD,
            1,
            jones.as_mut_ptr(),
        );
        assert_eq!(result, 0);

        free_fee_beam(beam);
    };

    let expected = array![
        0.051673288904250436,
        0.14798615369209014,
        -0.0029907711920181858,
        -0.008965331092654419,
        0.002309524016541907,
        0.006230549725189563,
        0.05144802517335513,
        0.14772685224822762
    ];
    assert_abs_diff_eq!(jones, expected);
}

#[test]
#[serial]
fn test_calc_jones_eng_via_ffi() {
    let file = std::ffi::CString::new("mwa_full_embedded_element_pattern.h5").unwrap();
    let mut jones = Array1::zeros(8);
    unsafe {
        let mut beam = null_mut();
        let result = new_fee_beam(file.into_raw(), &mut beam);
        assert_eq!(result, 0);
        let result = calc_jones(
            beam,
            45.0_f64.to_radians(),
            10.0_f64.to_radians(),
            51200000,
            [0; 16].as_ptr(),
            [1.0; 16].as_ptr(),
            16,
            1,
            null(),
            0,
            jones.as_mut_ptr(),
        );
        assert_eq!(result, 0);

        free_fee_beam(beam);
    };

    let expected = array![
        0.22172963270798213,
        0.6348455841660556,
        0.22462189266869015,
        0.6466211027377966,
        0.22219717621735582,
        0.6347084075291201,
        -0.22509296063362327,
        -0.6464547894620056
    ];
    assert_abs_diff_eq!(jones, expected);
}

#[test]
#[serial]
fn test_calc_jones_32_amps_via_ffi() {
    let file = std::ffi::CString::new("mwa_full_embedded_element_pattern.h5").unwrap();
    let mut jones = Array1::zeros(8);
    unsafe {
        let mut beam = null_mut();
        let result = new_fee_beam(file.into_raw(), &mut beam);
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
            null(),
            0,
            jones.as_mut_ptr(),
        );
        assert_eq!(result, 0);

        free_fee_beam(beam);
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
    let file = CString::new("mwa_full_embedded_element_pattern.h5").unwrap();
    let jones = unsafe {
        let mut beam = null_mut();
        let result = new_fee_beam(file.into_raw(), &mut beam);
        assert_eq!(result, 0);
        let az = vec![45.0_f64.to_radians(); num_directions];
        let za = vec![10.0_f64.to_radians(); num_directions];
        let mut jones = Array2::zeros((num_directions, 8));
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
            null(),
            0,
            jones.as_mut_ptr(),
        );
        assert_eq!(result, 0);

        free_fee_beam(beam);
        jones
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
    let file = CString::new("mwa_full_embedded_element_pattern.h5").unwrap();
    let jones = unsafe {
        let mut beam = null_mut();
        let result = new_fee_beam(file.into_raw(), &mut beam);
        assert_eq!(result, 0);
        let az = vec![45.0_f64.to_radians(); num_directions];
        let za = vec![10.0_f64.to_radians(); num_directions];
        let mut jones = Array2::zeros((num_directions, 8));
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
            null(),
            0,
            jones.as_mut_ptr(),
        );
        assert_eq!(result, 0);

        free_fee_beam(beam);
        jones
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
#[serial]
fn test_ffi_fee_freq_functions() {
    let file = "mwa_full_embedded_element_pattern.h5";
    let beam = FEEBeam::new(file).unwrap();
    assert!(!beam.get_freqs().is_empty());
    let freqs = beam.get_freqs();
    let search_freq = 150e6 as u32;
    let closest = beam.find_closest_freq(search_freq);

    unsafe {
        let file_cstr = CString::new("mwa_full_embedded_element_pattern.h5").unwrap();
        let mut ffi_beam = null_mut();
        let result = new_fee_beam(file_cstr.into_raw(), &mut ffi_beam);
        assert_eq!(result, 0);

        let mut freqs_ptr: *const u32 = null_mut();
        let mut num_freqs = 0;
        get_fee_beam_freqs(ffi_beam, &mut freqs_ptr, &mut num_freqs);
        assert_eq!(freqs.len(), num_freqs);
        let freqs_slice = std::slice::from_raw_parts(freqs_ptr, num_freqs);
        for (i, &freq) in freqs.iter().enumerate() {
            assert_eq!(freq, freqs_slice[i]);
        }

        let ffi_closest = closest_freq(ffi_beam, search_freq);
        assert_eq!(closest, ffi_closest);

        free_fee_beam(ffi_beam);
    }
}

#[test]
#[cfg(any(feature = "cuda", feature = "hip"))]
#[serial]
fn test_calc_jones_gpu_via_ffi() {
    let file = CString::new("mwa_full_embedded_element_pattern.h5").unwrap();
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

    let mut beam = null_mut();
    let jones_gpu = unsafe {
        let result = new_fee_beam(file.into_raw(), &mut beam);
        assert_eq!(result, 0);

        let num_freqs = freqs.len();
        let num_tiles = delays.dim().0;
        let num_amps = amps.dim().1;
        let mut gpu_beam = null_mut();

        let result = new_gpu_fee_beam(
            beam,
            freqs.as_ptr(),
            delays.as_ptr(),
            amps.as_ptr(),
            num_freqs as u32,
            num_tiles as u32,
            num_amps as u32,
            norm_to_zenith as u8,
            &mut gpu_beam,
        );
        assert_eq!(result, 0);

        let num_azza = az.len() as u32;
        let mut jones: Array3<Jones<GpuFloat>> = Array3::zeros((num_tiles, num_freqs, az.len()));

        let result = calc_jones_gpu(
            gpu_beam,
            num_azza,
            az.as_ptr(),
            za.as_ptr(),
            &MWA_LAT_RAD,
            1,
            jones.as_mut_ptr().cast(),
        );
        assert_eq!(result, 0);

        free_gpu_fee_beam(gpu_beam);

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
                        norm_to_zenith,
                        Some(MWA_LAT_RAD),
                        true,
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
        free_fee_beam(beam);
    }

    #[cfg(not(feature = "gpu-single"))]
    assert_abs_diff_eq!(jones_gpu, jones_cpu, epsilon = 1e-15);

    #[cfg(feature = "gpu-single")]
    // The errors are heavily dependent on the directions.
    assert_abs_diff_eq!(jones_gpu, jones_cpu, epsilon = 1e-6);
}

// Tests to expose errors follow.

#[test]
fn test_error_file_doesnt_exist() {
    let file = CString::new("unlikely-to-exist.h5").unwrap();
    let file_ptr = file.into_raw();
    unsafe {
        let mut beam = null_mut();
        let result = new_fee_beam(file_ptr, &mut beam);
        assert_ne!(result, 0);
        drop(CString::from_raw(file_ptr));

        let err_len = hb_last_error_length();
        let err = CString::from_vec_unchecked(vec![1; err_len as usize]);
        let err_ptr = err.into_raw();
        hb_last_error_message(err_ptr, err_len);
        let err = CString::from_raw(err_ptr);
        let err_str = err.to_str().unwrap();
        assert_eq!(
            err_str, "Specified beam file 'unlikely-to-exist.h5' doesn't exist",
            "error message: {err_str}"
        );
    }
}

#[test]
fn test_error_env_file_doesnt_exist() {
    std::env::set_var("MWA_BEAM_FILE", "unlikely-to-exist.h5");
    unsafe {
        let mut beam = null_mut();
        let result = new_fee_beam_from_env(&mut beam);
        assert_ne!(result, 0);

        let err_len = hb_last_error_length();
        let err = CString::from_vec_unchecked(vec![1; err_len as usize]);
        let err_ptr = err.into_raw();
        hb_last_error_message(err_ptr, err_len);
        let err = CString::from_raw(err_ptr);
        let err_str = err.to_str().unwrap();
        assert_eq!(
            err_str, "Specified beam file 'unlikely-to-exist.h5' doesn't exist",
            "error message: {err_str}"
        );
    }
}

#[test]
fn test_error_file_invalid_utf8() {
    let file = unsafe { CString::from_vec_unchecked(vec![1, 1, 1, 1, 0]) };
    let file_ptr = file.into_raw();

    unsafe {
        let mut beam = null_mut();
        let result = new_fee_beam(file_ptr, &mut beam);
        assert_ne!(result, 0);
        drop(CString::from_raw(file_ptr));

        let err_len = hb_last_error_length();
        let err = CString::from_vec_unchecked(vec![1; err_len as usize]);
        let err_ptr = err.into_raw();
        hb_last_error_message(err_ptr, err_len);
        let err = CString::from_raw(err_ptr);
        let err_str = err.to_str().unwrap();

        assert_eq!(
            err_str,
            "Specified beam file '\u{1}\u{1}\u{1}\u{1}' doesn't exist"
        );
    }
}

#[test]
fn test_bool_errors() {
    let file = CString::new("mwa_full_embedded_element_pattern.h5").unwrap();
    let file_ptr = file.into_raw();

    unsafe {
        let mut beam = null_mut();
        let result = new_fee_beam(file_ptr, &mut beam);
        assert_eq!(result, 0);
        drop(CString::from_raw(file_ptr));

        let mut jones = [0.0; 8];

        // Bad number of amps.
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
            5,
            2,
            null(),
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
            2,
            null(),
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
        assert_eq!(
            err_str,
            "A value other than 0 or 1 was used for norm_to_zenith"
        );

        // Do it all again for calc_jones_array.
        let az = [0.1];
        let za = [0.1];
        // Bad number of amps.
        let result = calc_jones_array(
            beam,
            az.len() as _,
            az.as_ptr(),
            za.as_ptr(),
            51200000,
            [0; 16].as_ptr(),
            [1.0; 16].as_ptr(),
            10,
            0,
            null(),
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
        let result = calc_jones_array(
            beam,
            az.len() as _,
            az.as_ptr(),
            za.as_ptr(),
            51200000,
            [0; 16].as_ptr(),
            [1.0; 16].as_ptr(),
            16,
            3,
            null(),
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
        assert_eq!(
            err_str,
            "A value other than 0 or 1 was used for norm_to_zenith"
        );
    };
}
