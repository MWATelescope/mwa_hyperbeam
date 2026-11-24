// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Tests for GPU FEE beam code.

use approx::{assert_abs_diff_eq, assert_abs_diff_ne};
use marlu::constants::MWA_LAT_RAD;
use ndarray::prelude::*;
use serial_test::serial;

use super::*;

#[test]
#[serial]
fn test_gpu_calc_jones_no_norm() {
    let beam = FEEBeam::new("mwa_full_embedded_element_pattern.h5").unwrap();
    let freqs = [150e6 as u32];
    let delays = array![[3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0]];
    let amps =
        array![[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]];
    let norm_to_zenith = false;
    let result = unsafe { beam.gpu_prepare(&freqs, delays.view(), amps.view(), norm_to_zenith) };
    assert!(result.is_ok(), "{}", result.unwrap_err());
    let cuda_beam = result.unwrap();
    assert_eq!(cuda_beam.num_coeffs, 1);
    assert_eq!(cuda_beam.num_unique_tiles, 1);
    assert_eq!(cuda_beam.num_unique_freqs, 1);

    let (az, za): (Vec<_>, Vec<_>) = (0..1025)
        .map(|i| {
            (
                0.45 + i as GpuFloat / 10000.0,
                0.45 + i as GpuFloat / 10000.0,
            )
        })
        .unzip();
    let latitude_rad = None;

    let result = cuda_beam.calc_jones_pair(&az, &za, latitude_rad, false);
    assert!(result.is_ok(), "{}", result.unwrap_err());
    let jones_gpu = result.unwrap();

    // Compare with CPU results.
    let mut jones_cpu =
        Array3::from_elem((delays.dim().0, freqs.len(), az.len()), Jones::default());
    // Maybe need to regenerate the directions, depending on the CUDA precision.
    let (az, za): (Vec<_>, Vec<_>) = (0..1025)
        .map(|i| (0.45 + i as f64 / 10000.0, 0.45 + i as f64 / 10000.0))
        .unzip();
    for ((mut out, delays), amps) in jones_cpu
        .outer_iter_mut()
        .zip(delays.outer_iter())
        .zip(amps.outer_iter())
    {
        for (mut out, freq) in out.outer_iter_mut().zip(freqs) {
            let cpu_results = beam
                .calc_jones_array_pair(
                    &az,
                    &za,
                    freq,
                    delays.as_slice().unwrap(),
                    amps.as_slice().unwrap(),
                    norm_to_zenith,
                    None,
                    false,
                )
                .unwrap();

            // Demote the CPU results if we have to.
            #[cfg(feature = "gpu-single")]
            let cpu_results: Vec<Jones<f32>> = cpu_results.into_iter().map(|j| j.into()).collect();

            out.assign(&Array1::from(cpu_results));
        }
    }

    #[cfg(not(feature = "gpu-single"))]
    assert_abs_diff_eq!(jones_gpu, jones_cpu, epsilon = 1e-15);

    #[cfg(feature = "gpu-single")]
    // The errors are heavily dependent on the directions.
    assert_abs_diff_eq!(jones_gpu, jones_cpu, epsilon = 1e-7);
}

#[test]
#[serial]
fn test_gpu_calc_jones_w_norm() {
    let beam = FEEBeam::new("mwa_full_embedded_element_pattern.h5").unwrap();
    let freqs = [150e6 as u32];
    let delays = array![[3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0]];
    let amps =
        array![[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]];
    let norm_to_zenith = true;
    let result = unsafe { beam.gpu_prepare(&freqs, delays.view(), amps.view(), norm_to_zenith) };
    assert!(result.is_ok(), "{}", result.unwrap_err());
    let cuda_beam = result.unwrap();
    assert_eq!(cuda_beam.num_coeffs, 1);
    assert_eq!(cuda_beam.num_unique_tiles, 1);
    assert_eq!(cuda_beam.num_unique_freqs, 1);

    let (az, za): (Vec<_>, Vec<_>) = (0..1025)
        .map(|i| {
            (
                0.45 + i as GpuFloat / 10000.0,
                0.45 + i as GpuFloat / 10000.0,
            )
        })
        .unzip();
    let latitude_rad = None;

    let result = cuda_beam.calc_jones_pair(&az, &za, latitude_rad, false);
    assert!(result.is_ok(), "{}", result.unwrap_err());
    let jones_gpu = result.unwrap();

    // Compare with CPU results.
    let mut jones_cpu =
        Array3::from_elem((delays.dim().0, freqs.len(), az.len()), Jones::default());
    // Maybe need to regenerate the directions, depending on the CUDA precision.
    let (az, za): (Vec<_>, Vec<_>) = (0..1025)
        .map(|i| (0.45 + i as f64 / 10000.0, 0.45 + i as f64 / 10000.0))
        .unzip();
    for ((mut out, delays), amps) in jones_cpu
        .outer_iter_mut()
        .zip(delays.outer_iter())
        .zip(amps.outer_iter())
    {
        for (mut out, freq) in out.outer_iter_mut().zip(freqs) {
            let cpu_results = beam
                .calc_jones_array_pair(
                    &az,
                    &za,
                    freq,
                    delays.as_slice().unwrap(),
                    amps.as_slice().unwrap(),
                    norm_to_zenith,
                    latitude_rad,
                    false,
                )
                .unwrap();

            // Demote the CPU results if we have to.
            #[cfg(feature = "gpu-single")]
            let cpu_results: Vec<Jones<f32>> = cpu_results.into_iter().map(|j| j.into()).collect();

            out.assign(&Array1::from(cpu_results));
        }
    }

    #[cfg(not(feature = "gpu-single"))]
    assert_abs_diff_eq!(jones_gpu, jones_cpu, epsilon = 1e-15);

    #[cfg(feature = "gpu-single")]
    // The errors are heavily dependent on the directions.
    assert_abs_diff_eq!(jones_gpu, jones_cpu, epsilon = 1e-6);
}

#[test]
#[serial]
fn test_gpu_calc_jones_w_norm_and_parallactic() {
    let beam = FEEBeam::new("mwa_full_embedded_element_pattern.h5").unwrap();
    let freqs = [150e6 as u32];
    let delays = array![[3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0]];
    let amps =
        array![[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]];
    let norm_to_zenith = true;
    let result = unsafe { beam.gpu_prepare(&freqs, delays.view(), amps.view(), norm_to_zenith) };
    assert!(result.is_ok(), "{}", result.unwrap_err());
    let cuda_beam = result.unwrap();
    assert_eq!(cuda_beam.num_coeffs, 1);
    assert_eq!(cuda_beam.num_unique_tiles, 1);
    assert_eq!(cuda_beam.num_unique_freqs, 1);

    let (az, za): (Vec<_>, Vec<_>) = (0..1025)
        .map(|i| {
            (
                0.45 + i as GpuFloat / 10000.0,
                0.45 + i as GpuFloat / 10000.0,
            )
        })
        .unzip();
    let latitude_rad = Some(MWA_LAT_RAD);

    let result = cuda_beam.calc_jones_pair(&az, &za, latitude_rad, true);
    assert!(result.is_ok(), "{}", result.unwrap_err());
    let jones_gpu = result.unwrap();

    // Compare with CPU results.
    let mut jones_cpu =
        Array3::from_elem((delays.dim().0, freqs.len(), az.len()), Jones::default());
    // Maybe need to regenerate the directions, depending on the CUDA precision.
    let (az, za): (Vec<_>, Vec<_>) = (0..1025)
        .map(|i| (0.45 + i as f64 / 10000.0, 0.45 + i as f64 / 10000.0))
        .unzip();
    for ((mut out, delays), amps) in jones_cpu
        .outer_iter_mut()
        .zip(delays.outer_iter())
        .zip(amps.outer_iter())
    {
        for (mut out, freq) in out.outer_iter_mut().zip(freqs) {
            let cpu_results = beam
                .calc_jones_array_pair(
                    &az,
                    &za,
                    freq,
                    delays.as_slice().unwrap(),
                    amps.as_slice().unwrap(),
                    norm_to_zenith,
                    latitude_rad,
                    true,
                )
                .unwrap();

            // Demote the CPU results if we have to.
            #[cfg(feature = "gpu-single")]
            let cpu_results: Vec<Jones<f32>> = cpu_results.into_iter().map(|j| j.into()).collect();

            out.assign(&Array1::from(cpu_results));
        }
    }

    #[cfg(not(feature = "gpu-single"))]
    assert_abs_diff_eq!(jones_gpu, jones_cpu, epsilon = 1e-15);

    #[cfg(feature = "gpu-single")]
    // The errors are heavily dependent on the directions.
    assert_abs_diff_eq!(jones_gpu, jones_cpu, epsilon = 1e-6);
}

#[test]
#[serial]
fn test_gpu_calc_jones_with_and_without_parallactic() {
    let beam = FEEBeam::new("mwa_full_embedded_element_pattern.h5").unwrap();
    let freqs = [150e6 as u32];
    let delays = array![[3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0]];
    let amps =
        array![[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]];
    let norm_to_zenith = true;
    let result = unsafe { beam.gpu_prepare(&freqs, delays.view(), amps.view(), norm_to_zenith) };
    assert!(result.is_ok(), "{}", result.unwrap_err());
    let cuda_beam = result.unwrap();
    assert_eq!(cuda_beam.num_coeffs, 1);
    assert_eq!(cuda_beam.num_unique_tiles, 1);
    assert_eq!(cuda_beam.num_unique_freqs, 1);

    let (az, za): (Vec<_>, Vec<_>) = (0..1025)
        .map(|i| {
            (
                0.45 + i as GpuFloat / 10000.0,
                0.45 + i as GpuFloat / 10000.0,
            )
        })
        .unzip();
    let latitude_rad = Some(MWA_LAT_RAD);

    let result = cuda_beam.calc_jones_pair(&az, &za, latitude_rad, false);
    assert!(result.is_ok(), "{}", result.unwrap_err());
    let pa = result.unwrap();
    let result = cuda_beam.calc_jones_pair(&az, &za, None, false);
    assert!(result.is_ok(), "{}", result.unwrap_err());
    let not_pa = result.unwrap();

    assert_abs_diff_ne!(pa, not_pa);
}

#[test]
#[serial]
fn test_gpu_calc_jones_deduplication() {
    let beam = FEEBeam::new("mwa_full_embedded_element_pattern.h5").unwrap();
    // 6 freqs here, but only 3 unique ones.
    let freqs = [
        150e6 as u32,
        200e6 as _,
        250e6 as _,
        150e6 as _,
        200e6 as _,
        250000001,
    ];
    // Tiles 0 and 3 are the same; 3 unique tiles.
    let delays = array![
        [3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0],
        [32, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0], // Delays of 32 are treated as distinct
        [3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0],
        [3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0],
    ];
    let amps = array![
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    ];
    let norm_to_zenith = false;
    let result = unsafe { beam.gpu_prepare(&freqs, delays.view(), amps.view(), norm_to_zenith) };
    assert!(result.is_ok(), "{}", result.unwrap_err());
    let cuda_beam = result.unwrap();
    assert_eq!(cuda_beam.num_coeffs, 9);
    assert_eq!(cuda_beam.num_unique_tiles, 3);
    assert_eq!(cuda_beam.num_unique_freqs, 3);

    let (az, za): (Vec<_>, Vec<_>) = (0..1025)
        .map(|i| {
            (
                0.45 + i as GpuFloat / 10000.0,
                0.45 + i as GpuFloat / 10000.0,
            )
        })
        .unzip();
    let latitude_rad = None;

    let result = cuda_beam.calc_jones_pair(&az, &za, latitude_rad, false);
    assert!(result.is_ok(), "{}", result.unwrap_err());
    let jones_gpu = result.unwrap();

    // Compare with CPU results.
    let mut jones_cpu =
        Array3::from_elem((delays.dim().0, freqs.len(), az.len()), Jones::default());
    // Maybe need to regenerate the directions, depending on the CUDA precision.
    let (az, za): (Vec<_>, Vec<_>) = (0..1025)
        .map(|i| (0.45 + i as f64 / 10000.0, 0.45 + i as f64 / 10000.0))
        .unzip();
    for ((mut out, delays), amps) in jones_cpu
        .outer_iter_mut()
        .zip(delays.outer_iter())
        .zip(amps.outer_iter())
    {
        for (mut out, freq) in out.outer_iter_mut().zip(freqs) {
            let cpu_results = beam
                .calc_jones_array_pair(
                    &az,
                    &za,
                    freq,
                    delays.as_slice().unwrap(),
                    amps.as_slice().unwrap(),
                    norm_to_zenith,
                    latitude_rad,
                    false,
                )
                .unwrap();

            // Demote the CPU results if we have to.
            #[cfg(feature = "gpu-single")]
            let cpu_results: Vec<Jones<f32>> = cpu_results.into_iter().map(|j| j.into()).collect();

            out.assign(&Array1::from(cpu_results));
        }
    }

    #[cfg(not(feature = "gpu-single"))]
    assert_abs_diff_eq!(jones_gpu, jones_cpu, epsilon = 1e-15);

    #[cfg(feature = "gpu-single")]
    // The errors are heavily dependent on the directions.
    assert_abs_diff_eq!(jones_gpu, jones_cpu, epsilon = 1e-6);
}

#[test]
#[serial]
fn test_gpu_calc_jones_deduplication_w_norm() {
    let beam = FEEBeam::new("mwa_full_embedded_element_pattern.h5").unwrap();
    // 6 freqs here, but only 3 unique ones.
    let freqs = [
        150e6 as u32,
        200e6 as _,
        250e6 as _,
        150e6 as _,
        200e6 as _,
        250000001,
    ];
    // Tiles 0 and 3 are the same; 3 unique tiles.
    let delays = array![
        [3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0],
        [32, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0], // Delays of 32 are treated as distinct
        [3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0],
        [3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0],
    ];
    let amps = array![
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    ];
    let norm_to_zenith = true;
    let result = unsafe { beam.gpu_prepare(&freqs, delays.view(), amps.view(), norm_to_zenith) };
    assert!(result.is_ok(), "{}", result.unwrap_err());
    let cuda_beam = result.unwrap();
    assert_eq!(cuda_beam.num_coeffs, 9);
    assert_eq!(cuda_beam.num_unique_tiles, 3);
    assert_eq!(cuda_beam.num_unique_freqs, 3);

    let (az, za): (Vec<_>, Vec<_>) = (0..1025)
        .map(|i| {
            (
                0.45 + i as GpuFloat / 10000.0,
                0.45 + i as GpuFloat / 10000.0,
            )
        })
        .unzip();
    let latitude_rad = None;

    let result = cuda_beam.calc_jones_pair(&az, &za, latitude_rad, false);
    assert!(result.is_ok(), "{}", result.unwrap_err());
    let jones_gpu = result.unwrap();

    // Compare with CPU results.
    let mut jones_cpu =
        Array3::from_elem((delays.dim().0, freqs.len(), az.len()), Jones::default());
    // Maybe need to regenerate the directions, depending on the CUDA precision.
    let (az, za): (Vec<_>, Vec<_>) = (0..1025)
        .map(|i| (0.45 + i as f64 / 10000.0, 0.45 + i as f64 / 10000.0))
        .unzip();
    for ((mut out, delays), amps) in jones_cpu
        .outer_iter_mut()
        .zip(delays.outer_iter())
        .zip(amps.outer_iter())
    {
        for (mut out, freq) in out.outer_iter_mut().zip(freqs) {
            let cpu_results = beam
                .calc_jones_array_pair(
                    &az,
                    &za,
                    freq,
                    delays.as_slice().unwrap(),
                    amps.as_slice().unwrap(),
                    norm_to_zenith,
                    latitude_rad,
                    false,
                )
                .unwrap();

            // Demote the CPU results if we have to.
            #[cfg(feature = "gpu-single")]
            let cpu_results: Vec<Jones<f32>> = cpu_results.into_iter().map(|j| j.into()).collect();

            out.assign(&Array1::from(cpu_results));
        }
    }

    #[cfg(not(feature = "gpu-single"))]
    assert_abs_diff_eq!(jones_gpu, jones_cpu, epsilon = 1e-15);

    #[cfg(feature = "gpu-single")]
    // The errors are heavily dependent on the directions.
    assert_abs_diff_eq!(jones_gpu, jones_cpu, epsilon = 1e-6);
}

#[test]
#[serial]
fn test_gpu_calc_jones_no_amps() {
    let beam = FEEBeam::new("mwa_full_embedded_element_pattern.h5").unwrap();
    let freqs: Vec<u32> = [50e6, 75e6, 100e6, 125e6, 150e6, 175e6, 200e6]
        .into_iter()
        .map(|f| f as u32)
        .collect();
    let delays = array![
        [3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0],
        [3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0]
    ];
    let amps = array![
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ];
    let norm_to_zenith = false;
    let result = unsafe { beam.gpu_prepare(&freqs, delays.view(), amps.view(), norm_to_zenith) };
    assert!(result.is_ok(), "{}", result.unwrap_err());
    let cuda_beam = result.unwrap();
    assert_eq!(cuda_beam.num_coeffs, 14);
    assert_eq!(cuda_beam.num_unique_tiles, 2);
    assert_eq!(cuda_beam.num_unique_freqs, 7);

    let (az, za): (Vec<_>, Vec<_>) = (0..1025)
        .map(|i| {
            (
                0.45 + i as GpuFloat / 10000.0,
                0.45 + i as GpuFloat / 10000.0,
            )
        })
        .unzip();
    let latitude_rad = None;

    let result = cuda_beam.calc_jones_pair(&az, &za, latitude_rad, false);
    assert!(result.is_ok(), "{}", result.unwrap_err());
    let jones_gpu = result.unwrap();

    // Compare with CPU results.
    let mut jones_cpu =
        Array3::from_elem((delays.dim().0, freqs.len(), az.len()), Jones::default());
    // Maybe need to regenerate the directions, depending on the CUDA precision.
    let (az, za): (Vec<_>, Vec<_>) = (0..1025)
        .map(|i| (0.45 + i as f64 / 10000.0, 0.45 + i as f64 / 10000.0))
        .unzip();
    for ((mut out, delays), amps) in jones_cpu
        .outer_iter_mut()
        .zip(delays.outer_iter())
        .zip(amps.outer_iter())
    {
        for (mut out, freq) in out.outer_iter_mut().zip(freqs.iter()) {
            let cpu_results = beam
                .calc_jones_array_pair(
                    &az,
                    &za,
                    *freq,
                    delays.as_slice().unwrap(),
                    amps.as_slice().unwrap(),
                    norm_to_zenith,
                    latitude_rad,
                    false,
                )
                .unwrap();

            // Demote the CPU results if we have to.
            #[cfg(feature = "gpu-single")]
            let cpu_results: Vec<Jones<f32>> = cpu_results.into_iter().map(|j| j.into()).collect();

            out.assign(&Array1::from(cpu_results));
        }
    }

    #[cfg(not(feature = "gpu-single"))]
    assert_abs_diff_eq!(jones_gpu, jones_cpu, epsilon = 1e-15);

    #[cfg(feature = "gpu-single")]
    // The errors are heavily dependent on the directions.
    assert_abs_diff_eq!(jones_gpu, jones_cpu, epsilon = 1e-6);

    // The results for this tile are all zero.
    assert_abs_diff_eq!(
        jones_gpu.slice(s![1, .., ..]),
        Array2::from_elem((jones_gpu.dim().1, jones_gpu.dim().2), Jones::default())
    );

    // The results for this tile are at least some non-zero.
    assert_abs_diff_ne!(
        jones_gpu.slice(s![0, .., ..]),
        Array2::from_elem((jones_gpu.dim().1, jones_gpu.dim().2), Jones::default())
    );
}

#[test]
#[serial]
fn test_gpu_calc_jones_iau_order() {
    let beam = FEEBeam::new("mwa_full_embedded_element_pattern.h5").unwrap();
    let freqs = [150e6 as u32];
    let delays = array![[3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0]];
    let amps =
        array![[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]];
    let norm_to_zenith = false;
    let result = unsafe { beam.gpu_prepare(&freqs, delays.view(), amps.view(), norm_to_zenith) };
    assert!(result.is_ok(), "{}", result.unwrap_err());
    let cuda_beam = result.unwrap();
    assert_eq!(cuda_beam.num_coeffs, 1);
    assert_eq!(cuda_beam.num_unique_tiles, 1);
    assert_eq!(cuda_beam.num_unique_freqs, 1);

    let (az, za): (Vec<_>, Vec<_>) = (vec![0.45 / 10000.0], vec![0.45 / 10000.0]);
    let latitude_rad = Some(MWA_LAT_RAD);

    let result = cuda_beam.calc_jones_pair(&az, &za, latitude_rad, true);
    assert!(result.is_ok(), "{}", result.unwrap_err());
    let j_iau = result.unwrap();

    let result = cuda_beam.calc_jones_pair(&az, &za, latitude_rad, false);
    assert!(result.is_ok(), "{}", result.unwrap_err());
    let j_not_iau = result.unwrap();

    assert_ne!(j_iau[(0, 0, 0)][0], j_not_iau[(0, 0, 0)][0]);
    assert_ne!(j_iau[(0, 0, 0)][1], j_not_iau[(0, 0, 0)][1]);
    assert_ne!(j_iau[(0, 0, 0)][2], j_not_iau[(0, 0, 0)][2]);
    assert_ne!(j_iau[(0, 0, 0)][3], j_not_iau[(0, 0, 0)][3]);
    assert_eq!(j_iau[(0, 0, 0)][0], j_not_iau[(0, 0, 0)][3]);
    assert_eq!(j_iau[(0, 0, 0)][1], j_not_iau[(0, 0, 0)][2]);
    assert_eq!(j_iau[(0, 0, 0)][2], j_not_iau[(0, 0, 0)][1]);
    assert_eq!(j_iau[(0, 0, 0)][3], j_not_iau[(0, 0, 0)][0]);
}

#[test]
#[serial]
fn test_cuda_calc_jones_pathological() {
    // I accidentally introduced a bug where Y dipole "m abs" values were used
    // for X dipole values. This test fails if the bug still exists.

    let beam = FEEBeam::new("mwa_full_embedded_element_pattern.h5").unwrap();
    let freqs = [119040000];
    let mut amps = Array2::ones((2, 32));
    amps[(0, 5)] = 0.0;
    let delays = Array2::zeros((amps.len_of(Axis(0)), 16));
    let norm_to_zenith = true;
    let cuda_beam =
        unsafe { beam.gpu_prepare(&freqs, delays.view(), amps.view(), norm_to_zenith) }.unwrap();

    let azs = [3.5279431];
    let zas = [0.19745648];
    let latitude_rad = Some(-0.4671829547325157);
    let iau_reorder = false;

    let result = cuda_beam.calc_jones_pair(&azs, &zas, latitude_rad, iau_reorder);
    assert!(result.is_ok());
}

#[test]
fn test_no_directions_doesnt_fail() {
    let beam = FEEBeam::new("mwa_full_embedded_element_pattern.h5").unwrap();
    let freqs = [150e6 as u32];
    let delays = array![[3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0]];
    let amps =
        array![[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]];
    let norm_to_zenith = true;
    let cuda_beam =
        unsafe { beam.gpu_prepare(&freqs, delays.view(), amps.view(), norm_to_zenith) }.unwrap();

    let latitude_rad = Some(MWA_LAT_RAD);
    let result = cuda_beam
        .calc_jones_pair(&[], &[], latitude_rad, false)
        .unwrap();
    assert!(result.is_empty());
}

/// A device only allows so many blocks in a kernel launch, and each set of
/// dipole coefficients (unique tile at a unique frequency) gets its own block
/// in y. More sets than that (65,535 for CUDA and HIP) need more than one
/// launch. Check that the results don't depend on how the sets are split over
/// launches, without needing tens of thousands of sets.
/// <https://github.com/MWATelescope/mwa_hyperbeam/issues/17>
#[test]
#[serial]
fn test_gpu_calc_jones_over_multiple_launches() {
    let beam = FEEBeam::new("mwa_full_embedded_element_pattern.h5").unwrap();
    let freqs = [150e6 as u32, 200e6 as _, 250e6 as _];
    // 3 unique tiles.
    let delays = array![
        [3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0],
        [32, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0],
        [3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0],
    ];
    let amps = array![
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    ];
    let (az, za): (Vec<_>, Vec<_>) = (0..1025)
        .map(|i| {
            (
                0.45 + i as GpuFloat / 10000.0,
                0.45 + i as GpuFloat / 10000.0,
            )
        })
        .unzip();
    let num_directions = az.len();

    for norm_to_zenith in [false, true] {
        let gpu_beam = unsafe {
            beam.gpu_prepare(&freqs, delays.view(), amps.view(), norm_to_zenith)
                .unwrap()
        };
        assert_eq!(gpu_beam.num_coeffs, 9);

        let calc = |max_coeffs_per_launch: Option<i32>| unsafe {
            let d_az = DevicePointer::copy_to_device(&az).unwrap();
            let d_za = DevicePointer::copy_to_device(&za).unwrap();
            let d_latitude_rad = DevicePointer::copy_to_device(&[MWA_LAT_RAD as GpuFloat]).unwrap();
            let mut results: Vec<Jones<GpuFloat>> = vec![Jones::default(); 9 * num_directions];
            let d_results: DevicePointer<Jones<GpuFloat>> =
                DevicePointer::copy_to_device(&results).unwrap();
            gpu_beam
                .calc_jones_device_pair_inner_max_coeffs_per_launch(
                    d_az.get(),
                    d_za.get(),
                    num_directions as i32,
                    d_latitude_rad.get(),
                    true,
                    max_coeffs_per_launch,
                    d_results.get_mut().cast(),
                )
                .map(|()| {
                    d_results.copy_from_device(&mut results).unwrap();
                    results
                })
        };

        let expected = calc(None).unwrap();
        // Every set of coefficients has had its results written.
        assert!(expected.iter().all(|j| *j != Jones::default()));
        // The sets give different results, so writing one set's results in
        // another's place would be noticed.
        for (i, set1) in expected.chunks_exact(num_directions).enumerate() {
            for set2 in expected.chunks_exact(num_directions).skip(i + 1) {
                assert_ne!(set1, set2);
            }
        }

        // 9 launches, 5 (the last with 1 set), 3 (the last with 1 set), 1
        // launch that's exactly big enough, and 1 that's bigger than needed.
        for max_coeffs_per_launch in [1, 2, 4, 9, 10] {
            let results = calc(Some(max_coeffs_per_launch)).unwrap();
            assert!(
                results == expected,
                "results differ with {max_coeffs_per_launch} sets of coefficients per launch"
            );
        }

        // A launch has to handle at least one set.
        assert!(calc(Some(0)).is_err());
    }
}
