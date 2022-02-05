// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Tests for GPU analytic beam code.

use approx::assert_abs_diff_eq;
use marlu::{constants::MWA_LAT_RAD, Jones};
use ndarray::prelude::*;

use super::AnalyticBeam;
use crate::gpu::GpuFloat;

fn test_analytic(
    beam: AnalyticBeam,
    delays: ArrayView2<u32>,
    amps: ArrayView2<f64>,
    freqs: &[u32],
    norm_to_zenith: bool,
) {
    let (az, za): (Vec<_>, Vec<_>) = (0..1025)
        .map(|i| (0.45 + i as f64 / 10000.0, 0.45 + i as f64 / 10000.0))
        .unzip();
    // Maybe need to regenerate the directions, depending on the GPU precision.
    let az_gpu: Vec<GpuFloat> = az.iter().copied().map(|f| f as GpuFloat).collect();
    let za_gpu: Vec<GpuFloat> = za.iter().copied().map(|f| f as GpuFloat).collect();
    let latitude_rad = MWA_LAT_RAD;
    let gpu_beam = unsafe { beam.gpu_prepare(delays.view(), amps.view()).unwrap() };

    let jones_gpu = gpu_beam
        .calc_jones_pair(
            &az_gpu,
            &za_gpu,
            freqs,
            latitude_rad as GpuFloat,
            norm_to_zenith,
        )
        .unwrap();

    // Compare with CPU results.
    let mut jones_cpu =
        Array3::from_elem((delays.dim().0, freqs.len(), az.len()), Jones::default());
    for ((mut out, delays), amps) in jones_cpu
        .outer_iter_mut()
        .zip(delays.outer_iter())
        .zip(amps.outer_iter())
    {
        for (mut out, &freq) in out.outer_iter_mut().zip(freqs) {
            let cpu_results = beam
                .calc_jones_array_pair(
                    &az,
                    &za,
                    freq,
                    delays.as_slice().unwrap(),
                    amps.as_slice().unwrap(),
                    latitude_rad,
                    norm_to_zenith,
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

    // Check de-duplication.
    let potentially_duplicated = gpu_beam
        .calc_jones_pair(&az_gpu, &za_gpu, freqs, latitude_rad as GpuFloat, true)
        .unwrap();
    assert_eq!(
        potentially_duplicated.len_of(Axis(0)),
        delays.len_of(Axis(0))
    );
}

#[test]
fn test_gpu_calc_jones() {
    let delays = array![[3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0]];
    let amps =
        array![[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]];
    let freqs = [150e6 as u32, 175e6 as _, 200e6 as _];

    for beam in [AnalyticBeam::new(), AnalyticBeam::new_rts()] {
        let norm_to_zenith = false;
        test_analytic(beam, delays.view(), amps.view(), &freqs, norm_to_zenith);
    }

    for beam in [AnalyticBeam::new(), AnalyticBeam::new_rts()] {
        let norm_to_zenith = true;
        test_analytic(beam, delays.view(), amps.view(), &freqs, norm_to_zenith);
    }
}

#[test]
fn test_gpu_calc_jones_multiple_tiles() {
    let delays = array![
        [3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ];
    let amps = array![
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    ];
    let freqs = [150e6 as u32, 175e6 as _, 200e6 as _];

    for beam in [AnalyticBeam::new(), AnalyticBeam::new_rts()] {
        let norm_to_zenith = false;
        test_analytic(beam, delays.view(), amps.view(), &freqs, norm_to_zenith);
    }

    for beam in [AnalyticBeam::new(), AnalyticBeam::new_rts()] {
        let norm_to_zenith = true;
        test_analytic(beam, delays.view(), amps.view(), &freqs, norm_to_zenith);
    }
}

#[test]
fn test_no_directions_doesnt_fail() {
    let beam = AnalyticBeam::new();
    let freqs = [150e6 as u32];
    let delays = array![[3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0]];
    let amps =
        array![[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]];
    let latitude_rad = MWA_LAT_RAD;
    let norm_to_zenith = true;

    let gpu_beam = unsafe { beam.gpu_prepare(delays.view(), amps.view()) }.unwrap();
    let result = gpu_beam
        .calc_jones_pair(&[], &[], &freqs, latitude_rad as GpuFloat, norm_to_zenith)
        .unwrap();
    assert!(result.is_empty());
}
