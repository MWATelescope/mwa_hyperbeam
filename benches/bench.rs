// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Benchmarks. FEE beam code benchmarks rely on the HDF5 file being present in
//! the project's root directory.

use criterion::*;
use marlu::constants::MWA_LAT_RAD;
use ndarray::prelude::*;
use rayon::prelude::*;

use mwa_hyperbeam::fee::*;
use mwa_hyperbeam::ska::*;

fn fee(c: &mut Criterion) {
    c.bench_function("new", |b| {
        b.iter(|| {
            FEEBeam::new("mwa_full_embedded_element_pattern.h5").unwrap();
        })
    });

    c.bench_function("find_closest_freq", |b| {
        let beam = FEEBeam::new("mwa_full_embedded_element_pattern.h5").unwrap();
        b.iter(|| beam.find_closest_freq(200_000_000));
    });

    c.bench_function("new + calc_jones_cold_cache", |b| {
        let az = 45.0_f64.to_radians();
        let za = 80.0_f64.to_radians();
        let freq = 51200000;
        let delays = [0; 16];
        let amps = [1.0; 16];
        let norm_to_zenith = false;
        let latitude_rad = Some(MWA_LAT_RAD);
        let iau_order = true;
        b.iter(|| {
            let beam = FEEBeam::new("mwa_full_embedded_element_pattern.h5").unwrap();
            beam.calc_jones_pair(
                az,
                za,
                freq,
                &delays,
                &amps,
                norm_to_zenith,
                latitude_rad,
                iau_order,
            )
            .unwrap();
        })
    });

    c.bench_function("calc_jones_cold_cache", |b| {
        let az = 45.0_f64.to_radians();
        let za = 80.0_f64.to_radians();
        let freq = 51200000;
        let delays = [0; 16];
        let amps = [1.0; 16];
        let norm_to_zenith = false;
        let latitude_rad = Some(MWA_LAT_RAD);
        let iau_order = true;
        let beam = FEEBeam::new("mwa_full_embedded_element_pattern.h5").unwrap();
        b.iter(|| {
            beam.calc_jones_pair(
                az,
                za,
                freq,
                &delays,
                &amps,
                norm_to_zenith,
                latitude_rad,
                iau_order,
            )
            .unwrap();
            beam.empty_cache();
        })
    });

    c.bench_function("calc_jones_hot_cache", |b| {
        let az = 45.0_f64.to_radians();
        let za = 80.0_f64.to_radians();
        let freq = 51200000;
        let delays = [0; 16];
        let amps = [1.0; 16];
        let norm_to_zenith = false;
        let latitude_rad = Some(MWA_LAT_RAD);
        let iau_order = true;
        let beam = FEEBeam::new("mwa_full_embedded_element_pattern.h5").unwrap();
        // Prime the cache.
        beam.calc_jones_pair(
            az,
            za,
            freq,
            &delays,
            &amps,
            norm_to_zenith,
            latitude_rad,
            iau_order,
        )
        .unwrap();
        b.iter(|| {
            beam.calc_jones_pair(
                az,
                za,
                freq,
                &delays,
                &amps,
                norm_to_zenith,
                latitude_rad,
                iau_order,
            )
            .unwrap();
        })
    });

    c.bench_function("calc_jones_array", |b| {
        let mut az = vec![];
        let mut za = vec![];
        for d in 5..85 {
            let rad = (d as f64).to_radians();
            az.push(rad);
            za.push(rad);
        }
        let freq = 51200000;
        let delays = [0; 16];
        let amps = [1.0; 16];
        let norm_to_zenith = false;
        let latitude_rad = Some(MWA_LAT_RAD);
        let iau_order = true;
        let beam = FEEBeam::new("mwa_full_embedded_element_pattern.h5").unwrap();
        // Prime the cache.
        beam.calc_jones_pair(
            az[0],
            za[0],
            freq,
            &delays,
            &amps,
            norm_to_zenith,
            latitude_rad,
            iau_order,
        )
        .unwrap();
        b.iter(|| {
            beam.calc_jones_array_pair(
                &az,
                &za,
                freq,
                &delays,
                &amps,
                norm_to_zenith,
                latitude_rad,
                iau_order,
            )
            .unwrap();
        })
    });

    // Similar to calc_jones_array, but many independent threads calling
    // calc_jones.
    c.bench_function("calc_jones in parallel", |b| {
        let mut az = vec![];
        let mut za = vec![];
        for d in 5..85 {
            let rad = (d as f64).to_radians();
            az.push(rad);
            za.push(rad);
        }
        let freq = 51200000;
        let delays = [0; 16];
        let amps = [1.0; 16];
        let norm_to_zenith = false;
        let latitude_rad = Some(MWA_LAT_RAD);
        let iau_order = true;
        let beam = FEEBeam::new("mwa_full_embedded_element_pattern.h5").unwrap();
        // Prime the cache.
        beam.calc_jones_pair(
            az[0],
            za[0],
            freq,
            &delays,
            &amps,
            norm_to_zenith,
            latitude_rad,
            iau_order,
        )
        .unwrap();
        b.iter(|| {
            az.par_iter()
                .zip(za.par_iter())
                .map(|(&a, &z)| {
                    beam.calc_jones_pair(
                        a,
                        z,
                        freq,
                        &delays,
                        &amps,
                        norm_to_zenith,
                        latitude_rad,
                        iau_order,
                    )
                    .unwrap()
                })
                .collect::<Vec<_>>();
        })
    });

    #[cfg(any(feature = "cuda", feature = "hip"))]
    c.bench_function("gpu_calc_jones", |b| {
        let freqs = [51200000];
        let delays = Array2::zeros((1, 16));
        let amps = Array2::ones((1, 32));
        let norm_to_zenith = false;
        let beam = FEEBeam::new("mwa_full_embedded_element_pattern.h5").unwrap();
        let gpu_beam = unsafe {
            beam.gpu_prepare(&freqs, delays.view(), amps.view(), norm_to_zenith)
                .unwrap()
        };

        let mut az = vec![];
        let mut za = vec![];
        for d in 5..85 {
            #[cfg(feature = "gpu-single")]
            let rad = (d as f32).to_radians();
            #[cfg(not(feature = "gpu-single"))]
            let rad = (d as f64).to_radians();
            az.push(rad);
            za.push(rad);
        }
        let latitude_rad = None;
        let iau_order = false;

        b.iter(|| {
            gpu_beam
                .calc_jones_pair(&az, &za, latitude_rad, iau_order)
                .unwrap();
        })
    });

    // Benchmarks with a fair few pointings!
    let num_directions = 100000;
    let mut az_double = vec![];
    let mut za_double = vec![];
    for i in 1..=num_directions {
        az_double.push(0.9 * std::f64::consts::TAU / i as f64);
        za_double.push(std::f64::consts::PI / i as f64);
    }
    let freqs = [51200000];
    let delays = Array2::zeros((1, 16));
    let amps = Array2::ones((1, 16));
    let norm_to_zenith = true;
    let latitude_rad = Some(MWA_LAT_RAD);
    let iau_order = true;

    c.bench_function("calc_jones_array 100000 dirs", |b| {
        let beam = FEEBeam::new("mwa_full_embedded_element_pattern.h5").unwrap();
        // Prime the cache.
        beam.calc_jones_pair(
            az_double[0],
            za_double[0],
            freqs[0],
            delays.as_slice().unwrap(),
            amps.as_slice().unwrap(),
            norm_to_zenith,
            latitude_rad,
            iau_order,
        )
        .unwrap();
        b.iter(|| {
            beam.calc_jones_array_pair(
                &az_double,
                &za_double,
                freqs[0],
                delays.as_slice().unwrap(),
                amps.as_slice().unwrap(),
                norm_to_zenith,
                latitude_rad,
                iau_order,
            )
            .unwrap();
        })
    });

    #[cfg(any(feature = "cuda", feature = "hip"))]
    c.bench_function("gpu_calc_jones 100000 dirs", |b| {
        let beam = FEEBeam::new("mwa_full_embedded_element_pattern.h5").unwrap();
        let gpu_beam = unsafe {
            beam.gpu_prepare(&freqs, delays.view(), amps.view(), norm_to_zenith)
                .unwrap()
        };
        let latitude_rad = Some(MWA_LAT_RAD);

        #[cfg(feature = "gpu-single")]
        let (az, za): (Vec<_>, Vec<_>) = az_double
            .iter()
            .zip(za_double.iter())
            .map(|(&az, &za)| (az as f32, za as f32))
            .unzip();
        #[cfg(not(feature = "gpu-single"))]
        let (az, za) = (az_double.clone(), za_double.clone());

        b.iter(|| {
            gpu_beam
                .calc_jones_pair(&az, &za, latitude_rad, true)
                .unwrap();
        })
    });

    // The following benchmarks require a few structs and methods to be public.
    // These benchmarks remain commented because those structs and/or methods
    // should not be made public in releases.

    // c.bench_function("calculating coefficients", |b| {
    //     let freq = 51200000;
    //     let delays = [0; 16];
    //     let amps = [1.0; 32];
    //     let beam = FEEBeam::new("mwa_full_embedded_element_pattern.h5").unwrap();
    //     b.iter(|| {
    //         beam.calc_modes(freq, &delays, &amps).unwrap();
    //     })
    // });

    // c.bench_function("getting coefficients from cache", |b| {
    //     let freq = 51200000;
    //     let delays = [0; 16];
    //     let amps = [1.0; 32];
    //     let beam = FEEBeam::new("mwa_full_embedded_element_pattern.h5").unwrap();
    //     // Prime the cache.
    //     let _ = beam.get_modes(freq, &delays, &amps).unwrap();
    //     b.iter(|| {
    //         // By calling populate_modes before the loop we are benchmarking a
    //         // hot cache.
    //         let _ = beam.get_modes(freq, &delays, &amps).unwrap();
    //     })
    // });
}

/// SKA-Low benchmarks. The analytic model needs no data; the EEP models are
/// only benchmarked if `SKA_LOW_EEP_GRID_FILE` and/or `SKA_LOW_EEP_SPH_PATH`
/// are set.
fn ska(c: &mut Criterion) {
    let n = 100_000;
    let az: Vec<f64> = (0..n)
        .map(|i| (i as f64 * 0.7731) % std::f64::consts::TAU)
        .collect();
    let za: Vec<f64> = (0..n)
        .map(|i| (i as f64 * 0.3117) % 89.0_f64.to_radians())
        .collect();

    let beam = SkaLowAnalytic::new();
    c.bench_function("ska-low-analytic calc_jones_array_pair 100k", |b| {
        b.iter(|| {
            beam.calc_jones_array_pair(&az, &za, 150e6, 0, Some(MWA_LAT_RAD), true)
                .unwrap()
        })
    });

    c.bench_function("ska-low-analytic calc_jones_array_pair 100k (no PA)", |b| {
        b.iter(|| {
            beam.calc_jones_array_pair(&az, &za, 150e6, 0, None, false)
                .unwrap()
        })
    });

    if let Ok(beam) = SkaLowEepGrid::new_from_env() {
        let freq = beam.get_freqs()[0] + 0.37 * (beam.get_freqs()[1] - beam.get_freqs()[0]);
        c.bench_function("ska-low-eep-grid calc_jones_array_pair 100k (no PA)", |b| {
            b.iter(|| {
                beam.calc_jones_array_pair(&az, &za, freq, 0, None, false)
                    .unwrap()
            })
        });
        c.bench_function("ska-low-eep-grid calc_jones_array_pair 100k", |b| {
            b.iter(|| {
                beam.calc_jones_array_pair(&az, &za, freq, 0, Some(MWA_LAT_RAD), true)
                    .unwrap()
            })
        });
    }

    if let Ok(beam) = SkaLowEepSph::new_from_env() {
        let freq = beam.get_freqs()[0];
        // Warm the coefficient cache.
        beam.calc_jones_pair(0.1, 0.1, freq, 0, None, false)
            .unwrap();
        c.bench_function("ska-low-eep-sph calc_jones_array_pair 10k", |b| {
            b.iter(|| {
                beam.calc_jones_array_pair(
                    &az[..10_000],
                    &za[..10_000],
                    freq,
                    0,
                    Some(MWA_LAT_RAD),
                    true,
                )
                .unwrap()
            })
        });
    }
}

criterion_group!(benches, fee, ska);
criterion_main!(benches);
