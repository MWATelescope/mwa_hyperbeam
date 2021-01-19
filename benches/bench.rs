// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Benchmarks. FEE beam code benchmarks rely on the HDF5 file being present in
//! the project's root directory.

use criterion::*;
use marlu::{ndarray, rayon};
use ndarray::prelude::*;
use rayon::prelude::*;

use mwa_hyperbeam::fee::*;

fn fee(c: &mut Criterion) {
    c.bench_function("new", |b| {
        b.iter(|| {
            FEEBeam::new("mwa_full_embedded_element_pattern.h5").unwrap();
        })
    });

    c.bench_function("new + calc_jones_cold_cache", |b| {
        let az = 45.0_f64.to_radians();
        let za = 80.0_f64.to_radians();
        let freq = 51200000;
        let delays = [0; 16];
        let gains = [1.0; 16];
        let norm_to_zenith = false;
        b.iter(|| {
            let beam = FEEBeam::new("mwa_full_embedded_element_pattern.h5").unwrap();
            beam.calc_jones(az, za, freq, &delays, &gains, norm_to_zenith)
                .unwrap();
        })
    });

    c.bench_function("calc_jones_cold_cache", |b| {
        let az = 45.0_f64.to_radians();
        let za = 80.0_f64.to_radians();
        let freq = 51200000;
        let delays = [0; 16];
        let gains = [1.0; 16];
        let norm_to_zenith = false;
        let beam = FEEBeam::new("mwa_full_embedded_element_pattern.h5").unwrap();
        b.iter(|| {
            beam.calc_jones(az, za, freq, &delays, &gains, norm_to_zenith)
                .unwrap();
            beam.empty_cache();
        })
    });

    c.bench_function("calc_jones_hot_cache", |b| {
        let az = 45.0_f64.to_radians();
        let za = 80.0_f64.to_radians();
        let freq = 51200000;
        let delays = [0; 16];
        let gains = [1.0; 16];
        let norm_to_zenith = false;
        let beam = FEEBeam::new("mwa_full_embedded_element_pattern.h5").unwrap();
        // Prime the cache.
        beam.calc_jones(az, za, freq, &delays, &gains, norm_to_zenith)
            .unwrap();
        b.iter(|| {
            beam.calc_jones(az, za, freq, &delays, &gains, norm_to_zenith)
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
        let gains = [1.0; 16];
        let norm_to_zenith = false;
        let beam = FEEBeam::new("mwa_full_embedded_element_pattern.h5").unwrap();
        // Prime the cache.
        beam.calc_jones(az[0], za[0], freq, &delays, &gains, norm_to_zenith)
            .unwrap();
        b.iter(|| {
            beam.calc_jones_array(&az, &za, freq, &delays, &gains, norm_to_zenith)
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
        let gains = [1.0; 16];
        let norm_to_zenith = false;
        let beam = FEEBeam::new("mwa_full_embedded_element_pattern.h5").unwrap();
        // Prime the cache.
        beam.calc_jones(az[0], za[0], freq, &delays, &gains, norm_to_zenith)
            .unwrap();
        b.iter(|| {
            az.par_iter()
                .zip(za.par_iter())
                .map(|(&a, &z)| {
                    beam.calc_jones(a, z, freq, &delays, &gains, norm_to_zenith)
                        .unwrap()
                })
                .collect::<Vec<_>>();
        })
    });

    #[cfg(feature = "cuda")]
    c.bench_function("cuda_calc_jones", |b| {
        let freqs = [51200000];
        let delays = Array2::zeros((1, 16));
        let amps = Array2::ones((1, 32));
        let norm_to_zenith = false;
        let beam = FEEBeam::new("mwa_full_embedded_element_pattern.h5").unwrap();
        let cuda_beam = unsafe {
            beam.cuda_prepare(&freqs, delays.view(), amps.view(), norm_to_zenith)
                .unwrap()
        };

        let mut az = vec![];
        let mut za = vec![];
        for d in 5..85 {
            #[cfg(feature = "cuda-single")]
            let rad = (d as f32).to_radians();
            #[cfg(not(feature = "cuda-single"))]
            let rad = (d as f64).to_radians();
            az.push(rad);
            za.push(rad);
        }
        let parallactic_correction = false;

        b.iter(|| {
            cuda_beam
                .calc_jones(&az, &za, parallactic_correction)
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

    c.bench_function("calc_jones_array 100000 dirs", |b| {
        let beam = FEEBeam::new("mwa_full_embedded_element_pattern.h5").unwrap();
        // Prime the cache.
        beam.calc_jones(
            az_double[0],
            za_double[0],
            freqs[0],
            delays.as_slice().unwrap(),
            amps.as_slice().unwrap(),
            norm_to_zenith,
        )
        .unwrap();
        b.iter(|| {
            beam.calc_jones_array(
                &az_double,
                &za_double,
                freqs[0],
                delays.as_slice().unwrap(),
                amps.as_slice().unwrap(),
                norm_to_zenith,
            )
            .unwrap();
        })
    });

    #[cfg(feature = "cuda")]
    c.bench_function("cuda_calc_jones 100000 dirs", |b| {
        let beam = FEEBeam::new("mwa_full_embedded_element_pattern.h5").unwrap();
        let cuda_beam = unsafe {
            beam.cuda_prepare(&freqs, delays.view(), amps.view(), norm_to_zenith)
                .unwrap()
        };
        let parallactic_correction = true;

        #[cfg(feature = "cuda-single")]
        let (az, za): (Vec<_>, Vec<_>) = az_double
            .iter()
            .zip(za_double.iter())
            .map(|(&az, &za)| (az as f32, za as f32))
            .unzip();
        #[cfg(not(feature = "cuda-single"))]
        let (az, za) = (az_double.clone(), za_double.clone());

        b.iter(|| {
            cuda_beam
                .calc_jones(&az, &za, parallactic_correction)
                .unwrap();
        })
    });

    // The following benchmarks require a few structs and methods to be public.
    // These benchmarks remain commented because those structs and/or methods
    // should not be made public in releases.

    // c.bench_function("calculating coefficients", |b| {
    //     let freq = 51200000;
    //     let delays = [0; 16];
    //     let gains = [1.0; 16];
    //     b.iter(|| {
    //         let beam = FEEBeam::new("mwa_full_embedded_element_pattern.h5").unwrap();
    //         beam.populate_modes(freq, &delays, &gains).unwrap();
    //     })
    // });

    // c.bench_function("getting coefficients from cache", |b| {
    //     let freq = 51200000;
    //     let delays = [0; 16];
    //     let gains = [1.0; 16];
    //     // Prime the cache.
    //     let beam = FEEBeam::new("mwa_full_embedded_element_pattern.h5").unwrap();
    //     beam.populate_modes(freq, &delays, &gains).unwrap();
    //     b.iter(|| {
    //         // By calling populate_modes before the loop we are benchmarking a
    //         // hot cache.
    //         beam.populate_modes(freq, &delays, &gains).unwrap();
    //     })
    // });
}

criterion_group!(benches, fee);
criterion_main!(benches);
