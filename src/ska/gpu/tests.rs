// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Tests comparing the GPU SKA-Low beams against the CPU code. These need a
//! GPU to run.

use std::f64::consts::{FRAC_PI_2, TAU};

use approx::assert_abs_diff_eq;
use marlu::{constants::MWA_LAT_RAD, AzEl, Jones};
use ndarray::prelude::*;
use serial_test::serial;

use super::*;
use crate::ska::{
    tests::{write_eep, write_feko_dir, write_feko_dir_with_l_maxes},
    SkaStation,
};

#[cfg(feature = "gpu-single")]
const EPSILON: f64 = 1e-4;
#[cfg(not(feature = "gpu-single"))]
const EPSILON: f64 = 1e-10;

fn directions() -> (Vec<f64>, Vec<f64>) {
    let az: Vec<f64> = (0..300).map(|i| (i as f64 * 0.7731) % TAU).collect();
    let mut za: Vec<f64> = (0..300).map(|i| (i as f64 * 0.3117) % 1.5).collect();
    // Include the zenith and directions very close to it, where single
    // precision needs care.
    for (i, z) in [1e-7, 1e-5, 1e-4, 3e-4, 1e-3, 1e-2].iter().enumerate() {
        za[1 + i] = *z;
    }
    (az, za)
}

#[allow(clippy::unnecessary_cast)] // Not unnecessary with gpu-single.
fn to_f64(j: Jones<GpuFloat>) -> Jones<f64> {
    Jones::from([
        marlu::c64::new(j[0].re as f64, j[0].im as f64),
        marlu::c64::new(j[1].re as f64, j[1].im as f64),
        marlu::c64::new(j[2].re as f64, j[2].im as f64),
        marlu::c64::new(j[3].re as f64, j[3].im as f64),
    ])
}

/// Compare element-mode GPU results against the CPU.
fn check_elements(beam: &SkaLowBeam, freqs: &[f64], element_ids: &[usize]) {
    let (az, za) = directions();
    let gpu = unsafe { beam.gpu_prepare(freqs, SkaGpuTiles::Elements(element_ids)) }.unwrap();
    let az_g: Vec<GpuFloat> = az.iter().map(|&x| x as _).collect();
    let za_g: Vec<GpuFloat> = za.iter().map(|&x| x as _).collect();
    for (latitude, iau) in [
        (None, false),
        (Some(MWA_LAT_RAD), true),
        (Some(MWA_LAT_RAD), false),
    ] {
        let results = gpu
            .calc_jones_pair(
                &az_g,
                &za_g,
                AzEl::from_radians(0.0, FRAC_PI_2),
                latitude,
                iau,
            )
            .unwrap();
        assert_eq!(results.dim(), (element_ids.len(), freqs.len(), az.len()));
        for (i_tile, &id) in element_ids.iter().enumerate() {
            for (i_freq, &f) in freqs.iter().enumerate() {
                let cpu = beam
                    .calc_jones_array_pair(&az, &za, f, id, latitude, iau)
                    .unwrap();
                for (c, g) in cpu.iter().zip(results.slice(s![i_tile, i_freq, ..])) {
                    let scale = c.iter().map(|x| x.norm()).fold(1.0, f64::max);
                    assert_abs_diff_eq!(*c, to_f64(*g), epsilon = EPSILON * scale);
                }
            }
        }
    }
}

/// Compare station-mode GPU results against the CPU.
fn check_stations(beam: &SkaLowBeam, freqs: &[f64], num_elements: usize) {
    // Unrotated stations, and two stations with different rotations (one of
    // them zero, which takes a different code path in the kernel).
    check_stations_inner(beam, freqs, num_elements, None, None, false);
    check_stations_inner(beam, freqs, num_elements, Some(&[0.0, 1.234]), None, false);
    // Stations with their own local frames (one tilted, one rotated about the
    // vertical), as for stations far from the array centre.
    let frames = [tilted_frame(0.006, -0.004), vertical_rotation_frame(0.3)];
    check_stations_inner(
        beam,
        freqs,
        num_elements,
        Some(&[0.0, 1.234]),
        Some(&frames),
        false,
    );
    // The X and Y dipoles with different flags.
    check_stations_inner(
        beam,
        freqs,
        num_elements,
        Some(&[0.0, 1.234]),
        Some(&frames),
        true,
    );
}

/// The rows of a local frame whose East is `beta` from the array's East
/// towards North.
fn vertical_rotation_frame(beta: f64) -> [[f64; 3]; 3] {
    let (s, c) = beta.sin_cos();
    [[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]]
}

/// The rows of a tilted local frame (vertical tilted by small angles east
/// `de` and north `dn`).
fn tilted_frame(de: f64, dn: f64) -> [[f64; 3]; 3] {
    let (se, ce) = de.sin_cos();
    let (sn, cn) = dn.sin_cos();
    let e = [ce, 0.0, -se];
    let u0 = [se, 0.0, ce];
    let n = [-sn * u0[0], cn, -sn * u0[2]];
    let u = [cn * u0[0], sn, cn * u0[2]];
    [e, n, u]
}

fn check_stations_inner(
    beam: &SkaLowBeam,
    freqs: &[f64],
    num_elements: usize,
    rotations: Option<&[f64]>,
    frames: Option<&[[[f64; 3]; 3]]>,
    separate_y_weights: bool,
) {
    let (az, za) = directions();
    let num_stations = 2;
    let positions = Array3::from_shape_fn((num_stations, num_elements, 3), |(s, e, d)| {
        let r = 1.5 * (e as f64).sqrt() + s as f64;
        let a = e as f64 * 2.399963 + s as f64;
        match d {
            0 => r * a.cos(),
            1 => r * a.sin(),
            _ => 0.01 * e as f64,
        }
    });
    let mut weights = Array2::ones((num_stations, num_elements));
    weights[(1, 0)] = 0.0;
    weights[(1, 1)] = 0.5;
    let weights_y = separate_y_weights.then(|| {
        let mut w = Array2::ones((num_stations, num_elements));
        w[(0, num_elements - 1)] = 0.0;
        w[(1, 1)] = 0.0;
        w
    });
    let pointing = AzEl::from_radians(0.6, FRAC_PI_2 - 0.3);
    let gpu = unsafe {
        beam.gpu_prepare(
            freqs,
            SkaGpuTiles::Stations {
                positions_enu: positions.view(),
                weights: Some(weights.view()),
                rotations_rad: rotations,
                weights_y: weights_y.as_ref().map(|w| w.view()),
                local_frames: frames,
            },
        )
    }
    .unwrap();
    let az_g: Vec<GpuFloat> = az.iter().map(|&x| x as _).collect();
    let za_g: Vec<GpuFloat> = za.iter().map(|&x| x as _).collect();
    let results = gpu
        .calc_jones_pair(&az_g, &za_g, pointing, Some(MWA_LAT_RAD), true)
        .unwrap();
    for i_station in 0..num_stations {
        let pos: Vec<[f64; 3]> = positions
            .slice(s![i_station, .., ..])
            .outer_iter()
            .map(|p| [p[0], p[1], p[2]])
            .collect();
        let make = |w: Vec<f64>| {
            let station = SkaStation::new(beam, pos.clone(), Some(w))
                .unwrap()
                .with_rotation(rotations.map(|r| r[i_station]).unwrap_or(0.0));
            match frames {
                Some(f) => station.with_local_frame(f[i_station]),
                None => station,
            }
        };
        let station = make(weights.slice(s![i_station, ..]).to_vec());
        let station_y = weights_y
            .as_ref()
            .map(|w| make(w.slice(s![i_station, ..]).to_vec()));
        for (i_freq, &f) in freqs.iter().enumerate() {
            let calc = |st: &SkaStation<&SkaLowBeam>| {
                st.calc_jones_array_pair(
                    &az,
                    &za,
                    f,
                    pointing.az,
                    pointing.za(),
                    Some(MWA_LAT_RAD),
                    true,
                )
                .unwrap()
            };
            let mut cpu = calc(&station);
            if let Some(st_y) = &station_y {
                // With the IAU order, the first row is the Y dipole's.
                for (c, y) in cpu.iter_mut().zip(calc(st_y)) {
                    *c = Jones::from([y[0], y[1], c[2], c[3]]);
                }
            }
            for (c, g) in cpu.iter().zip(results.slice(s![i_station, i_freq, ..])) {
                let scale = c.iter().map(|x| x.norm()).fold(1.0, f64::max);
                assert_abs_diff_eq!(*c, to_f64(*g), epsilon = EPSILON * scale);
            }
        }
    }
}

#[test]
#[serial]
fn gpu_analytic_matches_cpu() {
    for taper in [true, false] {
        let beam = SkaLowBeam::new_analytic(taper);
        check_elements(&beam, &[50e6, 150e6, 350e6], &[0]);
        check_stations(&beam, &[100e6, 200e6], 16);
    }
}

#[test]
#[serial]
fn gpu_eep_grid_matches_cpu() {
    let phis: Vec<f64> = (0..=72).map(|i| i as f64 * TAU / 72.0).collect();
    let thetas: Vec<f64> = (0..=18).map(|i| i as f64 * FRAC_PI_2 / 18.0).collect();
    let file = write_eep(
        "gpu_eep_grid.h5",
        &phis,
        &thetas,
        &[50e6, 100e6, 175e6, 350e6],
    );
    let beam = SkaLowBeam::new_eep_grid(Some(&file.0), true).unwrap();
    check_elements(&beam, &[50e6, 77.7e6, 350e6], &[0]);
    check_stations(&beam, &[123.4e6], 16);
}

#[test]
#[serial]
fn gpu_eep_sph_matches_cpu() {
    let dir = write_feko_dir("gpu_eep_sph", &[100.0, 110.0], 4, 6);
    let beam = SkaLowBeam::new_eep_sph(Some(&dir.0), None).unwrap();
    check_elements(&beam, &[100e6, 104e6, 112e6], &[0, 3, 1]);
    check_stations(&beam, &[100e6, 110e6], 4);
}

#[test]
#[serial]
fn gpu_eep_sph_large_degrees_match_cpu() {
    // Real SKA-Low FEKO files have degrees from ~140 to ~390, differing from
    // file to file.
    let dir = write_feko_dir_with_l_maxes("gpu_eep_sph_large", &[100.0, 110.0], 4, &[7, 300]);
    let beam = SkaLowBeam::new_eep_sph(Some(&dir.0), None).unwrap();
    check_elements(&beam, &[100e6, 110e6], &[0, 3]);
    check_stations(&beam, &[100e6, 110e6], 4);
}

#[test]
#[serial]
fn gpu_out_of_grid_is_zero() {
    let phis: Vec<f64> = (0..=36).map(|i| i as f64 * TAU / 36.0).collect();
    // Theta only up to 45 deg.
    let thetas: Vec<f64> = (0..=9).map(|i| i as f64 * FRAC_PI_2 / 18.0).collect();
    let file = write_eep("gpu_out_of_grid.h5", &phis, &thetas, &[50e6, 350e6]);
    let beam = SkaLowBeam::new_eep_grid(Some(&file.0), true).unwrap();
    let gpu = unsafe { beam.gpu_prepare(&[100e6], SkaGpuTiles::Elements(&[0])) }.unwrap();
    let results = gpu
        .calc_jones_pair(
            &[0.1, 0.1],
            &[0.2, 1.2],
            AzEl::from_radians(0.0, FRAC_PI_2),
            None,
            false,
        )
        .unwrap();
    assert!(results[(0, 0, 0)].iter().any(|c| c.norm() > 0.0));
    assert!(results[(0, 0, 1)].iter().all(|c| c.norm() == 0.0));
    // Out-of-range frequencies are an error when preparing.
    assert!(unsafe { beam.gpu_prepare(&[400e6], SkaGpuTiles::Elements(&[0])) }.is_err());
}

#[test]
fn max_l_max_matches_header() {
    // SKA_GPU_MAX_L_MAX must match SKA_MAX_L_MAX in ska.h, which sizes the
    // kernels' per-thread Legendre arrays.
    let header = include_str!("ska.h");
    let define = header
        .lines()
        .find_map(|l| l.strip_prefix("#define SKA_MAX_L_MAX "))
        .expect("ska.h defines SKA_MAX_L_MAX");
    assert_eq!(define.trim().parse::<usize>().unwrap(), SKA_GPU_MAX_L_MAX);
}
