// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI, TAU};

use approx::*;
use marlu::constants::{MWA_LAT_RAD, VEL_C};

use super::*;

/// A struct to hold all of the args to pass to a calculation, and the expected
/// results.
///
/// Rather than just putting these inside their own tests, we can test the FFI
/// and non-FFI code the exact same way and ensure consistency.
pub(crate) struct AnalyticArgsAndExpectation {
    pub(crate) az_rad: f64,
    pub(crate) za_rad: f64,
    pub(crate) freq_hz: u32,
    pub(crate) delays: [u32; 16],
    pub(crate) amps: [f64; 16],
    pub(crate) norm_to_zenith: bool,
    pub(crate) expected: [f64; 8],
}

pub(crate) const MWA_PB_1: AnalyticArgsAndExpectation = AnalyticArgsAndExpectation {
    az_rad: 91.459449355_f64 * PI / 180.0,
    za_rad: 56.5383409732_f64 * PI / 180.0,
    freq_hz: 180e6 as _,
    delays: [0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6],
    amps: [1.0; 16],
    norm_to_zenith: true,
    expected: [
        -0.0084149,
        0.08842726,
        0.00038883,
        -0.00408598,
        0.00021439,
        -0.00225292,
        0.01526155,
        -0.16037479,
    ],
};

pub(crate) const MWA_PB_2: AnalyticArgsAndExpectation = AnalyticArgsAndExpectation {
    az_rad: 356.5317707411 * PI / 180.0,
    za_rad: 35.7053945504 * PI / 180.0,
    freq_hz: 180e6 as _,
    delays: [0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6],
    amps: [1.0; 16],
    norm_to_zenith: true,
    expected: [
        0.00038578,
        -0.0040539,
        -0.00783877,
        0.08237309,
        -0.00636531,
        0.0668893,
        -0.00047508,
        0.00499231,
    ],
};

pub(crate) const MWA_PB_3: AnalyticArgsAndExpectation = AnalyticArgsAndExpectation {
    az_rad: 356.5317707411 * PI / 180.0,
    za_rad: 35.7053945504 * PI / 180.0,
    freq_hz: 180e6 as _,
    delays: [0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6],
    amps: [1.0; 16],
    norm_to_zenith: false,
    expected: [
        0.00066878,
        -0.00702778,
        -0.01358918,
        0.1428008,
        -0.0110348,
        0.11595833,
        -0.00082359,
        0.00865459,
    ],
};

pub(crate) const MWA_PB_4: AnalyticArgsAndExpectation = AnalyticArgsAndExpectation {
    az_rad: FRAC_PI_2,
    za_rad: FRAC_PI_4,
    freq_hz: 150e6 as _,
    delays: [0; 16],
    amps: [
        0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
    ],
    norm_to_zenith: true,
    expected: [
        -6.99192957e-02,
        -1.92627390e-01,
        -6.05470376e-18,
        -1.66806855e-17,
        -4.28132209e-18,
        -1.17950258e-17,
        9.88808163e-02,
        2.72416268e-01,
    ],
};

pub(crate) const MWA_PB_5: AnalyticArgsAndExpectation = AnalyticArgsAndExpectation {
    az_rad: 0.1,
    za_rad: 0.1,
    freq_hz: 200e6 as _,
    delays: [0; 16],
    amps: [1.0; 16],
    norm_to_zenith: true,
    expected: [
        0.08643823,
        1.37507661e-18,
        0.86582464,
        1.37737108e-17,
        0.86149913,
        1.37048996e-17,
        -0.08687223,
        -1.38198075e-18,
    ],
};

pub(crate) const RTS_1: AnalyticArgsAndExpectation = AnalyticArgsAndExpectation {
    az_rad: 1.5049529281106273,
    za_rad: 0.1213599693,
    freq_hz: 150e6 as _,
    delays: [0; 16],
    amps: [1.0; 16],
    norm_to_zenith: true,
    expected: [
        0.887545121,
        0.0,
        -0.053721261,
        0.0,
        0.052475453,
        0.0,
        0.881124746,
        0.0,
    ],
};

pub(crate) const RTS_2: AnalyticArgsAndExpectation = AnalyticArgsAndExpectation {
    az_rad: 1.5049529281106273,
    za_rad: 0.1213599693,
    freq_hz: 170e6 as _,
    delays: [0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6],
    amps: [1.0; 16],
    norm_to_zenith: true,
    expected: [
        0.868722009,
        0.0,
        -0.052581937,
        0.0,
        0.051362550,
        0.0,
        0.862437798,
        0.0,
    ],
};

pub(crate) const RTS_3: AnalyticArgsAndExpectation = AnalyticArgsAndExpectation {
    az_rad: 4.724779027649792,
    za_rad: 0.3230075857,
    freq_hz: 200e6 as _,
    delays: [0, 2, 4, 6, 0, 1, 2, 3, 10, 12, 14, 16, 0, 4, 8, 12],
    amps: [1.0; 16],
    norm_to_zenith: true,
    expected: [
        0.136849229,
        0.0,
        0.021808286,
        0.0,
        -0.020509968,
        0.0,
        0.129801306,
        0.0,
    ],
};

pub(crate) const RTS_4: AnalyticArgsAndExpectation = AnalyticArgsAndExpectation {
    az_rad: 4.724779027649792,
    za_rad: 0.3230075857,
    freq_hz: 200e6 as _,
    delays: [0, 2, 4, 6, 0, 1, 2, 3, 10, 12, 14, 16, 0, 4, 8, 12],
    amps: [1.0; 16],
    norm_to_zenith: false,
    expected: [
        0.260376182,
        0.0,
        0.041493535,
        0.0,
        -0.039023291,
        0.0,
        0.246966452,
        0.0,
    ],
};

pub(crate) const RTS_5: AnalyticArgsAndExpectation = AnalyticArgsAndExpectation {
    az_rad: 0.1,
    za_rad: 0.1,
    freq_hz: 200e6 as _,
    delays: [0; 16],
    amps: [1.0; 16],
    norm_to_zenith: true,
    expected: [
        0.866252349,
        0.0,
        -0.004176485,
        0.0,
        0.003330862,
        0.0,
        0.870518564,
        0.0,
    ],
};

macro_rules! test_analytic {
    ($beam:expr, $args:expr, $epsilon:expr) => {{
        let AnalyticArgsAndExpectation {
            az_rad,
            za_rad,
            freq_hz,
            delays,
            amps,
            norm_to_zenith,
            expected,
        } = $args;
        let result = $beam.calc_jones_pair(
            az_rad,
            za_rad,
            freq_hz,
            &delays,
            &amps,
            MWA_LAT_RAD,
            norm_to_zenith,
        );
        assert!(result.is_ok());
        let result = result.unwrap();
        let expected = Jones::from(expected);

        assert_abs_diff_eq!(result, expected, epsilon = $epsilon);
    }};
}

// Match the behaviour from mwa_pb.
#[test]
fn mwa_pb_1() {
    let beam = AnalyticBeam::new();
    test_analytic!(beam, MWA_PB_1, 1e-5);
}

#[test]
fn mwa_pb_2() {
    let beam = AnalyticBeam::new();
    test_analytic!(beam, MWA_PB_2, 1e-5);
}

#[test]
fn mwa_pb_3() {
    let beam = AnalyticBeam::new();
    test_analytic!(beam, MWA_PB_3, 1e-5);
}

#[test]
fn mwa_pb_4() {
    let beam = AnalyticBeam::new();
    test_analytic!(beam, MWA_PB_4, 1e-4);
}

#[test]
fn mwa_pb_5() {
    let beam = AnalyticBeam::new();
    test_analytic!(beam, MWA_PB_5, 1e-5);
}

#[test]
fn mwa_pb_single_matches_array() {
    let beam = AnalyticBeam::new();
    let result = beam.calc_jones_pair(
        4.724779027649792,
        0.3230075857,
        200e6 as _,
        &[0, 2, 4, 6, 0, 1, 2, 3, 10, 12, 14, 16, 0, 4, 8, 12],
        &[1.0; 16],
        MWA_LAT_RAD,
        false,
    );
    assert!(result.is_ok());
    let result = result.unwrap();

    let result_a = beam.calc_jones_array_pair(
        &[4.724779027649792],
        &[0.3230075857],
        200e6 as _,
        &[0, 2, 4, 6, 0, 1, 2, 3, 10, 12, 14, 16, 0, 4, 8, 12],
        &[1.0; 16],
        MWA_LAT_RAD,
        false,
    );
    assert!(result_a.is_ok());
    let result_a = result_a.unwrap()[0];

    assert_abs_diff_eq!(result, result_a);
}

// Match the behaviour from the RTS.
#[test]
fn rts_1() {
    let beam = AnalyticBeam::new_rts();
    test_analytic!(beam, RTS_1, 1e-9);
}

#[test]
fn rts_2() {
    let beam = AnalyticBeam::new_rts();
    test_analytic!(beam, RTS_2, 1e-9);
}

#[test]
fn rts_3() {
    let beam = AnalyticBeam::new_rts();
    test_analytic!(beam, RTS_3, 1e-9);
}

#[test]
fn rts_4() {
    let beam = AnalyticBeam::new_rts();
    test_analytic!(beam, RTS_4, 1e-9);
}

#[test]
fn rts_5() {
    let beam = AnalyticBeam::new_rts();
    test_analytic!(beam, RTS_5, 1e-9);
}

#[test]
fn rts_single_matches_array() {
    let beam = AnalyticBeam::new_rts();
    let result = beam.calc_jones_pair(
        4.724779027649792,
        0.3230075857,
        200e6 as _,
        &[0, 2, 4, 6, 0, 1, 2, 3, 10, 12, 14, 16, 0, 4, 8, 12],
        &[1.0; 16],
        MWA_LAT_RAD,
        false,
    );
    assert!(result.is_ok());
    let result = result.unwrap();

    let result_a = beam.calc_jones_array_pair(
        &[4.724779027649792],
        &[0.3230075857],
        200e6 as _,
        &[0, 2, 4, 6, 0, 1, 2, 3, 10, 12, 14, 16, 0, 4, 8, 12],
        &[1.0; 16],
        MWA_LAT_RAD,
        false,
    );
    assert!(result_a.is_ok());
    let result_a = result_a.unwrap()[0];

    assert_abs_diff_eq!(result, result_a);
}

#[test]
fn test_fix_amps_1() {
    let amps = fix_amps(&[1.0; 16], &[0; 16]);
    assert_eq!(amps, [1.0; 16]);

    let amps = fix_amps(&[1.0; 32], &[0; 16]);
    assert_eq!(amps, [1.0; 16]);
}

#[test]
fn test_fix_amps_2() {
    let amps = fix_amps(
        &[1.0; 16],
        &[32, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    );
    assert_eq!(
        amps,
        [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    );
}

#[test]
fn test_fix_amps_3() {
    let amps = fix_amps(
        &[
            1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        ],
        &[32, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    );
    assert_eq!(
        amps,
        [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    );
}

#[test]
fn test_fix_amps_4() {
    let amps = fix_amps(
        &[
            1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        ],
        &[32, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    );
    assert_eq!(
        amps,
        [0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    );
}

#[test]
fn test_cram() {
    let az_rad = 91.459449355_f64 * PI / 180.0;
    let za_rad = 56.5383409732_f64 * PI / 180.0;
    let freq_hz = 180e6 as _;
    let delays = [
        0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6, 0, 2,
        4, 6, 0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6,
        0, 2, 4, 6,
    ];
    let amps = [1.0; 64];
    let norm_to_zenith = true;

    let beam = AnalyticBeam::new_custom(AnalyticType::MwaPb, 0.3, 8);
    let result = beam.calc_jones_pair(
        az_rad,
        za_rad,
        freq_hz,
        &delays,
        &amps,
        MWA_LAT_RAD,
        norm_to_zenith,
    );
    result.unwrap();
}

/// The power of a beam response.
fn power(j: Jones<f64>) -> f64 {
    j.iter().map(|c| c.norm_sqr()).sum()
}

/// The zenith angle a tile points at when the delays of neighbouring bowties
/// differ by `delay_steps`.
fn steered_za(delay_steps: u32) -> f64 {
    (f64::from(delay_steps) * DELAY_STEP * VEL_C / MWA_DPL_SEP).asin()
}

// In the M&C order, the first row of bowties is the northernmost. Delays that
// grow from the first row to the last therefore point the tile south. This is
// the 1430233584 gridpoint, which is south of zenith.
// https://github.com/MWATelescope/mwa_hyperbeam/issues/23
#[test]
fn delays_growing_with_row_point_south() {
    let delays = [0, 0, 0, 0, 2, 2, 2, 2, 4, 4, 4, 4, 6, 6, 6, 6];
    let amps = [1.0; 16];
    let freq_hz = 150e6 as u32;
    let za = steered_za(2);

    for beam in [AnalyticBeam::new(), AnalyticBeam::new_rts()] {
        let calc = |az: f64| {
            beam.calc_jones_pair(az, za, freq_hz, &delays, &amps, MWA_LAT_RAD, false)
                .unwrap()
        };
        let (north, east, south, west) = (calc(0.0), calc(FRAC_PI_2), calc(PI), calc(-FRAC_PI_2));
        assert!(
            power(south) > 2.0 * power(north),
            "south: {}, north: {}",
            power(south),
            power(north)
        );
        assert!(power(south) > power(east));
        assert!(power(south) > power(west));
    }

    // For the mwa_pb beam, all bowties are in phase in the direction the tile
    // points at, so the response there is that of a single bowtie over its
    // ground screen.
    let beam = AnalyticBeam::new();
    let south = beam
        .calc_jones_pair(PI, za, freq_hz, &delays, &amps, MWA_LAT_RAD, false)
        .unwrap();
    let lambda_m = VEL_C / f64::from(freq_hz);
    let ground_plane = 2.0 * (TAU * 0.278 / lambda_m * za.cos()).sin();
    // Due south, the X (east-west) bowties only see the phi polarisation.
    assert_abs_diff_eq!(south[0].norm(), 0.0, epsilon = 1e-10);
    assert_abs_diff_eq!(south[1].norm(), ground_plane, epsilon = 1e-10);
    assert_abs_diff_eq!(south[2].norm(), ground_plane * za.cos(), epsilon = 1e-10);
    assert_abs_diff_eq!(south[3].norm(), 0.0, epsilon = 1e-10);
}

// The first bowtie in a row is the westernmost, so delays that grow along a
// row point the tile east.
#[test]
fn delays_growing_along_rows_point_east() {
    let delays = [0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6];
    let amps = [1.0; 16];
    let freq_hz = 150e6 as u32;
    let za = steered_za(2);

    for beam in [AnalyticBeam::new(), AnalyticBeam::new_rts()] {
        let calc = |az: f64| {
            beam.calc_jones_pair(az, za, freq_hz, &delays, &amps, MWA_LAT_RAD, false)
                .unwrap()
        };
        let (north, east, south, west) = (calc(0.0), calc(FRAC_PI_2), calc(PI), calc(-FRAC_PI_2));
        assert!(power(east) > 2.0 * power(west));
        assert!(power(east) > power(north));
        assert!(power(east) > power(south));
    }
}

// The same as `delays_growing_with_row_point_south`, but for a CRAM tile.
#[test]
fn cram_delays_growing_with_row_point_south() {
    let mut delays = [0; 64];
    for (i, delay) in delays.iter_mut().enumerate() {
        *delay = (i / 8) as u32;
    }
    let amps = [1.0; 64];
    let freq_hz = 150e6 as u32;
    let za = steered_za(1);

    for beam_type in [AnalyticType::MwaPb, AnalyticType::Rts] {
        let beam = AnalyticBeam::new_custom(beam_type, 0.3, 8);
        let calc = |az: f64| {
            beam.calc_jones_pair(az, za, freq_hz, &delays, &amps, MWA_LAT_RAD, false)
                .unwrap()
        };
        assert!(power(calc(PI)) > 2.0 * power(calc(0.0)));
    }
}
