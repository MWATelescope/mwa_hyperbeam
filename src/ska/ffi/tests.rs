// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Tests for the SKA-Low FFI functions.

use std::{ffi::CString, ptr::null};

use approx::assert_abs_diff_eq;

use super::*;
use crate::ffi::{hb_last_error_length, hb_last_error_message};

fn last_error() -> String {
    let len = hb_last_error_length();
    let mut buf = vec![0 as c_char; len.max(1) as usize];
    unsafe { hb_last_error_message(buf.as_mut_ptr(), len) };
    unsafe { CStr::from_ptr(buf.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

#[test]
fn ffi_analytic_matches_rust() {
    unsafe {
        let mut beam: *mut SkaLowBeam = std::ptr::null_mut();
        assert_eq!(new_ska_low_analytic_beam(1, &mut beam), 0);
        assert_eq!(ska_low_get_num_elements(beam), 0);

        let lat = -0.4660608448386394;
        let mut jones = [0.0; 8];
        assert_eq!(
            ska_low_calc_jones(beam, 0.7, 0.4, 150e6, 0, &lat, 1, jones.as_mut_ptr()),
            0
        );
        let expected = (*beam)
            .calc_jones_pair(0.7, 0.4, 150e6, 0, Some(lat), true)
            .unwrap();
        for k in 0..4 {
            assert_abs_diff_eq!(jones[2 * k], expected[k].re);
            assert_abs_diff_eq!(jones[2 * k + 1], expected[k].im);
        }

        let az = [0.1, 1.0, 2.0];
        let za = [0.2, 0.5, 1.2];
        let mut arr = [0.0; 24];
        assert_eq!(
            ska_low_calc_jones_array(
                beam,
                3,
                az.as_ptr(),
                za.as_ptr(),
                150e6,
                0,
                null(),
                0,
                arr.as_mut_ptr()
            ),
            0
        );
        for i in 0..3 {
            let e = (*beam)
                .calc_jones_pair(az[i], za[i], 150e6, 0, None, false)
                .unwrap();
            for k in 0..4 {
                assert_abs_diff_eq!(arr[8 * i + 2 * k], e[k].re);
                assert_abs_diff_eq!(arr[8 * i + 2 * k + 1], e[k].im);
            }
        }

        let mut raw = [0.0; 8];
        assert_eq!(
            ska_low_element_response(beam, 0.0, 0.0, 150e6, 0, raw.as_mut_ptr()),
            0
        );
        assert_abs_diff_eq!(raw[0], -1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(raw[6], 1.0, epsilon = 1e-12);

        // Station: at the pointing centre, the station response equals the
        // element response.
        let positions = [-1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 0.0];
        let mut st = [0.0; 8];
        assert_eq!(
            ska_low_calc_station_jones_array(
                beam,
                3,
                positions.as_ptr(),
                null(),
                0.0,
                1,
                [0.7].as_ptr(),
                [0.4].as_ptr(),
                150e6,
                0.7,
                0.4,
                &lat,
                1,
                st.as_mut_ptr()
            ),
            0
        );
        for (a, b) in st.iter().zip(jones.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-12);
        }

        free_ska_low_beam(beam);
    }
}

#[test]
fn ffi_errors() {
    unsafe {
        let mut beam: *mut SkaLowBeam = std::ptr::null_mut();
        assert_eq!(new_ska_low_analytic_beam(2, &mut beam), 1);
        assert!(last_error().contains("cos_taper"));

        let missing = CString::new("/does/not/exist.h5").unwrap();
        assert_eq!(new_ska_low_eep_grid_beam(missing.as_ptr(), 1, &mut beam), 1);
        assert!(last_error().contains("doesn't exist"));

        assert_eq!(new_ska_low_eep_grid_beam(null(), 0, &mut beam), 1);
        assert!(last_error().contains("requires an explicit"));

        let missing = CString::new("/does/not/exist").unwrap();
        assert_eq!(new_ska_low_eep_sph_beam(missing.as_ptr(), 0, &mut beam), 1);

        assert_eq!(new_ska_low_analytic_beam(1, &mut beam), 0);
        let mut jones = [0.0; 8];
        assert_eq!(
            ska_low_calc_jones(beam, 0.0, 0.1, 1e8, 0, null(), 5, jones.as_mut_ptr()),
            1
        );
        assert!(last_error().contains("iau_order"));
        // All elements flagged.
        let positions = [0.0; 6];
        let weights = [0.0; 2];
        assert_eq!(
            ska_low_calc_station_jones_array(
                beam,
                2,
                positions.as_ptr(),
                weights.as_ptr(),
                0.0,
                1,
                [0.0].as_ptr(),
                [0.1].as_ptr(),
                1e8,
                0.0,
                0.0,
                null(),
                0,
                jones.as_mut_ptr()
            ),
            1
        );
        assert!(last_error().contains("flagged"));
        free_ska_low_beam(beam);
        free_ska_low_beam(std::ptr::null_mut());
    }
}

#[test]
fn ffi_rotated_station_matches_rust() {
    unsafe {
        let mut beam: *mut SkaLowBeam = std::ptr::null_mut();
        assert_eq!(new_ska_low_analytic_beam(1, &mut beam), 0);
        let positions = [-1.0, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0, 2.0, 0.1];
        let (az, za) = ([0.3, 2.1], [0.4, 1.0]);
        let lat = -0.4660608448386394;
        let rotation = 0.83;
        let mut jones = [0.0; 16];
        assert_eq!(
            ska_low_calc_station_jones_array(
                beam,
                3,
                positions.as_ptr(),
                null(),
                rotation,
                2,
                az.as_ptr(),
                za.as_ptr(),
                1.2e8,
                0.5,
                0.2,
                &lat,
                1,
                jones.as_mut_ptr()
            ),
            0
        );
        let station = SkaStation::new(&*beam, positions.as_chunks::<3>().0.to_vec(), None)
            .unwrap()
            .with_rotation(rotation);
        for i in 0..2 {
            let e = station
                .calc_jones_pair(az[i], za[i], 1.2e8, 0.5, 0.2, Some(lat), true)
                .unwrap();
            for k in 0..4 {
                assert_abs_diff_eq!(jones[8 * i + 2 * k], e[k].re);
                assert_abs_diff_eq!(jones[8 * i + 2 * k + 1], e[k].im);
            }
        }
        free_ska_low_beam(beam);
    }
}
