// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code for allowing other languages to talk to this Rust library. See the
//! examples directory for usage.

// TODO: Error handling.

use std::ffi::CStr;

use crate::fee::FEEBeam;

/// Create a new MWA FEE beam.
///
/// # Arguments
///
/// `hdf5_file` - the path to the MWA FEE beam file.
///
/// # Returns
///
/// * A pointer to a Rust-owned `FEEBeam` struct. This struct must be freed by
/// calling `free_fee_beam`.
///
#[no_mangle]
pub unsafe extern "C" fn new_fee_beam(hdf5_file: *const std::os::raw::c_char) -> *mut FEEBeam {
    let m = CStr::from_ptr(hdf5_file).to_str().unwrap();
    let beam = FEEBeam::new(m).unwrap();
    Box::into_raw(Box::new(beam))
}

/// Create a new MWA FEE beam. Requires the HDF5 beam file path to be specified
/// by the environment variable MWA_BEAM_FILE.
///
/// # Returns
///
/// * A pointer to a Rust-owned `FEEBeam` struct. This struct must be freed by
/// calling `free_fee_beam`.
///
#[no_mangle]
pub unsafe extern "C" fn new_fee_beam_from_env() -> *mut FEEBeam {
    let beam = FEEBeam::new_from_env().unwrap();
    Box::into_raw(Box::new(beam))
}

/// Get the beam response Jones matrix for the given direction and pointing. Can
/// optionally re-define the X and Y polarisations and apply a parallactic-angle
/// correction; see Jack's thorough investigation at
/// https://github.com/JLBLine/polarisation_tests_for_FEE.
///
/// `delays` and `amps` apply to each dipole in a given MWA tile, and *must*
/// have 16 elements (each corresponds to an MWA dipole in a tile, in the M&C
/// order; see
/// https://wiki.mwatelescope.org/pages/viewpage.action?pageId=48005139). `amps`
/// being dipole gains (usually 1 or 0), not digital gains.
///
/// Note the return type (*double); we can't pass complex numbers across the FFI
/// boundary, so the real and imaginary components are unpacked into doubles.
/// The output contains 8 doubles, where the j00 is the first pair, j01 is the
/// second pair, etc.
///
/// # Arguments
///
/// `fee_beam` - A pointer to a `FEEBeam` struct created with the `new_fee_beam`
/// function
/// `az_rad` - The azimuth coordinate of the beam in radians
/// `za_rad` - The zenith angle coordinate of the beam in radians
/// `freq_hz` - The frequency used for the beam response in Hertz
/// `delays` - A pointer to a 16-element array of dipole delays for an MWA tile
/// `amps` - A pointer to a 16-element array of dipole gains for an MWA tile
/// `norm_to_zenith` - A boolean indicating whether the beam response should be
/// normalised with respect to zenith.
/// `parallactic` - A boolean indicating whether the parallactic angle
/// correction should be applied.
///
/// # Returns
///
/// * A pointer to an 8-element Jones matrix array on the heap. This array may
///   be freed by the caller.
///
#[no_mangle]
pub unsafe extern "C" fn calc_jones(
    fee_beam: *mut FEEBeam,
    az_rad: f64,
    za_rad: f64,
    freq_hz: u32,
    delays: *const u32,
    amps: *const f64,
    norm_to_zenith: u8,
    parallactic: u8,
) -> *mut f64 {
    let beam = &mut *fee_beam;
    let delays_s = std::slice::from_raw_parts(delays, 16);
    let amps_s = std::slice::from_raw_parts(amps, 16);
    let norm_bool = match norm_to_zenith {
        0 => false,
        1 => true,
        _ => panic!("A value other than 0 or 1 was used for norm_to_zenith"),
    };
    let para_bool = match parallactic {
        0 => false,
        1 => true,
        _ => panic!("A value other than 0 or 1 was used for parallactic"),
    };

    // Using the passed-in beam, get the beam response (Jones matrix).
    let jones = if para_bool {
        beam.calc_jones(az_rad, za_rad, freq_hz, delays_s, amps_s, norm_bool)
    } else {
        beam.calc_jones_eng(az_rad, za_rad, freq_hz, delays_s, amps_s, norm_bool)
    }
    .unwrap();

    // Because `jones` is a Rust array, it is on the stack and will be "freed"
    // at the end of this function. To transfer ownership of the array we put
    // `jones` onto the heap by putting it in a Box. By casting the array of
    // Complex64 into f64, we assume that the memory layout of a Complex64 is
    // the same as two f64s side by side.
    Box::into_raw(Box::new(jones)) as *mut f64
}

/// Get the beam response Jones matrix for several az/za directions for the
/// given pointing. The Jones matrix elements for each direction are put into a
/// single array. Can optionally re-define the X and Y polarisations and apply a
/// parallactic-angle correction; see Jack's thorough investigation at
/// https://github.com/JLBLine/polarisation_tests_for_FEE.
///
/// `delays` and `amps` apply to each dipole in a given MWA tile, and *must*
/// have 16 elements (each corresponds to an MWA dipole in a tile, in the M&C
/// order; see
/// https://wiki.mwatelescope.org/pages/viewpage.action?pageId=48005139). `amps`
/// being dipole gains (usually 1 or 0), not digital gains.
///
/// As there are 8 floats per Jones matrix, there are 8 * `num_azza` floats in
/// the array. Rust will calculate the Jones matrices in parallel. See the
/// documentation for `calc_jones` for more info.
#[no_mangle]
pub unsafe extern "C" fn calc_jones_array(
    fee_beam: *mut FEEBeam,
    num_azza: u32,
    az_rad: *const f64,
    za_rad: *const f64,
    freq_hz: u32,
    delays: *const u32,
    amps: *const f64,
    norm_to_zenith: u8,
    parallactic: u8,
) -> *mut f64 {
    let beam = &mut *fee_beam;
    let az = std::slice::from_raw_parts(az_rad, num_azza as usize);
    let za = std::slice::from_raw_parts(za_rad, num_azza as usize);
    let delays_s = std::slice::from_raw_parts(delays, 16);
    let amps_s = std::slice::from_raw_parts(amps, 16);
    let norm_bool = match norm_to_zenith {
        0 => false,
        1 => true,
        _ => panic!("A value other than 0 or 1 was used for norm_to_zenith"),
    };
    let para_bool = match parallactic {
        0 => false,
        1 => true,
        _ => panic!("A value other than 0 or 1 was used for parallactic"),
    };

    let mut jones = if para_bool {
        beam.calc_jones_array(az, za, freq_hz, delays_s, amps_s, norm_bool)
    } else {
        beam.calc_jones_eng_array(az, za, freq_hz, delays_s, amps_s, norm_bool)
    }
    .unwrap();
    let ptr = jones.as_mut_ptr();
    std::mem::forget(jones);
    ptr as *mut f64
}

/// The same as "calc_jones", except 32 elements are given to amps. The first 16
/// amps are for the X elements, the next 16 the Y elements.
#[no_mangle]
pub unsafe extern "C" fn calc_jones_all_amps(
    fee_beam: *mut FEEBeam,
    az_rad: f64,
    za_rad: f64,
    freq_hz: u32,
    delays: *const u32,
    amps: *const f64,
    norm_to_zenith: u8,
    parallactic: u8,
) -> *mut f64 {
    let beam = &mut *fee_beam;
    let delays_s = std::slice::from_raw_parts(delays, 16);
    let amps_s = std::slice::from_raw_parts(amps, 32);
    let norm_bool = match norm_to_zenith {
        0 => false,
        1 => true,
        _ => panic!("A value other than 0 or 1 was used for norm_to_zenith"),
    };
    let para_bool = match parallactic {
        0 => false,
        1 => true,
        _ => panic!("A value other than 0 or 1 was used for parallactic"),
    };

    let jones = if para_bool {
        beam.calc_jones(az_rad, za_rad, freq_hz, delays_s, amps_s, norm_bool)
    } else {
        beam.calc_jones_eng(az_rad, za_rad, freq_hz, delays_s, amps_s, norm_bool)
    }
    .unwrap();

    Box::into_raw(Box::new(jones)) as *mut f64
}

/// The same as "calc_jones_array", except 32 elements are given to amps. The
/// first 16 amps are for the X elements, the next 16 the Y elements.
#[no_mangle]
pub unsafe extern "C" fn calc_jones_array_all_amps(
    fee_beam: *mut FEEBeam,
    num_azza: u32,
    az_rad: *const f64,
    za_rad: *const f64,
    freq_hz: u32,
    delays: *const u32,
    amps: *const f64,
    norm_to_zenith: u8,
    parallactic: u8,
) -> *mut f64 {
    let beam = &mut *fee_beam;
    let az = std::slice::from_raw_parts(az_rad, num_azza as usize);
    let za = std::slice::from_raw_parts(za_rad, num_azza as usize);
    let delays_s = std::slice::from_raw_parts(delays, 16);
    let amps_s = std::slice::from_raw_parts(amps, 32);
    let norm_bool = match norm_to_zenith {
        0 => false,
        1 => true,
        _ => panic!("A value other than 0 or 1 was used for norm_to_zenith"),
    };
    let para_bool = match parallactic {
        0 => false,
        1 => true,
        _ => panic!("A value other than 0 or 1 was used for parallactic"),
    };

    let mut jones = if para_bool {
        beam.calc_jones_array(az, za, freq_hz, delays_s, amps_s, norm_bool)
    } else {
        beam.calc_jones_eng_array(az, za, freq_hz, delays_s, amps_s, norm_bool)
    }
    .unwrap();

    let ptr = jones.as_mut_ptr();
    std::mem::forget(jones);
    ptr as *mut f64
}

// Yeah, I wish I could just give the caller the number of frequencies and the
// array in one go, but I'm not sure it's possible.

/// Get the number of available frequencies inside the HDF5 file.
///
/// # Arguments
///
/// `fee_beam` - the pointer to the `FEEBeam` struct.
///
/// # Returns
///
/// * The number of frequencies in the array.
///
#[no_mangle]
pub unsafe extern "C" fn get_num_fee_beam_freqs(fee_beam: *mut FEEBeam) -> u32 {
    let beam = &mut *fee_beam;
    beam.freqs.len() as u32
}

/// Get the available frequencies inside the HDF5 file.
///
/// # Arguments
///
/// `fee_beam` - the pointer to the `FEEBeam` struct.
///
/// # Returns
///
/// * An ascending-sorted array with the available frequencies. Use
/// `get_num_fee_beam_freqs` to get the size of the array.
///
#[no_mangle]
pub unsafe extern "C" fn get_fee_beam_freqs(fee_beam: *mut FEEBeam) -> *mut u32 {
    let beam = &mut *fee_beam;
    let mut freqs = beam.freqs.clone();

    // Ensure that the vector doesn't have extra memory allocated.
    freqs.shrink_to_fit();
    // Get the pointer variable and tell Rust not to deallocate the associated
    // vector.
    let ptr = freqs.as_mut_ptr();
    std::mem::forget(freqs);
    ptr
}

/// Given a frequency in Hz, get the closest available frequency inside the HDF5
/// file.
///
/// # Arguments
///
/// `fee_beam` - the pointer to the `FEEBeam` struct.
///
/// # Returns
///
/// * The closest frequency to the specified frequency in Hz.
///
#[no_mangle]
pub unsafe extern "C" fn closest_freq(fee_beam: *mut FEEBeam, freq: u32) -> u32 {
    let beam = &mut *fee_beam;
    beam.find_closest_freq(freq)
}

/// Free the memory associated with an MWA FEE beam.
///
/// # Arguments
///
/// `fee_beam` - the pointer to the `FEEBeam` struct.
///
#[no_mangle]
pub unsafe extern "C" fn free_fee_beam(fee_beam: *mut FEEBeam) {
    Box::from_raw(fee_beam);
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::*;
    use marlu::ndarray::prelude::*;
    use serial_test::serial;

    #[test]
    #[serial]
    fn test_calc_jones_via_ffi() {
        let file = std::ffi::CString::new("mwa_full_embedded_element_pattern.h5").unwrap();
        let jones = unsafe {
            let beam = new_fee_beam(file.into_raw());
            let jones_ptr = calc_jones(
                beam,
                45.0_f64.to_radians(),
                10.0_f64.to_radians(),
                51200000,
                [0; 16].as_ptr(),
                [1.0; 16].as_ptr(),
                0,
                0,
            );
            Array1::from(Vec::from_raw_parts(jones_ptr, 8, 8))
        };

        let expected = array![
            0.036179, 0.103586, 0.036651, 0.105508, 0.036362, 0.103868, -0.036836, -0.105791,
        ];
        assert_abs_diff_eq!(jones, expected, epsilon = 1e-6);
    }

    #[test]
    #[serial]
    fn test_calc_jones_array_via_ffi() {
        let num_directions = 1000;
        let file = std::ffi::CString::new("mwa_full_embedded_element_pattern.h5").unwrap();
        let jones = unsafe {
            let beam = new_fee_beam(file.into_raw());
            let az = vec![45.0_f64.to_radians(); num_directions];
            let za = vec![10.0_f64.to_radians(); num_directions];
            let jones_ptr = calc_jones_array(
                beam,
                num_directions as _,
                az.as_ptr(),
                za.as_ptr(),
                51200000,
                [0; 16].as_ptr(),
                [1.0; 16].as_ptr(),
                0,
                0,
            );
            Array1::from(Vec::from_raw_parts(
                jones_ptr,
                8 * num_directions,
                8 * num_directions,
            ))
            .into_shape((num_directions, 8))
            .unwrap()
        };

        let expected = array![
            0.036179, 0.103586, 0.036651, 0.105508, 0.036362, 0.103868, -0.036836, -0.105791,
        ];
        assert_abs_diff_eq!(jones.slice(s![0, ..]), expected, epsilon = 1e-6);
        assert_abs_diff_eq!(jones.slice(s![-1, ..]), expected, epsilon = 1e-6);
    }
}
