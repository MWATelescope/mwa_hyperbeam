// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

/*!
Code for allowing other languages to talk to this Rust library. See the examples
directory for usage.
 */

// TODO: Error handling.

use std::ffi::CStr;

use crate::*;

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
    let m = CStr::from_ptr(hdf5_file).to_str().unwrap().to_string();
    let beam = FEEBeam::new(&m).unwrap();
    Box::into_raw(Box::new(beam))
}

/// Get the beam response Jones matrix for the given pointing.
///
/// Note the return type (*double); we can't pass complex numbers across the FFI
/// boundary, so the real and imaginary components are unpacked into
/// doubles. The output contains 8 doubles, where the j00 is the first pair, j01
/// is the second pair, etc.
///
/// # Arguments
///
/// `fee_beam` - A pointer to a `FEEBeam` struct created with the `new_fee_beam`
/// function
/// `az_rad` - The azimuth coordinate of the beam in radians
/// `za_rad` - The zenith angle coordinate of the beam in radians
/// `freq_hz` - The frequency used for the beam response in Hertz
/// `delays` - A pointer to a 16-element array of dipole delays for an MWA tile.
/// `amps` - A pointer to a 16-element array of dipole gains for an MWA tile.
/// `norm_to_zenith` - A boolean indicating whether the beam response should be
/// normalised with respect to zenith.
///
/// # Returns
///
/// * A pointer to an 8-element Jones matrix array on the heap. This array may
/// be freed by the caller.
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
) -> *mut f64 {
    let beam = &mut *fee_beam;
    let delays_s = std::slice::from_raw_parts(delays, 16);
    let amps_s = std::slice::from_raw_parts(amps, 16);
    let norm_bool = match norm_to_zenith {
        0 => false,
        1 => true,
        _ => todo!(),
    };

    // Using the passed-in beam, get the beam response (Jones matrix).
    let jones = beam
        .calc_jones(az_rad, za_rad, freq_hz, delays_s, amps_s, norm_bool)
        .unwrap();

    // Because `jones` is a Rust slice, it is on the stack. We cannot safely
    // pass this memory across the FFI boundary, so we put `jones` onto the heap
    // by putting it in a Box. By casting the array of Complex64 into f64, we
    // assume that the memory layout of a Complex64 is the same as two f64s side
    // by side.
    Box::into_raw(Box::new(jones)) as *mut f64
}

/// Get the beam response Jones matrix for several pointings. The Jones matrix
/// elements for each pointing are put into a single array. As there are 8
/// floats per Jones matrix, there are 8 * `num_pointings` floats in the array.
///
/// Rust will calculate the Jones matrices in parallel.
///
/// See the documentation for `calc_jones` for more info.
#[no_mangle]
pub unsafe extern "C" fn calc_jones_array(
    fee_beam: *mut FEEBeam,
    num_pointings: u32,
    az_rad: *const f64,
    za_rad: *const f64,
    freq_hz: u32,
    delays: *const u32,
    amps: *const f64,
    norm_to_zenith: u8,
) -> *mut f64 {
    let beam = &mut *fee_beam;
    let az = std::slice::from_raw_parts(az_rad, num_pointings as usize);
    let za = std::slice::from_raw_parts(za_rad, num_pointings as usize);
    let delays_s = std::slice::from_raw_parts(delays, 16);
    let amps_s = std::slice::from_raw_parts(amps, 16);
    let norm_bool = match norm_to_zenith {
        0 => false,
        1 => true,
        _ => todo!(),
    };

    let jones = beam
        .calc_jones_array(az, za, freq_hz, delays_s, amps_s, norm_bool)
        .unwrap();
    // Put all the Jones matrix elements into a flatten array on the heap.
    let mut jones_flattened = Vec::with_capacity(4 * jones.len());
    for j in jones.into_iter() {
        jones_flattened.push(j[0]);
        jones_flattened.push(j[1]);
        jones_flattened.push(j[2]);
        jones_flattened.push(j[3]);
    }
    // Ensure that the vector doesn't have extra memory allocated.
    jones_flattened.shrink_to_fit();
    // `jones_heap` is a vector. Rust will automatically deallocate it at the
    // end of this function. To stop that, get the pointer to the memory, then
    // tell Rust to forget about the vector.
    let ptr = jones_flattened.as_mut_ptr();
    std::mem::forget(jones_flattened);
    ptr as *mut f64
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
