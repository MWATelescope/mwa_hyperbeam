// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code for allowing other languages to talk to this Rust library. See the
//! examples directory for usage.

#[cfg(test)]
mod tests;

use std::error::Error;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::slice;

use super::FEEBeam;

cfg_if::cfg_if! {
    if #[cfg(feature = "cuda")] {
        use ndarray::prelude::*;
        use super::cuda::{CudaFloat, FEEBeamCUDA};
        use marlu::ndarray;
    }
}

/// Private function to conveniently write errors to error strings.
unsafe fn error_to_error_string<E: Error>(error: E, error_str: *mut c_char) {
    if !error_str.is_null() {
        let error = CString::new(error.to_string()).unwrap();
        let error_bytes = error.as_bytes_with_nul();
        // Rust wants to make a slice of [i8] with c_char, so cast the pointer
        // to get [u8] instead.
        let error_buf: &mut [u8] = slice::from_raw_parts_mut(error_str.cast(), error_bytes.len());
        error_buf[..].copy_from_slice(error_bytes);
    }
}

/// Private function to conveniently write strings to error strings.
unsafe fn string_to_error_string(error: &str, error_str: *mut c_char) {
    if !error_str.is_null() {
        let error = CString::new(error).unwrap();
        let error_bytes = error.as_bytes_with_nul();
        // Rust wants to make a slice of [i8] with c_char, so cast the pointer
        // to get [u8] instead.
        let error_buf: &mut [u8] = slice::from_raw_parts_mut(error_str.cast(), error_bytes.len());
        error_buf[..].copy_from_slice(error_bytes);
    }
}

/// Create a new MWA FEE beam.
///
/// # Arguments
///
/// `hdf5_file` - the path to the MWA FEE beam file.
/// `fee_beam` - a double pointer to the `FEEBeam` struct which is set by this
/// function. This struct must be freed by calling `free_fee_beam`.
/// `error_str` - a pointer to a character array which is set by this function
/// if an error occurs. If this pointer is null, no error message can be
/// reported if an error occurs.
///
/// # Returns
///
/// * An exit code integer. If this is non-zero and an error string was
///   provided, then the error string is set.
///
#[no_mangle]
pub unsafe extern "C" fn new_fee_beam(
    hdf5_file: *const c_char,
    fee_beam: *mut *mut FEEBeam,
    error_str: *mut c_char,
) -> i32 {
    let path = match CStr::from_ptr(hdf5_file).to_str() {
        Ok(p) => p,
        Err(e) => {
            error_to_error_string(e, error_str);
            return 2;
        }
    };
    let beam = match FEEBeam::new(path) {
        Ok(b) => b,
        Err(e) => {
            error_to_error_string(e, error_str);
            return 1;
        }
    };
    *fee_beam = Box::into_raw(Box::new(beam));
    0
}

/// Create a new MWA FEE beam. Requires the HDF5 beam file path to be specified
/// by the environment variable MWA_BEAM_FILE.
///
/// # Arguments
///
/// `fee_beam` - a double pointer to the `FEEBeam` struct which is set by this
/// function. This struct must be freed by calling `free_fee_beam`.
/// `error_str` - a pointer to a character array which is set by this function
/// if an error occurs. If this pointer is null, no error message can be
/// reported if an error occurs.
///
/// # Returns
///
/// * An exit code integer. If this is non-zero and an error string was
///   provided, then the error string is set.
///
#[no_mangle]
pub unsafe extern "C" fn new_fee_beam_from_env(
    fee_beam: *mut *mut FEEBeam,
    error_str: *mut c_char,
) -> i32 {
    let beam = match FEEBeam::new_from_env() {
        Ok(b) => b,
        Err(e) => {
            error_to_error_string(e, error_str);
            return 1;
        }
    };
    *fee_beam = Box::into_raw(Box::new(beam));
    0
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
/// 16 or 32 elements can be supplied for `amps`. If there are 16, then the
/// dipole gains apply to both X and Y elements of dipoles. If there are 32, the
/// first 16 amps are for the X elements, the next 16 the Y elements.
///
/// Note the type of `jones` (*double); we can't pass complex numbers across the
/// FFI boundary, so the real and imaginary components are unpacked into
/// doubles. The output contains 8 doubles, where the j00 is the first pair, j01
/// is the second pair, etc.
///
/// # Arguments
///
/// `fee_beam` - A pointer to a `FEEBeam` struct created with the `new_fee_beam`
/// function
/// `az_rad` - The azimuth direction to get the beam response (units of
/// radians)
/// `za_rad` - The zenith angle direction to get the beam response (units of
/// radians)
/// `freq_hz` - The frequency used for the beam response in Hertz
/// `delays` - A pointer to a 16-element array of dipole delays for an MWA tile
/// `amps` - A pointer to a 16- or 32-element array of dipole gains for an MWA
/// tile. The number of elements is indicated by `num_amps`.
/// `num_amps` - The number of dipole gains used (either 16 or 32).
/// `norm_to_zenith` - A boolean indicating whether the beam response should be
/// normalised with respect to zenith.
/// `parallactic` - A boolean indicating whether the parallactic angle
/// correction should be applied.
/// `jones` - A pointer to a buffer with at least 8 doubles available. The Jones
/// matrix beam response is written here.
/// `error_str` - a pointer to a character array which is set by this function
/// if an error occurs. If this pointer is null, no error message can be
/// reported if an error occurs.
///
/// # Returns
///
/// * An exit code integer. If this is non-zero and an error string was
///   provided, then the error string is set.
///
#[no_mangle]
pub unsafe extern "C" fn calc_jones(
    fee_beam: *mut FEEBeam,
    az_rad: f64,
    za_rad: f64,
    freq_hz: u32,
    delays: *const u32,
    amps: *const f64,
    num_amps: u32,
    norm_to_zenith: u8,
    parallactic: u8,
    jones: *mut f64,
    error_str: *mut c_char,
) -> i32 {
    let beam = &mut *fee_beam;
    let delays_s = slice::from_raw_parts(delays, 16);
    let amps_s = slice::from_raw_parts(amps, num_amps as usize);
    match num_amps {
        16 | 32 => (),
        _ => {
            string_to_error_string(
                "A value other than 16 or 32 was used for num_amps",
                error_str,
            );
            return 1;
        }
    };
    let norm_bool = match norm_to_zenith {
        0 => false,
        1 => true,
        _ => {
            string_to_error_string(
                "A value other than 0 or 1 was used for norm_to_zenith",
                error_str,
            );
            return 1;
        }
    };
    let para_bool = match parallactic {
        0 => false,
        1 => true,
        _ => {
            string_to_error_string(
                "A value other than 0 or 1 was used for parallactic",
                error_str,
            );
            return 1;
        }
    };

    // Using the passed-in beam, get the beam response (Jones matrix).
    let beam_jones_result = if para_bool {
        beam.calc_jones(az_rad, za_rad, freq_hz, delays_s, amps_s, norm_bool)
    } else {
        beam.calc_jones_eng(az_rad, za_rad, freq_hz, delays_s, amps_s, norm_bool)
    };
    let beam_jones = match beam_jones_result {
        Ok(j) => j,
        Err(e) => {
            error_to_error_string(e, error_str);
            return 1;
        }
    };

    let jones_buf = slice::from_raw_parts_mut(jones, 8);
    jones_buf[..].copy_from_slice(&[
        beam_jones[0].re,
        beam_jones[0].im,
        beam_jones[1].re,
        beam_jones[1].im,
        beam_jones[2].re,
        beam_jones[2].im,
        beam_jones[3].re,
        beam_jones[3].im,
    ]);
    0
}

/// Get the beam response Jones matrix for several az/za directions for the
/// given pointing. The Jones matrix elements for each direction are put into a
/// single array (made available with the output pointer `jones`). Can
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
/// As there are 8 doubles per Jones matrix, there are 8 * `num_azza` doubles in
/// the array. Rust will calculate the Jones matrices in parallel. See the
/// documentation for `calc_jones` for more info.
///
/// # Arguments
///
/// `fee_beam` - A pointer to a `FEEBeam` struct created with the `new_fee_beam`
/// function
/// `num_azza` - The number of directions within `az_rad` and `za_rad`
/// `az_rad` - The azimuth direction to get the beam response (units of
/// radians)
/// `za_rad` - The zenith angle direction to get the beam response (units of
/// radians)
/// `freq_hz` - The frequency used for the beam response in Hertz
/// `delays` - A pointer to a 16-element array of dipole delays for an MWA tile
/// `amps` - A pointer to a 16- or 32-element array of dipole gains for an MWA
/// tile. The number of elements is indicated by `num_amps`.
/// `num_amps` - The number of dipole gains used (either 16 or 32).
/// `norm_to_zenith` - A boolean indicating whether the beam response should be
/// normalised with respect to zenith.
/// `parallactic` - A boolean indicating whether the parallactic angle
/// correction should be applied.
/// `jones` - A double pointer to a buffer with at least 8 * num_azza *
/// sizeof(double) bytes available. The Jones matrix beam responses are written
/// here.
/// `error_str` - a pointer to a character array which is set by this function
/// if an error occurs. If this pointer is null, no error message can be
/// reported if an error occurs.
///
/// # Returns
///
/// * An exit code integer. If this is non-zero and an error string was
///   provided, then the error string is set.
///
#[no_mangle]
pub unsafe extern "C" fn calc_jones_array(
    fee_beam: *mut FEEBeam,
    num_azza: u32,
    az_rad: *const f64,
    za_rad: *const f64,
    freq_hz: u32,
    delays: *const u32,
    amps: *const f64,
    num_amps: u32,
    norm_to_zenith: u8,
    parallactic: u8,
    jones: *mut *mut f64,
    error_str: *mut c_char,
) -> i32 {
    let beam = &mut *fee_beam;
    let az = slice::from_raw_parts(az_rad, num_azza as usize);
    let za = slice::from_raw_parts(za_rad, num_azza as usize);
    let delays_s = slice::from_raw_parts(delays, 16);
    let amps_s = slice::from_raw_parts(amps, num_amps as usize);
    match num_amps {
        16 | 32 => (),
        _ => {
            string_to_error_string(
                "A value other than 16 or 32 was used for num_amps",
                error_str,
            );
            return 1;
        }
    };
    let norm_bool = match norm_to_zenith {
        0 => false,
        1 => true,
        _ => {
            string_to_error_string(
                "A value other than 0 or 1 was used for norm_to_zenith",
                error_str,
            );
            return 1;
        }
    };
    let para_bool = match parallactic {
        0 => false,
        1 => true,
        _ => {
            string_to_error_string(
                "A value other than 0 or 1 was used for parallactic",
                error_str,
            );
            return 1;
        }
    };

    let beam_jones_result = if para_bool {
        beam.calc_jones_array(az, za, freq_hz, delays_s, amps_s, norm_bool)
    } else {
        beam.calc_jones_eng_array(az, za, freq_hz, delays_s, amps_s, norm_bool)
    };
    let mut beam_jones = match beam_jones_result {
        Ok(j) => j,
        Err(e) => {
            error_to_error_string(e, error_str);
            return 1;
        }
    };

    let ptr = beam_jones.as_mut_ptr();
    std::mem::forget(beam_jones);
    *jones = ptr.cast();
    0
}

/// Get the available frequencies inside the HDF5 file.
///
/// # Arguments
///
/// `fee_beam` - the pointer to the `FEEBeam` struct.
/// `freqs_ptr` - a double pointer to the FEE beam frequencies. The `const`
/// annotation is deliberate; the caller does not own the frequencies.
/// `num_freqs` - a pointer to a `size_t` whose contents are set.
///
#[no_mangle]
pub unsafe extern "C" fn get_fee_beam_freqs(
    fee_beam: *mut FEEBeam,
    freqs_ptr: *mut *const u32,
    num_freqs: &mut usize,
) {
    let beam = &mut *fee_beam;
    let freqs = beam.get_freqs();
    let freqs_ptr_mutator = &mut *freqs_ptr;
    *freqs_ptr_mutator = freqs.as_ptr();
    *num_freqs = freqs.len();
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

/// Free the memory associated with an `FEEBeam`.
///
/// # Arguments
///
/// `fee_beam` - the pointer to the `FEEBeam` struct.
///
#[no_mangle]
pub unsafe extern "C" fn free_fee_beam(fee_beam: *mut FEEBeam) {
    Box::from_raw(fee_beam);
}

/// Get a `FEEBeamCUDA` struct, which is used to calculate beam responses on a
/// CUDA-capable device.
///
/// # Arguments
///
/// `fee_beam` - a pointer to a previously set `FEEBeam` struct.
/// `freqs_hz` - a pointer to an array of frequencies (units of Hz) at which the
/// beam responses will be calculated.
/// `delays` - a pointer to two-dimensional array of dipole delays. There must
/// be 16 delays per row; each row corresponds to a tile.
/// `amps` - a pointer to two-dimensional array of dipole amplitudes. There must
/// be 16 or 32 amps per row; each row corresponds to a tile. The number of amps
/// per row is specified by `num_amps`.
/// `num_freqs` - the number of frequencies in the array pointed to by `freqs_hz`.
/// `num_tiles` - the number of tiles in both `delays` and `amps`.
/// `num_amps` - either 16 or 32. See the documentation for `calc_jones` for
/// more explanation.
/// `norm_to_zenith` - A boolean indicating whether the beam responses should be
/// normalised with respect to zenith.
/// `cuda_fee_beam` - a double pointer to the `FEEBeamCUDA` struct which is set
/// by this function. This struct must be freed by calling `free_cuda_fee_beam`.
/// `error_str` - a pointer to a character array which is set by this function
/// if an error occurs. If this pointer is null, no error message can be
/// reported if an error occurs.
///
/// # Returns
///
/// * An exit code integer. If this is non-zero and an error string was
///   provided, then the error string is set.
///
#[cfg(feature = "cuda")]
#[no_mangle]
pub unsafe extern "C" fn new_cuda_fee_beam(
    fee_beam: *mut FEEBeam,
    freqs_hz: *const u32,
    delays: *const u32,
    amps: *const f64,
    num_freqs: u32,
    num_tiles: u32,
    num_amps: u32,
    norm_to_zenith: u8,
    cuda_fee_beam: *mut *mut FEEBeamCUDA,
    error_str: *mut c_char,
) -> i32 {
    match num_amps {
        16 | 32 => (),
        _ => {
            string_to_error_string(
                "A value other than 16 or 32 was used for num_amps",
                error_str,
            );
            return 1;
        }
    };
    // Turn the pointers into slices and/or arrays.
    let freqs = slice::from_raw_parts(freqs_hz, num_freqs as usize);
    let amps = ArrayView2::from_shape_ptr((num_tiles as usize, num_amps as usize), amps);
    let delays = ArrayView2::from_shape_ptr((num_tiles as usize, 16), delays);
    let norm_bool = match norm_to_zenith {
        0 => false,
        1 => true,
        _ => {
            string_to_error_string(
                "A value other than 0 or 1 was used for norm_to_zenith",
                error_str,
            );
            return 1;
        }
    };

    let beam = &mut *fee_beam;
    let cuda_beam = match beam.cuda_prepare(freqs, delays, amps, norm_bool) {
        Ok(b) => b,
        Err(e) => {
            error_to_error_string(e, error_str);
            return 1;
        }
    };
    *cuda_fee_beam = Box::into_raw(Box::new(cuda_beam));
    0
}

/// Get beam response Jones matrices for the given directions, using CUDA. The
/// Jones matrix elements for each direction are put into a single array (made
/// available with the output pointer `jones`). Can optionally re-define the X
/// and Y polarisations and apply a parallactic-angle correction; see Jack's
/// thorough investigation at
/// https://github.com/JLBLine/polarisation_tests_for_FEE.
///
/// # Arguments
///
/// `cuda_fee_beam` - A pointer to a `FEEBeamCUDA` struct created with the
/// `new_cuda_fee_beam` function
/// `az_rad` - The azimuth directions to get the beam response (units of
/// radians)
/// `za_rad` - The zenith angle directions to get the beam response (units of
/// radians)
/// `parallactic` - A boolean indicating whether the parallactic angle
/// correction should be applied.
/// `jones` - A double pointer to a buffer with resulting beam-response Jones
/// matrices.
/// `error_str` - a pointer to a character array which is set by this function
/// if an error occurs. If this pointer is null, no error message can be
/// reported if an error occurs.
///
/// # Returns
///
/// * An exit code integer. If this is non-zero and an error string was
///   provided, then the error string is set.
///
#[cfg(feature = "cuda")]
#[no_mangle]
pub unsafe extern "C" fn calc_jones_cuda(
    cuda_fee_beam: *mut FEEBeamCUDA,
    num_azza: u32,
    az_rad: *const CudaFloat,
    za_rad: *const CudaFloat,
    parallactic: u8,
    jones: *mut *mut CudaFloat,
    error_str: *mut c_char,
) -> i32 {
    let beam = &mut *cuda_fee_beam;
    let az = slice::from_raw_parts(az_rad, num_azza as usize);
    let za = slice::from_raw_parts(za_rad, num_azza as usize);
    let para_bool = match parallactic {
        0 => false,
        1 => true,
        _ => {
            string_to_error_string(
                "A value other than 0 or 1 was used for parallactic",
                error_str,
            );
            return 1;
        }
    };
    let mut beam_jones = match beam.calc_jones(az, za, para_bool) {
        Ok(j) => j,
        Err(e) => {
            error_to_error_string(e, error_str);
            return 1;
        }
    };

    let ptr = beam_jones.as_mut_ptr();
    std::mem::forget(beam_jones);
    *jones = ptr.cast();
    0
}

/// Get beam response Jones matrices for the given directions, using CUDA. The
/// Jones matrix elements for each direction are left on the device (the device
/// pointer is communicated via `d_jones`). Can optionally re-define the X and Y
/// polarisations and apply a parallactic-angle correction; see Jack's thorough
/// investigation at https://github.com/JLBLine/polarisation_tests_for_FEE.
///
/// # Arguments
///
/// `cuda_fee_beam` - A pointer to a `FEEBeamCUDA` struct created with the
/// `new_cuda_fee_beam` function
/// `az_rad` - The azimuth directions to get the beam response (units of
/// radians)
/// `za_rad` - The zenith angle directions to get the beam response (units of
/// radians)
/// `parallactic` - A boolean indicating whether the parallactic angle
/// correction should be applied.
/// `d_jones` - A double pointer to a device buffer with resulting beam-response
/// Jones matrices.
/// `error_str` - a pointer to a character array which is set by this function
/// if an error occurs. If this pointer is null, no error message can be
/// reported if an error occurs.
///
/// # Returns
///
/// * An exit code integer. If this is non-zero and an error string was
///   provided, then the error string is set.
///
#[cfg(feature = "cuda")]
#[no_mangle]
pub unsafe extern "C" fn calc_jones_cuda_device(
    cuda_fee_beam: *mut FEEBeamCUDA,
    num_azza: u32,
    az_rad: *const CudaFloat,
    za_rad: *const CudaFloat,
    parallactic: u8,
    d_jones: *mut *mut CudaFloat,
    error_str: *mut c_char,
) -> i32 {
    let beam = &mut *cuda_fee_beam;
    let az = slice::from_raw_parts(az_rad, num_azza as usize);
    let za = slice::from_raw_parts(za_rad, num_azza as usize);
    let para_bool = match parallactic {
        0 => false,
        1 => true,
        _ => {
            string_to_error_string(
                "A value other than 0 or 1 was used for parallactic",
                error_str,
            );
            return 1;
        }
    };
    let device_ptr = match beam.calc_jones_device(az, za, para_bool) {
        Ok(j) => j,
        Err(e) => {
            error_to_error_string(e, error_str);
            return 1;
        }
    };

    *d_jones = device_ptr.get_mut().cast();
    std::mem::forget(device_ptr);
    0
}

/// Get a pointer to the device beam Jones map. This is necessary to access
/// de-duplicated beam Jones matrices on the device.
///
/// # Arguments
///
/// `cuda_fee_beam` - the pointer to the `FEEBeamCUDA` struct.
///
/// # Returns
///
/// * A pointer to the device beam Jones map. The const annotation is
///   deliberate; the caller does not own the map.
///
#[cfg(feature = "cuda")]
#[no_mangle]
pub unsafe extern "C" fn get_cuda_map(cuda_fee_beam: *mut FEEBeamCUDA) -> *const u64 {
    let beam = &mut *cuda_fee_beam;
    beam.get_beam_jones_map()
}

/// Get the number of de-duplicated frequencies associated with this
/// `FEEBeamCUDA`.
///
/// # Arguments
///
/// `cuda_fee_beam` - the pointer to the `FEEBeamCUDA` struct.
///
/// # Returns
///
/// * The number of de-duplicated frequencies associated with this
///   `FEEBeamCUDA`.
///
#[cfg(feature = "cuda")]
#[no_mangle]
pub unsafe extern "C" fn get_num_unique_fee_freqs(cuda_fee_beam: *mut FEEBeamCUDA) -> i32 {
    let beam = &mut *cuda_fee_beam;
    beam.num_freqs
}

/// Free the memory associated with an `FEEBeamCUDA` beam.
///
/// # Arguments
///
/// `cuda_fee_beam` - the pointer to the `FEEBeamCUDA` struct.
///
#[cfg(feature = "cuda")]
#[no_mangle]
pub unsafe extern "C" fn free_cuda_fee_beam(fee_beam: *mut FEEBeamCUDA) {
    Box::from_raw(fee_beam);
}
