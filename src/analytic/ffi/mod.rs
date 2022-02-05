// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code for allowing other languages to talk to this Rust library's analytic
//! beam code. See the examples directory for usage.

#[cfg(test)]
mod tests;

use std::slice;

use super::{AnalyticBeam, AnalyticType};
use crate::ffi::{ffi_error, update_last_error};

cfg_if::cfg_if! {
    if #[cfg(any(feature = "cuda", feature = "hip"))] {
        use ndarray::prelude::*;

        use super::AnalyticBeamGpu;
        use crate::gpu::{DevicePointer, GpuFloat};
    }
}

/// Create a new MWA analytic beam.
///
/// # Arguments
///
/// * `rts_style` - a boolean to indicate whether to use RTS-style beam
///   responses. If this is true (a value of 1), RTS-style responses are
///   generated. The default is to use mwa_pb-style responses.
/// * `dipole_height_metres` - an optional pointer to a `double`. If this is not
///   null, the pointer is dereferenced and used as the dipole height (units of
///   metres). If it is null, then a default is used; the default depends on
///   whether this beam object is mwa_pb- or RTS-style.
/// * `analytic_beam` - a double pointer to the `AnalyticBeam` struct
///   which is set by this function. This struct must be freed by calling
///   `free_analytic_beam`.
///
/// # Returns
///
/// * An exit code integer. If this is non-zero then an error occurred; the
///   details can be obtained by (1) getting the length of the error string by
///   calling `hb_last_error_length` and (2) calling `hb_last_error_message`
///   with a string buffer with a length at least equal to the error length.
///
#[no_mangle]
pub unsafe extern "C" fn new_analytic_beam(
    rts_style: u8,
    dipole_height_metres: *const f64,
    analytic_beam: *mut *mut AnalyticBeam,
) -> i32 {
    let analytic_type = match rts_style {
        0 => AnalyticType::MwaPb,
        1 => AnalyticType::Rts,
        _ => {
            update_last_error("A value other than 0 or 1 was used for rts_style".to_string());
            return 1;
        }
    };
    let dipole_height_metres = dipole_height_metres.as_ref().copied();
    let beam = AnalyticBeam::new_custom(
        analytic_type,
        dipole_height_metres.unwrap_or_else(|| analytic_type.get_default_dipole_height()),
    );
    *analytic_beam = Box::into_raw(Box::new(beam));
    0
}

/// Get the beam response Jones matrix for the given direction and pointing.
///
/// `delays` and `amps` apply to each dipole in a given MWA tile, and *must*
/// have 16 elements (each corresponds to an MWA dipole in a tile, in the M&C
/// order; see
/// <https://wiki.mwatelescope.org/pages/viewpage.action?pageId=48005139>).
/// `amps` being dipole gains (usually 1 or 0), not digital gains.
///
/// 16 or 32 elements can be supplied for `amps`. If there are 16, then the
/// dipole gains apply to both X and Y elements of dipoles. If there are 32, the
/// first 16 amps are for the X elements, the next 16 the Y elements.
///
/// Note the type of `jones` (`*double`); we can't pass complex numbers across
/// the FFI boundary, so the real and imaginary components are unpacked into
/// doubles. The output contains 8 doubles, where the j00 is the first pair, j01
/// is the second pair, etc.
///
/// # Arguments
///
/// * `analytic_beam` - A pointer to a `AnalyticBeam` struct created with the
///   `new_analytic_beam` function
/// * `az_rad` - The azimuth direction to get the beam response (units of
///   radians)
/// * `za_rad` - The zenith angle direction to get the beam response (units of
///   radians)
/// * `freq_hz` - The frequency used for the beam response in Hertz
/// * `delays` - A pointer to a 16-element array of dipole delays for an MWA
///   tile
/// * `amps` - A pointer to a 16- or 32-element array of dipole gains for an MWA
///   tile. The number of elements is indicated by `num_amps`.
/// * `num_amps` - The number of dipole gains used (either 16 or 32).
/// * `latitude_rad` - The telescope latitude to use in beam calculations.
/// * `norm_to_zenith` - A boolean indicating whether the beam response should
///   be normalised with respect to zenith.
/// * `jones` - A pointer to a buffer with at least `8 * sizeof(double)`
///   allocated. The Jones matrix beam response is written here.
///
/// # Returns
///
/// * An exit code integer. If this is non-zero then an error occurred; the
///   details can be obtained by (1) getting the length of the error string by
///   calling `hb_last_error_length` and (2) calling `hb_last_error_message`
///   with a string buffer with a length at least equal to the error length.
///
#[no_mangle]
pub unsafe extern "C" fn analytic_calc_jones(
    analytic_beam: *mut AnalyticBeam,
    az_rad: f64,
    za_rad: f64,
    freq_hz: u32,
    delays: *const u32,
    amps: *const f64,
    num_amps: u32,
    latitude_rad: f64,
    norm_to_zenith: u8,
    jones: *mut f64,
) -> i32 {
    match num_amps {
        16 | 32 => (),
        _ => {
            update_last_error("A value other than 16 or 32 was used for num_amps".to_string());
            return 1;
        }
    };
    let norm_bool = match norm_to_zenith {
        0 => false,
        1 => true,
        _ => {
            update_last_error("A value other than 0 or 1 was used for norm_to_zenith".to_string());
            return 1;
        }
    };

    let beam = &*analytic_beam;
    let delays_s = slice::from_raw_parts(delays, 16);
    let amps_s = slice::from_raw_parts(amps, num_amps as usize);

    // Using the passed-in beam, get the beam response (Jones matrix).
    match beam.calc_jones_pair(
        az_rad,
        za_rad,
        freq_hz,
        delays_s,
        amps_s,
        latitude_rad,
        norm_bool,
    ) {
        Ok(j) => {
            let jones_buf = slice::from_raw_parts_mut(jones, 8);
            jones_buf[..].copy_from_slice(&[
                j[0].re, j[0].im, j[1].re, j[1].im, j[2].re, j[2].im, j[3].re, j[3].im,
            ]);
            0
        }
        Err(e) => {
            update_last_error(e.to_string());
            1
        }
    }
}

/// Get the beam response Jones matrix for several az/za directions for the
/// given pointing. The Jones matrix elements for each direction are put into a
/// single array (made available with the output pointer `jones`).
///
/// `delays` and `amps` apply to each dipole in a given MWA tile, and *must*
/// have 16 elements (each corresponds to an MWA dipole in a tile, in the M&C
/// order; see
/// <https://wiki.mwatelescope.org/pages/viewpage.action?pageId=48005139>).
/// `amps` being dipole gains (usually 1 or 0), not digital gains.
///
/// As there are 8 elements per Jones matrix, there must be at least `8 *
/// num_azza * sizeof(double)` allocated in the array. Rust will calculate the
/// Jones matrices in parallel. See the documentation for `calc_jones` for more
/// info.
///
/// # Arguments
///
/// * `analytic_beam` - A pointer to a `AnalyticBeam` struct created with the
///   `new_analytic_beam` function
/// * `num_azza` - The number of directions within `az_rad` and `za_rad`
/// * `az_rad` - The azimuth direction to get the beam response (units of
///   radians)
/// * `za_rad` - The zenith angle direction to get the beam response (units of
///   radians)
/// * `freq_hz` - The frequency used for the beam response in Hertz
/// * `delays` - A pointer to a 16-element array of dipole delays for an MWA
///   tile
/// * `amps` - A pointer to a 16- or 32-element array of dipole gains for an MWA
///   tile. The number of elements is indicated by `num_amps`.
/// * `num_amps` - The number of dipole gains used (either 16 or 32).
/// * `latitude_rad` - The telescope latitude to use in beam calculations.
/// * `norm_to_zenith` - A boolean indicating whether the beam response should
///   be normalised with respect to zenith.
/// * `jones` - A pointer to a buffer with at least `8 * num_azza *
///   sizeof(double)` bytes allocated. The Jones matrix beam responses are
///   written here.
///
/// # Returns
///
/// * An exit code integer. If this is non-zero then an error occurred; the
///   details can be obtained by (1) getting the length of the error string by
///   calling `hb_last_error_length` and (2) calling `hb_last_error_message`
///   with a string buffer with a length at least equal to the error length.
///
#[no_mangle]
pub unsafe extern "C" fn analytic_calc_jones_array(
    analytic_beam: *mut AnalyticBeam,
    num_azza: u32,
    az_rad: *const f64,
    za_rad: *const f64,
    freq_hz: u32,
    delays: *const u32,
    amps: *const f64,
    num_amps: u32,
    latitude_rad: f64,
    norm_to_zenith: u8,
    jones: *mut f64,
) -> i32 {
    match num_amps {
        16 | 32 => (),
        _ => {
            update_last_error("A value other than 16 or 32 was used for num_amps".to_string());
            return 1;
        }
    };
    let norm_bool = match norm_to_zenith {
        0 => false,
        1 => true,
        _ => {
            update_last_error("A value other than 0 or 1 was used for norm_to_zenith".to_string());
            return 1;
        }
    };

    let beam = &*analytic_beam;
    let az = slice::from_raw_parts(az_rad, num_azza as usize);
    let za = slice::from_raw_parts(za_rad, num_azza as usize);
    let delays_s = slice::from_raw_parts(delays, 16);
    let amps_s = slice::from_raw_parts(amps, num_amps as usize);
    let results_s = slice::from_raw_parts_mut(jones.cast(), num_azza as usize);

    ffi_error!(beam.calc_jones_array_pair_inner(
        az,
        za,
        freq_hz,
        delays_s,
        amps_s,
        latitude_rad,
        norm_bool,
        results_s
    ));
    0
}

/// Free the memory associated with an `AnalyticBeam`.
///
/// # Arguments
///
/// * `analytic_beam` - the pointer to the `AnalyticBeam` struct.
///
#[no_mangle]
pub unsafe extern "C" fn free_analytic_beam(analytic_beam: *mut AnalyticBeam) {
    drop(Box::from_raw(analytic_beam));
}

/// Get a `AnalyticBeamGpu` struct, which is used to calculate beam responses on
/// a GPU (CUDA- or HIP-capable device).
///
/// # Arguments
///
/// * `analytic_beam` - a pointer to a previously set `AnalyticBeam` struct.
/// * `delays` - a pointer to two-dimensional array of dipole delays. There must
///   be 16 delays per row; each row corresponds to a tile.
/// * `amps` - a pointer to two-dimensional array of dipole amplitudes. There
///   must be 16 or 32 amps per row; each row corresponds to a tile. The number
///   of amps per row is specified by `num_amps`.
/// * `num_freqs` - the number of frequencies in the array pointed to by
///   `freqs_hz`.
/// * `num_tiles` - the number of tiles in both `delays` and `amps`.
/// * `num_amps` - either 16 or 32. See the documentation for `calc_jones` for
///   more explanation.
/// * `norm_to_zenith` - A boolean indicating whether the beam responses should
///   be normalised with respect to zenith.
/// * `gpu_analytic_beam` - a double pointer to the `AnalyticBeamGpu` struct
///   which is set by this function. This struct must be freed by calling
///   `free_gpu_analytic_beam`.
///
/// # Returns
///
/// * An exit code integer. If this is non-zero then an error occurred; the
///   details can be obtained by (1) getting the length of the error string by
///   calling `hb_last_error_length` and (2) calling `hb_last_error_message`
///   with a string buffer with a length at least equal to the error length.
///
#[cfg(any(feature = "cuda", feature = "hip"))]
#[no_mangle]
pub unsafe extern "C" fn new_gpu_analytic_beam(
    analytic_beam: *mut AnalyticBeam,
    delays: *const u32,
    amps: *const f64,
    num_tiles: i32,
    num_amps: i32,
    gpu_analytic_beam: *mut *mut AnalyticBeamGpu,
) -> i32 {
    match num_amps {
        16 | 32 => (),
        _ => {
            update_last_error("A value other than 16 or 32 was used for num_amps".to_string());
            return 1;
        }
    };

    // Turn the pointers into slices.
    let amps = ArrayView2::from_shape_ptr((num_tiles as usize, num_amps as usize), amps);
    let delays = ArrayView2::from_shape_ptr((num_tiles as usize, 16), delays);

    let beam = &mut *analytic_beam;
    let gpu_beam = ffi_error!(beam.gpu_prepare(delays, amps));
    *gpu_analytic_beam = Box::into_raw(Box::new(gpu_beam));
    0
}

/// Get beam response Jones matrices for the given directions, using a GPU. The
/// Jones matrix elements for each direction are put into a host-memory buffer
/// `jones`.
///
/// # Arguments
///
/// * `gpu_beam` - A pointer to a `AnalyticBeamGpu` struct created with the
///   `new_gpu_analytic_beam` function
/// * `az_rad` - The azimuth directions to get the beam response (units of
///   radians)
/// * `za_rad` - The zenith angle directions to get the beam response (units of
///   radians)
/// * `jones` - A pointer to a buffer with at least `num_unique_tiles *
///   num_freqs * num_azza * 8 * sizeof(FLOAT)` bytes allocated.
///   `FLOAT` is either `float` or `double`, depending on how `hyperbeam` was
///   compiled. The Jones matrix beam responses are written here. This should be
///   set up with the `get_num_unique_tiles` function; see the examples for
///   more help.
///
/// # Returns
///
/// * An exit code integer. If this is non-zero then an error occurred; the
///   details can be obtained by (1) getting the length of the error string by
///   calling `hb_last_error_length` and (2) calling `hb_last_error_message`
///   with a string buffer with a length at least equal to the error length.
///
#[cfg(any(feature = "cuda", feature = "hip"))]
#[no_mangle]
pub unsafe extern "C" fn analytic_calc_jones_gpu(
    gpu_analytic_beam: *mut AnalyticBeamGpu,
    num_azza: u32,
    az_rad: *const GpuFloat,
    za_rad: *const GpuFloat,
    num_freqs: u32,
    freqs_hz: *const u32,
    latitude_rad: GpuFloat,
    norm_to_zenith: u8,
    jones: *mut GpuFloat,
) -> i32 {
    let num_azza_usize = match num_azza.try_into() {
        Ok(n) => n,
        Err(_) => {
            update_last_error("num_azza couldn't be converted to a usize".to_string());
            return 1;
        }
    };
    let num_freqs_usize = match num_freqs.try_into() {
        Ok(n) => n,
        Err(_) => {
            update_last_error("num_freqs couldn't be converted to a usize".to_string());
            return 1;
        }
    };
    let norm_to_zenith = match norm_to_zenith {
        0 => false,
        1 => true,
        _ => {
            update_last_error("A value other than 0 or 1 was used for norm_to_zenith".to_string());
            return 1;
        }
    };

    // Turn the pointers into slices and/or arrays.
    let beam = &mut *gpu_analytic_beam;
    let az = slice::from_raw_parts(az_rad, num_azza_usize);
    let za = slice::from_raw_parts(za_rad, num_azza_usize);
    let freqs = slice::from_raw_parts(freqs_hz, num_freqs_usize);
    let results = ArrayViewMut3::from_shape_ptr(
        (
            beam.num_unique_tiles as usize,
            num_freqs_usize,
            num_azza_usize,
        ),
        jones.cast(),
    );
    ffi_error!(beam.calc_jones_pair_inner(az, za, freqs, latitude_rad, norm_to_zenith, results));
    0
}

/// Get beam response Jones matrices for the given directions, using a GPU. The
/// Jones matrix elements for each direction are left on the device (the device
/// pointer is communicated via `d_jones`).
///
/// # Arguments
///
/// * `gpu_analytic_beam` - A pointer to a `AnalyticBeamGpu` struct created with
///   the `new_gpu_analytic_beam` function
/// * `az_rad` - The azimuth directions to get the beam response (units of
///   radians)
/// * `za_rad` - The zenith angle directions to get the beam response (units of
///   radians)
/// * `d_jones` - A pointer to a device buffer with at least `8 *
///   num_unique_tiles * num_freqs * num_azza * sizeof(FLOAT)` bytes
///   allocated. `FLOAT` is either `float` or `double`, depending on how
///   `hyperbeam` was compiled. The Jones matrix beam responses are written
///   here. This should be set up with the `get_num_unique_tiles` function; see
///   the examples for more help.
///
/// # Returns
///
/// * An exit code integer. If this is non-zero then an error occurred; the
///   details can be obtained by (1) getting the length of the error string by
///   calling `hb_last_error_length` and (2) calling `hb_last_error_message`
///   with a string buffer with a length at least equal to the error length.
///
#[cfg(any(feature = "cuda", feature = "hip"))]
#[no_mangle]
pub unsafe extern "C" fn analytic_calc_jones_gpu_device(
    gpu_analytic_beam: *mut AnalyticBeamGpu,
    num_azza: i32,
    az_rad: *const GpuFloat,
    za_rad: *const GpuFloat,
    num_freqs: i32,
    freqs_hz: *const u32,
    latitude_rad: GpuFloat,
    norm_to_zenith: u8,
    d_jones: *mut GpuFloat,
) -> i32 {
    let num_azza_usize = if num_azza < 0 {
        update_last_error("num_azza was less than 0; it must be positive".to_string());
        return 1;
    } else {
        match num_azza.try_into() {
            Ok(n) => n,
            Err(_) => {
                update_last_error("num_azza couldn't be converted to a usize".to_string());
                return 1;
            }
        }
    };
    let num_freqs_usize = if num_freqs < 0 {
        update_last_error("num_freqs was less than 0; it must be positive".to_string());
        return 1;
    } else {
        match num_freqs.try_into() {
            Ok(n) => n,
            Err(_) => {
                update_last_error("num_freqs couldn't be converted to a usize".to_string());
                return 1;
            }
        }
    };
    let norm_to_zenith = match norm_to_zenith {
        0 => false,
        1 => true,
        _ => {
            update_last_error("A value other than 0 or 1 was used for norm_to_zenith".to_string());
            return 1;
        }
    };

    let beam = &*gpu_analytic_beam;
    let az = slice::from_raw_parts(az_rad, num_azza_usize);
    let za = slice::from_raw_parts(za_rad, num_azza_usize);
    let freqs = slice::from_raw_parts(freqs_hz, num_freqs_usize);
    let d_az = ffi_error!(DevicePointer::copy_to_device(az));
    let d_za = ffi_error!(DevicePointer::copy_to_device(za));
    let d_freqs = ffi_error!(DevicePointer::copy_to_device(freqs));
    ffi_error!(beam.calc_jones_device_pair_inner(
        d_az.get(),
        d_za.get(),
        num_azza,
        d_freqs.get(),
        num_freqs,
        latitude_rad,
        norm_to_zenith,
        d_jones.cast()
    ));
    0
}

/// The same as `calc_jones_gpu_device`, but with the directions already
/// allocated on the device. As with `d_jones`, the precision of the floats
/// depends on how `hyperbeam` was compiled.
///
/// # Arguments
///
/// * `gpu_analytic_beam` - A pointer to a `AnalyticBeamGpu` struct created with
///   the `new_gpu_analytic_beam` function
/// * `d_az_rad` - The azimuth directions to get the beam response (units of
///   radians)
/// * `d_za_rad` - The zenith angle directions to get the beam response (units
///   of radians)
/// * `d_jones` - A pointer to a device buffer with at least `8 *
///   num_unique_tiles * num_freqs * num_azza * sizeof(FLOAT)` bytes
///   allocated. `FLOAT` is either `float` or `double`, depending on how
///   `hyperbeam` was compiled. The Jones matrix beam responses are written
///   here. This should be set up with the `get_num_unique_tiles` function; see
///   the examples for more help.
///
/// # Returns
///
/// * An exit code integer. If this is non-zero then an error occurred; the
///   details can be obtained by (1) getting the length of the error string by
///   calling `hb_last_error_length` and (2) calling `hb_last_error_message`
///   with a string buffer with a length at least equal to the error length.
///
#[cfg(any(feature = "cuda", feature = "hip"))]
#[no_mangle]
pub unsafe extern "C" fn analytic_calc_jones_gpu_device_inner(
    gpu_analytic_beam: *mut AnalyticBeamGpu,
    num_azza: i32,
    d_az_rad: *const GpuFloat,
    d_za_rad: *const GpuFloat,
    num_freqs: i32,
    d_freqs_hz: *const u32,
    latitude_rad: GpuFloat,
    norm_to_zenith: u8,
    d_jones: *mut GpuFloat,
) -> i32 {
    if num_azza < 0 {
        update_last_error("num_azza was less than 0; it must be positive".to_string());
        return 1;
    };
    if num_freqs < 0 {
        update_last_error("num_freqs was less than 0; it must be positive".to_string());
        return 1;
    };
    let norm_to_zenith = match norm_to_zenith {
        0 => false,
        1 => true,
        _ => {
            update_last_error("A value other than 0 or 1 was used for norm_to_zenith".to_string());
            return 1;
        }
    };

    let beam = &*gpu_analytic_beam;
    ffi_error!(beam.calc_jones_device_pair_inner(
        d_az_rad,
        d_za_rad,
        num_azza,
        d_freqs_hz,
        num_freqs,
        latitude_rad,
        norm_to_zenith,
        d_jones.cast()
    ));
    0
}

/// Get a pointer to the tile map. This is necessary to access de-duplicated
/// beam Jones matrices.
///
/// # Arguments
///
/// * `gpu_analytic_beam` - the pointer to the `AnalyticBeamGpu` struct.
///
/// # Returns
///
/// * A pointer to the tile map. The const annotation is deliberate; the caller
///   does not own the map.
///
#[cfg(any(feature = "cuda", feature = "hip"))]
#[no_mangle]
pub unsafe extern "C" fn get_analytic_tile_map(
    gpu_analytic_beam: *mut AnalyticBeamGpu,
) -> *const i32 {
    let beam = &*gpu_analytic_beam;
    beam.get_tile_map()
}

/// Get a pointer to the device tile map. This is necessary to access
/// de-duplicated beam Jones matrices on the device.
///
/// # Arguments
///
/// * `gpu_analytic_beam` - the pointer to the `AnalyticBeamGpu` struct.
///
/// # Returns
///
/// * A pointer to the device tile map. The const annotation is deliberate; the
///   caller does not own the map.
///
#[cfg(any(feature = "cuda", feature = "hip"))]
#[no_mangle]
pub unsafe extern "C" fn get_analytic_device_tile_map(
    gpu_analytic_beam: *mut AnalyticBeamGpu,
) -> *const i32 {
    let beam = &*gpu_analytic_beam;
    beam.get_device_tile_map()
}

/// Get the number of de-duplicated tiles associated with this `AnalyticBeamGpu`.
///
/// # Arguments
///
/// * `gpu_analytic_beam` - the pointer to the `AnalyticBeamGpu` struct.
///
/// # Returns
///
/// * The number of de-duplicated tiles associated with this `AnalyticBeamGpu`.
///
#[cfg(any(feature = "cuda", feature = "hip"))]
#[no_mangle]
pub unsafe extern "C" fn get_num_unique_analytic_tiles(
    gpu_analytic_beam: *mut AnalyticBeamGpu,
) -> i32 {
    let beam = &*gpu_analytic_beam;
    beam.num_unique_tiles
}

/// Free the memory associated with an `AnalyticBeamGpu` beam.
///
/// # Arguments
///
/// * `gpu_analytic_beam` - the pointer to the `AnalyticBeamGpu` struct.
///
#[cfg(any(feature = "cuda", feature = "hip"))]
#[no_mangle]
pub unsafe extern "C" fn free_gpu_analytic_beam(analytic_beam: *mut AnalyticBeamGpu) {
    drop(Box::from_raw(analytic_beam));
}
