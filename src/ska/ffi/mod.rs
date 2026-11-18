// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code for allowing other languages to talk to this Rust library's SKA-Low
//! beam code. See the examples directory for usage.
//!
//! All SKA-Low models are used through a single opaque `SkaLowBeam` handle,
//! created with one of `new_ska_low_analytic_beam`, `new_ska_low_eep_grid_beam`
//! or `new_ska_low_eep_sph_beam`, and freed with `free_ska_low_beam`.

#[cfg(test)]
mod tests;

use std::{
    ffi::{c_char, CStr},
    path::PathBuf,
    slice,
};

use marlu::Jones;

use super::{SkaElementResponse, SkaLowBeam, SkaStation};
use crate::ffi::{ffi_error, update_last_error};

fn bool_arg(value: u8, name: &str) -> Result<bool, String> {
    match value {
        0 => Ok(false),
        1 => Ok(true),
        _ => Err(format!("A value other than 0 or 1 was used for {name}")),
    }
}

/// Convert a nullable C string to an optional path.
unsafe fn path_arg(s: *const c_char) -> Result<Option<PathBuf>, String> {
    if s.is_null() {
        return Ok(None);
    }
    CStr::from_ptr(s)
        .to_str()
        .map(|s| Some(PathBuf::from(s)))
        .map_err(|e| format!("Path is not valid UTF-8: {e}"))
}

fn write_jones(j: &Jones<f64>, out: &mut [f64]) {
    for (k, c) in j.iter().enumerate() {
        out[2 * k] = c.re;
        out[2 * k + 1] = c.im;
    }
}

fn write_jones_array(js: &[Jones<f64>], out: &mut [f64]) {
    for (j, o) in js.iter().zip(out.as_chunks_mut::<8>().0) {
        write_jones(j, o);
    }
}

/// Create a new `ska-low-analytic` beam (EveryBeam's `oskar_dipole_cos`
/// element model).
///
/// # Arguments
///
/// * `cos_taper` - 1 to apply the `cos(theta)` taper (EveryBeam's
///   `oskar_dipole_cos`; this is what you want), 0 for no taper (EveryBeam's
///   `oskar_dipole`).
/// * `ska_low_beam` - a double pointer to the `SkaLowBeam` struct which is set
///   by this function. This struct must be freed by calling
///   `free_ska_low_beam`.
///
/// # Returns
///
/// * An exit code integer. If this is non-zero then an error occurred; the
///   details can be obtained by (1) getting the length of the error string by
///   calling `hb_last_error_length` and (2) calling `hb_last_error_message`
///   with a string buffer with a length at least equal to the error length.
#[no_mangle]
pub unsafe extern "C" fn new_ska_low_analytic_beam(
    cos_taper: u8,
    ska_low_beam: *mut *mut SkaLowBeam,
) -> i32 {
    let cos_taper = ffi_error!(bool_arg(cos_taper, "cos_taper"));
    *ska_low_beam = Box::into_raw(Box::new(SkaLowBeam::new_analytic(cos_taper)));
    0
}

/// Create a new `ska-low-eep-grid` beam (EveryBeam's `gridded_eep` element
/// model).
///
/// # Arguments
///
/// * `hdf5_file` - a null-terminated path to the EEP grid HDF5 file. If this is
///   null, the `SKA_LOW_EEP_GRID_FILE` environment variable is used, or failing
///   that `$EVERYBEAM_DATADIR/oskar_eep_grid.h5`.
/// * `ska_low_convention` - 1 to apply EveryBeam's SKA-Low (OSKAR) sign
///   convention (conjugation and negation of the Y dipole; this is what you
///   want), 0 to use the grid values as they are. Must be 1 if `hdf5_file` is
///   null.
/// * `ska_low_beam` - a double pointer to the `SkaLowBeam` struct which is set
///   by this function. This struct must be freed by calling
///   `free_ska_low_beam`.
///
/// # Returns
///
/// * An exit code integer. If this is non-zero then an error occurred; the
///   details can be obtained by (1) getting the length of the error string by
///   calling `hb_last_error_length` and (2) calling `hb_last_error_message`
///   with a string buffer with a length at least equal to the error length.
#[no_mangle]
pub unsafe extern "C" fn new_ska_low_eep_grid_beam(
    hdf5_file: *const c_char,
    ska_low_convention: u8,
    ska_low_beam: *mut *mut SkaLowBeam,
) -> i32 {
    let convention = ffi_error!(bool_arg(ska_low_convention, "ska_low_convention"));
    let file = ffi_error!(path_arg(hdf5_file));
    if file.is_none() && !convention {
        update_last_error("ska_low_convention = 0 requires an explicit hdf5_file".to_string());
        return 1;
    }
    let beam = ffi_error!(SkaLowBeam::new_eep_grid(file, convention));
    *ska_low_beam = Box::into_raw(Box::new(beam));
    0
}

/// Create a new `ska-low-eep-sph` beam (EveryBeam's `skalow_feko` element
/// model).
///
/// # Arguments
///
/// * `coeff_dir` - a null-terminated path to a directory of FEKO coefficient
///   HDF5 files (one per frequency). If this is null, the
///   `SKA_LOW_EEP_SPH_PATH` environment variable is used, or failing that
///   `$EVERYBEAM_DATADIR/skalow`.
/// * `l_max` - the maximum spherical-wave degree to use. 0 uses all available
///   degrees.
/// * `ska_low_beam` - a double pointer to the `SkaLowBeam` struct which is set
///   by this function. This struct must be freed by calling
///   `free_ska_low_beam`.
///
/// # Returns
///
/// * An exit code integer. If this is non-zero then an error occurred; the
///   details can be obtained by (1) getting the length of the error string by
///   calling `hb_last_error_length` and (2) calling `hb_last_error_message`
///   with a string buffer with a length at least equal to the error length.
#[no_mangle]
pub unsafe extern "C" fn new_ska_low_eep_sph_beam(
    coeff_dir: *const c_char,
    l_max: u32,
    ska_low_beam: *mut *mut SkaLowBeam,
) -> i32 {
    let dir = ffi_error!(path_arg(coeff_dir));
    let l_max = if l_max == 0 {
        None
    } else {
        Some(l_max as usize)
    };
    let beam = ffi_error!(SkaLowBeam::new_eep_sph(dir, l_max));
    *ska_low_beam = Box::into_raw(Box::new(beam));
    0
}

/// Get the number of elements with distinct responses in an SKA-Low beam. 0
/// means all elements share the same response (`element_id` is ignored).
///
/// # Arguments
///
/// * `ska_low_beam` - A pointer to a `SkaLowBeam` struct.
#[no_mangle]
pub unsafe extern "C" fn ska_low_get_num_elements(ska_low_beam: *const SkaLowBeam) -> u32 {
    (*ska_low_beam).num_elements().unwrap_or(0) as u32
}

/// Get the SKA-Low element-response Jones matrix for a direction, in
/// hyperbeam's convention (the same as the FEE beam).
///
/// Note the type of `jones` (`*double`); we can't pass complex numbers across
/// the FFI boundary, so the real and imaginary components are unpacked into
/// doubles. The output contains 8 doubles, where the j00 is the first pair, j01
/// is the second pair, etc. Without a latitude, the Jones matrix is [X_theta,
/// X_az, Y_theta, Y_az].
///
/// # Arguments
///
/// * `ska_low_beam` - A pointer to a `SkaLowBeam` struct.
/// * `az_rad` - The azimuth (North through East) \[radians\].
/// * `za_rad` - The zenith angle \[radians\].
/// * `freq_hz` - The frequency \[Hz\].
/// * `element_id` - The element (only used by `ska-low-eep-sph`).
/// * `latitude_rad` - A pointer to the telescope latitude \[radians\]. If not
///   null, the parallactic-angle correction is applied.
/// * `iau_order` - 1 for IAU polarisation order (only used with a latitude).
/// * `jones` - A pointer to a buffer with at least `8 * sizeof(double)`
///   allocated. The Jones matrix is written here.
///
/// # Returns
///
/// * An exit code integer. If this is non-zero then an error occurred; the
///   details can be obtained by (1) getting the length of the error string by
///   calling `hb_last_error_length` and (2) calling `hb_last_error_message`
///   with a string buffer with a length at least equal to the error length.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ska_low_calc_jones(
    ska_low_beam: *const SkaLowBeam,
    az_rad: f64,
    za_rad: f64,
    freq_hz: f64,
    element_id: u32,
    latitude_rad: *const f64,
    iau_order: u8,
    jones: *mut f64,
) -> i32 {
    let iau_order = ffi_error!(bool_arg(iau_order, "iau_order"));
    let beam = &*ska_low_beam;
    let j = ffi_error!(beam.calc_jones_pair(
        az_rad,
        za_rad,
        freq_hz,
        element_id as usize,
        latitude_rad.as_ref().copied(),
        iau_order,
    ));
    write_jones(&j, slice::from_raw_parts_mut(jones, 8));
    0
}

/// Get SKA-Low element-response Jones matrices for many directions (computed
/// in parallel). See `ska_low_calc_jones`.
///
/// # Arguments
///
/// * `ska_low_beam` - A pointer to a `SkaLowBeam` struct.
/// * `num_azza` - The number of directions in `az_rad` and `za_rad`.
/// * `az_rad` - The azimuths (North through East) \[radians\].
/// * `za_rad` - The zenith angles \[radians\].
/// * `freq_hz` - The frequency \[Hz\].
/// * `element_id` - The element (only used by `ska-low-eep-sph`).
/// * `latitude_rad` - A pointer to the telescope latitude \[radians\]. If not
///   null, the parallactic-angle correction is applied.
/// * `iau_order` - 1 for IAU polarisation order (only used with a latitude).
/// * `jones` - A pointer to a buffer with at least `8 * num_azza *
///   sizeof(double)` allocated.
///
/// # Returns
///
/// * An exit code integer. If this is non-zero then an error occurred; the
///   details can be obtained by (1) getting the length of the error string by
///   calling `hb_last_error_length` and (2) calling `hb_last_error_message`
///   with a string buffer with a length at least equal to the error length.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ska_low_calc_jones_array(
    ska_low_beam: *const SkaLowBeam,
    num_azza: u32,
    az_rad: *const f64,
    za_rad: *const f64,
    freq_hz: f64,
    element_id: u32,
    latitude_rad: *const f64,
    iau_order: u8,
    jones: *mut f64,
) -> i32 {
    let iau_order = ffi_error!(bool_arg(iau_order, "iau_order"));
    let beam = &*ska_low_beam;
    let n = num_azza as usize;
    let js = ffi_error!(beam.calc_jones_array_pair(
        slice::from_raw_parts(az_rad, n),
        slice::from_raw_parts(za_rad, n),
        freq_hz,
        element_id as usize,
        latitude_rad.as_ref().copied(),
        iau_order,
    ));
    write_jones_array(&js, slice::from_raw_parts_mut(jones, 8 * n));
    0
}

/// Get the raw SKA-Low element response in EveryBeam's convention: exactly
/// what EveryBeam's `ElementResponse::Response(element_id, freq, theta, phi)`
/// returns ([X_theta, X_phi, Y_theta, Y_phi], 8 doubles).
///
/// # Arguments
///
/// * `ska_low_beam` - A pointer to a `SkaLowBeam` struct.
/// * `theta_rad` - The zenith angle \[radians\].
/// * `phi_rad` - The angle from East through North \[radians\].
/// * `freq_hz` - The frequency \[Hz\].
/// * `element_id` - The element (only used by `ska-low-eep-sph`).
/// * `jones` - A pointer to a buffer with at least `8 * sizeof(double)`
///   allocated.
///
/// # Returns
///
/// * An exit code integer. If this is non-zero then an error occurred; the
///   details can be obtained by (1) getting the length of the error string by
///   calling `hb_last_error_length` and (2) calling `hb_last_error_message`
///   with a string buffer with a length at least equal to the error length.
#[no_mangle]
pub unsafe extern "C" fn ska_low_element_response(
    ska_low_beam: *const SkaLowBeam,
    theta_rad: f64,
    phi_rad: f64,
    freq_hz: f64,
    element_id: u32,
    jones: *mut f64,
) -> i32 {
    let beam = &*ska_low_beam;
    let j = ffi_error!(beam.element_response(element_id as usize, freq_hz, theta_rad, phi_rad));
    write_jones(&j, slice::from_raw_parts_mut(jones, 8));
    0
}

/// Get SKA-Low station-beam Jones matrices for many directions, given a
/// pointing and the station's element positions. See `ska_low_calc_jones` for
/// the output convention.
///
/// # Arguments
///
/// * `ska_low_beam` - A pointer to a `SkaLowBeam` struct.
/// * `num_elements` - The number of station elements. For `ska-low-eep-sph`,
///   this must equal `ska_low_get_num_elements`, and element `i` uses
///   `element_id = i`.
/// * `positions_enu` - `3 * num_elements` doubles: East, North, Up \[metres\]
///   for each element.
/// * `weights` - `num_elements` doubles (0 flags an element), or null for all
///   ones.
/// * `rotation_rad` - The station's rotation about the local vertical: the
///   angle of its X dipoles from East towards North \[radians\]. SKA-Low
///   stations are rigidly rotated; use 0 for an unrotated station.
/// * `num_azza` - The number of directions in `az_rad` and `za_rad`.
/// * `az_rad` - The azimuths (North through East) \[radians\].
/// * `za_rad` - The zenith angles \[radians\].
/// * `freq_hz` - The frequency \[Hz\].
/// * `pointing_az_rad` - The station pointing azimuth \[radians\].
/// * `pointing_za_rad` - The station pointing zenith angle \[radians\].
/// * `latitude_rad` - A pointer to the telescope latitude \[radians\]. If not
///   null, the parallactic-angle correction is applied.
/// * `iau_order` - 1 for IAU polarisation order (only used with a latitude).
/// * `jones` - A pointer to a buffer with at least `8 * num_azza *
///   sizeof(double)` allocated.
///
/// # Returns
///
/// * An exit code integer. If this is non-zero then an error occurred; the
///   details can be obtained by (1) getting the length of the error string by
///   calling `hb_last_error_length` and (2) calling `hb_last_error_message`
///   with a string buffer with a length at least equal to the error length.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ska_low_calc_station_jones_array(
    ska_low_beam: *const SkaLowBeam,
    num_elements: u32,
    positions_enu: *const f64,
    weights: *const f64,
    rotation_rad: f64,
    num_azza: u32,
    az_rad: *const f64,
    za_rad: *const f64,
    freq_hz: f64,
    pointing_az_rad: f64,
    pointing_za_rad: f64,
    latitude_rad: *const f64,
    iau_order: u8,
    jones: *mut f64,
) -> i32 {
    let iau_order = ffi_error!(bool_arg(iau_order, "iau_order"));
    let beam = &*ska_low_beam;
    let ne = num_elements as usize;
    let positions = slice::from_raw_parts(positions_enu, 3 * ne)
        .as_chunks::<3>()
        .0
        .to_vec();
    let weights = if weights.is_null() {
        None
    } else {
        Some(slice::from_raw_parts(weights, ne).to_vec())
    };
    let station = ffi_error!(SkaStation::new(beam, positions, weights)).with_rotation(rotation_rad);
    let n = num_azza as usize;
    let js = ffi_error!(station.calc_jones_array_pair(
        slice::from_raw_parts(az_rad, n),
        slice::from_raw_parts(za_rad, n),
        freq_hz,
        pointing_az_rad,
        pointing_za_rad,
        latitude_rad.as_ref().copied(),
        iau_order,
    ));
    write_jones_array(&js, slice::from_raw_parts_mut(jones, 8 * n));
    0
}

/// Free the memory associated with an SKA-Low beam.
///
/// # Arguments
///
/// * `ska_low_beam` - the pointer to the `SkaLowBeam` struct.
#[no_mangle]
pub unsafe extern "C" fn free_ska_low_beam(ska_low_beam: *mut SkaLowBeam) {
    if !ska_low_beam.is_null() {
        drop(Box::from_raw(ska_low_beam));
    }
}

cfg_if::cfg_if! {
    if #[cfg(any(feature = "cuda", feature = "hip"))] {
        use marlu::AzEl;
        use ndarray::prelude::*;

        use super::{SkaGpuTiles, SkaLowBeamGpu};
        use crate::gpu::{DevicePointer, GpuFloat};
    }
}

/// Prepare a GPU SKA-Low beam for the given frequencies and tiles. A "tile" is
/// either a single element or a station.
///
/// # Arguments
///
/// * `ska_low_beam` - A pointer to a `SkaLowBeam` struct.
/// * `num_freqs` - The number of frequencies in `freqs_hz`.
/// * `freqs_hz` - The frequencies \[Hz\].
/// * `num_tiles` - The number of tiles.
/// * `element_ids` - If `positions_enu` is null, each tile is a single element
///   and this gives each tile's element ID (`num_tiles` values; only used by
///   `ska-low-eep-sph`). May be null, in which case element 0 is used for all
///   tiles.
/// * `num_station_elements` - The number of elements per station (only used
///   with `positions_enu`).
/// * `positions_enu` - If not null, each tile is a station, and this holds `3 *
///   num_station_elements * num_tiles` doubles (East, North, Up \[metres\],
///   station-major).
/// * `weights` - If not null (and `positions_enu` is not null),
///   `num_station_elements * num_tiles` element weights (0 flags an element).
/// * `rotations_rad` - If not null (and `positions_enu` is not null),
///   `num_tiles` station rotations about the local vertical (the angle of each
///   station's X dipoles from East towards North \[radians\]).
/// * `gpu_ska_low_beam` - a double pointer to the `SkaLowBeamGpu` struct which
///   is set by this function. This struct must be freed by calling
///   `free_gpu_ska_low_beam`.
///
/// # Returns
///
/// * An exit code integer. If this is non-zero then an error occurred; the
///   details can be obtained by (1) getting the length of the error string by
///   calling `hb_last_error_length` and (2) calling `hb_last_error_message`
///   with a string buffer with a length at least equal to the error length.
#[cfg(any(feature = "cuda", feature = "hip"))]
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn new_gpu_ska_low_beam(
    ska_low_beam: *const SkaLowBeam,
    num_freqs: u32,
    freqs_hz: *const f64,
    num_tiles: u32,
    element_ids: *const u32,
    num_station_elements: u32,
    positions_enu: *const f64,
    weights: *const f64,
    rotations_rad: *const f64,
    gpu_ska_low_beam: *mut *mut SkaLowBeamGpu,
) -> i32 {
    let beam = &*ska_low_beam;
    let freqs = slice::from_raw_parts(freqs_hz, num_freqs as usize);
    let (nt, ne) = (num_tiles as usize, num_station_elements as usize);
    let ids: Vec<usize>;
    let tiles = if positions_enu.is_null() {
        ids = if element_ids.is_null() {
            vec![0; nt]
        } else {
            slice::from_raw_parts(element_ids, nt)
                .iter()
                .map(|&i| i as usize)
                .collect()
        };
        SkaGpuTiles::Elements(&ids)
    } else {
        SkaGpuTiles::Stations {
            positions_enu: ArrayView3::from_shape_ptr((nt, ne, 3), positions_enu),
            weights: if weights.is_null() {
                None
            } else {
                Some(ArrayView2::from_shape_ptr((nt, ne), weights))
            },
            rotations_rad: if rotations_rad.is_null() {
                None
            } else {
                Some(slice::from_raw_parts(rotations_rad, nt))
            },
            local_frames: None,
        }
    };
    let gpu_beam = ffi_error!(beam.gpu_prepare(freqs, tiles));
    *gpu_ska_low_beam = Box::into_raw(Box::new(gpu_beam));
    0
}

/// Calculate GPU SKA-Low beam responses and copy them to the host. `jones`
/// must have room for `8 * num_tiles * num_freqs * num_azza` floats (`float`
/// or `double`, depending on how hyperbeam was compiled), ordered tile, then
/// frequency, then direction.
///
/// # Arguments
///
/// * `gpu_ska_low_beam` - A pointer to a `SkaLowBeamGpu` struct.
/// * `num_azza` - The number of directions.
/// * `az_rad` - Host pointer to azimuths \[radians\].
/// * `za_rad` - Host pointer to zenith angles \[radians\].
/// * `pointing_az_rad` - The station pointing azimuth (stations only).
/// * `pointing_za_rad` - The station pointing zenith angle (stations only).
/// * `latitude_rad` - A pointer to the telescope latitude. If not null, the
///   parallactic-angle correction is applied.
/// * `iau_order` - 1 for IAU order (only used with a latitude).
/// * `jones` - Host pointer to the output buffer.
///
/// # Returns
///
/// * An exit code integer. If this is non-zero then an error occurred; the
///   details can be obtained by (1) getting the length of the error string by
///   calling `hb_last_error_length` and (2) calling `hb_last_error_message`
///   with a string buffer with a length at least equal to the error length.
#[cfg(any(feature = "cuda", feature = "hip"))]
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ska_low_calc_jones_gpu(
    gpu_ska_low_beam: *const SkaLowBeamGpu,
    num_azza: u32,
    az_rad: *const GpuFloat,
    za_rad: *const GpuFloat,
    pointing_az_rad: f64,
    pointing_za_rad: f64,
    latitude_rad: *const f64,
    iau_order: u8,
    jones: *mut GpuFloat,
) -> i32 {
    let iau_order = ffi_error!(bool_arg(iau_order, "iau_order"));
    let beam = &*gpu_ska_low_beam;
    let n = num_azza as usize;
    let results = ffi_error!(beam.calc_jones_pair(
        slice::from_raw_parts(az_rad, n),
        slice::from_raw_parts(za_rad, n),
        AzEl::from_radians(
            pointing_az_rad,
            std::f64::consts::FRAC_PI_2 - pointing_za_rad
        ),
        latitude_rad.as_ref().copied(),
        iau_order,
    ));
    let out = slice::from_raw_parts_mut(jones, results.len() * 8);
    for (j, o) in results.iter().zip(out.as_chunks_mut::<8>().0) {
        for k in 0..4 {
            o[2 * k] = j[k].re;
            o[2 * k + 1] = j[k].im;
        }
    }
    0
}

/// Calculate GPU SKA-Low beam responses into a device buffer, given host
/// directions. `d_jones` must be a device buffer with room for `num_tiles *
/// num_freqs * num_azza` Jones matrices (8 floats each). See
/// `ska_low_calc_jones_gpu` for the other arguments.
///
/// # Returns
///
/// * An exit code integer. If this is non-zero then an error occurred; the
///   details can be obtained by (1) getting the length of the error string by
///   calling `hb_last_error_length` and (2) calling `hb_last_error_message`
///   with a string buffer with a length at least equal to the error length.
#[cfg(any(feature = "cuda", feature = "hip"))]
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ska_low_calc_jones_gpu_device(
    gpu_ska_low_beam: *const SkaLowBeamGpu,
    num_azza: u32,
    az_rad: *const GpuFloat,
    za_rad: *const GpuFloat,
    pointing_az_rad: f64,
    pointing_za_rad: f64,
    latitude_rad: *const f64,
    iau_order: u8,
    d_jones: *mut GpuFloat,
) -> i32 {
    let iau_order = ffi_error!(bool_arg(iau_order, "iau_order"));
    let beam = &*gpu_ska_low_beam;
    let n = num_azza as usize;
    let d_az = ffi_error!(DevicePointer::copy_to_device(slice::from_raw_parts(
        az_rad, n
    )));
    let d_za = ffi_error!(DevicePointer::copy_to_device(slice::from_raw_parts(
        za_rad, n
    )));
    let d_lat = ffi_error!(latitude_rad
        .as_ref()
        .map(|&l| DevicePointer::copy_to_device(&[l as GpuFloat]))
        .transpose());
    ffi_error!(beam.calc_jones_device_pair_inner(
        d_az.get(),
        d_za.get(),
        n as i32,
        pointing_az_rad as GpuFloat,
        pointing_za_rad as GpuFloat,
        d_lat.as_ref().map(|d| d.get()).unwrap_or(std::ptr::null()),
        iau_order,
        d_jones.cast(),
    ));
    0
}

/// Calculate GPU SKA-Low beam responses with all inputs and outputs already
/// on the device. `d_latitude_rad` is a device pointer (or null for no
/// parallactic-angle correction). See `ska_low_calc_jones_gpu_device`.
///
/// # Returns
///
/// * An exit code integer. If this is non-zero then an error occurred; the
///   details can be obtained by (1) getting the length of the error string by
///   calling `hb_last_error_length` and (2) calling `hb_last_error_message`
///   with a string buffer with a length at least equal to the error length.
#[cfg(any(feature = "cuda", feature = "hip"))]
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ska_low_calc_jones_gpu_device_inner(
    gpu_ska_low_beam: *const SkaLowBeamGpu,
    num_azza: i32,
    d_az_rad: *const GpuFloat,
    d_za_rad: *const GpuFloat,
    pointing_az_rad: GpuFloat,
    pointing_za_rad: GpuFloat,
    d_latitude_rad: *const GpuFloat,
    iau_order: u8,
    d_jones: *mut GpuFloat,
) -> i32 {
    let iau_order = ffi_error!(bool_arg(iau_order, "iau_order"));
    let beam = &*gpu_ska_low_beam;
    ffi_error!(beam.calc_jones_device_pair_inner(
        d_az_rad,
        d_za_rad,
        num_azza,
        pointing_az_rad,
        pointing_za_rad,
        d_latitude_rad,
        iau_order,
        d_jones.cast(),
    ));
    0
}

/// Free the memory associated with a GPU SKA-Low beam.
///
/// # Arguments
///
/// * `gpu_ska_low_beam` - the pointer to the `SkaLowBeamGpu` struct.
#[cfg(any(feature = "cuda", feature = "hip"))]
#[no_mangle]
pub unsafe extern "C" fn free_gpu_ska_low_beam(gpu_ska_low_beam: *mut SkaLowBeamGpu) {
    if !gpu_ska_low_beam.is_null() {
        drop(Box::from_raw(gpu_ska_low_beam));
    }
}
