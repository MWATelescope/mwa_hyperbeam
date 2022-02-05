// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! GPU code to implement the MWA analytic beam.

#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

// Include Rust bindings to the GPU code, depending on the precision used.
#[cfg(feature = "gpu-single")]
include!("single.rs");
#[cfg(not(feature = "gpu-single"))]
include!("double.rs");

#[cfg(test)]
mod tests;

use std::{
    collections::hash_map::DefaultHasher,
    convert::TryInto,
    ffi::CStr,
    hash::{Hash, Hasher},
};

use marlu::{AzEl, Jones};
use ndarray::prelude::*;

use super::{delay_ints_to_floats, reorder_to_rts, AnalyticBeam, AnalyticBeamError};
use crate::gpu::{DevicePointer, GpuError, GpuFloat};

/// A GPU beam object ready to calculate beam responses.
pub struct AnalyticBeamGpu {
    analytic_type: super::AnalyticType,
    dipole_height: GpuFloat,

    d_delays: DevicePointer<GpuFloat>,
    d_amps: DevicePointer<GpuFloat>,

    /// The number of unique tiles according to the delays and amps.
    pub(super) num_unique_tiles: i32,

    /// This is used to access de-duplicated Jones matrices.
    tile_map: Vec<i32>,

    /// The device pointer to the `tile_map` (same as the host's memory
    /// equivalent above).
    d_tile_map: DevicePointer<i32>,
}

impl AnalyticBeamGpu {
    /// Prepare a GPU-capable device for beam-response computations given the
    /// frequencies, delays and amps to be used. The resulting object takes
    /// directions and computes the beam responses on the device.
    ///
    /// This function is intentionally kept private. Use
    /// [`AnalyticBeam::gpu_prepare`] to create a `AnalyticBeamGpu`.
    pub(super) unsafe fn new(
        analytic_beam: &AnalyticBeam,
        delays_array: ArrayView2<u32>,
        amps_array: ArrayView2<f64>,
    ) -> Result<AnalyticBeamGpu, AnalyticBeamError> {
        if delays_array.len_of(Axis(1)) != 16 {
            return Err(AnalyticBeamError::IncorrectDelaysArrayColLength {
                rows: delays_array.len_of(Axis(0)),
                num_delays: delays_array.len_of(Axis(1)),
            });
        }
        if !(amps_array.len_of(Axis(1)) == 16 || amps_array.len_of(Axis(1)) == 32) {
            return Err(AnalyticBeamError::IncorrectAmpsLength(
                amps_array.len_of(Axis(1)),
            ));
        }

        // Determine the unique tiles according to the gains and delays. Unlike
        // FEE, all frequencies give different results, so there's no need to
        // consider them.
        let mut unique_tiles = vec![];
        let mut tile_map = vec![];
        let mut i_tile = 0;
        let mut unique_delays = vec![];
        let mut unique_amps = vec![];
        for (delays, amps) in delays_array.outer_iter().zip(amps_array.outer_iter()) {
            let mut unique_tile_hasher = DefaultHasher::new();
            delays.hash(&mut unique_tile_hasher);
            // We can't hash f64 values, but we can hash their bits.
            for amp in amps {
                amp.to_bits().hash(&mut unique_tile_hasher);
            }
            let unique_tile_hash = unique_tile_hasher.finish();

            let (amps, delays) = fix_amps_ndarray(amps, delays);
            let (amps, delays) = if matches!(analytic_beam.beam_type, super::AnalyticType::Rts) {
                reorder_to_rts(&amps, &delays)
            } else {
                (amps, delay_ints_to_floats(&delays))
            };

            let this_tile_index = if let Some((index, _)) = unique_tiles
                .iter()
                .enumerate()
                .find(|(_, t)| **t == unique_tile_hash)
            {
                index.try_into().expect("smaller than i32::MAX")
            } else {
                unique_tiles.push(unique_tile_hash);
                unique_delays.extend(delays.iter().copied().map(|d| d as GpuFloat));
                unique_amps.extend(amps.iter().map(|&f| f as GpuFloat));
                i_tile += 1;
                i_tile - 1
            };
            tile_map.push(this_tile_index);
        }

        let d_tile_map = DevicePointer::copy_to_device(&tile_map)?;
        Ok(AnalyticBeamGpu {
            analytic_type: analytic_beam.beam_type,
            dipole_height: analytic_beam.dipole_height as GpuFloat,
            d_delays: DevicePointer::copy_to_device(&unique_delays)?,
            d_amps: DevicePointer::copy_to_device(&unique_amps)?,
            num_unique_tiles: unique_tiles
                .len()
                .try_into()
                .expect("smaller than i32::MAX"),
            tile_map,
            d_tile_map,
        })
    }

    /// Given directions, calculate beam-response Jones matrices on the device
    /// and return a pointer to them.
    ///
    /// Note that this function needs to allocate two vectors for azimuths and
    /// zenith angles from the supplied `azels`.
    pub fn calc_jones_device(
        &self,
        azels: &[AzEl],
        freqs_hz: &[u32],
        latitude_rad: f64,
        norm_to_zenith: bool,
    ) -> Result<DevicePointer<Jones<GpuFloat>>, AnalyticBeamError> {
        unsafe {
            // Allocate a buffer on the device for results.
            let d_results = DevicePointer::malloc(
                self.num_unique_tiles as usize
                    * freqs_hz.len()
                    * azels.len()
                    * std::mem::size_of::<Jones<GpuFloat>>(),
            )?;

            // Also copy the directions to the device.
            let (azs, zas): (Vec<GpuFloat>, Vec<GpuFloat>) = azels
                .iter()
                .map(|&azel| (azel.az as GpuFloat, azel.za() as GpuFloat))
                .unzip();
            let d_azs = DevicePointer::copy_to_device(&azs)?;
            let d_zas = DevicePointer::copy_to_device(&zas)?;
            let d_freqs = DevicePointer::copy_to_device(freqs_hz)?;

            self.calc_jones_device_pair_inner(
                d_azs.get(),
                d_zas.get(),
                azels.len().try_into().expect("much fewer than i32::MAX"),
                d_freqs.get(),
                freqs_hz.len().try_into().expect("much fewer than i32::MAX"),
                latitude_rad as GpuFloat,
                norm_to_zenith,
                d_results.get_mut() as *mut std::ffi::c_void,
            )?;
            Ok(d_results)
        }
    }

    /// Given directions, calculate beam-response Jones matrices on the device
    /// and return a pointer to them.
    pub fn calc_jones_device_pair(
        &self,
        az_rad: &[GpuFloat],
        za_rad: &[GpuFloat],
        freqs_hz: &[u32],
        latitude_rad: GpuFloat,
        norm_to_zenith: bool,
    ) -> Result<DevicePointer<Jones<GpuFloat>>, AnalyticBeamError> {
        unsafe {
            // Allocate a buffer on the device for results.
            let d_results = DevicePointer::malloc(
                self.num_unique_tiles as usize
                    * freqs_hz.len()
                    * az_rad.len()
                    * std::mem::size_of::<Jones<GpuFloat>>(),
            )?;

            // Also copy the directions to the device.
            let d_azs = DevicePointer::copy_to_device(az_rad)?;
            let d_zas = DevicePointer::copy_to_device(za_rad)?;
            let d_freqs = DevicePointer::copy_to_device(freqs_hz)?;

            self.calc_jones_device_pair_inner(
                d_azs.get(),
                d_zas.get(),
                az_rad.len().try_into().expect("much fewer than i32::MAX"),
                d_freqs.get(),
                freqs_hz.len().try_into().expect("much fewer than i32::MAX"),
                latitude_rad,
                norm_to_zenith,
                d_results.get_mut() as *mut std::ffi::c_void,
            )?;
            Ok(d_results)
        }
    }

    /// Given directions, calculate beam-response Jones matrices
    /// into the supplied pre-allocated device pointer. This buffer
    /// should have a shape of (`num_unique_tiles`, `num_freqs`,
    /// `az_rad_length`). The number of unique tiles can be accessed with
    /// [`AnalyticBeamGpu::get_num_unique_tiles`]. `d_latitude_rad` is
    /// populated with the array latitude, if the caller wants the parallactic-
    /// angle correction to be applied. If the pointer is null, then no
    /// correction is applied.
    ///
    /// # Safety
    ///
    /// If `d_results` is too small (correct size described above), then
    /// undefined behaviour looms.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn calc_jones_device_pair_inner(
        &self,
        d_az_rad: *const GpuFloat,
        d_za_rad: *const GpuFloat,
        num_directions: i32,
        d_freqs_hz: *const u32,
        num_freqs: i32,
        latitude_rad: GpuFloat,
        norm_to_zenith: bool,
        d_results: *mut std::ffi::c_void,
    ) -> Result<(), AnalyticBeamError> {
        // Don't do anything if there aren't any directions.
        if num_directions == 0 {
            return Ok(());
        }

        // The return value is a pointer to a CUDA/HIP error string. If it's
        // null then everything is fine.
        let error_message_ptr = gpu_analytic_calc_jones(
            match self.analytic_type {
                super::AnalyticType::MwaPb => ANALYTIC_TYPE_MWA_PB,
                super::AnalyticType::Rts => ANALYTIC_TYPE_RTS,
            },
            self.dipole_height,
            d_az_rad,
            d_za_rad,
            num_directions,
            d_freqs_hz,
            num_freqs,
            self.d_delays.get(),
            self.d_amps.get(),
            self.num_unique_tiles,
            latitude_rad,
            norm_to_zenith as _,
            d_results,
        );
        if error_message_ptr.is_null() {
            Ok(())
        } else {
            let error_message = CStr::from_ptr(error_message_ptr)
                .to_str()
                .unwrap_or("<cannot read GPU error string>");
            let our_error_str =
                format!("analytic.h:analytic_calc_jones_gpu failed with: {error_message}");
            Err(AnalyticBeamError::Gpu(GpuError::Kernel {
                msg: our_error_str.into(),
                file: file!(),
                line: line!(),
            }))
        }
    }

    /// Given directions, calculate beam-response Jones matrices on the device,
    /// copy them to the host, and free the device memory. The returned array
    /// is "expanded"; tile and frequency de-duplication is undone to give
    /// an array with the same number of tiles as was specified when this
    /// [`AnalyticBeamGpu`] was created and frequencies specified to this
    /// function.
    ///
    /// Note that this function needs to allocate two vectors for azimuths and
    /// zenith angles from the supplied `azels`.
    pub fn calc_jones(
        &self,
        azels: &[AzEl],
        freqs_hz: &[u32],
        latitude_rad: f64,
        norm_to_zenith: bool,
    ) -> Result<Array3<Jones<GpuFloat>>, AnalyticBeamError> {
        let mut results = Array3::from_elem(
            (self.tile_map.len(), freqs_hz.len(), azels.len()),
            Jones::default(),
        );

        let (azs, zas): (Vec<GpuFloat>, Vec<GpuFloat>) = azels
            .iter()
            .map(|&azel| (azel.az as GpuFloat, azel.za() as GpuFloat))
            .unzip();
        self.calc_jones_pair_inner(
            &azs,
            &zas,
            freqs_hz,
            latitude_rad as GpuFloat,
            norm_to_zenith,
            results.view_mut(),
        )?;
        Ok(results)
    }

    /// Given directions, calculate beam-response Jones matrices on the device,
    /// copy them to the host, and free the device memory. The returned array
    /// is "expanded"; tile and frequency de-duplication is undone to give
    /// an array with the same number of tiles as was specified when this
    /// [`AnalyticBeamGpu`] was created and frequencies specified to this
    /// function.
    pub fn calc_jones_pair(
        &self,
        az_rad: &[GpuFloat],
        za_rad: &[GpuFloat],
        freqs_hz: &[u32],
        latitude_rad: GpuFloat,
        norm_to_zenith: bool,
    ) -> Result<Array3<Jones<GpuFloat>>, AnalyticBeamError> {
        let mut results = Array3::from_elem(
            (self.tile_map.len(), freqs_hz.len(), az_rad.len()),
            Jones::default(),
        );

        self.calc_jones_pair_inner(
            az_rad,
            za_rad,
            freqs_hz,
            latitude_rad,
            norm_to_zenith,
            results.view_mut(),
        )?;
        Ok(results)
    }

    /// Given directions, calculate beam-response Jones matrices on the device,
    /// copy them to the host, and free the device memory. This function is
    /// the same as [`AnalyticBeamGpu::calc_jones_pair`], but the results are
    /// stored in a pre-allocated array. This array should have a shape of
    /// (`total_num_tiles`, `total_num_freqs`, `az_rad_length`). The first
    /// dimension can be accessed with `AnalyticBeamGpu::get_total_num_tiles`.
    pub fn calc_jones_pair_inner(
        &self,
        az_rad: &[GpuFloat],
        za_rad: &[GpuFloat],
        freqs_hz: &[u32],
        latitude_rad: GpuFloat,
        norm_to_zenith: bool,
        mut results: ArrayViewMut3<Jones<GpuFloat>>,
    ) -> Result<(), AnalyticBeamError> {
        // Allocate an array matching the deduplicated device memory.
        let mut dedup_results: Array3<Jones<GpuFloat>> = Array3::from_elem(
            (self.num_unique_tiles as usize, freqs_hz.len(), az_rad.len()),
            Jones::default(),
        );
        // Calculate the beam responses. and copy them to the host.
        let device_ptr =
            self.calc_jones_device_pair(az_rad, za_rad, freqs_hz, latitude_rad, norm_to_zenith)?;
        unsafe {
            device_ptr.copy_from_device(dedup_results.as_slice_mut().expect("is contiguous"))?;
        }
        // Free the device memory.
        drop(device_ptr);

        // Expand the results according to the map.
        results
            .outer_iter_mut()
            .zip(self.tile_map.iter())
            .for_each(|(mut jones_row, &i_row)| {
                let i_row: usize = i_row.try_into().expect("is a positive int");
                jones_row.assign(&dedup_results.slice(s![i_row, .., ..]));
            });
        Ok(())
    }

    /// Get the number of tiles that this [`AnalyticBeamGpu`] applies to.
    pub fn get_total_num_tiles(&self) -> usize {
        self.tile_map.len()
    }

    /// Get a pointer to the tile map associated with this
    /// [`AnalyticBeamGpu`]. This is necessary to access de-duplicated beam
    /// Jones matrices.
    pub fn get_tile_map(&self) -> *const i32 {
        self.tile_map.as_ptr()
    }

    /// Get a pointer to the device tile map associated with this
    /// [`AnalyticBeamGpu`]. This is necessary to access de-duplicated beam
    /// Jones matrices on the device.
    pub fn get_device_tile_map(&self) -> *const i32 {
        self.d_tile_map.get()
    }

    /// Get the number of de-duplicated tiles associated with this
    /// [`AnalyticBeamGpu`].
    pub fn get_num_unique_tiles(&self) -> i32 {
        self.num_unique_tiles
    }
}

/// Ensure that any delays of 32 have an amplitude (dipole gain) of 0. The
/// results are bad otherwise! Also ensure that we have 32 dipole gains (amps)
/// here. Also return a Rust array of delays for convenience.
pub(super) fn fix_amps_ndarray(
    amps: ArrayView1<f64>,
    delays: ArrayView1<u32>,
) -> ([f64; 16], [u32; 16]) {
    // The lengths of `amps` and `delays` should be checked before calling this
    // functions; the asserts are a last resort guard.
    assert_eq!(delays.len(), 16);
    assert!(amps.len() == 16 || amps.len() == 32);

    let mut fixed_amps = [0.0; 16];
    fixed_amps
        .iter_mut()
        .zip(amps.iter())
        .zip(delays.iter().cycle())
        .for_each(|((out_amp, &in_amp), &delay)| {
            if delay == 32 {
                *out_amp = 0.0;
            } else {
                *out_amp = in_amp;
            }
        });
    // Handle 32 amps.
    fixed_amps
        .iter_mut()
        .zip(amps.iter().skip(16))
        .for_each(|(out_amp, &in_amp)| {
            *out_amp = out_amp.min(in_amp);
        });

    // So that we don't have to do .as_slice().unwrap() on our ndarrays outside
    // of this function, return a Rust array of delays here.
    let mut delays_a: [u32; 16] = [0; 16];
    delays_a.iter_mut().zip(delays).for_each(|(da, d)| *da = *d);

    (fixed_amps, delays_a)
}
