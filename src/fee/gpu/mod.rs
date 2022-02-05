// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! GPU code to implement the MWA Fully Embedded Element (FEE) beam, a.k.a. "the
//! 2016 beam".

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

use super::{FEEBeam, FEEBeamError};
use crate::{
    gpu::{DevicePointer, GpuError, GpuFloat},
    types::CacheKey,
};

/// A GPU beam object ready to calculate beam responses.
#[derive(Debug)]
pub struct FEEBeamGpu {
    x_q1_accum: DevicePointer<GpuFloat>,
    x_q2_accum: DevicePointer<GpuFloat>,

    x_m_accum: DevicePointer<i8>,
    x_n_accum: DevicePointer<i8>,
    x_m_signs: DevicePointer<i8>,
    x_m_abs_m: DevicePointer<i8>,
    x_lengths: DevicePointer<i32>,
    x_offsets: DevicePointer<i32>,

    y_q1_accum: DevicePointer<GpuFloat>,
    y_q2_accum: DevicePointer<GpuFloat>,

    y_m_accum: DevicePointer<i8>,
    y_n_accum: DevicePointer<i8>,
    y_m_signs: DevicePointer<i8>,
    y_m_abs_m: DevicePointer<i8>,
    y_lengths: DevicePointer<i32>,
    y_offsets: DevicePointer<i32>,

    /// The number of unique tile coefficients.
    pub(super) num_coeffs: i32,

    /// The number of tiles used to generate the [`FEECoeffs`].
    pub(super) num_unique_tiles: i32,

    /// The number of frequencies used to generate [`FEECoeffs`].
    pub(super) num_unique_freqs: i32,

    /// The biggest `n_max` across all sets of coefficients, for both x and y
    /// dipoles.
    n_max: u8,

    /// Tile map. This is used to access de-duplicated Jones matrices.
    /// Not using this would mean that Jones matrices have to be 1:1 with
    /// threads, and that potentially uses a huge amount of compute and memory!
    tile_map: Vec<i32>,

    /// Tile map. This is used to access de-duplicated Jones matrices.
    /// Not using this would mean that Jones matrices have to be 1:1 with
    /// threads, and that potentially uses a huge amount of compute and memory!
    freq_map: Vec<i32>,

    /// The device pointer to the `tile_map` (same as the host's memory
    /// equivalent above).
    d_tile_map: DevicePointer<i32>,

    /// The device pointer to the `freq_map` (same as the host's memory
    /// equivalent above).
    d_freq_map: DevicePointer<i32>,

    /// Jones matrices for normalising the beam response. Has a shape
    /// `num_unique_tiles`, `num_unique_freqs`, `num_directions`, in that order.
    /// If this is `None`, then no normalisation is done (a null pointer is
    /// given to the CUDA code).
    pub(super) d_norm_jones: Option<DevicePointer<GpuFloat>>,
}

impl FEEBeamGpu {
    /// Prepare a GPU-capable device for beam-response computations given the
    /// frequencies, delays and amps to be used. The resulting object takes
    /// directions and computes the beam responses on the device.
    ///
    /// `delays_array` and `amps_array` must have the same number of rows; these
    /// correspond to tile configurations (i.e. each tile is allowed to have
    /// distinct delays and amps). `delays_array` must have 16 elements per row,
    /// but `amps_array` can have 16 or 32 elements per row (see
    /// [`FEEBeam::calc_jones`] for an explanation).
    ///
    /// The code will automatically de-duplicate tile configurations so that no
    /// redundant calculations are done.
    ///
    /// This function is intentionally kept private. Use
    /// [`FEEBeam::gpu_prepare`] to create a `FEEBeamGpu`.
    pub(super) unsafe fn new(
        fee_beam: &FEEBeam,
        freqs_hz: &[u32],
        delays_array: ArrayView2<u32>,
        amps_array: ArrayView2<f64>,
        norm_to_zenith: bool,
    ) -> Result<FEEBeamGpu, FEEBeamError> {
        if delays_array.len_of(Axis(1)) != 16 {
            return Err(FEEBeamError::IncorrectDelaysArrayColLength {
                rows: delays_array.len_of(Axis(0)),
                num_delays: delays_array.len_of(Axis(1)),
            });
        }
        if !(amps_array.len_of(Axis(1)) == 16 || amps_array.len_of(Axis(1)) == 32) {
            return Err(FEEBeamError::IncorrectAmpsLength(
                amps_array.len_of(Axis(1)),
            ));
        }

        // Prepare the cache with all unique combinations of tiles and
        // frequencies. Track all of the unique tiles and frequencies to allow
        // de-duplication.
        let mut unique_hashes = vec![];
        let mut unique_tiles = vec![];
        let mut unique_fee_freqs = vec![];
        let mut tile_map = vec![];
        let mut freq_map = vec![];
        let mut i_tile = 0;
        let mut i_freq = 0;
        for (i, (delays, amps)) in delays_array
            .outer_iter()
            .zip(amps_array.outer_iter())
            .enumerate()
        {
            let (full_amps, delays) = fix_amps_ndarray(amps, delays);

            let mut unique_tile_hasher = DefaultHasher::new();
            delays.hash(&mut unique_tile_hasher);
            // We can't hash f64 values, but we can hash their bits.
            for amp in full_amps {
                amp.to_bits().hash(&mut unique_tile_hasher);
            }
            let unique_tile_hash = unique_tile_hasher.finish();
            let this_tile_index = if let Some((index, _)) = unique_tiles
                .iter()
                .enumerate()
                .find(|(_, t)| **t == unique_tile_hash)
            {
                index.try_into().expect("smaller than i32::MAX")
            } else {
                unique_tiles.push(unique_tile_hash);
                i_tile += 1;
                i_tile - 1
            };
            tile_map.push(this_tile_index);

            for &freq in freqs_hz {
                // If we're normalising the beam responses, cache the
                // normalisation Jones matrices too.
                if norm_to_zenith {
                    fee_beam.get_norm_jones(freq)?;
                }

                drop(fee_beam.get_modes(freq, &delays, &full_amps)?);

                let fee_freq = fee_beam.find_closest_freq(freq);
                let hash = CacheKey::new(fee_freq, &delays, &full_amps);
                if !unique_hashes.contains(&(hash, fee_freq)) {
                    unique_hashes.push((hash, fee_freq));
                }

                // No need to do this code more than once; frequency redundancy
                // applies to all tiles.
                if i == 0 {
                    let this_freq_index = if let Some((index, _)) = unique_fee_freqs
                        .iter()
                        .enumerate()
                        .find(|(_, f)| **f == fee_freq)
                    {
                        index.try_into().expect("smaller than i32::MAX")
                    } else {
                        unique_fee_freqs.push(fee_freq);
                        i_freq += 1;
                        i_freq - 1
                    };
                    freq_map.push(this_freq_index);
                }
            }
        }

        // Now populate the GPU-flavoured dipole coefficients. Start by
        // determining the lengths of the following vectors (saves a lot of
        // re-allocs) as well as the largest `n_max` (We don't need information
        // on x or y n_max, only the biggest one across all coefficients).
        let (x_len, y_len, n_max) = unique_hashes.iter().fold((0, 0, 0), |acc, (hash, _)| {
            let coeffs = &fee_beam.coeff_cache.read()[hash];

            let current_x_len = coeffs.x.q1_accum.len().min(coeffs.x.m_accum.len());
            let current_y_len = coeffs.y.q1_accum.len().min(coeffs.y.m_accum.len());
            let current_n_max = coeffs.x.n_max.max(coeffs.y.n_max);
            (
                acc.0 + current_x_len,
                acc.1 + current_y_len,
                acc.2.max(current_n_max),
            )
        });

        // The "accum" vectors actually hold complex numbers, so their lengths
        // are doubled.
        let mut x_q1_accum = Vec::with_capacity(x_len * 2);
        let mut x_q2_accum = Vec::with_capacity(x_len * 2);
        let mut x_m_accum = Vec::with_capacity(x_len);
        let mut x_n_accum = Vec::with_capacity(x_len);
        let mut x_m_signs = Vec::with_capacity(x_len);
        let mut x_m_abs_m = Vec::with_capacity(x_len);
        let mut x_lengths = Vec::with_capacity(x_len);
        let mut x_offsets = Vec::with_capacity(x_len);
        let mut y_q1_accum = Vec::with_capacity(y_len * 2);
        let mut y_q2_accum = Vec::with_capacity(y_len * 2);
        let mut y_m_accum = Vec::with_capacity(y_len);
        let mut y_n_accum = Vec::with_capacity(y_len);
        let mut y_m_signs = Vec::with_capacity(y_len);
        let mut y_m_abs_m = Vec::with_capacity(y_len);
        let mut y_lengths = Vec::with_capacity(y_len);
        let mut y_offsets = Vec::with_capacity(y_len);
        let mut norm_jones = Vec::with_capacity(unique_hashes.len());

        unique_hashes.iter().for_each(|(hash, fee_freq)| {
            let coeffs = &fee_beam.coeff_cache.read()[hash];
            let x_offset = x_offsets
                .last()
                .and_then(|&o| x_lengths.last().map(|&l| l + o))
                .unwrap_or(0);
            let y_offset = y_offsets
                .last()
                .and_then(|&o| y_lengths.last().map(|&l| l + o))
                .unwrap_or(0);
            // If m/n_accum has fewer elements than q1/q2_accum, then we
            // only need that many elements.
            let current_x_len = coeffs.x.q1_accum.len().min(coeffs.x.m_accum.len());
            let current_y_len = coeffs.y.q1_accum.len().min(coeffs.y.m_accum.len());

            // We may need to convert the floats before copying to the device.
            x_q1_accum.extend(
                coeffs.x.q1_accum[..current_x_len]
                    .iter()
                    .flat_map(|c| [c.re as GpuFloat, c.im as GpuFloat]),
            );
            x_q2_accum.extend(
                coeffs.x.q2_accum[..current_x_len]
                    .iter()
                    .flat_map(|c| [c.re as GpuFloat, c.im as GpuFloat]),
            );
            x_m_accum.extend(&coeffs.x.m_accum[..current_x_len]);
            x_n_accum.extend(&coeffs.x.n_accum[..current_x_len]);
            x_m_signs.extend(&coeffs.x.m_signs[..current_x_len]);
            x_m_abs_m.extend(coeffs.x.m_accum[..current_x_len].iter().map(|i| i.abs()));
            x_lengths.push(
                current_x_len
                    .try_into()
                    .expect("much smaller than i32::MAX"),
            );
            x_offsets.push(x_offset);

            y_q1_accum.extend(
                coeffs.y.q1_accum[..current_y_len]
                    .iter()
                    .flat_map(|c| [c.re as GpuFloat, c.im as GpuFloat]),
            );
            y_q2_accum.extend(
                coeffs.y.q2_accum[..current_y_len]
                    .iter()
                    .flat_map(|c| [c.re as GpuFloat, c.im as GpuFloat]),
            );
            y_m_accum.extend(&coeffs.y.m_accum[..current_y_len]);
            y_n_accum.extend(&coeffs.y.n_accum[..current_y_len]);
            y_m_signs.extend(&coeffs.y.m_signs[..current_y_len]);
            y_m_abs_m.extend(coeffs.y.m_accum[..current_y_len].iter().map(|i| i.abs()));
            y_lengths.push(
                current_y_len
                    .try_into()
                    .expect("much smaller than i32::MAX"),
            );
            y_offsets.push(y_offset);

            if norm_to_zenith {
                norm_jones.push(*fee_beam.norm_cache.read()[fee_freq]);
            }
        });

        let d_norm_jones = if norm_jones.is_empty() {
            None
        } else {
            let norm_jones_unpacked: Vec<GpuFloat> = norm_jones
                .into_iter()
                .flat_map(|j| {
                    [
                        j[0].re as _,
                        j[0].im as _,
                        j[1].re as _,
                        j[1].im as _,
                        j[2].re as _,
                        j[2].im as _,
                        j[3].re as _,
                        j[3].im as _,
                    ]
                })
                .collect();
            Some(DevicePointer::copy_to_device(&norm_jones_unpacked)?)
        };

        let d_tile_map = DevicePointer::copy_to_device(&tile_map)?;
        let d_freq_map = DevicePointer::copy_to_device(&freq_map)?;
        Ok(FEEBeamGpu {
            x_q1_accum: DevicePointer::copy_to_device(&x_q1_accum)?,
            x_q2_accum: DevicePointer::copy_to_device(&x_q2_accum)?,
            x_m_accum: DevicePointer::copy_to_device(&x_m_accum)?,
            x_n_accum: DevicePointer::copy_to_device(&x_n_accum)?,
            x_m_signs: DevicePointer::copy_to_device(&x_m_signs)?,
            x_m_abs_m: DevicePointer::copy_to_device(&x_m_abs_m)?,
            x_lengths: DevicePointer::copy_to_device(&x_lengths)?,
            x_offsets: DevicePointer::copy_to_device(&x_offsets)?,

            y_q1_accum: DevicePointer::copy_to_device(&y_q1_accum)?,
            y_q2_accum: DevicePointer::copy_to_device(&y_q2_accum)?,
            y_m_accum: DevicePointer::copy_to_device(&y_m_accum)?,
            y_n_accum: DevicePointer::copy_to_device(&y_n_accum)?,
            y_m_signs: DevicePointer::copy_to_device(&y_m_signs)?,
            y_m_abs_m: DevicePointer::copy_to_device(&y_m_abs_m)?,
            y_lengths: DevicePointer::copy_to_device(&y_lengths)?,
            y_offsets: DevicePointer::copy_to_device(&y_offsets)?,

            num_coeffs: unique_hashes
                .len()
                .try_into()
                .expect("many fewer coeffs than i32::MAX"),
            num_unique_tiles: unique_tiles
                .len()
                .try_into()
                .expect("expected much fewer than i32::MAX"),
            num_unique_freqs: unique_fee_freqs
                .len()
                .try_into()
                .expect("expected much fewer than i32::MAX"),
            n_max,

            tile_map,
            freq_map,
            d_tile_map,
            d_freq_map,
            d_norm_jones,
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
        latitude_rad: Option<f64>,
        iau_reorder: bool,
    ) -> Result<DevicePointer<Jones<GpuFloat>>, FEEBeamError> {
        unsafe {
            // Allocate a buffer on the device for results.
            let d_results = DevicePointer::malloc(
                self.num_unique_tiles as usize
                    * self.num_unique_freqs as usize
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

            // Allocate the latitude if we have to.
            let d_latitude_rad = latitude_rad
                .map(|f| DevicePointer::copy_to_device(&[f as GpuFloat]))
                .transpose()?;

            self.calc_jones_device_pair_inner(
                d_azs.get(),
                d_zas.get(),
                azels.len().try_into().expect("much fewer than i32::MAX"),
                d_latitude_rad.map(|p| p.get()).unwrap_or(std::ptr::null()),
                iau_reorder,
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
        latitude_rad: Option<f64>,
        iau_reorder: bool,
    ) -> Result<DevicePointer<Jones<GpuFloat>>, FEEBeamError> {
        unsafe {
            // Allocate a buffer on the device for results.
            let d_results = DevicePointer::malloc(
                self.num_unique_tiles as usize
                    * self.num_unique_freqs as usize
                    * az_rad.len()
                    * std::mem::size_of::<Jones<GpuFloat>>(),
            )?;

            // Also copy the directions to the device.
            let d_azs = DevicePointer::copy_to_device(az_rad)?;
            let d_zas = DevicePointer::copy_to_device(za_rad)?;

            // Allocate the latitude if we have to.
            let d_latitude_rad = latitude_rad
                .map(|f| DevicePointer::copy_to_device(&[f as GpuFloat]))
                .transpose()?;

            self.calc_jones_device_pair_inner(
                d_azs.get(),
                d_zas.get(),
                az_rad.len().try_into().expect("much fewer than i32::MAX"),
                d_latitude_rad.map(|p| p.get()).unwrap_or(std::ptr::null()),
                iau_reorder,
                d_results.get_mut() as *mut std::ffi::c_void,
            )?;
            Ok(d_results)
        }
    }

    /// Given directions, calculate beam-response Jones matrices into the
    /// supplied pre-allocated device pointer. This buffer should have a shape
    /// of (`num_unique_tiles`, `num_unique_freqs`, `az_rad_length`). The first
    /// two dimensions can be accessed with [`FEEBeamGpu::get_num_unique_tiles`]
    /// and [`FEEBeamGpu::get_num_unique_freqs`]. `d_latitude_rad` is
    /// populated with the array latitude, if the caller wants the parallactic-
    /// angle correction to be applied. If the pointer is null, then no
    /// correction is applied.
    ///
    /// # Safety
    ///
    /// If `d_results` is too small (correct size described above), then
    /// undefined behaviour looms.
    pub unsafe fn calc_jones_device_pair_inner(
        &self,
        d_az_rad: *const GpuFloat,
        d_za_rad: *const GpuFloat,
        num_directions: i32,
        d_latitude_rad: *const GpuFloat,
        iau_reorder: bool,
        d_results: *mut std::ffi::c_void,
    ) -> Result<(), FEEBeamError> {
        // Don't do anything if there aren't any directions.
        if num_directions == 0 {
            return Ok(());
        }

        // The return value is a pointer to a CUDA/HIP error string. If it's null
        // then everything is fine.
        let error_message_ptr = gpu_calc_jones(
            d_az_rad,
            d_za_rad,
            num_directions,
            &self.get_fee_coeffs(),
            self.num_coeffs,
            match self.d_norm_jones.as_ref() {
                Some(n) => (*n).get().cast(),
                None => std::ptr::null(),
            },
            d_latitude_rad,
            iau_reorder.into(),
            d_results,
        );
        if error_message_ptr.is_null() {
            Ok(())
        } else {
            let error_message = CStr::from_ptr(error_message_ptr)
                .to_str()
                .unwrap_or("<cannot read GPU error string>");
            let our_error_str = format!("fee.h:gpu_calc_jones failed with: {error_message}");
            Err(FEEBeamError::Gpu(GpuError::Kernel {
                msg: our_error_str.into(),
                file: file!(),
                line: line!(),
            }))
        }
    }

    /// Given directions, calculate beam-response Jones matrices on the device,
    /// copy them to the host, and free the device memory. The returned array is
    /// "expanded"; tile and frequency de-duplication is undone to give an array
    /// with the same number of tiles and frequencies as was specified when this
    /// [`FEEBeamGpu`] was created.
    ///
    /// Note that this function needs to allocate two vectors for azimuths and
    /// zenith angles from the supplied `azels`.
    pub fn calc_jones(
        &self,
        azels: &[AzEl],
        latitude_rad: Option<f64>,
        iau_reorder: bool,
    ) -> Result<Array3<Jones<GpuFloat>>, FEEBeamError> {
        let mut results = Array3::from_elem(
            (self.tile_map.len(), self.freq_map.len(), azels.len()),
            Jones::default(),
        );

        let (azs, zas): (Vec<GpuFloat>, Vec<GpuFloat>) = azels
            .iter()
            .map(|&azel| (azel.az as GpuFloat, azel.za() as GpuFloat))
            .unzip();
        self.calc_jones_pair_inner(&azs, &zas, latitude_rad, iau_reorder, results.view_mut())?;
        Ok(results)
    }

    /// Given directions, calculate beam-response Jones matrices on the device,
    /// copy them to the host, and free the device memory. The returned array is
    /// "expanded"; tile and frequency de-duplication is undone to give an array
    /// with the same number of tiles and frequencies as was specified when this
    /// [`FEEBeamGpu`] was created.
    pub fn calc_jones_pair(
        &self,
        az_rad: &[GpuFloat],
        za_rad: &[GpuFloat],
        latitude_rad: Option<f64>,
        iau_reorder: bool,
    ) -> Result<Array3<Jones<GpuFloat>>, FEEBeamError> {
        let mut results = Array3::from_elem(
            (self.tile_map.len(), self.freq_map.len(), az_rad.len()),
            Jones::default(),
        );

        self.calc_jones_pair_inner(
            az_rad,
            za_rad,
            latitude_rad,
            iau_reorder,
            results.view_mut(),
        )?;
        Ok(results)
    }

    /// Given directions, calculate beam-response Jones matrices on the device,
    /// copy them to the host, and free the device memory. This function is the
    /// same as [`FEEBeamGpu::calc_jones_pair`], but the results are stored in
    /// a pre-allocated array. This array should have a shape of
    /// (`total_num_tiles`, `total_num_freqs`, `az_rad_length`). The first two
    /// dimensions can be accessed with `FEEBeamGpu::get_total_num_tiles` and
    /// `FEEBeamGpu::get_total_num_freqs`.
    pub fn calc_jones_pair_inner(
        &self,
        az_rad: &[GpuFloat],
        za_rad: &[GpuFloat],
        latitude_rad: Option<f64>,
        iau_reorder: bool,
        mut results: ArrayViewMut3<Jones<GpuFloat>>,
    ) -> Result<(), FEEBeamError> {
        // Allocate an array matching the deduplicated device memory.
        let mut dedup_results: Array3<Jones<GpuFloat>> = Array3::from_elem(
            (
                self.num_unique_tiles as usize,
                self.num_unique_freqs as usize,
                az_rad.len(),
            ),
            Jones::default(),
        );
        // Calculate the beam responses. and copy them to the host.
        let device_ptr = self.calc_jones_device_pair(az_rad, za_rad, latitude_rad, iau_reorder)?;
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
                jones_row
                    .outer_iter_mut()
                    .zip(self.freq_map.iter())
                    .for_each(|(mut jones_col, &i_col)| {
                        let i_col: usize = i_col.try_into().expect("is a positive int");
                        jones_col.assign(&dedup_results.slice(s![i_row, i_col, ..]));
                    })
            });
        Ok(())
    }

    /// Convenience function to get the `FEECoeffs` C struct that the GPU code
    /// wants.
    fn get_fee_coeffs(&self) -> FEECoeffs {
        FEECoeffs {
            x_q1_accum: self.x_q1_accum.get(),
            x_q2_accum: self.x_q2_accum.get(),
            x_m_accum: self.x_m_accum.get(),
            x_n_accum: self.x_n_accum.get(),
            x_m_signs: self.x_m_signs.get(),
            x_m_abs_m: self.x_m_abs_m.get(),
            x_lengths: self.x_lengths.get(),
            x_offsets: self.x_offsets.get(),
            y_q1_accum: self.y_q1_accum.get(),
            y_q2_accum: self.y_q2_accum.get(),
            y_m_accum: self.y_m_accum.get(),
            y_n_accum: self.y_n_accum.get(),
            y_m_signs: self.y_m_signs.get(),
            y_m_abs_m: self.y_m_abs_m.get(),
            y_lengths: self.y_lengths.get(),
            y_offsets: self.y_offsets.get(),
            n_max: self.n_max,
        }
    }

    /// Get the number of tiles that this [`FEEBeamGpu`] applies to.
    pub fn get_total_num_tiles(&self) -> usize {
        self.tile_map.len()
    }

    /// Get the number of frequencies that this [`FEEBeamGpu`] applies to.
    pub fn get_total_num_freqs(&self) -> usize {
        self.freq_map.len()
    }

    /// Get a pointer to the device tile map associated with this
    /// [`FEEBeamGpu`]. This is necessary to access de-duplicated beam Jones
    /// matrices on the device.
    pub fn get_tile_map(&self) -> *const i32 {
        self.d_tile_map.get()
    }

    /// Get a pointer to the device freq map associated with this
    /// [`FEEBeamGpu`]. This is necessary to access de-duplicated beam Jones
    /// matrices on the device.
    pub fn get_freq_map(&self) -> *const i32 {
        self.d_freq_map.get()
    }

    /// Get the number of de-duplicated tiles associated with this
    /// [`FEEBeamGpu`].
    pub fn get_num_unique_tiles(&self) -> i32 {
        self.num_unique_tiles
    }

    /// Get the number of de-duplicated frequencies associated with this
    /// [`FEEBeamGpu`].
    pub fn get_num_unique_freqs(&self) -> i32 {
        self.num_unique_freqs
    }
}

/// Ensure that any delays of 32 have an amplitude (dipole gain) of 0. The
/// results are bad otherwise! Also ensure that we have 32 dipole gains (amps)
/// here. Also return a Rust array of delays for convenience.
pub(super) fn fix_amps_ndarray(
    amps: ArrayView1<f64>,
    delays: ArrayView1<u32>,
) -> ([f64; 32], [u32; 16]) {
    let mut full_amps: [f64; 32] = [1.0; 32];
    full_amps
        .iter_mut()
        .zip(amps.iter().cycle())
        .zip(delays.iter().cycle())
        .for_each(|((out_amp, &in_amp), &delay)| {
            if delay == 32 {
                *out_amp = 0.0;
            } else {
                *out_amp = in_amp;
            }
        });

    // So that we don't have to do .as_slice().unwrap() on our ndarrays outside
    // of this function, return a Rust array of delays here.
    let mut delays_a: [u32; 16] = [0; 16];
    delays_a.iter_mut().zip(delays).for_each(|(da, d)| *da = *d);

    (full_amps, delays_a)
}
