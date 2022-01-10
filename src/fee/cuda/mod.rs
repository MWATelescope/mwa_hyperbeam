// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! CUDA code to implement the MWA Fully Embedded Element (FEE) beam, a.k.a.
//! "the 2016 beam".

#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
// This is needed only because tests inside bindgen-produced files (cuda_*.rs)
// trigger the warning.
#![allow(deref_nullptr)]
// Link hyperbeam_cu produced by build.rs
#![link(name = "hyperbeam_cu", kind = "static")]

// Include Rust bindings to the CUDA code and set a compile-time variable type
// (this makes things a lot cleaner than having a #[cfg(...)] on many struct
// members).
cfg_if::cfg_if! {
    if #[cfg(feature = "cuda-single")] {
        include!("single.rs");
        pub(crate) type CudaFloat = f32;
    } else {
        include!("double.rs");
        pub(crate) type CudaFloat = f64;
    }
}

#[cfg(test)]
mod tests;

use std::collections::hash_map::DefaultHasher;
use std::convert::TryInto;
use std::ffi::CString;
use std::hash::{Hash, Hasher};

use marlu::{
    c64,
    cuda::{cuda_status_to_error, DevicePointer},
    ndarray, Jones,
};
use ndarray::prelude::*;

use super::{fix_amps, FEEBeam, FEEBeamError};
use crate::types::CacheKey;

/// Device pointers to coefficients for FEE beam calculations.
#[derive(Debug)]
pub struct FEEBeamCUDA {
    x_q1_accum: DevicePointer<CudaFloat>,
    x_q2_accum: DevicePointer<CudaFloat>,

    x_m_accum: DevicePointer<i8>,
    x_n_accum: DevicePointer<i8>,
    x_m_signs: DevicePointer<i8>,
    x_n_max: DevicePointer<u8>,
    x_lengths: DevicePointer<i32>,
    x_offsets: DevicePointer<i32>,

    y_q1_accum: DevicePointer<CudaFloat>,
    y_q2_accum: DevicePointer<CudaFloat>,

    y_m_accum: DevicePointer<i8>,
    y_n_accum: DevicePointer<i8>,
    y_m_signs: DevicePointer<i8>,
    y_n_max: DevicePointer<u8>,
    y_lengths: DevicePointer<i32>,
    y_offsets: DevicePointer<i32>,

    /// The number of unique tile coefficients.
    pub(super) num_coeffs: i32,

    /// The number of tiles used to generate the `FEECoeffs`. Also one of the indices
    /// used to make `(d_)coeff_map`.
    pub(super) num_unique_tiles: i32,

    /// The number of frequencies used to generate `FEECoeffs`. Also one of the
    /// indices used to make `(d_)coeff_map`.
    pub(super) num_unique_freqs: i32,

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

    /// Jones matrices for normalising the beam response. This array has the
    /// same shape as `coeff_map`. If this is `None`, then no normalisation is
    /// done (a null pointer is given to the CUDA code).
    pub(super) d_norm_jones: Option<DevicePointer<CudaFloat>>,
}

impl FEEBeamCUDA {
    /// Prepare a CUDA-capable device for beam-response computations given the
    /// frequencies, delays and amps to be used. The resulting object takes
    /// directions and computes the beam responses on the device.
    ///
    /// `delays_array` and `amps_array` must have the same number of rows; these
    /// correspond to tile configurations (i.e. each tile is allowed to have
    /// distinct delays and amps). `delays_array` must have 16 elements per row,
    /// but `amps_array` can have 16 or 32 elements per row (see `calc_jones`
    /// for an explanation).
    ///
    /// The code will automatically de-duplicate tile configurations so that no
    /// redundant calculations are done.
    pub(super) unsafe fn new(
        fee_beam: &FEEBeam,
        freqs_hz: &[u32],
        delays_array: ArrayView2<u32>,
        amps_array: ArrayView2<f64>,
        norm_to_zenith: bool,
    ) -> Result<FEEBeamCUDA, FEEBeamError> {
        // `delays` must have 16 elements...
        debug_assert_eq!(delays_array.dim().1, 16);
        // ... but `amps` may have either 16 or 32. 32 elements corresponds to
        // each element of each dipole; i.e. 16 X amplitudes followed by 16 Y
        // amplitudes.
        debug_assert!(amps_array.dim().1 == 16 || amps_array.dim().1 == 32);

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
            // unwrap is safe as these collections are contiguous (outer iter).
            let delays = delays.as_slice().unwrap();
            let full_amps = fix_amps(amps.as_slice().unwrap(), delays);

            let mut unique_tile_hasher = DefaultHasher::new();
            delays.hash(&mut unique_tile_hasher);
            // We can't hash f64 values, but we can hash their bits.
            for amp in full_amps {
                amp.to_bits().hash(&mut unique_tile_hasher);
            }
            let unique_tile_hash = unique_tile_hasher.finish();
            let this_tile_index = if unique_tiles.contains(&unique_tile_hash) {
                unique_tiles
                    .iter()
                    .enumerate()
                    .find(|(_, t)| **t == unique_tile_hash)
                    .unwrap()
                    .0 as i32
            } else {
                unique_tiles.push(unique_tile_hash);
                i_tile += 1;
                i_tile - 1
            };
            tile_map.push(this_tile_index);

            for freq in freqs_hz {
                // If we're normalising the beam responses, cache the
                // normalisation Jones matrices too.
                if norm_to_zenith {
                    fee_beam.get_norm_jones(*freq)?;
                }

                let _ = fee_beam.get_modes(*freq, delays, &full_amps)?;

                let fee_freq = fee_beam.find_closest_freq(*freq);
                let hash = CacheKey::new(fee_freq, delays, &full_amps);
                if !unique_hashes.contains(&(hash, fee_freq)) {
                    unique_hashes.push((hash, fee_freq));
                }

                // No need to do this code more than once; frequency redundancy
                // applies to all tiles.
                if i == 0 {
                    let this_freq_index = if unique_fee_freqs.contains(&fee_freq) {
                        unique_fee_freqs
                            .iter()
                            .enumerate()
                            .find(|(_, f)| **f == fee_freq)
                            .unwrap()
                            .0 as i32
                    } else {
                        unique_fee_freqs.push(fee_freq);
                        i_freq += 1;
                        i_freq - 1
                    };
                    freq_map.push(this_freq_index);
                }
            }
        }

        // Now populate the CUDA-flavoured dipole coefficients.
        let mut x_q1_accum = vec![];
        let mut x_q2_accum = vec![];
        let mut x_m_accum = vec![];
        let mut x_n_accum = vec![];
        let mut x_m_signs = vec![];
        let mut x_n_max = vec![];
        let mut x_lengths = vec![];
        let mut x_offsets = vec![];
        let mut y_q1_accum = vec![];
        let mut y_q2_accum = vec![];
        let mut y_m_accum = vec![];
        let mut y_n_accum = vec![];
        let mut y_m_signs = vec![];
        let mut y_n_max = vec![];
        let mut y_lengths = vec![];
        let mut y_offsets = vec![];
        let mut norm_jones = vec![];

        // We can't safely pass complex numbers over the FFI boundary, so
        // collect new vectors of floats and use those pointers instead.
        let unpack_complex = |v: &[c64]| -> Vec<CudaFloat> {
            v.iter()
                .flat_map(|c| [c.re as CudaFloat, c.im as CudaFloat])
                .collect()
        };

        let num_coeffs = unique_hashes.len().try_into().unwrap();
        unique_hashes.into_iter().for_each(|(hash, fee_freq)| {
            let coeffs = &fee_beam.coeff_cache.read()[&hash];
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

            x_q1_accum.append(&mut unpack_complex(&coeffs.x.q1_accum[..current_x_len]));
            x_q2_accum.append(&mut unpack_complex(&coeffs.x.q2_accum[..current_x_len]));
            x_m_accum.extend_from_slice(&coeffs.x.m_accum[..current_x_len]);
            x_n_accum.extend_from_slice(&coeffs.x.n_accum[..current_x_len]);
            x_m_signs.extend_from_slice(&coeffs.x.m_signs[..current_x_len]);
            x_n_max.push(coeffs.x.n_max.try_into().unwrap());
            x_lengths.push(current_x_len.try_into().unwrap());
            x_offsets.push(x_offset);

            y_q1_accum.append(&mut unpack_complex(&coeffs.y.q1_accum[..current_y_len]));
            y_q2_accum.append(&mut unpack_complex(&coeffs.y.q2_accum[..current_y_len]));
            y_m_accum.extend_from_slice(&coeffs.y.m_accum[..current_y_len]);
            y_n_accum.extend_from_slice(&coeffs.y.n_accum[..current_y_len]);
            y_m_signs.extend_from_slice(&coeffs.y.m_signs[..current_y_len]);
            y_n_max.push(coeffs.y.n_max.try_into().unwrap());
            y_lengths.push(current_y_len.try_into().unwrap());
            y_offsets.push(y_offset);

            if norm_to_zenith {
                norm_jones.push(*fee_beam.norm_cache.read()[&fee_freq]);
            }
        });

        let x_q1_accum = DevicePointer::copy_to_device(&x_q1_accum)?;
        let x_q2_accum = DevicePointer::copy_to_device(&x_q2_accum)?;
        let x_m_accum = DevicePointer::copy_to_device(&x_m_accum)?;
        let x_n_accum = DevicePointer::copy_to_device(&x_n_accum)?;
        let x_m_signs = DevicePointer::copy_to_device(&x_m_signs)?;
        let x_n_max = DevicePointer::copy_to_device(&x_n_max)?;
        let x_lengths = DevicePointer::copy_to_device(&x_lengths)?;
        let x_offsets = DevicePointer::copy_to_device(&x_offsets)?;

        let y_q1_accum = DevicePointer::copy_to_device(&y_q1_accum)?;
        let y_q2_accum = DevicePointer::copy_to_device(&y_q2_accum)?;
        let y_m_accum = DevicePointer::copy_to_device(&y_m_accum)?;
        let y_n_accum = DevicePointer::copy_to_device(&y_n_accum)?;
        let y_m_signs = DevicePointer::copy_to_device(&y_m_signs)?;
        let y_n_max = DevicePointer::copy_to_device(&y_n_max)?;
        let y_lengths = DevicePointer::copy_to_device(&y_lengths)?;
        let y_offsets = DevicePointer::copy_to_device(&y_offsets)?;

        let num_unique_tiles = unique_tiles.len().try_into().unwrap();
        let num_unique_freqs = unique_fee_freqs.len().try_into().unwrap();

        let d_tile_map = DevicePointer::copy_to_device(&tile_map)?;
        let d_freq_map = DevicePointer::copy_to_device(&freq_map)?;

        let d_norm_jones = if norm_jones.is_empty() {
            None
        } else {
            let norm_jones_unpacked: Vec<CudaFloat> = norm_jones
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

        Ok(Self {
            x_q1_accum,
            x_q2_accum,
            x_m_accum,
            x_n_accum,
            x_m_signs,
            x_n_max,
            x_lengths,
            x_offsets,

            y_q1_accum,
            y_q2_accum,
            y_m_accum,
            y_n_accum,
            y_m_signs,
            y_n_max,
            y_lengths,
            y_offsets,

            num_coeffs,
            num_unique_tiles,
            num_unique_freqs,

            tile_map,
            freq_map,
            d_tile_map,
            d_freq_map,
            d_norm_jones,
        })
    }

    /// Given directions, calculate beam-response Jones matrices on the device
    /// and return a pointer to them.
    pub fn calc_jones_device(
        &self,
        az_rad: &[CudaFloat],
        za_rad: &[CudaFloat],
        parallactic: bool,
    ) -> Result<DevicePointer<Jones<CudaFloat>>, FEEBeamError> {
        unsafe {
            // Allocate a buffer on the device for results.
            let d_results = DevicePointer::malloc(
                self.num_unique_tiles as usize
                    * self.num_unique_freqs as usize
                    * az_rad.len()
                    * std::mem::size_of::<Jones<CudaFloat>>(),
            )?;

            let d_azs = DevicePointer::copy_to_device(az_rad)?;
            let d_zas = DevicePointer::copy_to_device(za_rad)?;
            let error_str =
                CString::from_vec_unchecked(vec![1; marlu::cuda::ERROR_STR_LENGTH]).into_raw();

            let result = cuda_calc_jones(
                d_azs.get(),
                d_zas.get(),
                az_rad.len().try_into().unwrap(),
                &self.get_fee_coeffs(),
                self.num_coeffs,
                match self.d_norm_jones.as_ref() {
                    Some(n) => (*n).get().cast(),
                    None => std::ptr::null_mut(),
                },
                parallactic as i8,
                d_results.get_mut() as *mut std::ffi::c_void,
                error_str,
            );
            cuda_status_to_error(result, error_str)?;

            Ok(d_results)
        }
    }

    /// Given directions, calculate beam-response Jones matrices on the device,
    /// copy them to the host, and free the device memory. The returned array is
    /// "expanded"; tile and frequency de-duplication is undone to give an array
    /// with the same number of tiles and frequencies as was specified when this
    /// [FEEBeamCUDA] was created.
    pub fn calc_jones(
        &self,
        az_rad: &[CudaFloat],
        za_rad: &[CudaFloat],
        parallactic: bool,
    ) -> Result<Array3<Jones<CudaFloat>>, FEEBeamError> {
        let device_ptr = self.calc_jones_device(az_rad, za_rad, parallactic)?;
        let mut jones_results: Array3<Jones<CudaFloat>> = Array3::from_elem(
            (
                self.num_unique_tiles as usize,
                self.num_unique_freqs as usize,
                az_rad.len(),
            ),
            Jones::default(),
        );
        unsafe {
            device_ptr.copy_from_device(jones_results.as_slice_mut().unwrap())?;
        }
        // Free the device memory.
        drop(device_ptr);

        // Expand the results according to the map.
        let mut jones_expanded = Array3::from_elem(
            (self.tile_map.len(), self.freq_map.len(), az_rad.len()),
            Jones::default(),
        );
        jones_expanded
            .outer_iter_mut()
            .zip(self.tile_map.iter())
            .for_each(|(mut jones_row, &i_row)| {
                let i_row: usize = i_row.try_into().unwrap();
                jones_row
                    .outer_iter_mut()
                    .zip(self.freq_map.iter())
                    .for_each(|(mut jones_col, &i_col)| {
                        let i_col: usize = i_col.try_into().unwrap();
                        jones_col.assign(&jones_results.slice(s![i_row, i_col, ..]));
                    })
            });
        Ok(jones_expanded)
    }

    /// Convenience function to get the `FEECoeffs` C struct that the CUDA code
    /// wants.
    fn get_fee_coeffs(&self) -> FEECoeffs {
        FEECoeffs {
            x_q1_accum: self.x_q1_accum.get_mut(),
            x_q2_accum: self.x_q2_accum.get_mut(),
            x_m_accum: self.x_m_accum.get_mut(),
            x_n_accum: self.x_n_accum.get_mut(),
            x_m_signs: self.x_m_signs.get_mut(),
            x_n_max: self.x_n_max.get_mut(),
            x_lengths: self.x_lengths.get_mut(),
            x_offsets: self.x_offsets.get_mut(),
            y_q1_accum: self.y_q1_accum.get_mut(),
            y_q2_accum: self.y_q2_accum.get_mut(),
            y_m_accum: self.y_m_accum.get_mut(),
            y_n_accum: self.y_n_accum.get_mut(),
            y_m_signs: self.y_m_signs.get_mut(),
            y_n_max: self.y_n_max.get_mut(),
            y_lengths: self.y_lengths.get_mut(),
            y_offsets: self.y_offsets.get_mut(),
        }
    }

    /// Get a pointer to the device tile map. This is necessary to access
    /// de-duplicated beam Jones matrices on the device.
    pub fn get_tile_map(&self) -> *const i32 {
        self.d_tile_map.get()
    }

    /// Get a pointer to the device freq map. This is necessary to access
    /// de-duplicated beam Jones matrices on the device.
    pub fn get_freq_map(&self) -> *const i32 {
        self.d_freq_map.get()
    }

    /// Get the number of de-duplicated frequencies associated with this
    /// [FEEBeamCUDA].
    pub fn get_num_unique_freqs(&self) -> i32 {
        self.num_unique_freqs
    }
}
