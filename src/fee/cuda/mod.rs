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

use std::collections::{hash_map::Entry, HashMap, HashSet};
use std::convert::TryInto;
use std::ffi::CString;

use marlu::{
    c64,
    cuda::{cuda_status_to_error, DevicePointer},
    ndarray, rayon, Jones,
};
use ndarray::prelude::*;
use rayon::prelude::*;

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
    pub(super) num_tiles: i32,

    /// The number of frequencies used to generate `FEECoeffs`. Also one of the
    /// indices used to make `(d_)coeff_map`.
    pub(super) num_freqs: i32,

    /// Coefficients map. This is used to access de-duplicated Jones matrices.
    /// Not using this would mean that Jones matrices have to be 1:1 with
    /// threads, and that potentially uses a huge amount of compute and memory!
    coeff_map: Array2<(usize, usize)>,

    /// Device pointer to the coefficients map. This is used to access
    /// de-duplicated Jones matrices. Not using this would mean that Jones
    /// matrices have to be 1:1 with threads, and that potentially uses a huge
    /// amount of compute and memory!
    d_coeff_map: DevicePointer<u64>,

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

        // Prepare the cache with all unique combinations of tiles and frequencies
        let hash_results: Vec<Vec<Result<_, _>>> = delays_array
            .outer_iter()
            .into_par_iter()
            .zip(amps_array.outer_iter().into_par_iter())
            .enumerate()
            .map(|(i_tile, (delays, amps))| {
                // unwrap is safe as these collections are contiguous.
                let delays = delays.as_slice().unwrap();
                let full_amps = fix_amps(amps.as_slice().unwrap(), delays);
                freqs_hz
                    .iter()
                    .enumerate()
                    .map(|(i_freq, freq)| {
                        // If we're normalising the beam responses, cache the
                        // normalisation Jones matrices too.
                        if norm_to_zenith {
                            fee_beam.populate_norm_jones(*freq).and_then(|fee_freq| {
                                fee_beam
                                    .populate_modes(*freq, delays, &full_amps)
                                    .map(|h| ((i_tile, i_freq, fee_freq), h))
                            })
                        } else {
                            fee_beam
                                .populate_modes(*freq, delays, &full_amps)
                                .map(|h| ((i_tile, i_freq, 0), h))
                        }
                    })
                    .collect::<Vec<Result<_, _>>>()
            })
            .collect();

        let mut unique_hash_set = HashSet::new();
        let mut unique_tile_set = HashSet::new();
        let mut unique_freq_set = HashMap::new();
        let mut unique_hashes = vec![];
        let mut row = 0;
        let mut col = 0;
        for hash_collection in hash_results {
            for hash_result in hash_collection {
                let ((i_tile, i_freq, fee_freq), hash) = hash_result?;

                if !unique_hash_set.contains(&hash) {
                    unique_hash_set.insert(hash);
                    // Unique hash found. Have we seen this tile before?
                    if !unique_tile_set.contains(&i_tile) {
                        // Put it in the set and increment the unique row
                        // dimension.
                        unique_tile_set.insert(i_tile);
                        row += 1;
                    }

                    // Have we seen this frequency before?
                    if let Entry::Vacant(e) = unique_freq_set.entry(i_freq) {
                        // Put it in the set and increment the unique column
                        // dimension.
                        e.insert(col);
                        col += 1;
                    }

                    unique_hashes.push((hash, (row - 1, unique_freq_set[&i_freq], fee_freq)));
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

        unique_hashes.iter().for_each(|(hash, (_, _, fee_freq))| {
            let coeffs = fee_beam.coeff_cache.get(hash).unwrap();
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
                norm_jones.push(*fee_beam.norm_cache.get(fee_freq).unwrap());
            }
        });

        let coeff_dedup_map: HashMap<CacheKey, (usize, usize)> = unique_hashes
            .iter()
            .map(|(hash, (tile_index, freq_index, _))| (*hash, (*tile_index, *freq_index)))
            .collect();

        // Now map each combination of frequency and tile parameters to its
        // unique dipole coefficients. The mapping is an index into the 1D array
        // `unique_hashes`, which corresponds exactly to `cuda_coeffs`.
        let mut coeff_map = Array2::from_elem((delays_array.dim().0, freqs_hz.len()), (0, 0));
        coeff_map
            .outer_iter_mut()
            .into_par_iter()
            .zip(delays_array.outer_iter().into_par_iter())
            .zip(amps_array.outer_iter().into_par_iter())
            .try_for_each(|((mut freq_coeff_indices, delays), amps)| {
                let delays = delays.as_slice().unwrap();
                let full_amps = fix_amps(amps.as_slice().unwrap(), delays);
                freq_coeff_indices
                    .iter_mut()
                    .zip(freqs_hz.iter())
                    .try_for_each(|(freq_coeff_index, freq)| {
                        match fee_beam.populate_modes(*freq, delays, &full_amps) {
                            Ok(hash) => {
                                *freq_coeff_index = coeff_dedup_map[&hash];
                                Ok(())
                            }
                            Err(e) => Err(e),
                        }
                    })
            })?;

        let num_tiles = unique_tile_set.len().try_into().unwrap();
        let num_freqs = unique_freq_set.len().try_into().unwrap();

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

        // Put the map's tuple elements into a single int, while demoting
        // the size of the ints; CUDA goes a little faster with "int"
        // instead of "size_t". Overflowing ints? Well, that'd be a big
        // problem, but there's almost certainly insufficient memory before
        // that happens.
        let coeff_map_ints = coeff_map.mapv(|(i_row, i_col)| ((i_row << 32) + i_col) as u64);
        let d_coeff_map = DevicePointer::copy_to_device(coeff_map_ints.as_slice().unwrap())?;

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

            num_coeffs: unique_hashes.len().try_into().unwrap(),
            num_tiles,
            num_freqs,

            coeff_map,
            d_coeff_map,
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
                self.num_tiles as usize
                    * self.num_freqs as usize
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
                self.num_tiles as usize,
                self.num_freqs as usize,
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
            (self.coeff_map.dim().0, self.coeff_map.dim().1, az_rad.len()),
            Jones::default(),
        );
        jones_expanded
            .outer_iter_mut()
            .zip(self.coeff_map.outer_iter())
            .for_each(|(mut jones_row, index_row)| {
                jones_row.outer_iter_mut().zip(index_row.iter()).for_each(
                    |(mut jones_col, &(i_row, i_col))| {
                        jones_col.assign(&jones_results.slice(s![i_row, i_col, ..]));
                    },
                )
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

    /// Get a pointer to the device beam Jones map. This is necessary to access
    /// de-duplicated beam Jones matrices on the device.
    pub fn get_beam_jones_map(&self) -> *const u64 {
        self.d_coeff_map.get()
    }

    /// Get the number of de-duplicated frequencies associated with this
    /// [FEEBeamCUDA].
    pub fn get_num_unique_freqs(&self) -> i32 {
        self.num_freqs
    }
}
