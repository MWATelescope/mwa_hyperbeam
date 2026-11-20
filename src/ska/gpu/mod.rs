// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! GPU code for the SKA-Low element and station beams.

#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

// Include Rust bindings to the GPU code, depending on the precision used.
#[cfg(feature = "gpu-single")]
include!("single.rs");
#[cfg(not(feature = "gpu-single"))]
include!("double.rs");

#[cfg(test)]
mod tests;

use std::{convert::TryInto, ffi::CStr};

use marlu::{AzEl, Jones};
use ndarray::prelude::*;

use super::{beam::SkaLowBeamInner, SkaBeamError, SkaElementResponse, SkaLowBeam};
use crate::gpu::{DevicePointer, GpuComplex, GpuError, GpuFloat};

/// The maximum spherical-wave degree supported by the GPU code (must match
/// `SKA_MAX_L_MAX` in ska.h).
pub const SKA_GPU_MAX_L_MAX: usize = 64;

/// What each "tile" of a [`SkaLowBeamGpu`] is.
#[derive(Clone, Copy, Debug)]
pub enum SkaGpuTiles<'a> {
    /// Each tile is a single element, with the given element ID (only
    /// meaningful for `ska-low-eep-sph`; use `&[0]` for other models).
    Elements(&'a [usize]),

    /// Each tile is a station. `positions_enu` has shape `(num_stations,
    /// num_elements, 3)` (East, North, Up in metres) and `weights` (if given)
    /// shape `(num_stations, num_elements)`; a weight of 0 flags an element.
    /// For `ska-low-eep-sph`, `num_elements` must equal the number of elements
    /// in the coefficient files, and element `i` uses element ID `i`. Stations
    /// with fewer real elements can be padded with zero-weight elements
    /// (except for `ska-low-eep-sph`). `rotations_rad` (if given) has one
    /// rotation per station, as for
    /// [`SkaStation::with_rotation`](super::SkaStation::with_rotation).
    /// `local_frames` (if given) has one local frame per station, as for
    /// [`SkaStation::with_local_frame`](super::SkaStation::with_local_frame).
    Stations {
        positions_enu: ArrayView3<'a, f64>,
        weights: Option<ArrayView2<'a, f64>>,
        /// If given, the weights of the Y dipoles (the same shape as
        /// `weights`); otherwise both dipoles use `weights`.
        weights_y: Option<ArrayView2<'a, f64>>,
        rotations_rad: Option<&'a [f64]>,
        local_frames: Option<&'a [[[f64; 3]; 3]]>,
    },
}

/// A GPU SKA-Low beam object ready to calculate beam responses for a set of
/// frequencies and tiles (elements or stations).
///
/// Results have shape `(num_tiles, num_freqs, num_directions)`, the same
/// layout as [`FEEBeamGpu`](crate::fee::FEEBeamGpu), and the same
/// conventions as the CPU `calc_jones*` methods. Directions outside of a
/// model's domain (e.g. outside an EEP grid) give zero responses on the GPU,
/// rather than an error.
pub struct SkaLowBeamGpu {
    model: SkaGpuModel,
    num_freqs: i32,
    d_freqs: DevicePointer<GpuFloat>,
    num_tiles: i32,
    d_element_ids: Option<DevicePointer<i32>>,
    num_station_elements: i32,
    d_positions: Option<DevicePointer<GpuFloat>>,
    d_weights: Option<DevicePointer<GpuFloat>>,
    d_weights_y: Option<DevicePointer<GpuFloat>>,
    d_rotations: Option<DevicePointer<GpuFloat>>,
    d_frames: Option<DevicePointer<GpuFloat>>,

    // Device memory referenced by `model`; kept here so it lives as long as
    // `model`.
    _d_phis: Option<DevicePointer<GpuFloat>>,
    _d_thetas: Option<DevicePointer<GpuFloat>>,
    _d_planes: Option<DevicePointer<Jones<GpuFloat>>>,
    _d_sph_coeffs: Option<DevicePointer<GpuComplex>>,
    _d_coeff_set_offsets: Option<DevicePointer<i64>>,
    _d_coeff_set_l_max: Option<DevicePointer<i32>>,
    _d_freq_to_coeff_set: Option<DevicePointer<i32>>,
}

// The raw pointers in `SkaGpuModel` are device pointers owned by this struct.
unsafe impl Send for SkaLowBeamGpu {}
unsafe impl Sync for SkaLowBeamGpu {}

fn to_gpu_float(v: &[f64]) -> Vec<GpuFloat> {
    v.iter().map(|&x| x as GpuFloat).collect()
}

impl SkaLowBeam {
    /// Prepare a GPU-capable device for beam-response computations given the
    /// frequencies and tiles (elements or stations) to be used.
    ///
    /// # Safety
    ///
    /// This function interfaces directly with the CUDA/HIP API. Rust errors
    /// attempt to catch problems but there are no guarantees.
    pub unsafe fn gpu_prepare(
        &self,
        freqs_hz: &[f64],
        tiles: SkaGpuTiles,
    ) -> Result<SkaLowBeamGpu, SkaBeamError> {
        SkaLowBeamGpu::new(self, freqs_hz, tiles)
    }
}

impl SkaLowBeamGpu {
    unsafe fn new(
        beam: &SkaLowBeam,
        freqs_hz: &[f64],
        tiles: SkaGpuTiles,
    ) -> Result<SkaLowBeamGpu, SkaBeamError> {
        let mut model = SkaGpuModel {
            model: SKA_MODEL_SKA_ANALYTIC,
            cos_taper: 0,
            phis: std::ptr::null(),
            num_phis: 0,
            thetas: std::ptr::null(),
            num_thetas: 0,
            planes: std::ptr::null(),
            l_max: 0,
            num_sph_elements: 0,
            sph_coeffs: std::ptr::null(),
            coeff_set_offsets: std::ptr::null(),
            coeff_set_l_max: std::ptr::null(),
            freq_to_coeff_set: std::ptr::null(),
        };
        let mut d_phis = None;
        let mut d_thetas = None;
        let mut d_planes = None;
        let mut d_sph_coeffs = None;
        let mut d_coeff_set_offsets = None;
        let mut d_coeff_set_l_max = None;
        let mut d_freq_to_coeff_set = None;

        match &beam.inner {
            SkaLowBeamInner::Analytic(b) => {
                model.model = SKA_MODEL_SKA_ANALYTIC;
                model.cos_taper = b.has_cos_taper() as _;
            }

            SkaLowBeamInner::EepGrid(b) => {
                model.model = SKA_MODEL_SKA_EEP_GRID;
                // Interpolate in frequency on the host, one plane per
                // frequency. This also checks that the frequencies are in
                // range.
                let mut planes: Vec<Jones<GpuFloat>> = vec![];
                for &f in freqs_hz {
                    let plane = b.freq_plane(f)?;
                    planes.extend(plane.iter().map(|j| {
                        Jones::from([
                            GpuComplex::new(j[0].re as _, j[0].im as _),
                            GpuComplex::new(j[1].re as _, j[1].im as _),
                            GpuComplex::new(j[2].re as _, j[2].im as _),
                            GpuComplex::new(j[3].re as _, j[3].im as _),
                        ])
                    }));
                }
                let phis = DevicePointer::copy_to_device(&to_gpu_float(b.get_phis()))?;
                let thetas = DevicePointer::copy_to_device(&to_gpu_float(b.get_thetas()))?;
                let planes = DevicePointer::copy_to_device(&planes)?;
                model.phis = phis.get();
                model.num_phis = b.get_phis().len().try_into().expect("fits in i32");
                model.thetas = thetas.get();
                model.num_thetas = b.get_thetas().len().try_into().expect("fits in i32");
                model.planes = planes.get() as *const _;
                d_phis = Some(phis);
                d_thetas = Some(thetas);
                d_planes = Some(planes);
            }

            SkaLowBeamInner::EepSph(b) => {
                model.model = SKA_MODEL_SKA_EEP_SPH;
                // Map each frequency to a (unique) coefficient file.
                let mut file_indices: Vec<usize> = vec![];
                let mut freq_to_set = vec![];
                for &f in freqs_hz {
                    let fi = b.closest_file(f);
                    let set = match file_indices.iter().position(|&x| x == fi) {
                        Some(s) => s,
                        None => {
                            file_indices.push(fi);
                            file_indices.len() - 1
                        }
                    };
                    freq_to_set.push(set as i32);
                }
                // Each coefficient set (file) is stored at its own degree;
                // files can be gigabytes each, so they're uploaded one at a
                // time (without caching them on the host, if they aren't
                // already).
                let num_elements = file_indices
                    .first()
                    .map(|&fi| b.file_num_elements(fi))
                    .unwrap_or(0);
                if file_indices
                    .iter()
                    .any(|&fi| b.file_num_elements(fi) != num_elements)
                {
                    return Err(SkaBeamError::GpuInconsistentElements);
                }
                let set_l_max: Vec<usize> =
                    file_indices.iter().map(|&fi| b.file_l_max(fi)).collect();
                let l_max = set_l_max.iter().copied().max().unwrap_or(0);
                if l_max > SKA_GPU_MAX_L_MAX {
                    return Err(SkaBeamError::GpuLMax {
                        got: l_max,
                        max: SKA_GPU_MAX_L_MAX,
                    });
                }
                let set_sizes: Vec<usize> = set_l_max
                    .iter()
                    .map(|&l| num_elements * l * (l + 2) * 4)
                    .collect();
                let mut offsets = Vec::with_capacity(set_sizes.len());
                let mut total = 0usize;
                for &size in &set_sizes {
                    offsets.push(total as i64);
                    total += size;
                }
                let mut d_coeffs =
                    DevicePointer::<GpuComplex>::malloc(total * std::mem::size_of::<GpuComplex>())?;
                for ((&fi, &offset), &size) in file_indices.iter().zip(&offsets).zip(&set_sizes) {
                    let c = match b.cached_coeffs(fi) {
                        Some(c) => c,
                        None => std::sync::Arc::new(b.read_coeffs(fi)?),
                    };
                    let set: Vec<GpuComplex> = c
                        .as_standard_layout()
                        .iter()
                        .map(|v| GpuComplex::new(v.re as _, v.im as _))
                        .collect();
                    debug_assert_eq!(set.len(), size);
                    d_coeffs.overwrite_at(offset as usize, &set)?;
                }
                let d_offsets = DevicePointer::copy_to_device(&offsets)?;
                let set_l_max_i32: Vec<i32> = set_l_max
                    .iter()
                    .map(|&l| l.try_into().expect("fits in i32"))
                    .collect();
                let d_set_l_max = DevicePointer::copy_to_device(&set_l_max_i32)?;
                let d_map = DevicePointer::copy_to_device(&freq_to_set)?;
                model.l_max = l_max as _;
                model.num_sph_elements = num_elements.try_into().expect("fits in i32");
                model.sph_coeffs = d_coeffs.get() as *const _;
                model.coeff_set_offsets = d_offsets.get();
                model.coeff_set_l_max = d_set_l_max.get();
                model.freq_to_coeff_set = d_map.get();
                d_sph_coeffs = Some(d_coeffs);
                d_coeff_set_offsets = Some(d_offsets);
                d_coeff_set_l_max = Some(d_set_l_max);
                d_freq_to_coeff_set = Some(d_map);
            }
        }

        let num_beam_elements = beam.num_elements();
        let (
            num_tiles,
            d_element_ids,
            num_station_elements,
            d_positions,
            d_weights,
            d_weights_y,
            d_rotations,
            d_frames,
        ) = match tiles {
            SkaGpuTiles::Elements(ids) => {
                if let Some(n) = num_beam_elements {
                    if let Some(&bad) = ids.iter().find(|&&id| id >= n) {
                        return Err(SkaBeamError::ElementOutOfRange {
                            got: bad,
                            num_elements: n,
                        });
                    }
                }
                let ids: Vec<i32> = ids.iter().map(|&i| i as i32).collect();
                (
                    ids.len(),
                    Some(DevicePointer::copy_to_device(&ids)?),
                    0,
                    None,
                    None,
                    None,
                    None,
                    None,
                )
            }
            SkaGpuTiles::Stations {
                positions_enu,
                weights,
                weights_y,
                rotations_rad,
                local_frames,
            } => {
                let (num_stations, num_elements, three) = positions_enu.dim();
                assert_eq!(
                    three, 3,
                    "positions_enu must have shape (stations, elements, 3)"
                );
                if let Some(n) = num_beam_elements {
                    if num_elements != n {
                        return Err(SkaBeamError::PositionsLength {
                            got: num_elements,
                            expected: n,
                        });
                    }
                }
                let check_weights = |w: ArrayView2<f64>| -> Result<Array2<f64>, SkaBeamError> {
                    if w.dim() != (num_stations, num_elements) {
                        return Err(SkaBeamError::WeightsLength {
                            got: w.len(),
                            expected: num_stations * num_elements,
                        });
                    }
                    if w.outer_iter().any(|w| w.sum() == 0.0) {
                        return Err(SkaBeamError::AllElementsFlagged);
                    }
                    Ok(w.to_owned())
                };
                let weights = match weights {
                    Some(w) => check_weights(w)?,
                    None => Array2::ones((num_stations, num_elements)),
                };
                let d_weights_y = match weights_y {
                    Some(w) => {
                        let w: Vec<GpuFloat> = check_weights(w)?.iter().map(|&x| x as _).collect();
                        Some(DevicePointer::copy_to_device(&w)?)
                    }
                    None => None,
                };
                let d_rotations = match rotations_rad {
                    Some(r) => {
                        if r.len() != num_stations {
                            return Err(SkaBeamError::RotationsLength {
                                got: r.len(),
                                expected: num_stations,
                            });
                        }
                        Some(DevicePointer::copy_to_device(&to_gpu_float(r))?)
                    }
                    None => None,
                };
                let d_frames = match local_frames {
                    Some(f) => {
                        if f.len() != num_stations {
                            return Err(SkaBeamError::LocalFramesLength {
                                got: f.len(),
                                expected: num_stations,
                            });
                        }
                        let flat: Vec<f64> = f.iter().flatten().flatten().copied().collect();
                        Some(DevicePointer::copy_to_device(&to_gpu_float(&flat))?)
                    }
                    None => None,
                };
                let positions: Vec<GpuFloat> = positions_enu.iter().map(|&x| x as _).collect();
                let weights: Vec<GpuFloat> = weights.iter().map(|&x| x as _).collect();
                (
                    num_stations,
                    None,
                    num_elements,
                    Some(DevicePointer::copy_to_device(&positions)?),
                    Some(DevicePointer::copy_to_device(&weights)?),
                    d_weights_y,
                    d_rotations,
                    d_frames,
                )
            }
        };

        Ok(SkaLowBeamGpu {
            model,
            num_freqs: freqs_hz.len().try_into().expect("fits in i32"),
            d_freqs: DevicePointer::copy_to_device(&to_gpu_float(freqs_hz))?,
            num_tiles: num_tiles.try_into().expect("fits in i32"),
            d_element_ids,
            num_station_elements: num_station_elements.try_into().expect("fits in i32"),
            d_positions,
            d_weights,
            d_weights_y,
            d_rotations,
            d_frames,
            _d_phis: d_phis,
            _d_thetas: d_thetas,
            _d_planes: d_planes,
            _d_sph_coeffs: d_sph_coeffs,
            _d_coeff_set_offsets: d_coeff_set_offsets,
            _d_coeff_set_l_max: d_coeff_set_l_max,
            _d_freq_to_coeff_set: d_freq_to_coeff_set,
        })
    }

    /// The number of tiles (elements or stations).
    pub fn get_num_tiles(&self) -> i32 {
        self.num_tiles
    }

    /// The number of frequencies.
    pub fn get_num_freqs(&self) -> i32 {
        self.num_freqs
    }

    /// Calculate beam-response Jones matrices into a pre-allocated device
    /// buffer with shape `(num_tiles, num_freqs, num_directions)`.
    /// `pointing_az_rad` and `pointing_za_rad` are only used for stations. If
    /// `d_latitude_rad` is not null, the parallactic-angle correction is
    /// applied.
    ///
    /// # Safety
    ///
    /// All pointers must be valid device pointers, and `d_results` must be big
    /// enough (see above).
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn calc_jones_device_pair_inner(
        &self,
        d_az_rad: *const GpuFloat,
        d_za_rad: *const GpuFloat,
        num_directions: i32,
        pointing_az_rad: GpuFloat,
        pointing_za_rad: GpuFloat,
        d_latitude_rad: *const GpuFloat,
        iau_order: bool,
        d_results: *mut std::ffi::c_void,
    ) -> Result<(), SkaBeamError> {
        if num_directions == 0 {
            return Ok(());
        }
        let error_message_ptr = gpu_ska_calc_jones(
            &self.model,
            d_az_rad,
            d_za_rad,
            num_directions,
            self.d_freqs.get(),
            self.num_freqs,
            self.num_tiles,
            self.d_element_ids
                .as_ref()
                .map(|d| d.get())
                .unwrap_or(std::ptr::null()),
            self.num_station_elements,
            self.d_positions
                .as_ref()
                .map(|d| d.get())
                .unwrap_or(std::ptr::null()),
            self.d_weights
                .as_ref()
                .map(|d| d.get())
                .unwrap_or(std::ptr::null()),
            self.d_weights_y
                .as_ref()
                .map(|d| d.get())
                .unwrap_or(std::ptr::null()),
            self.d_rotations
                .as_ref()
                .map(|d| d.get())
                .unwrap_or(std::ptr::null()),
            self.d_frames
                .as_ref()
                .map(|d| d.get())
                .unwrap_or(std::ptr::null()),
            pointing_az_rad,
            pointing_za_rad,
            d_latitude_rad,
            iau_order as _,
            d_results,
        );
        if error_message_ptr.is_null() {
            Ok(())
        } else {
            let error_message = CStr::from_ptr(error_message_ptr)
                .to_str()
                .unwrap_or("<cannot read GPU error string>");
            Err(SkaBeamError::Gpu(GpuError::Kernel {
                msg: format!("ska.h:gpu_ska_calc_jones failed with: {error_message}").into(),
                file: file!(),
                line: line!(),
            }))
        }
    }

    /// Calculate beam-response Jones matrices on the device and return a
    /// device pointer to them (shape `(num_tiles, num_freqs,
    /// num_directions)`).
    pub fn calc_jones_device_pair(
        &self,
        az_rad: &[GpuFloat],
        za_rad: &[GpuFloat],
        pointing: AzEl,
        latitude_rad: Option<f64>,
        iau_order: bool,
    ) -> Result<DevicePointer<Jones<GpuFloat>>, SkaBeamError> {
        let n = az_rad.len().min(za_rad.len());
        unsafe {
            let d_results = DevicePointer::malloc(
                self.num_tiles as usize
                    * self.num_freqs as usize
                    * n
                    * std::mem::size_of::<Jones<GpuFloat>>(),
            )?;
            let d_azs = DevicePointer::copy_to_device(&az_rad[..n])?;
            let d_zas = DevicePointer::copy_to_device(&za_rad[..n])?;
            let d_latitude = latitude_rad
                .map(|l| DevicePointer::copy_to_device(&[l as GpuFloat]))
                .transpose()?;
            self.calc_jones_device_pair_inner(
                d_azs.get(),
                d_zas.get(),
                n.try_into().expect("fits in i32"),
                pointing.az as _,
                pointing.za() as _,
                d_latitude
                    .as_ref()
                    .map(|d| d.get())
                    .unwrap_or(std::ptr::null()),
                iau_order,
                d_results.get_mut() as *mut std::ffi::c_void,
            )?;
            Ok(d_results)
        }
    }

    /// Calculate beam-response Jones matrices on the device and copy them to
    /// the host. The result has shape `(num_tiles, num_freqs,
    /// num_directions)`.
    pub fn calc_jones_pair(
        &self,
        az_rad: &[GpuFloat],
        za_rad: &[GpuFloat],
        pointing: AzEl,
        latitude_rad: Option<f64>,
        iau_order: bool,
    ) -> Result<Array3<Jones<GpuFloat>>, SkaBeamError> {
        let n = az_rad.len().min(za_rad.len());
        let d_results =
            self.calc_jones_device_pair(az_rad, za_rad, pointing, latitude_rad, iau_order)?;
        let mut results = Array3::from_elem(
            (self.num_tiles as usize, self.num_freqs as usize, n),
            Jones::default(),
        );
        unsafe {
            d_results.copy_from_device(results.as_slice_mut().expect("is contiguous"))?;
        }
        Ok(results)
    }

    /// Calculate beam-response Jones matrices on the device for [`AzEl`]
    /// directions and copy them to the host. See
    /// [`calc_jones_pair`](Self::calc_jones_pair).
    pub fn calc_jones(
        &self,
        azels: &[AzEl],
        pointing: AzEl,
        latitude_rad: Option<f64>,
        iau_order: bool,
    ) -> Result<Array3<Jones<GpuFloat>>, SkaBeamError> {
        let (az, za): (Vec<GpuFloat>, Vec<GpuFloat>) = azels
            .iter()
            .map(|a| (a.az as GpuFloat, a.za() as GpuFloat))
            .unzip();
        self.calc_jones_pair(&az, &za, pointing, latitude_rad, iau_order)
    }
}
