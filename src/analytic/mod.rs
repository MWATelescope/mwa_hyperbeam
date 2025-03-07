// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code for the analytic MWA beam.

mod error;
mod ffi;
#[cfg(any(feature = "cuda", feature = "hip"))]
mod gpu;
#[cfg(test)]
mod tests;

pub use error::AnalyticBeamError;

#[cfg(any(feature = "cuda", feature = "hip"))]
pub use gpu::AnalyticBeamGpu;

use std::f64::consts::{FRAC_PI_2, TAU};

use marlu::{c64, constants::VEL_C, rayon, AzEl, Jones};
use rayon::prelude::*;

use crate::{
    constants::{DELAY_STEP, MWA_DPL_HGT, MWA_DPL_HGT_RTS, MWA_DPL_SEP},
    direction::HorizCoord,
};

#[cfg(any(feature = "cuda", feature = "hip"))]
use ndarray::prelude::*;

/// Which analytic beam code are we emulating?
#[derive(Clone, Copy, Debug)]
pub enum AnalyticType {
    /// Behaviour derived from [mwa_pb](https://github.com/MWATelescope/mwa_pb).
    MwaPb,

    /// Behaviour derived from the RTS.
    Rts,
}

impl AnalyticType {
    /// Different analytic beam types use different MWA dipole heights by
    /// default. This method returns the default height given a analytic beam
    /// type.
    pub fn get_default_dipole_height(self) -> f64 {
        match self {
            AnalyticType::MwaPb => MWA_DPL_HGT,
            AnalyticType::Rts => MWA_DPL_HGT_RTS,
        }
    }
}

/// The struct used to calculate beam-response Jones matrices for the analytic
/// beam implementation.
pub struct AnalyticBeam {
    /// The height of the MWA dipoles we're simulating \[metres\].
    ///
    /// The RTS uses an old value, presumably derived from early MWA dipoles.
    /// The up-to-date value is 0.278m, and is used by default.
    dipole_height: f64,

    /// Which analytic beam code are we emulating?
    beam_type: AnalyticType,

    /// The number of bowties in a row of an MWA tile. Almost all MWA tiles
    /// have 4 bowties per row, for a total of 16 bowties. As of October 2023,
    /// the only exception is the CRAM tile, which has 8 bowties per row, for a
    /// total of 64 bowties.
    pub(crate) bowties_per_row: u8,
}

impl Default for AnalyticBeam {
    fn default() -> Self {
        let beam_type = AnalyticType::MwaPb;
        AnalyticBeam {
            dipole_height: beam_type.get_default_dipole_height(),
            beam_type,
            bowties_per_row: 4,
        }
    }
}

impl AnalyticBeam {
    /// Create a new [`AnalyticBeam`] struct using mwa_pb analytic beam code and
    /// the [default](MWA_DPL_HGT) MWA dipole height.
    pub fn new() -> AnalyticBeam {
        AnalyticBeam::default()
    }

    /// Create a new [`AnalyticBeam`] struct using RTS analytic beam code and
    /// the MWA dipole height from the [RTS](MWA_DPL_HGT_RTS).
    pub fn new_rts() -> AnalyticBeam {
        let beam_type = AnalyticType::Rts;
        AnalyticBeam {
            dipole_height: beam_type.get_default_dipole_height(),
            beam_type,
            bowties_per_row: 4,
        }
    }

    /// Create a new [`AnalyticBeam`] struct with custom behaviour, MWA
    /// dipole height and variable bowties per row (you want this to be 4 for
    /// normal MWA tiles, 8 for the CRAM).
    pub fn new_custom(
        beam_type: AnalyticType,
        dipole_height_metres: f64,
        bowties_per_row: u8,
    ) -> AnalyticBeam {
        if bowties_per_row == 0 {
            panic!("bowties_per_row was 0, why would you do that?");
        }
        // We want the number of bowties to fit in a u8; complain if there are
        // too many damn bowties.
        if bowties_per_row >= 16 {
            panic!("bowties_per_row is restricted to be less than 16");
        }
        AnalyticBeam {
            dipole_height: dipole_height_metres,
            beam_type,
            bowties_per_row,
        }
    }

    /// Calculate the beam-response Jones matrix for a given direction, pointing
    /// and latitude.
    ///
    /// `delays` and `amps` apply to each dipole in an MWA tile in the M&C
    /// order; see
    /// <https://wiki.mwatelescope.org/pages/viewpage.action?pageId=48005139>.
    /// `delays` *must* have `bowties_per_row * bowties_per_row` elements (which
    /// was declared when `AnalyticBeam` was created), whereas `amps` can have
    /// this number or double elements; if the former is given, then these map
    /// 1:1 with bowties. If double are given, then the *smallest* of the two
    /// amps corresponding to a bowtie's dipoles is used.
    ///
    /// e.g. A normal MWA tile has 4 bowties per row. `delays` must then have
    /// 16 elements, and `amps` can have 16 or 32 elements. A CRAM tile has 8
    /// bowties per row; `delays` must have 64 elements, and `amps` can have 64
    /// or 128 elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::f64::consts::FRAC_PI_2;
    ///
    /// use marlu::{AzEl, Jones, constants::MWA_LAT_RAD};
    /// use mwa_hyperbeam::analytic::AnalyticBeam;
    ///
    /// let direction = AzEl::from_radians(0.4, 0.7);
    /// let freq_hz = 150e6 as u32;
    /// let delays = vec![0; 16];
    /// let amps = vec![1.0; 16];
    /// let latitude_rad = MWA_LAT_RAD;
    /// let norm_to_zenith = true;
    /// let beam = AnalyticBeam::new_rts();
    /// let result = beam.calc_jones(direction, freq_hz, &delays, &amps, latitude_rad, norm_to_zenith).unwrap();
    ///
    /// // Floats can be used too.
    /// let direction = (0.4, FRAC_PI_2 - 0.7);
    /// let result2 = beam.calc_jones(direction, freq_hz, &delays, &amps, latitude_rad, norm_to_zenith).unwrap();
    /// assert_eq!(result, result2);
    /// ```
    pub fn calc_jones<C: HorizCoord>(
        &self,
        direction: C,
        freq_hz: u32,
        delays: &[u32],
        amps: &[f64],
        latitude_rad: f64,
        norm_to_zenith: bool,
    ) -> Result<Jones<f64>, AnalyticBeamError> {
        let az_rad = direction.get_az();
        let za_rad = direction.get_za();
        if za_rad > FRAC_PI_2 {
            return Err(AnalyticBeamError::BelowHorizon { za: za_rad });
        }
        let num_bowties = usize::from(self.bowties_per_row * self.bowties_per_row);
        if delays.len() != num_bowties {
            return Err(AnalyticBeamError::IncorrectDelaysLength {
                got: delays.len(),
                expected: num_bowties,
            });
        }
        if amps.len() != num_bowties && amps.len() != num_bowties * 2 {
            return Err(AnalyticBeamError::IncorrectAmpsLength {
                got: amps.len(),
                expected1: num_bowties,
                expected2: num_bowties * 2,
            });
        }

        let amps = fix_amps(amps, delays);
        let (amps, delays) = if matches!(self.beam_type, AnalyticType::Rts) {
            reorder_to_rts(&amps, delays)
        } else {
            (amps.to_vec(), delay_ints_to_floats(delays))
        };

        let lambda_m = VEL_C / freq_hz as f64;
        let (s_lat, c_lat) = latitude_rad.sin_cos();
        let jones = self.calc_jones_inner(
            az_rad,
            za_rad,
            lambda_m,
            latitude_rad,
            s_lat,
            c_lat,
            &delays,
            &amps,
            norm_to_zenith,
        );
        Ok(jones)
    }

    /// Calculate the beam-response Jones matrices for many directions given a
    /// pointing and latitude. This is basically a wrapper around `calc_jones`
    /// that efficiently calculates the Jones matrices in parallel. The number
    /// of parallel threads used can be controlled by setting
    /// `RAYON_NUM_THREADS`.
    ///
    /// `delays` and `amps` apply to each dipole in an MWA tile in the M&C
    /// order; see
    /// <https://wiki.mwatelescope.org/pages/viewpage.action?pageId=48005139>.
    /// `delays` *must* have `bowties_per_row * bowties_per_row` elements (which
    /// was declared when `AnalyticBeam` was created), whereas `amps` can have
    /// this number or double elements; if the former is given, then these map
    /// 1:1 with bowties. If double are given, then the *smallest* of the two
    /// amps corresponding to a bowtie's dipoles is used.
    ///
    /// e.g. A normal MWA tile has 4 bowties per row. `delays` must then have
    /// 16 elements, and `amps` can have 16 or 32 elements. A CRAM tile has 8
    /// bowties per row; `delays` must have 64 elements, and `amps` can have 64
    /// or 128 elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::f64::consts::FRAC_PI_2;
    ///
    /// use marlu::{AzEl, Jones, constants::MWA_LAT_RAD};
    /// use mwa_hyperbeam::analytic::AnalyticBeam;
    ///
    /// let directions = vec![AzEl::from_radians(0.4, 0.7), AzEl::from_radians(0.5, 0.8)];
    /// let freq_hz = 150e6 as u32;
    /// let delays = vec![0; 16];
    /// let amps = vec![1.0; 16];
    /// let latitude_rad = MWA_LAT_RAD;
    /// let norm_to_zenith = true;
    /// let beam = AnalyticBeam::new_rts();
    /// let results = beam.calc_jones_array(directions, freq_hz, &delays, &amps, latitude_rad, norm_to_zenith).unwrap();
    ///
    /// // Floats can be used, but these need to be grouped by azimuth and ZA.
    /// let azimuths = vec![0.4, 0.5];
    /// let zenith_angles = vec![FRAC_PI_2 - 0.7, FRAC_PI_2 - 0.8];
    /// let results2 = beam.calc_jones_array((&azimuths, &zenith_angles), freq_hz, &delays, &amps, latitude_rad, norm_to_zenith).unwrap();
    /// assert_eq!(results, results2);
    /// ```
    pub fn calc_jones_array<C, I, I2>(
        &self,
        directions: I,
        freq_hz: u32,
        delays: &[u32],
        amps: &[f64],
        latitude_rad: f64,
        norm_to_zenith: bool,
    ) -> Result<Vec<Jones<f64>>, AnalyticBeamError>
    where
        C: HorizCoord,
        I: IntoParallelIterator<Iter = I2>,
        I2: IndexedParallelIterator<Item = C>,
    {
        let directions = directions.into_par_iter();
        let mut results = vec![Jones::default(); directions.len()];
        self.calc_jones_array_inner(
            directions,
            freq_hz,
            delays,
            amps,
            latitude_rad,
            norm_to_zenith,
            &mut results,
        )?;
        Ok(results)
    }

    /// Calculate the beam-response Jones matrices for many directions given a
    /// pointing and latitude. This is the same as `calc_jones_array` but uses
    /// pre-allocated memory. `results` should have a length equal to or greater
    /// than `directions`.
    ///
    /// `delays` and `amps` apply to each dipole in an MWA tile in the M&C
    /// order; see
    /// <https://wiki.mwatelescope.org/pages/viewpage.action?pageId=48005139>.
    /// `delays` *must* have `bowties_per_row * bowties_per_row` elements (which
    /// was declared when `AnalyticBeam` was created), whereas `amps` can have
    /// this number or double elements; if the former is given, then these map
    /// 1:1 with bowties. If double are given, then the *smallest* of the two
    /// amps corresponding to a bowtie's dipoles is used.
    ///
    /// e.g. A normal MWA tile has 4 bowties per row. `delays` must then have
    /// 16 elements, and `amps` can have 16 or 32 elements. A CRAM tile has 8
    /// bowties per row; `delays` must have 64 elements, and `amps` can have 64
    /// or 128 elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::f64::consts::FRAC_PI_2;
    ///
    /// use marlu::{AzEl, Jones, constants::MWA_LAT_RAD};
    /// use mwa_hyperbeam::analytic::AnalyticBeam;
    ///
    /// let directions = vec![AzEl::from_radians(0.4, 0.7), AzEl::from_radians(0.5, 0.8)];
    /// let freq_hz = 150e6 as u32;
    /// let delays = vec![0; 16];
    /// let amps = vec![1.0; 16];
    /// let latitude_rad = MWA_LAT_RAD;
    /// let norm_to_zenith = true;
    /// // Make the results buffer the right size, fill with default values which will be overwritten
    /// let mut results = vec![Jones::default(); directions.len()];
    /// assert_eq!(results[0][0].re, 0.0);
    /// let beam = AnalyticBeam::new_rts();
    /// beam.calc_jones_array_inner(&directions, freq_hz, &delays, &amps, latitude_rad, norm_to_zenith, &mut results).unwrap();
    /// assert_ne!(results[0][0].re, 0.0);
    ///
    /// // Floats can be used, but these need to be grouped by azimuth and ZA.
    /// let azimuths = vec![0.4, 0.5];
    /// let zenith_angles = vec![FRAC_PI_2 - 0.7, FRAC_PI_2 - 0.8];
    /// let mut results2 = vec![Jones::default(); directions.len()];
    /// beam.calc_jones_array_inner((&azimuths, &zenith_angles), freq_hz, &delays, &amps, latitude_rad, norm_to_zenith, &mut results2).unwrap();
    /// assert_eq!(results, results2);
    /// ```
    #[allow(clippy::too_many_arguments)]
    pub fn calc_jones_array_inner<C, I, I2>(
        &self,
        directions: I,
        freq_hz: u32,
        delays: &[u32],
        amps: &[f64],
        latitude_rad: f64,
        norm_to_zenith: bool,
        results: &mut [Jones<f64>],
    ) -> Result<(), AnalyticBeamError>
    where
        C: HorizCoord,
        I: IntoParallelIterator<Iter = I2>,
        I2: IndexedParallelIterator<Item = C>,
    {
        let num_bowties = usize::from(self.bowties_per_row * self.bowties_per_row);
        if delays.len() != num_bowties {
            return Err(AnalyticBeamError::IncorrectDelaysLength {
                got: delays.len(),
                expected: num_bowties,
            });
        }
        if amps.len() != num_bowties && amps.len() != num_bowties * 2 {
            return Err(AnalyticBeamError::IncorrectAmpsLength {
                got: amps.len(),
                expected1: num_bowties,
                expected2: num_bowties * 2,
            });
        }

        let amps = fix_amps(amps, delays);
        let (amps, delays) = if matches!(self.beam_type, AnalyticType::Rts) {
            reorder_to_rts(&amps, delays)
        } else {
            (amps.to_vec(), delay_ints_to_floats(delays))
        };

        let lambda_m = VEL_C / freq_hz as f64;
        let (s_lat, c_lat) = latitude_rad.sin_cos();
        directions
            .into_par_iter()
            .zip(results.par_iter_mut())
            .try_for_each(|(dir, result)| {
                let az_rad = dir.get_az();
                let za_rad = dir.get_za();
                if za_rad > FRAC_PI_2 {
                    return Err(AnalyticBeamError::BelowHorizon { za: za_rad });
                }

                let j = self.calc_jones_inner(
                    az_rad,
                    za_rad,
                    lambda_m,
                    latitude_rad,
                    s_lat,
                    c_lat,
                    &delays,
                    &amps,
                    norm_to_zenith,
                );
                *result = j;

                Ok(())
            })
    }

    /// Helper function.
    // The code here was derived with the help of primary_beam.py in mwa_pb,
    // commit 8619797, and Jack's WODEN.
    #[allow(clippy::too_many_arguments)]
    fn calc_jones_inner(
        &self,
        az_rad: f64,
        za_rad: f64,
        lambda_m: f64,
        latitude_rad: f64,
        sin_latitude: f64,
        cos_latitude: f64,
        delays: &[f64],
        amps: &[f64],
        norm_to_zenith: bool,
    ) -> Jones<f64> {
        // The following logic could probably be significantly cleaned up, but
        // I'm out of time.

        let (s_az, c_az) = az_rad.sin_cos();
        let (s_za, c_za) = za_rad.sin_cos();

        let mut jones = match self.beam_type {
            AnalyticType::MwaPb => Jones::from([
                c64::new(c_za * s_az, 0.0),
                c64::new(c_az, 0.0),
                c64::new(c_za * c_az, 0.0),
                c64::new(-s_az, 0.0),
            ]),
            AnalyticType::Rts => {
                let hadec = AzEl::from_radians(az_rad, FRAC_PI_2 - za_rad).to_hadec(latitude_rad);
                let (s_ha, c_ha) = hadec.ha.sin_cos();
                let (s_dec, c_dec) = hadec.dec.sin_cos();

                Jones::from([
                    c64::new(cos_latitude * c_dec + sin_latitude * s_dec * c_ha, 0.0),
                    c64::new(-sin_latitude * s_ha, 0.0),
                    c64::new(s_dec * s_ha, 0.0),
                    c64::new(c_ha, 0.0),
                ])
            }
        };

        let proj_e = s_za * s_az;
        let proj_n = s_za * c_az;
        // The RTS code uses proj_z as below, but dip_z is always set to 0.0, so
        // we don't actually need proj_z. lmao
        // let proj_z = c_za;

        let multiplier = -TAU / lambda_m;

        // Loop over each dipole.
        let mut array_factor = c64::new(0.0, 0.0);
        for (k, (&delay, &amp)) in delays.iter().zip(amps.iter()).enumerate() {
            let col = k % usize::from(self.bowties_per_row);
            let row = k / usize::from(self.bowties_per_row);
            let (dip_e, dip_n) = match self.beam_type {
                AnalyticType::MwaPb => (
                    (col as f64 - 1.5) * MWA_DPL_SEP,
                    (row as f64 - 1.5) * MWA_DPL_SEP,
                ),
                AnalyticType::Rts => (
                    (row as f64 - 1.5) * MWA_DPL_SEP,
                    (col as f64 - 1.5) * MWA_DPL_SEP,
                ),
            };
            // let dip_z = 0.0;

            let phase = match self.beam_type {
                AnalyticType::MwaPb => {
                    -multiplier
                        * (dip_e * proj_e
                         + dip_n * proj_n
                         // + dip_z * proj_z
                         - delay)
                }
                AnalyticType::Rts => {
                    multiplier
                        * (dip_e * proj_e
                         + dip_n * proj_n
                         // + dip_z * proj_z
                         - delay)
                }
            };
            let (s_phase, c_phase) = phase.sin_cos();
            array_factor += amp * c64::new(c_phase, s_phase);
        }

        let mut ground_plane = 2.0 * (TAU * self.dipole_height / lambda_m * c_za).sin()
            / usize::from(self.bowties_per_row).pow(2) as f64;
        if norm_to_zenith {
            ground_plane /= 2.0 * (TAU * self.dipole_height / lambda_m).sin();
        }

        jones[0] *= ground_plane * array_factor;
        jones[1] *= ground_plane * array_factor;
        jones[2] *= ground_plane * array_factor;
        jones[3] *= ground_plane * array_factor;

        // The RTS deliberately sets the imaginary parts to 0.
        if matches!(self.beam_type, AnalyticType::Rts) {
            for j in jones.iter_mut() {
                *j = c64::new(j.re, 0.0);
            }
        }

        jones
    }

    /// Prepare a compute-capable GPU device for beam-response computations
    /// given the delays and amps to be used. The resulting object takes
    /// directions and frequencies to compute the beam responses on the device.
    ///
    /// `delays_array` and `amps_array` must have the same number of rows;
    /// these correspond to tile configurations (i.e. each tile is allowed
    /// to have distinct delays and amps). The number of elements per row of
    /// `delays_array` and `amps_array` have the same restrictions as `delays`
    /// and `amps` in `calc_jones`.
    ///
    /// The code will automatically de-duplicate tile configurations so that no
    /// redundant calculations are done.
    ///
    /// # Safety
    ///
    /// This function interfaces directly with the CUDA/HIP API. Rust errors
    /// attempt to catch problems but there are no guarantees.
    #[cfg(any(feature = "cuda", feature = "hip"))]
    pub unsafe fn gpu_prepare(
        &self,
        delays: ArrayView2<u32>,
        amps: ArrayView2<f64>,
    ) -> Result<gpu::AnalyticBeamGpu, AnalyticBeamError> {
        // This function is deliberately kept thin to keep the focus of this
        // module on the CPU code.
        gpu::AnalyticBeamGpu::new(self, delays, amps)
    }
}

/// Ensure that any delays of 32 have an amplitude (dipole gain) of 0. The
/// results are bad otherwise! Also potentially halve the number of amps (e.g.
/// if 32 are given for a 16-bowtie tile, yield 16); we use the smaller of the
/// two gains associated with a bowtie.
fn fix_amps(amps: &[f64], delays: &[u32]) -> Vec<f64> {
    // The lengths of `amps` and `delays` should be checked before calling this
    // functions; the asserts are a last resort guard.
    assert!(amps.len() == delays.len() || amps.len() == delays.len() * 2);

    let mut fixed_amps = vec![0.0; delays.len()];
    fixed_amps
        .iter_mut()
        .zip(amps.iter())
        .zip(delays.iter())
        .for_each(|((fixed, &amp), &delay)| *fixed = if delay == 32 { 0.0 } else { amp });
    if amps.len() == delays.len() * 2 {
        fixed_amps
            .iter_mut()
            .zip(amps.iter().skip(delays.len()))
            .for_each(|(fixed, &amp)| {
                *fixed = fixed.min(amp);
            });
    }
    fixed_amps
}

/// The RTS doesn't use the M&C order. This function takes in the M&C-ordered
/// amps and delays, and returns RTS-ordered amps and delays. It also does
/// extra... things.
// Several thousand upside down emojis.
fn reorder_to_rts(amps: &[f64], delays: &[u32]) -> (Vec<f64>, Vec<f64>) {
    // Assume that the number of delays is the number of bowties.
    let num_bowties = delays.len();
    // Get the number of bowties per row from the number of bowties. This
    // assumes that the number is a perfect square.
    let bowties_per_row = (num_bowties as f64).sqrt().round() as usize;
    assert_eq!(bowties_per_row * bowties_per_row, num_bowties);

    let mut indices = Vec::with_capacity(num_bowties);
    for i_col in 0..bowties_per_row {
        for i_row in (0..bowties_per_row).rev() {
            indices.push(i_row * bowties_per_row + i_col);
        }
    }

    // Convert to "RTS order".
    let mut rts_amps = vec![0.0; num_bowties];
    let mut rts_delays = vec![0.0; num_bowties];
    indices
        .into_iter()
        .zip(rts_amps.iter_mut())
        .zip(rts_delays.iter_mut())
        .for_each(|((i, rts_amp), rts_delay)| {
            *rts_amp = amps[i];
            *rts_delay = f64::from(delays[i]);
        });

    // Do this crazy stuff.
    let delay_0 = rts_delays.iter().sum::<f64>() * VEL_C * DELAY_STEP / num_bowties as f64;
    rts_delays.iter_mut().for_each(|d| {
        *d = *d * VEL_C * DELAY_STEP - delay_0;
    });
    (rts_amps, rts_delays)
}

fn delay_ints_to_floats(delays: &[u32]) -> Vec<f64> {
    delays
        .iter()
        .copied()
        .map(|d| d as f64 * VEL_C * DELAY_STEP)
        .collect()
}
