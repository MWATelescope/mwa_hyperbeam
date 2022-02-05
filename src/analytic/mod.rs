// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code for the analytic MWA beam.
//!
//! While 32 amps are accepted for each dipole, only 1 amp is used per bowtie
//! (of which there are 16). If 32 amps are given, then the *smallest* of the
//! two amps corresponding to a bowtie's dipoles is used.

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

use crate::constants::{DELAY_STEP, MWA_DPL_SEP, NUM_DIPOLES};

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
            AnalyticType::MwaPb => 0.278,
            AnalyticType::Rts => 0.30,
        }
    }
}

/// The main struct to be used for calculating analytic pointings.
///
/// Methods can accept 32 amps for each dipole, only 1 amp is used per bowtie
/// (of which there are 16). If 32 amps are given, then the *smallest* of the
/// two amps corresponding to a bowtie's dipoles is used.
pub struct AnalyticBeam {
    /// The height of the MWA dipoles we're simulating \[metres\].
    ///
    /// The RTS uses an old value, presumably derived from early MWA dipoles.
    /// The up-to-date value is 0.278m, and is used by default.
    dipole_height: f64,

    /// Which analytic beam code are we emulating?
    beam_type: AnalyticType,
}

impl Default for AnalyticBeam {
    fn default() -> Self {
        let beam_type = AnalyticType::MwaPb;
        AnalyticBeam {
            dipole_height: beam_type.get_default_dipole_height(),
            beam_type,
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
        }
    }

    /// Create a new [`AnalyticBeam`] struct with custom behaviour and MWA
    /// dipole height.
    pub fn new_custom(beam_type: AnalyticType, dipole_height_metres: f64) -> AnalyticBeam {
        AnalyticBeam {
            dipole_height: dipole_height_metres,
            beam_type,
        }
    }

    /// Calculate the beam-response Jones matrix for a given direction, pointing
    /// and latitude.
    ///
    /// `delays` and `amps` apply to each dipole in an MWA tile in the M&C
    /// order; see
    /// <https://wiki.mwatelescope.org/pages/viewpage.action?pageId=48005139>.
    /// `delays` *must* have 16 elements, whereas `amps` can have 16 or 32
    /// elements; if 16 are given, then these map 1:1 with dipoles. If 32 are
    /// given, then the *smallest* of the two amps corresponding to a bowtie's
    /// dipoles is used.
    pub fn calc_jones(
        &self,
        azel: AzEl,
        freq_hz: u32,
        delays: &[u32],
        amps: &[f64],
        latitude_rad: f64,
        norm_to_zenith: bool,
    ) -> Result<Jones<f64>, AnalyticBeamError> {
        self.calc_jones_pair(
            azel.az,
            azel.za(),
            freq_hz,
            delays,
            amps,
            latitude_rad,
            norm_to_zenith,
        )
    }

    /// Calculate the beam-response Jones matrix for a given direction and
    /// pointing.
    ///
    /// `delays` and `amps` apply to each dipole in an MWA tile in the M&C
    /// order; see
    /// <https://wiki.mwatelescope.org/pages/viewpage.action?pageId=48005139>.
    /// `delays` *must* have 16 elements, whereas `amps` can have 16 or 32
    /// elements; if 16 are given, then these map 1:1 with dipoles. If 32 are
    /// given, then the *smallest* of the two amps corresponding to a bowtie's
    /// dipoles is used.
    #[allow(clippy::too_many_arguments)]
    pub fn calc_jones_pair(
        &self,
        az_rad: f64,
        za_rad: f64,
        freq_hz: u32,
        delays: &[u32],
        amps: &[f64],
        latitude_rad: f64,
        norm_to_zenith: bool,
    ) -> Result<Jones<f64>, AnalyticBeamError> {
        if za_rad > FRAC_PI_2 {
            return Err(AnalyticBeamError::BelowHorizon { za: za_rad });
        }
        let delays: &[u32; 16] = delays
            .try_into()
            .map_err(|_| AnalyticBeamError::IncorrectDelaysLength(delays.len()))?;
        if !(amps.len() == 16 || amps.len() == 32) {
            return Err(AnalyticBeamError::IncorrectAmpsLength(amps.len()));
        }

        let amps = fix_amps(amps, delays);
        let (amps, delays) = if matches!(self.beam_type, AnalyticType::Rts) {
            reorder_to_rts(&amps, delays)
        } else {
            (amps, delay_ints_to_floats(delays))
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

    /// Calculate the beam-response Jones matrices for many directions
    /// given a pointing and latitude. This is basically a wrapper around
    /// `calc_jones` that efficiently calculates the Jones matrices in
    /// parallel. The number of parallel threads used can be controlled by
    /// setting `RAYON_NUM_THREADS`.
    ///
    /// `delays` and `amps` apply to each dipole in an MWA tile in the M&C
    /// order; see
    /// <https://wiki.mwatelescope.org/pages/viewpage.action?pageId=48005139>.
    /// `delays` *must* have 16 elements, whereas `amps` can have 16 or 32
    /// elements; if 16 are given, then these map 1:1 with dipoles. If 32 are
    /// given, then the *smallest* of the two amps corresponding to a bowtie's
    /// dipoles is used.
    pub fn calc_jones_array(
        &self,
        azels: &[AzEl],
        freq_hz: u32,
        delays: &[u32],
        amps: &[f64],
        latitude_rad: f64,
        norm_to_zenith: bool,
    ) -> Result<Vec<Jones<f64>>, AnalyticBeamError> {
        let mut results = vec![Jones::default(); azels.len()];
        self.calc_jones_array_inner(
            azels,
            freq_hz,
            delays,
            amps,
            latitude_rad,
            norm_to_zenith,
            &mut results,
        )?;
        Ok(results)
    }

    /// Calculate the Jones matrices for many directions given a pointing and
    /// latitude. This is the same as `calc_jones_array` but uses pre-allocated
    /// memory.
    ///
    /// `delays` and `amps` apply to each dipole in an MWA tile in the M&C
    /// order; see
    /// <https://wiki.mwatelescope.org/pages/viewpage.action?pageId=48005139>.
    /// `delays` *must* have 16 elements, whereas `amps` can have 16 or 32
    /// elements; if 16 are given, then these map 1:1 with dipoles. If 32 are
    /// given, then the *smallest* of the two amps corresponding to a bowtie's
    /// dipoles is used.
    #[allow(clippy::too_many_arguments)]
    pub fn calc_jones_array_inner(
        &self,
        azels: &[AzEl],
        freq_hz: u32,
        delays: &[u32],
        amps: &[f64],
        latitude_rad: f64,
        norm_to_zenith: bool,
        results: &mut [Jones<f64>],
    ) -> Result<(), AnalyticBeamError> {
        for azel in azels {
            let za = azel.za();
            if za > FRAC_PI_2 {
                return Err(AnalyticBeamError::BelowHorizon { za });
            }
        }
        let delays: &[u32; 16] = delays
            .try_into()
            .map_err(|_| AnalyticBeamError::IncorrectDelaysLength(delays.len()))?;
        if !(amps.len() == 16 || amps.len() == 32) {
            return Err(AnalyticBeamError::IncorrectAmpsLength(amps.len()));
        }

        let amps = fix_amps(amps, delays);
        let (amps, delays) = if matches!(self.beam_type, AnalyticType::Rts) {
            reorder_to_rts(&amps, delays)
        } else {
            (amps, delay_ints_to_floats(delays))
        };

        let lambda_m = VEL_C / freq_hz as f64;
        let (s_lat, c_lat) = latitude_rad.sin_cos();
        azels
            .par_iter()
            .zip(results.par_iter_mut())
            .try_for_each(|(&azel, result)| {
                if azel.za() > FRAC_PI_2 {
                    return Err(AnalyticBeamError::BelowHorizon { za: azel.za() });
                }

                let j = self.calc_jones_inner(
                    azel.az,
                    azel.za(),
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

    /// Calculate the beam-response Jones matrices for many directions given a
    /// pointing. This is basically a wrapper around `calc_jones` that
    /// efficiently calculates the Jones matrices in parallel. The number of
    /// parallel threads used can be controlled by setting `RAYON_NUM_THREADS`.
    ///
    /// `delays` and `amps` apply to each dipole in an MWA tile in the M&C
    /// order; see
    /// <https://wiki.mwatelescope.org/pages/viewpage.action?pageId=48005139>.
    /// `delays` *must* have 16 elements, whereas `amps` can have 16 or 32
    /// elements; if 16 are given, then these map 1:1 with dipoles. If 32 are
    /// given, then the *smallest* of the two amps corresponding to a bowtie's
    /// dipoles is used.
    #[allow(clippy::too_many_arguments)]
    pub fn calc_jones_array_pair(
        &self,
        az_rad: &[f64],
        za_rad: &[f64],
        freq_hz: u32,
        delays: &[u32],
        amps: &[f64],
        latitude_rad: f64,
        norm_to_zenith: bool,
    ) -> Result<Vec<Jones<f64>>, AnalyticBeamError> {
        for &za in za_rad {
            if za > FRAC_PI_2 {
                return Err(AnalyticBeamError::BelowHorizon { za });
            }
        }
        let delays: &[u32; 16] = delays
            .try_into()
            .map_err(|_| AnalyticBeamError::IncorrectDelaysLength(delays.len()))?;
        if !(amps.len() == 16 || amps.len() == 32) {
            return Err(AnalyticBeamError::IncorrectAmpsLength(amps.len()));
        }

        let amps = fix_amps(amps, delays);
        let (amps, delays) = if matches!(self.beam_type, AnalyticType::Rts) {
            reorder_to_rts(&amps, delays)
        } else {
            (amps, delay_ints_to_floats(delays))
        };

        let lambda_m = VEL_C / freq_hz as f64;
        let (s_lat, c_lat) = latitude_rad.sin_cos();
        let out = az_rad
            .par_iter()
            .zip(za_rad.par_iter())
            .map(|(&az, &za)| {
                self.calc_jones_inner(
                    az,
                    za,
                    lambda_m,
                    latitude_rad,
                    s_lat,
                    c_lat,
                    &delays,
                    &amps,
                    norm_to_zenith,
                )
            })
            .collect();
        Ok(out)
    }

    /// Calculate the Jones matrices for many directions given a pointing. This
    /// is the same as `calc_jones_array_pair` but uses pre-allocated memory.
    ///
    /// `delays` and `amps` apply to each dipole in an MWA tile in the M&C
    /// order; see
    /// <https://wiki.mwatelescope.org/pages/viewpage.action?pageId=48005139>.
    /// `delays` *must* have 16 elements, whereas `amps` can have 16 or 32
    /// elements; if 16 are given, then these map 1:1 with dipoles. If 32 are
    /// given, then the *smallest* of the two amps corresponding to a bowtie's
    /// dipoles is used.
    #[allow(clippy::too_many_arguments)]
    pub fn calc_jones_array_pair_inner(
        &self,
        az_rad: &[f64],
        za_rad: &[f64],
        freq_hz: u32,
        delays: &[u32],
        amps: &[f64],
        latitude_rad: f64,
        norm_to_zenith: bool,
        results: &mut [Jones<f64>],
    ) -> Result<(), AnalyticBeamError> {
        for &za in za_rad {
            if za > FRAC_PI_2 {
                return Err(AnalyticBeamError::BelowHorizon { za });
            }
        }
        let delays: &[u32; 16] = delays
            .try_into()
            .map_err(|_| AnalyticBeamError::IncorrectDelaysLength(delays.len()))?;
        if !(amps.len() == 16 || amps.len() == 32) {
            return Err(AnalyticBeamError::IncorrectAmpsLength(amps.len()));
        }

        let amps = fix_amps(amps, delays);
        let (amps, delays) = if matches!(self.beam_type, AnalyticType::Rts) {
            reorder_to_rts(&amps, delays)
        } else {
            (amps, delay_ints_to_floats(delays))
        };

        let lambda_m = VEL_C / freq_hz as f64;
        let (s_lat, c_lat) = latitude_rad.sin_cos();
        az_rad
            .par_iter()
            .zip(za_rad.par_iter())
            .zip(results.par_iter_mut())
            .try_for_each(|((&az, &za), result)| {
                if za > FRAC_PI_2 {
                    return Err(AnalyticBeamError::BelowHorizon { za });
                }

                let j = self.calc_jones_inner(
                    az,
                    za,
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
        delays: &[f64; 16],
        amps: &[f64; 16],
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
            let col = k % 4;
            let row = k / 4;
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

        let mut ground_plane =
            2.0 * (TAU * self.dipole_height / lambda_m * c_za).sin() / NUM_DIPOLES as f64;
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
    /// `delays_array` and `amps_array` must have the same number of rows; these
    /// correspond to tile configurations (i.e. each tile is allowed to have
    /// distinct delays and amps). `delays_array` must have 16 elements per row,
    /// but `amps_array` can have 16 or 32 elements per row (see `calc_jones`
    /// for an explanation).
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
/// results are bad otherwise! Also convert 32 amps to 16; we use the smaller of
/// the two gains associated with a bowtie.
fn fix_amps(amps: &[f64], delays: &[u32]) -> [f64; 16] {
    // The lengths of `amps` and `delays` should be checked before calling this
    // functions; the asserts are a last resort guard.
    assert_eq!(delays.len(), 16);
    assert!(amps.len() == 16 || amps.len() == 32);

    let mut fixed_amps: [f64; 16] = [0.0; 16];
    fixed_amps
        .iter_mut()
        .zip(amps)
        .zip(amps.iter().cycle().skip(16))
        .zip(delays.iter().cycle())
        .for_each(|(((fixed, &x_amp), &y_amp), &delay)| {
            *fixed = if delay == 32 { 0.0 } else { x_amp.min(y_amp) };
        });
    fixed_amps
}

/// The RTS doesn't use the M&C order. This function takes in the M&C-ordered
/// amps and delays, and returns RTS-ordered amps and delays. It also does
/// extra... things.
// Several thousand upside down emojis.
fn reorder_to_rts(amps: &[f64; 16], delays: &[u32; 16]) -> ([f64; 16], [f64; 16]) {
    let indices = [12, 8, 4, 0, 13, 9, 5, 1, 14, 10, 6, 2, 15, 11, 7, 3];

    // Convert to "RTS order".
    let mut rts_amps = [0.0; 16];
    let mut rts_delays = [0.0; 16];
    indices
        .into_iter()
        .zip(rts_amps.iter_mut())
        .zip(rts_delays.iter_mut())
        .for_each(|((i, rts_amp), rts_delay)| {
            *rts_amp = amps[i];
            *rts_delay = f64::from(delays[i]);
        });

    // Do this crazy stuff.
    let delay_0 = rts_delays.iter().sum::<f64>() * VEL_C * DELAY_STEP / NUM_DIPOLES as f64;
    rts_delays.iter_mut().for_each(|d| {
        *d = *d * VEL_C * DELAY_STEP - delay_0;
    });
    (rts_amps, rts_delays)
}

fn delay_ints_to_floats(delays: &[u32]) -> [f64; 16] {
    assert_eq!(delays.len(), 16);
    [
        delays[0] as f64 * VEL_C * DELAY_STEP,
        delays[1] as f64 * VEL_C * DELAY_STEP,
        delays[2] as f64 * VEL_C * DELAY_STEP,
        delays[3] as f64 * VEL_C * DELAY_STEP,
        delays[4] as f64 * VEL_C * DELAY_STEP,
        delays[5] as f64 * VEL_C * DELAY_STEP,
        delays[6] as f64 * VEL_C * DELAY_STEP,
        delays[7] as f64 * VEL_C * DELAY_STEP,
        delays[8] as f64 * VEL_C * DELAY_STEP,
        delays[9] as f64 * VEL_C * DELAY_STEP,
        delays[10] as f64 * VEL_C * DELAY_STEP,
        delays[11] as f64 * VEL_C * DELAY_STEP,
        delays[12] as f64 * VEL_C * DELAY_STEP,
        delays[13] as f64 * VEL_C * DELAY_STEP,
        delays[14] as f64 * VEL_C * DELAY_STEP,
        delays[15] as f64 * VEL_C * DELAY_STEP,
    ]
}
