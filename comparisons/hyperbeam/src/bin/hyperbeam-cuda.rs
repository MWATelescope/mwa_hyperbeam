use mwa_hyperbeam::{
    fee::{FEEBeam, FEEBeamGpu},
    AzEl,
};
use ndarray::Array2;

const NUM_POINTS_TO_SIMULATE: usize = 300_000;
const NUM_POINTS_TO_SIMULATE_BIG: usize = 999_999;
const FREQ_HZ: f64 = 150e6;
const EVERYBEAM_AZ_RAD: f64 = 1.745998843813605;
const EVERYBEAM_EL_RAD: f64 = 1.548676626223685;
const MS_DELAYS: &[u32; 16] = &[0; 16];
const DIPOLE_GAINS: &[f64; 16] = &[1.0; 16];
const _EVERYBEAM_MWA_LATITUDE_RAD: f64 = -0.466018978039551;

fn main() {
    let beam = FEEBeam::new_from_env().unwrap();
    let gpu_beam: FEEBeamGpu;
    let delays_array = Array2::from_shape_vec((1, MS_DELAYS.len()), Vec::from(MS_DELAYS)).unwrap();
    let gains_array =
        Array2::from_shape_vec((1, DIPOLE_GAINS.len()), Vec::from(DIPOLE_GAINS)).unwrap();
    let azel = AzEl::from_radians(EVERYBEAM_AZ_RAD, EVERYBEAM_EL_RAD);
    let azels = vec![azel; NUM_POINTS_TO_SIMULATE];

    {
        // In order to more fairly compare the cold cache time here with others,
        // initialise the GPU beam after the timer starts.
        let start = std::time::Instant::now();
        gpu_beam = unsafe {
            beam.gpu_prepare(
                &[FREQ_HZ as u32],
                delays_array.view(),
                gains_array.view(),
                true,
            )
        }
        .unwrap();
        let _jones = gpu_beam.calc_jones(&azels, None, false).unwrap();
        println!(
            "time taken to produce {} simulations (cold cache): {:?}",
            azels.len(),
            start.elapsed()
        );
    }
    {
        let start = std::time::Instant::now();
        let _jones = gpu_beam.calc_jones(&azels, None, false).unwrap();
        println!(
            "time taken to produce {} simulations (hot cache):  {:?}",
            azels.len(),
            start.elapsed()
        );
    }
    {
        let azels_big = vec![azel; NUM_POINTS_TO_SIMULATE_BIG];
        let start = std::time::Instant::now();
        let jones = gpu_beam.calc_jones(&azels_big, None, false).unwrap();
        println!(
            "time taken to produce {} simulations (hot cache):  {:?}",
            azels_big.len(),
            start.elapsed()
        );

        println!("First and last MWA beam responses:");
        let first = jones.first().unwrap();
        let last = jones.last().unwrap();
        println!(
            "[{:+.6}{:+.6}i, {:+.6}{:+.6}i",
            first[0].re, first[0].im, first[1].re, first[1].im
        );
        println!(
            " {:+.6}{:+.6}i, {:+.6}{:+.6}i]",
            first[2].re, first[2].im, first[3].re, first[3].im
        );
        println!(
            "[{:+.6}{:+.6}i, {:+.6}{:+.6}i",
            last[0].re, last[0].im, last[1].re, last[1].im
        );
        println!(
            " {:+.6}{:+.6}i, {:+.6}{:+.6}i]",
            last[2].re, last[2].im, last[3].re, last[3].im
        );
    }
}
