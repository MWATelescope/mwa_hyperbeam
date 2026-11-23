use std::sync::atomic::{AtomicBool, Ordering};

use mwa_hyperbeam::{fee::FEEBeam, AzEl};

const FREQ_HZ: f64 = 150e6;
const EVERYBEAM_AZ_RAD: f64 = 1.745998843813605;
const EVERYBEAM_EL_RAD: f64 = 1.548676626223685;
const MS_DELAYS: &[u32; 16] = &[0; 16];
const DIPOLE_GAINS: &[f64; 16] = &[1.0; 16];
const _EVERYBEAM_MWA_LATITUDE_RAD: f64 = -0.466018978039551;

static CACHE_HOT: AtomicBool = AtomicBool::new(false);

fn bench(n: usize, beam: &FEEBeam, azel: AzEl, print_first_and_last: bool) {
    let azels = vec![azel; n];
    let start = std::time::Instant::now();

    let jones = if n == 1 {
        beam.calc_jones(
            azel,
            FREQ_HZ as u32,
            MS_DELAYS,
            DIPOLE_GAINS,
            true,
            None,
            false,
        )
        .unwrap();
        None
    } else {
        let jones = beam
            .calc_jones_array(
                &azels,
                FREQ_HZ as u32,
                MS_DELAYS,
                DIPOLE_GAINS,
                true,
                None,
                false,
            )
            .unwrap();
        Some(jones)
    };
    let duration = start.elapsed();

    let cache_state = if CACHE_HOT.load(Ordering::Relaxed) {
        "hot"
    } else {
        CACHE_HOT.store(true, Ordering::Relaxed);
        "cold"
    };
    let plural = if n == 1 { "" } else { "s" };
    println!(
        "time taken to produce {n} simulation{plural} ({cache_state} cache): {:?}",
        duration
    );

    if let Some(jones) = jones {
        if print_first_and_last {
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
}

fn main() {
    let beam = FEEBeam::new_from_env().unwrap();
    let azel = AzEl::from_radians(EVERYBEAM_AZ_RAD, EVERYBEAM_EL_RAD);

    // Check for a CLI argument. If it's there, we'll do only one benchmark
    // with the indicated amount of simulations. This is mostly useful to see
    // memory usage as a one-off.
    if let Some(arg) = std::env::args().nth(1) {
        // Verify it's a number.
        let n = arg.parse().expect("is a number");
        bench(n, &beam, azel, true);
        std::process::exit(0);
    }

    bench(1, &beam, azel, false);
    bench(1, &beam, azel, false);
    bench(1000, &beam, azel, false);
    bench(300_000, &beam, azel, false);
    bench(999_999, &beam, azel, true);
}
