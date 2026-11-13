#!/usr/bin/env python3

import sys
import time

import numpy as np
import mwa_hyperbeam


N = 1000
FREQ_HZ = 150e6
DELAYS = [0] * 16
AMPS = [1.0] * 16


def get_pointings(n):
    # az = np.linspace(0, 0.9 * np.pi, n)
    # za = np.linspace(0.1, 0.9 * np.pi / 2, n)
    az = np.ones(n) * 1.745998843813605
    za = np.ones(n) * (np.pi / 2 - 1.548676626223685)
    return az, za


az, za = get_pointings(N)
beam = mwa_hyperbeam.FEEBeam()

start_time = time.time()
jones = beam.calc_jones(
    az[0],
    za[0],
    FREQ_HZ,
    delays=DELAYS,
    amps=AMPS,
    norm_to_zenith=True,
    latitude_rad=None,
    iau_order=False,
)
duration = time.time() - start_time
print(f"time taken to produce 1 simulation (cold cache): {duration}s")

start_time = time.time()
jones = beam.calc_jones(
    az[0],
    za[0],
    FREQ_HZ,
    delays=DELAYS,
    amps=AMPS,
    norm_to_zenith=True,
    latitude_rad=None,
    iau_order=False,
)
duration = time.time() - start_time
print(f"time taken to produce 1 simulation (hot cache):  {duration}s")

start_time = time.time()
jones = beam.calc_jones_array(
    az,
    za,
    FREQ_HZ,
    delays=DELAYS,
    amps=AMPS,
    norm_to_zenith=True,
    latitude_rad=None,
    iau_order=False,
)
duration = time.time() - start_time
print(f"time taken to produce {len(az)} simulations (hot cache):  {duration}s")

az, za = get_pointings(300000)
start_time = time.time()
jones = beam.calc_jones_array(
    az,
    za,
    FREQ_HZ,
    delays=DELAYS,
    amps=AMPS,
    norm_to_zenith=True,
    latitude_rad=None,
    iau_order=False,
)
duration = time.time() - start_time
print(f"time taken to produce {len(az)} simulations (hot cache):  {duration}s")

az, za = get_pointings(999999)
start_time = time.time()
jones = beam.calc_jones_array(
    az,
    za,
    FREQ_HZ,
    delays=DELAYS,
    amps=AMPS,
    norm_to_zenith=True,
    latitude_rad=None,
    iau_order=False,
)
duration = time.time() - start_time
print(f"time taken to produce {len(az)} simulations (hot cache):  {duration}s")

print("First and last MWA beam responses:")
print(jones[0])
print(jones[-1])

if len(sys.argv) >= 2 and sys.argv[1] == "cuda":
    print("\n*** hyperbeam Python results with CUDA ***")
    start_time = time.time()
    jones = beam.calc_jones_gpu(
        az,
        za,
        [FREQ_HZ],
        DELAYS,
        AMPS,
        norm_to_zenith=True,
        latitude_rad=None,
        iau_order=False,
    )
    duration = time.time() - start_time
    print(f"time taken to produce {len(az)} simulations (hot cache):  {duration}s")
