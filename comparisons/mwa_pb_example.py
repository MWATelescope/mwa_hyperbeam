#!/usr/bin/env python3

import time

import numpy as np
from mwa_pb.primary_beam import MWA_Tile_full_EE

N = 1000
FREQ_HZ = 150e6
DELAYS = np.array([0] * 16)


def get_pointings(n):
    az = np.linspace(0, 0.9 * np.pi, n)
    za = np.linspace(0.1, 0.9 * np.pi / 2, n)
    # az = np.ones(n) * 1.745998843813605
    # za = np.ones(n) * (np.pi / 2 - 1.548676626223685)
    return az, za


az, za = get_pointings(N)

start_time = time.time()
jones = MWA_Tile_full_EE(
    za[0],
    az[0],
    int(FREQ_HZ),
    delays=DELAYS,
    zenithnorm=True,
    interp=False,
    power=False,
    jones=True,
)
duration = time.time() - start_time
print(f"time taken to produce 1 simulation (cold cache): {duration}s")

start_time = time.time()
jones = MWA_Tile_full_EE(
    za[0],
    az[0],
    int(FREQ_HZ),
    delays=DELAYS,
    zenithnorm=True,
    interp=False,
    power=False,
    jones=True,
)
duration = time.time() - start_time
print(f"time taken to produce 1 simulation (hot cache):  {duration}s")

start_time = time.time()
jones = MWA_Tile_full_EE(
    za,
    az,
    int(FREQ_HZ),
    delays=DELAYS,
    zenithnorm=True,
    interp=False,
    power=False,
    jones=True,
)
duration = time.time() - start_time
print(f"time taken to produce {len(az)} simulations (hot cache):  {duration}s")

az, za = get_pointings(300000)
start_time = time.time()
jones = MWA_Tile_full_EE(
    za,
    az,
    int(FREQ_HZ),
    delays=DELAYS,
    zenithnorm=True,
    interp=False,
    power=False,
    jones=True,
)
duration = time.time() - start_time
print(f"time taken to produce {len(az)} simulations (hot cache):  {duration}s")

# Not printing these because they're different to other packages
# print("First and last MWA beam responses:")
# print(jones[0])
# print(jones[-1])
