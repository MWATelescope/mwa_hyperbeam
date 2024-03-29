#!/usr/bin/env python

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import sys
import time
import numpy as np
import mwa_hyperbeam

if len(sys.argv) > 1 and sys.argv[1] == "rts":
    beam = mwa_hyperbeam.AnalyticBeam(rts_behaviour=True)
else:
    beam = mwa_hyperbeam.AnalyticBeam()

# Make a lot of directions for hyperbeam to calculate in parallel.
n = 1000000
az_rad = np.linspace(0, 0.9 * np.pi, n)
za_rad = np.linspace(0.1, 0.9 * np.pi / 2, n)
freq_hz = 167000000
# Delays and amps correspond to dipoles in the "M&C order". See
# https://wiki.mwatelescope.org/pages/viewpage.action?pageId=48005139) for more
# info.
delays = [0] * 16
amps = [1.0] * 16
# MWA latitude
latitude_rad = -0.4660608448386394
# Should we normalise the beam response?
norm_to_zenith = True

# Pass the values to hyperbeam and get a numpy array back. Each element is a
# 4-element Jones matrix.
start_time = time.time()
# beam.calc_jones is also available, but that makes a single Jones matrix at a
# time, so one would need to iterate over az and za. calc_jones_array is done in
# parallel with Rust (so it's fast).
jones = beam.calc_jones_array(
    az_rad, za_rad, freq_hz, delays, amps, latitude_rad, norm_to_zenith
)
duration = time.time() - start_time
print("Time to calculate {} directions: {:.3}s".format(n, duration))
print("First Jones matrix:")
print(jones[0])

# It's also possible to supply amps for all dipole elements. The first 16 amps
# are for X elements, the second 16 are for Y elements.
amps = np.ones(32)
amps[-1] = 0
jones = beam.calc_jones_array(
    az_rad[:1], za_rad[:1], freq_hz, delays, amps, latitude_rad, norm_to_zenith
)
print("First Jones matrix with altered Y amps:")
print(jones[0])

# BONUS ROUND - CRAM TILE
beam = mwa_hyperbeam.AnalyticBeam(bowties_per_row=8)
az_rad = 45.0 * np.pi / 180.0
za_rad = 80.0 * np.pi / 180.0
freq_hz = 51200000
# fmt: off
delays_cram = [
    3, 2, 1, 0, 3, 2, 1, 0,
    3, 2, 1, 0, 3, 2, 1, 0,
    3, 2, 1, 0, 3, 2, 1, 0,
    3, 2, 1, 0, 3, 2, 1, 0,
    3, 2, 1, 0, 3, 2, 1, 0,
    3, 2, 1, 0, 3, 2, 1, 0,
    3, 2, 1, 0, 3, 2, 1, 0,
    3, 2, 1, 0, 3, 2, 1, 0,
]
amps_cram = [
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 0, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 0, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 0, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 0, 1,
]
jones = beam.calc_jones(
    az_rad, za_rad, freq_hz, delays_cram, amps_cram, latitude_rad, norm_to_zenith
)
print("The CRAM Jones matrix:")
print(jones)
