#!/usr/bin/env python

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import sys
import time
import numpy as np
import mwa_hyperbeam

# We can make a new beam object with a path to the HDF5 file, or, if not given,
# use whatever is specified in MWA_BEAM_FILE.
if len(sys.argv) > 1:
    beam = mwa_hyperbeam.FEEBeam(sys.argv[1])
else:
    beam = mwa_hyperbeam.FEEBeam()

# Make a lot of directions for hyperbeam to calculate in parallel.
n = 100000
az = np.linspace(0, 0.9 * np.pi, n)
za = np.linspace(0.1, 0.9 * np.pi / 2, n)
freq = 167000000
# Delays and amps correspond to dipoles in the "M&C order". See
# https://wiki.mwatelescope.org/pages/viewpage.action?pageId=48005139) for more
# info.
delays = [0] * 16
amps = [1.0] * 16
# Should we normalise the beam response?
norm_to_zenith = True
# Should we apply the parallactic angle correction? If so, give the array
# latitude here. Read more here:
# https://github.com/MWATelescope/mwa_hyperbeam/blob/main/fee_pols.pdf
latitude_rad = None
# Do we want an "IAU ordered" beam response? This value doesn't matter if we
# don't do a parallactic angle correction.
iau_order = False

# Pass the values to hyperbeam and get a numpy array back. Each element is a
# 4-element Jones matrix.
start_time = time.time()
# beam.calc_jones is also available, but that makes a single Jones matrix at a
# time, so one would need to iterate over az and za. calc_jones_array is done in
# parallel with Rust (so it's fast).
jones = beam.calc_jones_array(
    az, za, freq, delays, amps, norm_to_zenith, latitude_rad, iau_order
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
    az[:1], za[:1], freq, delays, amps, norm_to_zenith, latitude_rad, iau_order
)
print("First Jones matrix with altered Y amps:")
print(jones[0])

# Get another beam response, but this time with the parallactic-angle
# correction.
latitude_rad = -0.4660608448386394
iau_order = True
jones = beam.calc_jones(
    az[0], za[0], freq, delays, amps, norm_to_zenith, latitude_rad, iau_order
)
print("Parallactic-angle corrected, IAU-ordered beam response:")
print(jones)

# Supply only mandatory arguments (latitude_rad and iau_order are optional).
jones = beam.calc_jones(az[0], za[0], freq, delays, amps, norm_to_zenith)
