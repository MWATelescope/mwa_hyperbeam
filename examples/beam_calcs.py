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

# Make a lot of pointings for hyperbeam to calculate in parallel.
n = 1000000
az = np.linspace(0, 0.9 * np.pi, n)
za = np.linspace(0.1, 0.9 * np.pi / 2, n)
freq = 167000000
delays = [0] * 16
amps = [1.0] * 16
beam_norm = True

# Pass the values to hyperbeam and get a numpy array back. Each element is a
# 4-element Jones matrix.
start_time = time.time()
jones = beam.calc_jones_array(az, za, freq, delays, amps, beam_norm)
duration = time.time() - start_time
print("Time to calculate {} pointings: {:.3}s".format(n, duration))
print("First Jones matrix:")
print(jones[0])
