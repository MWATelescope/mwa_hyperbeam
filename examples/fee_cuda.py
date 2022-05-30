#!/usr/bin/env python

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

# Build with something like:
# maturin develop --release -b pyo3 --cargo-extra-args="--features=python,all-static,cuda-single" --strip

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
n = 1000000
az = np.linspace(0, 0.9 * np.pi, n)
za = np.linspace(0.1, 0.9 * np.pi / 2, n)
# Multiple frequencies can be specified.
freq = [167000000]
# Multiple tiles can also be specified. Each is allowed to have their own delays
# and amps, but each row (tile) must have 16 delays, 16 or 32 amps.
# Unfortunately the arrays must be flattened. Delays and amps correspond to
# dipoles in the "M&C order". See
# https://wiki.mwatelescope.org/pages/viewpage.action?pageId=48005139) for more
# info.
delays = np.zeros((2, 16), dtype=np.uint).flatten()
amps = np.ones((2, 16)).flatten()
# Should we normalise the beam response?
norm_to_zenith = True
# Should we apply the parallactic angle correction? Read more here:
# https://github.com/JLBLine/polarisation_tests_for_FEE
parallactic = True

# Pass the values to hyperbeam and get a numpy array back.
start_time = time.time()
jones = beam.calc_jones_cuda(
    az, za, freq, delays, amps, norm_to_zenith, parallactic)
duration = time.time() - start_time
print("Time to calculate {} directions: {:.3}s".format(n, duration))
print("First Jones matrix:")
print(jones[0, 0, 0])
