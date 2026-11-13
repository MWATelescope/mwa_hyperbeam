#!/usr/bin/env python3

# Written against pyuvdata 2.4.1

import os
import time

import numpy as np
from pyuvdata.uvbeam import UVBeam


FREQ_HZ = 150e6
DELAYS = np.zeros((2, 16), dtype=int)
AMPS = np.ones((2, 16))


mwa_beam_file = os.environ.get("MWA_BEAM_FILE")

for res in [1, 2]:
    start_time = time.time()
    beam = UVBeam.from_file(
        mwa_beam_file,
        delays=DELAYS,
        amplitudes=AMPS,
        frequency=[FREQ_HZ],
        pixels_per_deg=res,
    )
    duration = time.time() - start_time
    print(
        f"time taken to produce {beam.data_array.shape[-1]*beam.data_array.shape[-2]} simulations: {duration}s"
    )
