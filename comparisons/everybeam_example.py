#!/usr/bin/env python3

import os
import sys

import everybeam as eb


ms_path = sys.argv[1]
# This thorws an error on v0.5.3 of EveryBeam (the latest at the time of
# writing), because the MWA isn't supported.
telescope = eb.load_telescope(ms_path, os.environ.get("MWA_BEAM_FILE"))
