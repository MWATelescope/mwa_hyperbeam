This is a release of `mwa_hyperbeam`, primary beam code for the MWA, obtained
from the [GitHub releases
page](https://github.com/MWATelescope/mwa_hyperbeam/releases).

The FEE beam code in `hyperbeam` requires an HDF5 file to function. This can be
obtained with:

  `wget http://ws.mwatelescope.org/static/mwa_full_embedded_element_pattern.h5`

Move the `h5` file anywhere you like, and put the file path in `MWA_BEAM_FILE`:

  `export MWA_BEAM_FILE=/path/to/mwa_full_embedded_element_pattern.h5`

# Licensing

`hyperbeam` is licensed under the [Mozilla Public License 2.0 (MPL
2.0)](https://www.mozilla.org/en-US/MPL/2.0/). The LICENSE file is the relevant
copy.

Another license from the `hdf5` library is provided (COPYING-hdf5). This is
included because (as per the terms of that license) `hdf5` is compiled inside
the `hyperbeam` products in this tarball.

# What are these different x86-64 versions?

They are [microarchitecture
levels](https://en.wikipedia.org/wiki/X86-64#Microarchitecture_levels). By
default, Rust compiles for all x86-64 CPUs; this allows maximum compatibility,
but potentially limits the runtime performance because many modern CPU features
can't be used. Compiling at different levels allows the code to be optimised for
different classes of CPUs so users can get something that works best for them.

## Rule of thumb for which microarchitecture to use

You're probably safe with x86-64-v2, especially if your CPU was made after 2010.
If your CPU is older than that, you may want x86-64. If your CPU is a little
newer, x86-64-v3 is also likely to work. x86-64-v4 isn't widely supported
(AVX-512), so only use that if you know it works.

# CUDA?

The releases with "CUDA" in the name are CUDA enabled. The `hyperbeam` libraries
have been dynamically linked against CUDA 11.2.0; to use them, a CUDA
installation on version 11 is required.

There is also a double- or single-precision version of `hyperbeam` provided. If
you're running a desktop NVIDIA GPU (e.g. RTX 2070), then you probably want the
single-precision version. This is because desktop GPUs have a lot less
double-precision computation capability. It is still possible to use the
double-precision version, but the extra precision comes at the expensive of
speed.

Other GPUs, like the V100s hosted by the Pawsey Supercomputing Centre, are
capable of running the double-precision code much faster, so there is little
incentive for running single-precision code on these GPUs.
