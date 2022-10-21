The CUDA code here was written by Cristian Di Pietrantonio and Maciej Cytowski
on behalf of the Pawsey Supercomputing Centre. It is provided under the terms
given in the LICENSE file in this directory.

The bulk of the CUDA code lives in fee.h - this is done so that in the future,
the device code may be utilised directly by code outside of hyperbeam. This is
also the recommended way of distributing CUDA functions "across libraries" -
nvcc does not allow device functions to be called from other compilation units.
