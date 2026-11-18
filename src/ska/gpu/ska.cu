// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "ska.h"

extern "C" const char *gpu_ska_calc_jones(const SkaGpuModel *model, const FLOAT *d_azs, const FLOAT *d_zas,
                                          const int num_directions, const FLOAT *d_freqs_hz, const int num_freqs,
                                          const int num_tiles, const int *d_element_ids,
                                          const int num_station_elements, const FLOAT *d_positions,
                                          const FLOAT *d_weights, const FLOAT *d_rotations, const FLOAT pointing_az,
                                          const FLOAT pointing_za,
                                          const FLOAT *d_latitude_rad, const int iau_order, void *d_results) {
    if (num_directions == 0)
        return NULL;
    dim3 gridDim, blockDim;
    blockDim.x = gpu_warp_size() * 4;
    gridDim.x = (int)ceil((double)num_directions / (double)blockDim.x);
    ska_kernel<<<gridDim, blockDim>>>(*model, d_azs, d_zas, num_directions, d_freqs_hz, num_freqs, num_tiles,
                                      d_element_ids, num_station_elements, d_positions, d_weights, d_rotations, pointing_az,
                                      pointing_za, d_latitude_rad, iau_order, (JONES *)d_results);

    gpuError_t error_id;
#ifdef DEBUG
    error_id = gpuDeviceSynchronize();
    if (error_id != gpuSuccess) {
        return gpuGetErrorString(error_id);
    }
#endif
    error_id = gpuGetLastError();
    if (error_id != gpuSuccess) {
        return gpuGetErrorString(error_id);
    }

    return NULL;
}
