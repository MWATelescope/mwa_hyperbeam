// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "ska.h"

extern "C" const char *gpu_ska_calc_jones(const SkaGpuModel *model, const FLOAT *d_azs, const FLOAT *d_zas,
                                          const int num_directions, const FLOAT *d_freqs_hz, const int num_freqs,
                                          const int num_tiles, const int *d_element_ids,
                                          const int num_station_elements, const FLOAT *d_positions,
                                          const FLOAT *d_weights, const FLOAT *d_weights_y,
                                          const FLOAT *d_rotations, const FLOAT *d_frames, const FLOAT pointing_az,
                                          const FLOAT pointing_za, const FLOAT *d_latitude_rad, const int iau_order,
                                          void *d_results) {
    if (num_directions == 0 || num_freqs == 0 || num_tiles == 0)
        return NULL;
    if (model->model == SKA_EEP_SPH && d_positions != NULL) {
        // Stations of distinct elements: a block per (direction, tile,
        // frequency); the kernel loops if there are more than fit in a grid.
        const size_t num_blocks = (size_t)num_directions * num_tiles * num_freqs;
        dim3 gridDim, blockDim;
        blockDim.x = SKA_SPH_STATION_THREADS;
        gridDim.x = (unsigned int)(num_blocks < (1u << 30) ? num_blocks : (1u << 30));
        ska_sph_station_kernel<<<gridDim, blockDim>>>(*model, d_azs, d_zas, num_directions, d_freqs_hz, num_freqs,
                                                      num_tiles, num_station_elements, d_positions, d_weights,
                                                      d_weights_y, d_rotations, d_frames, pointing_az, pointing_za,
                                                      d_latitude_rad, iau_order, (JONES *)d_results);
    } else {
        // A thread per response (direction, tile, frequency); the kernel
        // loops if there are more than fit in a grid.
        const size_t num_responses = (size_t)num_directions * num_tiles * num_freqs;
        dim3 gridDim, blockDim;
        blockDim.x = gpu_warp_size() * 4;
        const size_t num_blocks = (num_responses + blockDim.x - 1) / blockDim.x;
        gridDim.x = (unsigned int)(num_blocks < (1u << 24) ? num_blocks : (1u << 24));
        ska_kernel<<<gridDim, blockDim>>>(*model, d_azs, d_zas, num_directions, d_freqs_hz, num_freqs, num_tiles,
                                          d_element_ids, num_station_elements, d_positions, d_weights, d_weights_y,
                                          d_rotations, d_frames, pointing_az, pointing_za, d_latitude_rad,
                                          iau_order, (JONES *)d_results);
    }

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
