#pragma once

#include "gpu_common.cuh"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

typedef enum ANALYTIC_TYPE { MWA_PB, RTS } ANALYTIC_TYPE;

const char *gpu_analytic_calc_jones(const ANALYTIC_TYPE at, const FLOAT dipole_height_m, const FLOAT *d_azs,
                                    const FLOAT *d_zas, int num_directions, const unsigned int *d_freqs_hz,
                                    const int num_freqs, const FLOAT *d_delays, const FLOAT *d_amps,
                                    const int num_tiles, const FLOAT latitude_rad, const uint8_t norm_to_zenith,
                                    void *d_results);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

// To allow bindgen to run on this file we have to hide a bunch of stuff behind
// a macro.
#ifndef BINDGEN

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

__global__ void analytic_kernel(const ANALYTIC_TYPE at, const FLOAT dipole_height_m, const FLOAT *azs, const FLOAT *zas,
                                const int num_directions, const unsigned int *freqs_hz, const int num_freqs,
                                const FLOAT *delays, const FLOAT *amps, const int num_tiles, const FLOAT latitude_rad,
                                const bool norm_to_zenith, JONES *results) {
    for (int i_direction = blockIdx.x * blockDim.x + threadIdx.x; i_direction < num_directions;
         i_direction += gridDim.x * blockDim.x) {
        const FLOAT az = azs[i_direction];
        const FLOAT za = zas[i_direction];

        FLOAT s_az, c_az, s_za, c_za;
        SINCOS(az, &s_az, &c_az);
        SINCOS(za, &s_za, &c_za);

        JONES jones_original = JONES{
            .j00 = MAKE_COMPLEX(0.0, 0.0),
            .j01 = MAKE_COMPLEX(0.0, 0.0),
            .j10 = MAKE_COMPLEX(0.0, 0.0),
            .j11 = MAKE_COMPLEX(0.0, 0.0),
        };
        if (at == MWA_PB) {
            jones_original.j00.x = c_za * s_az;
            jones_original.j01.x = c_az;
            jones_original.j10.x = c_za * c_az;
            jones_original.j11.x = -s_az;
        } else if (at == RTS) {
            HADec hadec = azel_to_hadec(az, M_PI_2 - za, latitude_rad);
            FLOAT s_ha, c_ha, s_dec, c_dec, s_latitude, c_latitude;
            SINCOS(hadec.ha, &s_ha, &c_ha);
            SINCOS(hadec.dec, &s_dec, &c_dec);
            SINCOS(latitude_rad, &s_latitude, &c_latitude);

            jones_original.j00.x = c_latitude * c_dec + s_latitude * s_dec * c_ha;
            jones_original.j01.x = -s_latitude * s_ha;
            jones_original.j10.x = s_dec * s_ha;
            jones_original.j11.x = c_ha;
        }

        FLOAT proj_e = s_za * s_az;
        FLOAT proj_n = s_za * c_az;

        for (int i_tile = 0; i_tile < num_tiles; i_tile++) {
            for (int i_freq = 0; i_freq < num_freqs; i_freq++) {
                FLOAT lambda_m = VEL_C / freqs_hz[i_freq];
                FLOAT multiplier = -M_2PI / lambda_m;
                JONES jones = jones_original;

                COMPLEX array_factor = MAKE_COMPLEX(0, 0);
                // Pray to your deity that this unrolls. Or, ya know, don't, because
                // this is still heaps faster than FEE.
                for (int row = 0; row < 4; row++) {
                    for (int col = 0; col < 4; col++) {
                        FLOAT dip_e = 0.0;
                        FLOAT dip_n = 0.0;
                        FLOAT phase = 0.0;
                        FLOAT delay = delays[i_tile * 16 + row * 4 + col];

                        if (at == MWA_PB) {
                            dip_e = ((FLOAT)col - 1.5) * MWA_DPL_SEP;
                            dip_n = ((FLOAT)row - 1.5) * MWA_DPL_SEP;
                            phase = -multiplier * (dip_e * proj_e + dip_n * proj_n - delay);
                        } else if (at == RTS) {
                            dip_e = ((FLOAT)row - 1.5) * MWA_DPL_SEP;
                            dip_n = ((FLOAT)col - 1.5) * MWA_DPL_SEP;
                            phase = multiplier * (dip_e * proj_e + dip_n * proj_n - delay);
                        }

                        FLOAT s_phase, c_phase;
                        SINCOS(phase, &s_phase, &c_phase);
                        FLOAT amp = amps[i_tile * 16 + row * 4 + col];
                        array_factor += MAKE_COMPLEX(c_phase, s_phase) * amp;
                    }
                }

                FLOAT ground_plane = 2.0 * SIN(M_2PI * dipole_height_m / lambda_m * c_za) / NUM_DIPOLES;
                if (norm_to_zenith) {
                    ground_plane /= 2.0 * SIN(M_2PI * dipole_height_m / lambda_m);
                }

                array_factor *= ground_plane;
                jones.j00 *= array_factor;
                jones.j01 *= array_factor;
                jones.j10 *= array_factor;
                jones.j11 *= array_factor;

                if (at == RTS) {
                    jones.j00.y = 0.0;
                    jones.j01.y = 0.0;
                    jones.j10.y = 0.0;
                    jones.j11.y = 0.0;
                }

                // Copy the Jones matrix to global memory.
                results[((num_directions * num_freqs * i_tile) + num_directions * i_freq) + i_direction] = jones;
            }
        }
    }
}

extern "C" const char *gpu_analytic_calc_jones(const ANALYTIC_TYPE at, const FLOAT dipole_height_m, const FLOAT *d_azs,
                                               const FLOAT *d_zas, int num_directions, const unsigned int *d_freqs_hz,
                                               const int num_freqs, const FLOAT *d_delays, const FLOAT *d_amps,
                                               const int num_tiles, const FLOAT latitude_rad,
                                               const uint8_t norm_to_zenith, void *d_results) {
    dim3 gridDim, blockDim;
    blockDim.x = warpSize;
    gridDim.x = (int)ceil((double)num_directions / (double)blockDim.x);
    analytic_kernel<<<gridDim, blockDim>>>(at, dipole_height_m, d_azs, d_zas, num_directions, d_freqs_hz, num_freqs,
                                           d_delays, d_amps, num_tiles, latitude_rad, (bool)norm_to_zenith,
                                           (JONES *)d_results);

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

#endif // BINDGEN

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus
