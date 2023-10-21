// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

// Get beam responses using a GPU. This is mostly useful when you want to get
// Jones matrices for different tile configurations (e.g. different delays, dead
// dipoles) toward a huge number of directions.
//
// hyperbeam lets you run the GPU code at single- or double-precision. The
// reason is that desktop GPUs have poor double-precision performance.
//
// Compile with something like:
// # Double precision with CUDA
// cargo build --release --features=cuda
// # Double precision with HIP
// cargo build --release --features=hip
// # or single precision
// cargo build --release --features=cuda,gpu-single
// cargo build --release --features=hip,gpu-single
//
// Note the precision you're using; if it's single then supply "-D SINGLE"
// below. Otherwise, no flag is needed.
//
// Compile and run this file with something like:
// gcc -O3 -D SINGLE -I ../include/ -L ../target/release/ -l mwa_hyperbeam ./analytic_hip.c -o analytic_hip
// LD_LIBRARY_PATH=../target/release ./analytic_hip

#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "mwa_hyperbeam.h"

#ifdef SINGLE
#define FLOAT float
#define CREAL crealf
#define CIMAG cimagf
#else
#define FLOAT double
#define CREAL creal
#define CIMAG cimag
#endif

void handle_hyperbeam_error(char file[], int line_num, const char function_name[]) {
    int err_length = hb_last_error_length();
    char *err = malloc(err_length * sizeof(char));
    int err_status = hb_last_error_message(err, err_length);
    if (err_status == -1) {
        printf("Something really bad happened!\n");
        exit(EXIT_FAILURE);
    }
    printf("File %s:%d: hyperbeam error in %s: %s\n", file, line_num, function_name, err);

    exit(EXIT_FAILURE);
}

int main(int argc, char *argv[]) {
    // Get a new beam object from hyperbeam.
    AnalyticBeam *beam;
    char rts_style = 0;                  // 1 or RTS style, 0 for mwa_pb
    double *dipole_height_metres = NULL; // Point to a valid float if you want a custom height
    // Point to a valid int if you want a custom number of bowties per row. You
    // almost certainly want this to be 4, unless you're simulating the CRAM
    // tile.
    uint8_t *bowties_per_row = NULL;
    if (new_analytic_beam(rts_style, dipole_height_metres, bowties_per_row, &beam))
        handle_hyperbeam_error(__FILE__, __LINE__, "new_analytic_beam");

    // Set up our telescope array. Here, we are using two distinct tiles
    // (different dead dipoles). The first 16 values are the first tile, second
    // 16 second tile. When giving 16 values per tile, each value is used for
    // the X and Y dipoles. It's possible to supply 32 values per tile; in that
    // case, the first 16 values are for the X dipoles and the last 16 are for
    // the Y dipoles.

    // Delays and amps correspond to dipoles in the "M&C order". See
    // https://wiki.mwatelescope.org/pages/viewpage.action?pageId=48005139) for
    // more info. Amps refer to dipole gains, and are usually set to 1 or 0 (if
    // a dipole is dead).
    unsigned delays[32] = {3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0,
                           3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0};
    double dip_amps[32] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1,
                           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0};
    int freqs_hz[3] = {150e6, 175e6, 200e6};
    int num_freqs = 3;
    int num_tiles = 2;
    // Number of specified amps per tile.
    int num_amps = 16;
    // MWA latitude
    double latitude_rad = -0.4660608448386394;
    // Should we normalise the beam response?
    int norm_to_zenith = 1;

    // Now get a new GPU analytic beam object.
    AnalyticBeamGpu *gpu_beam;
    if (new_gpu_analytic_beam(beam, delays, dip_amps, num_tiles, num_amps, &gpu_beam))
        handle_hyperbeam_error(__FILE__, __LINE__, "new_gpu_analytic_beam");

    // Set up the directions to get the beam responses.
    int num_azzas = 1000000;
    FLOAT *az = malloc(num_azzas * sizeof(FLOAT));
    FLOAT *za = malloc(num_azzas * sizeof(FLOAT));
    for (int i = 0; i < num_azzas; i++) {
        az[i] = (-170.0 + i * 340.0 / num_azzas) * M_PI / 180.0;
        za[i] = (10.0 + i * 70.0 / num_azzas) * M_PI / 180.0;
    }

    // Allocate a buffer for the results.
    size_t num_unique_tiles = (size_t)get_num_unique_analytic_tiles(gpu_beam);
    complex FLOAT *jones = malloc(num_unique_tiles * num_freqs * num_azzas * 8 * sizeof(FLOAT));
    // hyperbeam expects a pointer to our FLOAT macro. Casting the pointer works
    // fine.
    if (analytic_calc_jones_gpu(gpu_beam, num_azzas, az, za, num_freqs, freqs_hz, latitude_rad, norm_to_zenith,
                                (FLOAT *)jones))
        handle_hyperbeam_error(__FILE__, __LINE__, "analytic_calc_jones_gpu");

    printf("The first Jones matrix:\n");
    printf("[[%+.8f%+.8fi,", CREAL(jones[0]), CIMAG(jones[0]));
    printf(" %+.8f%+.8fi]\n", CREAL(jones[1]), CIMAG(jones[1]));
    printf(" [%+.8f%+.8fi,", CREAL(jones[2]), CIMAG(jones[2]));
    printf(" %+.8f%+.8fi]]\n", CREAL(jones[3]), CIMAG(jones[3]));

    // Free the heap-allocated Jones matrices.
    free(jones);
    // Free the beam objects - we must use special functions to do this.
    free_gpu_analytic_beam(gpu_beam);
    free_analytic_beam(beam);

    return EXIT_SUCCESS;
}
