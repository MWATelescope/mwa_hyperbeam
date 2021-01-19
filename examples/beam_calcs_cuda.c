// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

// Get beam responses using CUDA. This is mostly useful when you want to get
// Jones matrices for different tile configurations (e.g. different delays, dead
// dipoles) toward a huge number of directions.
//
// hyperbeam lets you run the CUDA code at single- or double-precision. The
// reason is that desktop GPUs have poor double-precision performance.
//
// Compile with something like:
// # Double precision
// cargo build --release --features=cuda
// # or single precision
// cargo build --release --features=cuda-single
//
// Note the precision you're using; if it's single then supply "-D SINGLE"
// below. Otherwise, no flag is needed.
//
// Compile and run this file with something like:
// gcc -O3 -D SINGLE -I ../include/ -L ../target/release/ -l mwa_hyperbeam ./beam_calcs_cuda.c -o beam_calcs_cuda
// LD_LIBRARY_PATH=../target/release ./beam_calcs_cuda ../mwa_full_embedded_element_pattern.h5

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

int main(int argc, char *argv[]) {
    if (argc == 1) {
        fprintf(stderr, "Expected one argument - the path to the HDF5 file.\n");
        exit(1);
    }

    // Get a new FEE beam object from hyperbeam.
    FEEBeam *beam;
    char error[200];
    if (new_fee_beam(argv[1], &beam, error)) {
        printf("Got an error when trying to make an FEEBeam: %s\n", error);
        return EXIT_FAILURE;
    }
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
    // Should we normalise the beam response?
    int norm_to_zenith = 1;

    // Now get a new CUDA FEE beam object.
    FEEBeamCUDA *cuda_beam;
    if (new_cuda_fee_beam(beam, freqs_hz, delays, dip_amps, num_freqs, num_tiles, num_amps, norm_to_zenith, &cuda_beam,
                          error)) {
        printf("Got an error when trying to make an FEEBeamCUDA: %s\n", error);
        return EXIT_FAILURE;
    }

    // Set up the directions to get the beam responses.
    int num_azzas = 1000000;
    FLOAT *az = malloc(num_azzas * sizeof(FLOAT));
    FLOAT *za = malloc(num_azzas * sizeof(FLOAT));
    for (int i = 0; i < num_azzas; i++) {
        az[i] = (-170.0 + i * 340.0 / num_azzas) * M_PI / 180.0;
        za[i] = (10.0 + i * 70.0 / num_azzas) * M_PI / 180.0;
    }
    // Should we apply the parallactic angle correction? Read more here:
    // https://github.com/JLBLine/polarisation_tests_for_FEE
    int parallactic = 1;

    complex FLOAT *jones;
    // hyperbeam expects a pointer to our FLOAT macro. Casting the pointer works
    // fine.
    if (calc_jones_cuda(cuda_beam, num_azzas, az, za, parallactic, (FLOAT **)&jones, error)) {
        printf("Got an error when running calc_jones_cuda: %s\n", error);
        return EXIT_FAILURE;
    }
    printf("The first Jones matrix:\n");
    printf("[[%+.8f%+.8fi,", CREAL(jones[0]), CIMAG(jones[0]));
    printf(" %+.8f%+.8fi]\n", CREAL(jones[1]), CIMAG(jones[1]));
    printf(" [%+.8f%+.8fi,", CREAL(jones[2]), CIMAG(jones[2]));
    printf(" %+.8f%+.8fi]]\n", CREAL(jones[3]), CIMAG(jones[3]));

    // Free the heap-allocated Jones matrices.
    free(jones);
    // Free the beam objects - we must use special functions to do this.
    free_cuda_fee_beam(cuda_beam);
    free_fee_beam(beam);

    return EXIT_SUCCESS;
}
