// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

// Build and run with something like:
// gcc -O3 -I ../include/ -L ../target/release/ -l mwa_hyperbeam ./beam_calcs_many.c -o beam_calcs_many
// LD_LIBRARY_PATH=../target/release ./beam_calcs_many ../mwa_full_embedded_element_pattern.h5

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "mwa_hyperbeam.h"

int main(int argc, char *argv[]) {
    if (argc == 1) {
        fprintf(stderr, "Expected one argument - the path to the HDF5 file.\n");
        exit(1);
    }

    // Get a new beam from hyperbeam.
    FEEBeam *beam = new_fee_beam(argv[1]);

    // Set up the directions to test.
    int num_directions = 5000000;
    double *az = malloc(num_directions * sizeof(double));
    double *za = malloc(num_directions * sizeof(double));
    for (int i = 0; i < num_directions; i++) {
        az[i] = 45.0 * M_PI / 180.0;
        za[i] = 10.0 * M_PI / 180.0;
    }
    unsigned delays[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    double amps[16] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    int freq_hz = 51200000;
    int norm_to_zenith = 0;

    /* Calculate the Jones matrices for all directions. Rust will do this in */
    /* parallel. */
    double *jones = calc_jones_array(beam, num_directions, az, za, freq_hz, delays, amps, norm_to_zenith);
    printf("The first Jones matrix:\n");
    printf("[[%+.8f%+.8fi,", jones[0], jones[1]);
    printf(" %+.8f%+.8fi]\n", jones[2], jones[3]);
    printf(" [%+.8f%+.8fi,", jones[4], jones[5]);
    printf(" %+.8f%+.8fi]]\n", jones[6], jones[7]);

    // Freeing memory.
    free(az);
    free(za);
    free(jones);

    // Free the beam - we must use a special function to do this.
    free_fee_beam(beam);

    return 0;
}
