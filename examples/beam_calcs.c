// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

// Build and run with something like:
// gcc -O3 -I ../include/ -L ../target/release/ -l mwa_hyperbeam ./beam_calcs.c -o beam_calcs
// LD_LIBRARY_PATH=../target/release ./beam_calcs ../mwa_full_embedded_element_pattern.h5

#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "mwa_hyperbeam.h"

int main(int argc, char *argv[]) {
    if (argc == 1) {
        fprintf(stderr, "Expected one argument - the path to the HDF5 file.\n");
        exit(1);
    }

    // Get a new beam object from hyperbeam.
    FEEBeam *beam;
    char error[200];
    if (new_fee_beam(argv[1], &beam, error)) {
        printf("Got an error when trying to make an FEEBeam: %s\n", error);
        return EXIT_FAILURE;
    }

    // Set up the direction and pointing to test.
    double az = 45.0 * M_PI / 180.0;
    double za = 80.0 * M_PI / 180.0;
    // Delays and amps correspond to dipoles in the "M&C order". See
    // https://wiki.mwatelescope.org/pages/viewpage.action?pageId=48005139) for
    // more info. Amps refer to dipole gains, and are usually set to 1 or 0 (if
    // a dipole is dead).
    unsigned delays[16] = {3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0};
    double amps[16] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1};
    int freq_hz = 51200000;
    // Should we normalise the beam response?
    int norm_to_zenith = 1;
    // Should we apply the parallactic angle correction? Read more here:
    // https://github.com/JLBLine/polarisation_tests_for_FEE
    int parallactic = 1;

    // Calculate the Jones matrix for this direction and pointing. This Jones
    // matrix is on the stack.
    complex double jones[4];
    // hyperbeam expects a pointer to doubles. Casting the pointer works fine.
    if (calc_jones(beam, az, za, freq_hz, delays, amps, 16, norm_to_zenith, parallactic, (double *)&jones, error)) {
        printf("Got an error when running calc_jones: %s\n", error);
        return EXIT_FAILURE;
    }
    printf("The returned Jones matrix:\n");
    printf("[[%+.8f%+.8fi,", creal(jones[0]), cimag(jones[0]));
    printf(" %+.8f%+.8fi]\n", creal(jones[1]), cimag(jones[1]));
    printf(" [%+.8f%+.8fi,", creal(jones[2]), cimag(jones[2]));
    printf(" %+.8f%+.8fi]]\n", creal(jones[3]), cimag(jones[3]));

    // Amps can have 32 elements to specify X and Y elements of the dipoles. The
    // first 16 elements are X elements, second 16 are Y elements.
    double amps_2[32] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1,
                         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    // This Jones matrix is on the heap.
    complex double *jones_2 = malloc(4 * sizeof(complex double));
    if (calc_jones(beam, az, za, freq_hz, delays, amps_2, 32, norm_to_zenith, parallactic, (double *)jones_2, error)) {
        printf("Got an error when running calc_jones_all_amps: %s\n", error);
        return EXIT_FAILURE;
    }
    // The resulting Jones matrix has different elements on the second row,
    // corresponding to the Y element; this is because we only altered the Y
    // amps.
    printf("The returned Jones matrix with altered Y amps:\n");
    printf("[[%+.8f%+.8fi,", creal(jones_2[0]), cimag(jones_2[0]));
    printf(" %+.8f%+.8fi]\n", creal(jones_2[1]), cimag(jones_2[1]));
    printf(" [%+.8f%+.8fi,", creal(jones_2[2]), cimag(jones_2[2]));
    printf(" %+.8f%+.8fi]]\n", creal(jones_2[3]), cimag(jones_2[3]));

    // Free the heap-allocated Jones matrix.
    free(jones_2);
    // Free the beam - we must use a special function to do this.
    free_fee_beam(beam);

    return EXIT_SUCCESS;
}
