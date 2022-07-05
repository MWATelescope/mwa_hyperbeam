// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

// Build and run with something like:
// gcc -O3 -I ../include/ -L ../target/release/ -l mwa_hyperbeam ./fee.c -o fee
// LD_LIBRARY_PATH=../target/release ./fee ../mwa_full_embedded_element_pattern.h5

#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "mwa_hyperbeam.h"

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
    if (argc == 1) {
        fprintf(stderr, "Expected one argument - the path to the HDF5 file.\n");
        exit(EXIT_FAILURE);
    }

    // Get a new beam object from hyperbeam.
    FEEBeam *beam;
    if (new_fee_beam(argv[1], &beam))
        handle_hyperbeam_error(__FILE__, __LINE__, "new_fee_beam");

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
    // Should we apply the parallactic angle correction? If so, use this
    // latitude for the MWA. Read more here:
    // https://github.com/MWATelescope/mwa_hyperbeam/blob/main/fee_pols.pdf
    double latitude_rad = -0.4660608448386394;
    // Should the beam-response Jones matrix be in the IAU polarisation order?
    int iau_order = 1;

    // Calculate the Jones matrix for this direction and pointing. This Jones
    // matrix is on the stack.
    complex double jones[4];
    // hyperbeam expects a pointer to doubles. Casting the pointer works fine.
    if (calc_jones(beam, az, za, freq_hz, delays, amps, 16, norm_to_zenith, &latitude_rad, iau_order, (double *)&jones))
        handle_hyperbeam_error(__FILE__, __LINE__, "calc_jones");

    printf("The returned Jones matrix:\n");
    printf("[[%+.8f%+.8fi,", creal(jones[0]), cimag(jones[0]));
    printf(" %+.8f%+.8fi]\n", creal(jones[1]), cimag(jones[1]));
    printf(" [%+.8f%+.8fi,", creal(jones[2]), cimag(jones[2]));
    printf(" %+.8f%+.8fi]]\n", creal(jones[3]), cimag(jones[3]));

    // Amps can have 32 elements to specify amps of the X and Y dipoles. The
    // first 16 elements are X amps, the second 16 are Y amps.
    double amps_2[32] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1,
                         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    // This Jones matrix is on the heap.
    complex double *jones_2 = malloc(4 * sizeof(complex double));
    if (calc_jones(beam, az, za, freq_hz, delays, amps_2, 32, norm_to_zenith, &latitude_rad, iau_order,
                   (double *)jones_2))
        handle_hyperbeam_error(__FILE__, __LINE__, "calc_jones");

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
