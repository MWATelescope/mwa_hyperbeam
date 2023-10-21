// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

// Get analytic beam responses from hyperbeam in parallel. See analytic.c for a more
// thorough discussion.

// Build and run with something like:
// gcc -O3 -I ../include/ -L ../target/release/ -l mwa_hyperbeam ./analytic_parallel.c -o analytic_parallel
// LD_LIBRARY_PATH=../target/release ./analytic_parallel

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
    // Get a new beam object from hyperbeam.
    AnalyticBeam *beam;
    // 1 or RTS style, 0 for mwa_pb
    char rts_style = 0;
    // Point to a valid float if you want a custom height
    double dipole_height_metres = 0.5;
    // Point to a valid int if you want a custom number of bowties per row. You
    // almost certainly want this to be 4, unless you're simulating the CRAM
    // tile.
    uint8_t bowties_per_row = 4;
    if (new_analytic_beam(rts_style, &dipole_height_metres, &bowties_per_row, &beam))
        handle_hyperbeam_error(__FILE__, __LINE__, "new_analytic_beam");

    // Set up the directions to test.
    int num_directions = 5000;
    double *az = malloc(num_directions * sizeof(double));
    double *za = malloc(num_directions * sizeof(double));
    for (int i = 0; i < num_directions; i++) {
        az[i] = 45.0 * M_PI / 180.0;
        za[i] = 10.0 * M_PI / 180.0;
    }
    // Delays and amps correspond to dipoles in the "M&C order". See
    // https://wiki.mwatelescope.org/pages/viewpage.action?pageId=48005139) for
    // more info.
    unsigned delays[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    double amps[16] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    int freq_hz = 51200000;
    // MWA latitude
    double latitude_rad = -0.4660608448386394;
    // Should we normalise the beam response?
    int norm_to_zenith = 1;

    // Calculate the Jones matrices for all directions. Rust will do this in
    // parallel.
    complex double *jones = malloc(num_directions * 4 * sizeof(complex double));
    // hyperbeam expects a pointer to doubles. Casting the pointer works fine.
    if (analytic_calc_jones_array(beam, num_directions, az, za, freq_hz, delays, amps, 16, latitude_rad, norm_to_zenith,
                                  (double *)jones))
        handle_hyperbeam_error(__FILE__, __LINE__, "analytic_calc_jones_array");

    printf("The first Jones matrix:\n");
    printf("[[%+.8f%+.8fi,", creal(jones[0]), cimag(jones[0]));
    printf(" %+.8f%+.8fi]\n", creal(jones[1]), cimag(jones[1]));
    printf(" [%+.8f%+.8fi,", creal(jones[2]), cimag(jones[2]));
    printf(" %+.8f%+.8fi]]\n", creal(jones[3]), cimag(jones[3]));

    double amps_2[32] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0};
    complex double *jones_2 = malloc(num_directions * 4 * sizeof(complex double));
    if (analytic_calc_jones_array(beam, num_directions, az, za, freq_hz, delays, amps_2, 32, latitude_rad,
                                  norm_to_zenith, (double *)jones_2))
        handle_hyperbeam_error(__FILE__, __LINE__, "analytic_calc_jones_array");

    printf("The first Jones matrix with altered Y amps:\n");
    printf("[[%+.8f%+.8fi,", creal(jones_2[0]), cimag(jones_2[0]));
    printf(" %+.8f%+.8fi]\n", creal(jones_2[1]), cimag(jones_2[1]));
    printf(" [%+.8f%+.8fi,", creal(jones_2[2]), cimag(jones_2[2]));
    printf(" %+.8f%+.8fi]]\n", creal(jones_2[3]), cimag(jones_2[3]));

    // Freeing memory.
    free(az);
    free(za);
    free(jones);
    free(jones_2);

    // Free the beam - we must use a special function to do this.
    free_analytic_beam(beam);

    return EXIT_SUCCESS;
}
