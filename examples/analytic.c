// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

// Build and run with something like:
// gcc -O3 -I ../include/ -L ../target/release/ -l mwa_hyperbeam ./analytic.c -o analytic
// LD_LIBRARY_PATH=../target/release ./analytic

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
    char rts_style = 0;                  // 1 or RTS style, 0 for mwa_pb
    double *dipole_height_metres = NULL; // Point to a valid float if you want a custom height
    // Point to a valid int if you want a custom number of bowties per row. You
    // almost certainly want this to be 4, unless you're simulating the CRAM
    // tile.
    uint8_t *bowties_per_row = NULL;
    if (new_analytic_beam(rts_style, dipole_height_metres, bowties_per_row, &beam))
        handle_hyperbeam_error(__FILE__, __LINE__, "new_analytic_beam");

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
    // MWA latitude
    double latitude_rad = -0.4660608448386394;
    // Should we normalise the beam response?
    int norm_to_zenith = 1;

    // Calculate the Jones matrix for this direction and pointing. This Jones
    // matrix is on the stack.
    complex double jones[4];
    // hyperbeam expects a pointer to doubles. Casting the pointer works fine.
    if (analytic_calc_jones(beam, az, za, freq_hz, delays, amps, 16, latitude_rad, norm_to_zenith, (double *)&jones))
        handle_hyperbeam_error(__FILE__, __LINE__, "analytic_calc_jones");

    printf("The returned Jones matrix:\n");
    printf("[[%+.8f%+.8fi,", creal(jones[0]), cimag(jones[0]));
    printf(" %+.8f%+.8fi]\n", creal(jones[1]), cimag(jones[1]));
    printf(" [%+.8f%+.8fi,", creal(jones[2]), cimag(jones[2]));
    printf(" %+.8f%+.8fi]]\n", creal(jones[3]), cimag(jones[3]));

    // Amps can have 32 elements to specify amps of the X and Y dipoles. The
    // first 16 elements are X amps, the second 16 are Y amps.
    double amps_2[32] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
                         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    // This Jones matrix is on the heap.
    complex double *jones_2 = malloc(4 * sizeof(complex double));
    if (analytic_calc_jones(beam, az, za, freq_hz, delays, amps_2, 32, latitude_rad, norm_to_zenith, (double *)jones_2))
        handle_hyperbeam_error(__FILE__, __LINE__, "analytic_calc_jones");

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
    free_analytic_beam(beam);

    /* BONUS ROUND - CRAM TILE */
    uint8_t cram_bowties_per_row = 8;
    if (new_analytic_beam(rts_style, dipole_height_metres, &cram_bowties_per_row, &beam))
        handle_hyperbeam_error(__FILE__, __LINE__, "new_analytic_beam");
    unsigned delays_cram[64] = {3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0, 3, 2,
                                1, 0, 3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0,
                                3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0};
    double amps_cram[64] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1,
                            1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                            1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1};
    complex double *jones_cram = malloc(4 * sizeof(complex double));
    if (analytic_calc_jones(beam, az, za, freq_hz, delays_cram, amps_cram, 64, latitude_rad, norm_to_zenith,
                            (double *)jones_cram))
        handle_hyperbeam_error(__FILE__, __LINE__, "analytic_calc_jones");
    printf("The CRAM Jones matrix:\n");
    printf("[[%+.8f%+.8fi,", creal(jones_cram[0]), cimag(jones_cram[0]));
    printf(" %+.8f%+.8fi]\n", creal(jones_cram[1]), cimag(jones_cram[1]));
    printf(" [%+.8f%+.8fi,", creal(jones_cram[2]), cimag(jones_cram[2]));
    printf(" %+.8f%+.8fi]]\n", creal(jones_cram[3]), cimag(jones_cram[3]));
    free(jones_cram);
    free_analytic_beam(beam);

    return EXIT_SUCCESS;
}
