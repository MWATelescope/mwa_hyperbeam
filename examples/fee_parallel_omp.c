// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

// Use OpenMP for running hyperbeam in parallel, rather than letting hyperbeam
// calculate beam responses in parallel. See fee.c for a more thorough
// discussion.

// See beam_calcs.c for a more thorough discussion. This example merely
// illustrates using hyperbeam with OpenMP.

// Build and run with something like:
// gcc -O3 -fopenmp -I ../include/ -L ../target/release/ -l mwa_hyperbeam ./fee_parallel_omp.c -o fee_parallel_omp
// LD_LIBRARY_PATH=../target/release ./fee_parallel_omp ../mwa_full_embedded_element_pattern.h5

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
        exit(1);
    }

    // Get a new beam object from hyperbeam.
    FEEBeam *beam;
    if (new_fee_beam(argv[1], &beam))
        handle_hyperbeam_error(__FILE__, __LINE__, "new_fee_beam");

    // Set up the directions to test.
    int num_directions = 5000;
    double *az = malloc(num_directions * sizeof(double));
    double *za = malloc(num_directions * sizeof(double));
    for (int i = 0; i < num_directions; i++) {
        double coord_deg = 5.0 + (double)i * 80.0 / (double)num_directions;
        double coord_rad = coord_deg * M_PI / 180.0;
        az[i] = coord_rad;
        za[i] = coord_rad;
    }
    // Delays and amps correspond to dipoles in the "M&C order". See
    // https://wiki.mwatelescope.org/pages/viewpage.action?pageId=48005139) for
    // more info.
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

    // Calculate the Jones matrices for all directions.
    complex double *jones = malloc(num_directions * 4 * sizeof(complex double));
#pragma omp parallel for
    for (int i = 0; i < num_directions; i++) {
        // hyperbeam expects a pointer to doubles. Casting the pointer works fine.
        if (calc_jones(beam, az[i], za[i], freq_hz, delays, amps, 16, norm_to_zenith, &latitude_rad, iau_order,
                       (double *)(jones + i * 4)))
            handle_hyperbeam_error(__FILE__, __LINE__, "calc_jones");
    }

    printf("The first Jones matrix:\n");
    printf("[[%+.8f%+.8fi,", creal(jones[0]), cimag(jones[0]));
    printf(" %+.8f%+.8fi]\n", creal(jones[1]), cimag(jones[1]));
    printf(" [%+.8f%+.8fi,", creal(jones[2]), cimag(jones[2]));
    printf(" %+.8f%+.8fi]]\n", creal(jones[3]), cimag(jones[3]));

    // Freeing memory.
    free(az);
    free(za);
    free(jones);

    // Free the beam - we must use a special function to do this.
    free_fee_beam(beam);

    return EXIT_SUCCESS;
}
