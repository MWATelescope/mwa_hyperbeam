// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

// Build and run with something like:
// gcc -O3 -I ../include/ -L ../target/release/ -l mwa_hyperbeam ./beam_calcs.c -o beam_calcs
// LD_LIBRARY_PATH=../target/release ./beam_calcs ../mwa_full_embedded_element_pattern.h5

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
    int norm_to_zenith = 0;

    // Calculate the Jones matrix for this direction and pointing.
    double *jones = calc_jones(beam, az, za, freq_hz, delays, amps, norm_to_zenith);
    printf("The returned Jones matrix:\n");
    printf("[[%+.8f%+.8fi,", jones[0], jones[1]);
    printf(" %+.8f%+.8fi]\n", jones[2], jones[3]);
    printf(" [%+.8f%+.8fi,", jones[4], jones[5]);
    printf(" %+.8f%+.8fi]]\n", jones[6], jones[7]);

    // Amps can have 32 elements to specify X and Y elements of the dipoles, but
    // another function is needed to use this extra info. The first 16 elements
    // are X elements, second 16 are Y elements.
    double amps_2[32] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1,
                         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    double *jones_2 = calc_jones_all_amps(beam, az, za, freq_hz, delays, amps_2, norm_to_zenith);
    // The resulting Jones matrix has different elements on the second row,
    // corresponding to the Y element; this is because we only altered the Y
    // amps.
    printf("The returned Jones matrix with altered Y amps:\n");
    printf("[[%+.8f%+.8fi,", jones_2[0], jones_2[1]);
    printf(" %+.8f%+.8fi]\n", jones_2[2], jones_2[3]);
    printf(" [%+.8f%+.8fi,", jones_2[4], jones_2[5]);
    printf(" %+.8f%+.8fi]]\n", jones_2[6], jones_2[7]);

    // Free the Jones matrices.
    free(jones);
    free(jones_2);
    // Free the beam - we must use a special function to do this.
    free_fee_beam(beam);

    return 0;
}
