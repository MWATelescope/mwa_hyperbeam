// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

// Get the frequencies defined in the FEE beam file. hyperbeam can only use
// these frequencies for beam responses.

// Build and run with something like:
// gcc -O3 -I ../include/ -L ../target/release/ -l mwa_hyperbeam ./get_freqs.c -o get_freqs
// LD_LIBRARY_PATH=../target/release ./get_freqs ../mwa_full_embedded_element_pattern.h5

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

    // Of the available frequencies, which is closest to 255 MHz?
    printf("Closest freq. to 255 MHz: %.2f MHz\n", (double)closest_freq(beam, 255000000) / 1e6);

    // Get the frequencies from the FEEBeam struct.
    size_t num_freqs;
    const unsigned *freqs;
    get_fee_beam_freqs(beam, &freqs, &num_freqs);

    // Print them out.
    printf("All frequencies:\n");
    for (int i = 0; i < num_freqs; i++) {
        double f = (double)freqs[i] / 1e6;
        printf("%*.2f MHz", 6, f);
        if (i > 0 && (i + 1) % 4 == 0) {
            printf(",\n");
        } else if (i == num_freqs - 1) {
            printf("\n");
        } else {
            printf(", ");
        }
    }

    // We DON'T own the freqs array - don't free it.
    // Free the beam - we must use a special function to do this.
    free_fee_beam(beam);

    return 0;
}
