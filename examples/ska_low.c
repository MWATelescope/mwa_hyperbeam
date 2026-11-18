// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

// Build and run with something like:
// gcc -O3 -I ../include/ -L ../target/release/ -l mwa_hyperbeam ./ska_low.c -o ska_low
// LD_LIBRARY_PATH=../target/release ./ska_low [ska-low-eep-grid FILE | ska-low-eep-sph DIR]
//
// With no arguments, the ska-low-analytic model is used.

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
    // Choose a model. All models use the same SkaLowBeam handle.
    SkaLowBeam *beam;
    double freq_hz = 100e6;
    if (argc >= 3 && strcmp(argv[1], "ska-low-eep-grid") == 0) {
        // 1: apply the SKA-Low sign convention (as EveryBeam does for OSKAR).
        if (new_ska_low_eep_grid_beam(argv[2], 1, &beam))
            handle_hyperbeam_error(__FILE__, __LINE__, "new_ska_low_eep_grid_beam");
    } else if (argc >= 3 && strcmp(argv[1], "ska-low-eep-sph") == 0) {
        // 0: use all available spherical-wave degrees.
        if (new_ska_low_eep_sph_beam(argv[2], 0, &beam))
            handle_hyperbeam_error(__FILE__, __LINE__, "new_ska_low_eep_sph_beam");
    } else {
        // 1: cos(theta) taper (EveryBeam's oskar_dipole_cos).
        if (new_ska_low_analytic_beam(1, &beam))
            handle_hyperbeam_error(__FILE__, __LINE__, "new_ska_low_analytic_beam");
    }
    unsigned num_elements = ska_low_get_num_elements(beam);
    printf("Elements with distinct responses: %u\n", num_elements);

    // A single direction, with the parallactic-angle correction for SKA-Low.
    double latitude_rad = -26.82472208 * M_PI / 180.0;
    double jones[8];
    if (ska_low_calc_jones(beam, 45.0 * M_PI / 180.0, 30.0 * M_PI / 180.0, freq_hz, 0, &latitude_rad, 1, jones))
        handle_hyperbeam_error(__FILE__, __LINE__, "ska_low_calc_jones");
    printf("Element 0 (IAU order, PA-corrected):\n");
    printf("[[%+.6f%+.6fi, %+.6f%+.6fi]\n", jones[0], jones[1], jones[2], jones[3]);
    printf(" [%+.6f%+.6fi, %+.6f%+.6fi]]\n", jones[4], jones[5], jones[6], jones[7]);

    // Many directions at once (computed in parallel).
    int num_directions = 1000;
    double *az = malloc(num_directions * sizeof(double));
    double *za = malloc(num_directions * sizeof(double));
    for (int i = 0; i < num_directions; i++) {
        az[i] = 0.4 * M_PI * i / num_directions;
        za[i] = 0.3 * M_PI * i / num_directions;
    }
    double *jones_array = malloc(8 * num_directions * sizeof(double));
    if (ska_low_calc_jones_array(beam, num_directions, az, za, freq_hz, 0, &latitude_rad, 1, jones_array))
        handle_hyperbeam_error(__FILE__, __LINE__, "ska_low_calc_jones_array");
    printf("Direction 999 XX: %+.6f%+.6fi\n", jones_array[8 * 999], jones_array[8 * 999 + 1]);

    // A station beam: a 4x4 grid of elements 1.5 m apart, pointed at zenith,
    // with the station rotated by 30 degrees (its X dipoles are 30 degrees
    // from East towards North; SKA-Low stations are rigidly rotated, see the
    // COORDINATE_AXES of the MS PHASED_ARRAY table). (For ska-low-eep-sph, the
    // number of positions must match the number of elements in the
    // coefficient files.)
    if (num_elements == 0) {
        int n_el = 16;
        double positions[3 * 16];
        for (int i = 0; i < n_el; i++) {
            positions[3 * i] = 1.5 * (i % 4) - 2.25;
            positions[3 * i + 1] = 1.5 * (i / 4) - 2.25;
            positions[3 * i + 2] = 0.0;
        }
        double rotation_rad = 30.0 * M_PI / 180.0;
        if (ska_low_calc_station_jones_array(beam, n_el, positions, NULL, rotation_rad, num_directions, az, za, freq_hz,
                                             0.0, 0.0, &latitude_rad, 1, jones_array))
            handle_hyperbeam_error(__FILE__, __LINE__, "ska_low_calc_station_jones_array");
        printf("Station beam, direction 0 XX: %+.6f%+.6fi\n", jones_array[0], jones_array[1]);
        printf("Station beam, direction 999 XX: %+.6f%+.6fi\n", jones_array[8 * 999], jones_array[8 * 999 + 1]);
    }

    free(az);
    free(za);
    free(jones_array);
    free_ska_low_beam(beam);
    printf("Done!\n");
    return EXIT_SUCCESS;
}
