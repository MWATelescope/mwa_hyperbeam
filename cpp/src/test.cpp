#include <stdio.h>

#include "beam2016implementation.h"

int main(int argc, char *argv[]) {
    const double delays[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    const double amps[16] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    Beam2016Implementation fullee(delays, amps, "/home/chj/Software/personal/mwa_hyperbeam/");

    double az = 0.78539816;
    double za = 0;
    int freq_hz = 51200000;
    int zenith_norm = 0;
    JonesMatrix jones_matrix = fullee.CalcJones(az, za, freq_hz, zenith_norm);

    return 0;
}
