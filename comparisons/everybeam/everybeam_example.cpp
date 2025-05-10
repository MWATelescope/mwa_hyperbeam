#include <EveryBeam/beammode.h>
#include <EveryBeam/beamnormalisationmode.h>
#include <EveryBeam/griddedresponse/mwagrid.h>
#include <EveryBeam/pointresponse/pointresponse.h>
#include <EveryBeam/telescope/mwa.h>
#include <EveryBeam/telescope/telescope.h>
#include <aocommon/coordinatesystem.h>
#include <casacore/ms/MeasurementSets.h>

#include <chrono>
#include <ctime>
#include <memory>

const size_t NUM_POINTS_TO_SIMULATE = 300'000;
const double RA_RAD = 0.0 * M_PI / 180.0;
const double DEC_RAD = -27.0 * M_PI / 180.0;
const double FREQ_HZ = 150e6;

int main(int argc, char *argv[]) {
    if (argc == 1) {
        printf("Expected an argument (path to a measurement set for EveryBeam to use)\n");
        return 1;
    }
    const char *ms_path = argv[1];

    casacore::MeasurementSet ms(ms_path);
    casacore::MEpoch::ScalarColumn time_column(ms, ms.columnName(casacore::MSMainEnums::TIME));
    casacore::MEpoch first_time = time_column(0);
    casacore::Quantity first_utc_time = first_time.get(casacore::Unit("s"));

    everybeam::Options options;
    // options.coeff_path = "/usr/local/mwa_full_embedded_element_pattern.h5";
    options.coeff_path = std::getenv("MWA_BEAM_FILE");
    options.beam_normalisation_mode = everybeam::BeamNormalisationMode::kFull;
    options.beam_mode = everybeam::BeamMode::kFull;
    options.frequency_interpolation = false;
    everybeam::telescope::MWA beam = everybeam::telescope::MWA(ms, options);

    std::complex<float> *jones =
        (std::complex<float> *)malloc(NUM_POINTS_TO_SIMULATE * 4 * sizeof(std::complex<float>));

    std::unique_ptr<everybeam::pointresponse::PointResponse> pr = beam.GetPointResponse(first_utc_time.getBaseValue());

    auto start = std::chrono::high_resolution_clock::now();
    pr->Response(everybeam::BeamMode::kFull, jones, RA_RAD, DEC_RAD, FREQ_HZ, 0, 0);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    printf("time taken to produce 1 simulation (cold cache): %fs\n", (double)duration.count() / 1e6);

    start = std::chrono::high_resolution_clock::now();
    pr->Response(everybeam::BeamMode::kFull, jones, RA_RAD, DEC_RAD, FREQ_HZ, 0, 0);
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    printf("time taken to produce 1 simulation (hot cache):  %fs\n", (double)duration.count() / 1e6);

    start = std::chrono::high_resolution_clock::now();
    // Attempting to use OpenMP will either cause segfaults or expose the lack
    // of thread safety in HDF5
    // #pragma omp parallel for
    for (size_t i_jones = 0; i_jones < 1000; ++i_jones) {
        std::complex<float> *e = jones + 4 * i_jones;
        pr->Response(everybeam::BeamMode::kFull, e, RA_RAD, DEC_RAD, FREQ_HZ, 0, 0);
    }
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    printf("time taken to produce %ld simulations: %fs\n", 1000, (double)duration.count() / 1e6);

    start = std::chrono::high_resolution_clock::now();
    // Attempting to use OpenMP will either cause segfaults or expose the lack
    // of thread safety in HDF5
    // #pragma omp parallel for
    for (size_t i_jones = 0; i_jones < NUM_POINTS_TO_SIMULATE; ++i_jones) {
        std::complex<float> *e = jones + 4 * i_jones;
        pr->Response(everybeam::BeamMode::kFull, e, RA_RAD, DEC_RAD, FREQ_HZ, 0, 0);
    }
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    printf("time taken to produce %ld simulations: %fs\n", NUM_POINTS_TO_SIMULATE, (double)duration.count() / 1e6);

    printf("\nFirst and last MWA beam responses:\n");
    printf("[%+f%+fi, %+f%+fi\n", jones[0].real(), jones[0].imag(), jones[1].real(), jones[1].imag());
    printf(" %+f%+fi, %+f%+fi]\n", jones[2].real(), jones[2].imag(), jones[3].real(), jones[3].imag());
    printf("[%+f%+fi, %+f%+fi\n", jones[(NUM_POINTS_TO_SIMULATE - 1) * 4 + 0].real(),
           jones[(NUM_POINTS_TO_SIMULATE - 1) * 4 + 0].imag(), jones[(NUM_POINTS_TO_SIMULATE - 1) * 4 + 1].real(),
           jones[(NUM_POINTS_TO_SIMULATE - 1) * 4 + 1].imag());
    printf(" %+f%+fi, %+f%+fi]\n", jones[(NUM_POINTS_TO_SIMULATE - 1) * 4 + 2].real(),
           jones[(NUM_POINTS_TO_SIMULATE - 1) * 4 + 2].imag(), jones[(NUM_POINTS_TO_SIMULATE - 1) * 4 + 3].real(),
           jones[(NUM_POINTS_TO_SIMULATE - 1) * 4 + 3].imag());

    return 0;
}
