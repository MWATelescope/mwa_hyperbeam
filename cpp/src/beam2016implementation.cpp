/**
 * C++ implementation of Full Embeded Element beam model for MWA based on beam_full_EE.py script
 * and Sokolowski et al (2016) paper
 * Implemented by Marcin Sokolowski (May 2017) - marcin.sokolowski@curtin.edu.au
 * 20 May 2017 : Somewhat optimized by André Offringa.
 */

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include <boost/math/special_functions/legendre.hpp>

#include <boost/filesystem.hpp>

#include <H5Cpp.h>

#include "beam2016implementation.h"
#include "system.h"

using namespace std;
using namespace H5;

// constants :
static const double deg2rad = M_PI / 180.00;

// beamformer step in pico-seconds
#define DELAY_STEP 435.0e-12

#define DEFAULT_H5_FILE      "mwa_full_embedded_element_pattern.h5"
#define DEFAULT_H5_FILE_PATH "mwapy/data/"
#define N_ANT_COUNT          16

bool Beam2016Implementation::has_freq(int freq_hz) const {
    return std::find(m_freq_list.begin(), m_freq_list.end(), freq_hz) != m_freq_list.end();
}

int Beam2016Implementation::find_closest_freq(int freq_hz) const {
    double min_diff = 1e20;
    int best_idx = -1;

    for (size_t i = 0; i < m_freq_list.size(); i++) {
        double diff = abs(m_freq_list[i] - freq_hz);
        if (diff < min_diff) {
            min_diff = diff;
            best_idx = i;
        }
    }

    if (best_idx >= 0) {
        return m_freq_list[best_idx];
    }

    return m_freq_list[0];
}

Beam2016Implementation::Beam2016Implementation(const double *delays, const double *amps, const std::string &searchPath)
    : _calcModesLastFreqHz(-1), _calcModesLastDelays(), _calcModesLastAmps(), _h5File(), _searchPath(searchPath),
      _factorial(100) {
    if (delays == nullptr)
        std::fill(_delays, _delays + N_ANT_COUNT, 0.0);
    else
        std::copy_n(delays, N_ANT_COUNT, _delays);

    if (amps == nullptr)
        std::fill(_amps, _amps + N_ANT_COUNT, 1.0);
    else
        std::copy_n(amps, N_ANT_COUNT, _amps);

    Read();
    _jPowerTable[0] = std::complex<double>(1.0, 0.0);
    _jPowerTable[1] = std::complex<double>(0, 1.0);
    _jPowerTable[2] = std::complex<double>(-1.0, 0.0);
    _jPowerTable[3] = std::complex<double>(0.0, -1.0);
}

//-------------------------------------------------------------------- Calculation of Jones matrix
//------------------------------------------------------------------------------------------------------------------
void Beam2016Implementation::CalcJonesArray(vector<vector<double>> &azim_arr, vector<vector<double>> &za_arr,
                                            vector<vector<JonesMatrix>> &jones, int freq_hz_param, bool bZenithNorm) {
    // convert AZIM -> FEKO PHI phi=90-azim or azim=90-phi :
    // python : phi_arr=math.pi/2-phi_arr #Convert to East through North (FEKO coords)
    for (size_t y = 0; y < azim_arr.size(); y++) {
        vector<double> &image_row = azim_arr[y];

        for (size_t x = 0; x < image_row.size(); x++) {
            image_row[x] = M_PI / 2.00 - image_row[x];

            // phi_arr[phi_arr < 0] += 2*math.pi #360 wrap
            if (image_row[x] < 0) {
                image_row[x] += 2.0 * M_PI;
            }
        }
    }

    JonesMatrix::zeros(jones, azim_arr[0].size(), azim_arr.size());

    for (size_t y = 0; y < azim_arr.size(); y++) {
        vector<double> &image_row = azim_arr[y];
        for (size_t x = 0; x < image_row.size(); x++) {
            double azim_deg = image_row[x];
            double za_deg = (za_arr[y])[x];

            jones[y][x] = CalcJones(azim_deg, za_deg, freq_hz_param, bZenithNorm);
        }
    }
}

JonesMatrix Beam2016Implementation::CalcJonesDirect(double az_rad, double za_rad, const Coefficients &coefsX,
                                                    const Coefficients &coefsY) {
    // convert AZIM -> FEKO PHI phi=90-azim or azim=90-phi :
    // python : phi_arr=math.pi/2-phi_arr #Convert to East through North (FEKO coords)
    JonesMatrix jones;
    double phi_rad = M_PI / 2.00 - az_rad;

    CalcSigmas(phi_rad, za_rad, coefsX, 'X', jones);
    CalcSigmas(phi_rad, za_rad, coefsY, 'Y', jones);

    return jones;
}

JonesMatrix Beam2016Implementation::CalcJones(double az_deg, double za_deg, int freq_hz_param, bool bZenithNorm) {
    recursive_lock<std::mutex> lock(_mutex, std::defer_lock);
    return CalcJones(az_deg, za_deg, freq_hz_param, _delays, _amps, lock, bZenithNorm);
}

JonesMatrix Beam2016Implementation::CalcJones(double az_deg, double za_deg, int freq_hz, const double *delays,
                                              const double *amps, recursive_lock<std::mutex> &lock, bool bZenithNorm) {
    if (!has_freq(freq_hz)) {
        freq_hz = find_closest_freq(freq_hz);
    }

    Coefficients coefsX, coefsY;
    GetModes(freq_hz, N_ANT_COUNT, delays, amps, coefsX, coefsY, lock);

    JonesMatrix result = CalcJonesDirect(az_deg * deg2rad, za_deg * deg2rad, coefsX, coefsY);

    if (bZenithNorm) {
        JonesMatrix normMatrix;

        std::lock_guard<recursive_lock<std::mutex>> glock(lock);
        std::map<int, JonesMatrix>::const_iterator iter = _normJonesCache.find(freq_hz);
        if (iter == _normJonesCache.end()) {
            normMatrix = CalcZenithNormMatrix(freq_hz, lock);
            _normJonesCache.insert(std::make_pair(freq_hz, normMatrix));
        } else {
            normMatrix = iter->second;
        }

        result.j00 = result.j00 / normMatrix.j00;
        result.j01 = result.j01 / normMatrix.j01;
        result.j10 = result.j10 / normMatrix.j10;
        result.j11 = result.j11 / normMatrix.j11;
    }

    return result;
}

JonesMatrix Beam2016Implementation::CalcZenithNormMatrix(int freq_hz, recursive_lock<std::mutex> &lock) {
    // std::cout << "INFO : calculating Jones matrix for frequency = " << freq_hz << " Hz\n";

    // Azimuth angles at which Jones components are maximum (see beam_full_EE.py for comments):
    //  max_phis=[[math.pi/2,math.pi],[0,math.pi/2]] #phi where each Jones vector is max
    double j00_max_az = 90.00;
    double j01_max_az = 180.00;
    double j10_max_az = 0.00;
    double j11_max_az = 90.00;

    JonesMatrix tmp_jones = JonesMatrix(1, 1, 1, 1);
    JonesMatrix jonesMatrix;

    // default delays at zenith
    const double defaultDelays[N_ANT_COUNT] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    const double defaultAmps[N_ANT_COUNT] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

    // j00 :
    tmp_jones = CalcJones(j00_max_az, 0, freq_hz, defaultDelays, defaultAmps, lock, false);
    jonesMatrix.j00 = abs(tmp_jones.j00);

    // j01 :
    tmp_jones = CalcJones(j01_max_az, 0, freq_hz, defaultDelays, defaultAmps, lock, false);
    jonesMatrix.j01 = abs(tmp_jones.j01);

    // j10 :
    tmp_jones = CalcJones(j10_max_az, 0, freq_hz, defaultDelays, defaultAmps, lock, false);
    jonesMatrix.j10 = abs(tmp_jones.j10);

    // j11 :
    tmp_jones = CalcJones(j11_max_az, 0, freq_hz, defaultDelays, defaultAmps, lock, false);
    jonesMatrix.j11 = abs(tmp_jones.j11);

    return jonesMatrix;
}

//-------------------------------------------------------------------- Calculation of Jones matrix components for given
//polarisation (eq. 4 and 5 in the Sokolowski et al (2017) paper --------------------------------
void Beam2016Implementation::CalcSigmas(double phi, double theta, const Coefficients &coefs, char pol,
                                        JonesMatrix &jones_matrix) const {
    int nmax = int(coefs.Nmax);

    double u = cos(theta);

    vector<double> P1sin_arr, P1_arr;

    P1sin(5, 0.0, P1sin_arr, P1_arr);
    printf("nmax: 5\ntheta: 0\n");
    for (int i = 0; i < nmax * nmax + 2 * nmax; i++) {
        printf("P1sin_arr[%d]: %f\n", i, P1sin_arr[i]);
    }
    for (int i = 0; i < nmax * nmax + 2 * nmax; i++) {
        printf("P1_arr[%d]: %f\n", i, P1_arr[i]);
    }
    printf("\n\n");
    P1sin(nmax, theta, P1sin_arr, P1_arr);

    if (coefs.N_accum.size() != coefs.M_accum.size()) {
        throw std::runtime_error("ERROR : size of N_accnum != M_accum ( " + std::to_string(coefs.N_accum.size()) +
                                 " != " + std::to_string(coefs.M_accum.size()) + ")");
    }

    complex<double> sigma_P(0, 0), sigma_T(0, 0);
    for (size_t i = 0; i < coefs.N_accum.size(); i++) {
        double N = coefs.N_accum[i];
        int n = int(N);

        double M = coefs.M_accum[i];
        // int    m = int(M);
        double m_abs_m = coefs.MabsM[i];

        double c_mn_sqr = (0.5 * (2 * N + 1) * _factorial(N - abs(M)) / _factorial(N + abs(M)));
        double c_mn = sqrt(c_mn_sqr);
        // Optimisation comment :
        // Possibly this might be a faster version - but does not seem so, so I leave it as it was
        // MS tested it (2017-05-17), but does not seem to give a significant speed up, so for now the old version left
        // If tested again comment the 2 lines above and uncomment the line below ,
        // also verify that lines in CalcModes ( after comment "Intialisation of Cmn vector :") are un-commented - FOR
        // NOW THEY ARE COMMENTED OUT NOT TO CALCULATE SOMETHING WHICH IS NOT USED I've tested it on 2017-05-19 and the
        // version with line below and lines at the end of CalcModes uncommented runs in ~9m30sec and the current one
        // was ~9m40sec so I leave the current version as I've tested it for longer time (MS), but it can be restored if
        // some further optimisation is needed (but it will not be a breaktrough). double c_mn = Cmn[i];

        complex<double> ejm_phi(cos(M * phi), sin(M * phi));
        complex<double> phi_comp = (ejm_phi * c_mn) / (sqrt(N * (N + 1))) * m_abs_m;

        complex<double> j_power_n = JPower(n);
        complex<double> E_theta_mn =
            j_power_n *
            (P1sin_arr[i] * (fabs(M) * coefs.Q2_accum[i] * u - M * coefs.Q1_accum[i]) + coefs.Q2_accum[i] * P1_arr[i]);

        complex<double> j_power_np1 = JPower(n + 1);
        complex<double> E_phi_mn =
            j_power_np1 *
            (P1sin_arr[i] * (M * coefs.Q2_accum[i] - fabs(M) * coefs.Q1_accum[i] * u) - coefs.Q1_accum[i] * P1_arr[i]);

        sigma_P = sigma_P + phi_comp * E_phi_mn;
        sigma_T = sigma_T + phi_comp * E_theta_mn;
    }

    if (pol == 'X') {
        jones_matrix.j00 = sigma_T;
        jones_matrix.j01 = -sigma_P; // as it is now in python code in sign_fix branch
    } else {
        jones_matrix.j10 = sigma_T;
        jones_matrix.j11 = -sigma_P; // as it is now in python code in sign_fix branch
    }
}

//-------------------------------------------------------------------- Calculation of spherical harmonics coefficients
//(eq. 3-6 in the Sokolowski et al 2016 paper)  --------------------------------
// function comparing current parameters : frequency, delays and amplitudes with those previously used to calculate
// spherical waves coefficients (stored in the 3 variables : m_CalcModesLastFreqHz , , m_CalcModesLastAmps )
bool Beam2016Implementation::IsCalcModesRequired(int freq_hz, int n_ant, const double *delays, const double *amps) {
    if (freq_hz != _calcModesLastFreqHz) {
        return true;
    }

    if (_calcModesLastDelays.empty() || _calcModesLastAmps.empty()) {
        return true;
    }

    for (int i = 0; i < n_ant; i++) {
        if (delays[i] != _calcModesLastDelays[i])
            return true;
        if (amps[i] != _calcModesLastAmps[i])
            return true;
    }

    return false;
}

// function calculating coefficients for X and Y and storing parameters frequency, delays and amplitudes
void Beam2016Implementation::GetModes(int freq_hz, size_t n_ant, const double *delays, const double *amps,
                                      Coefficients &coefsX, Coefficients &coefsY, recursive_lock<std::mutex> &lock) {
    std::unique_lock<recursive_lock<std::mutex>> glock(lock);
    if (IsCalcModesRequired(freq_hz, n_ant, delays, amps)) {
        glock.unlock();
        coefsX.Nmax = CalcModes(freq_hz, n_ant, delays, amps, 'X', coefsX, lock);
        coefsY.Nmax = CalcModes(freq_hz, n_ant, delays, amps, 'Y', coefsY, lock);

        glock.lock();
        _coefX = coefsX;
        _coefY = coefsY;
        _calcModesLastFreqHz = freq_hz;
        _calcModesLastDelays.assign(delays, delays + n_ant);
        _calcModesLastAmps.assign(amps, amps + n_ant);
    } else {
        coefsX = _coefX;
        coefsY = _coefY;
    }
}

// function calculating all coefficients Q1, Q2, N, M and derived MabsM, Nmax for a given polarisation (X or Y)
double Beam2016Implementation::CalcModes(int freq_hz, size_t n_ant, const double *delays, const double *amp, char pol,
                                         Coefficients &coefs, recursive_lock<std::mutex> &lock) {
    vector<double> phases(n_ant);
    double Nmax = 0;
    coefs.M_accum.clear();
    coefs.N_accum.clear();
    coefs.MabsM.clear();
    coefs.Cmn.clear();

    int modes_size = m_Modes[0].size();
    coefs.Q1_accum.assign(modes_size, 0.0);
    coefs.Q2_accum.assign(modes_size, 0.0);

    for (size_t a = 0; a < n_ant; a++) {
        double phase = 2 * M_PI * freq_hz * (-double(delays[a]) * DELAY_STEP);

        phases[a] = phase;

        // complex excitation voltage:
        // self.amps[pol]*np.exp(1.0j*phases)
        complex<double> phase_factor(cos(phase), sin(phase));
        complex<double> Vcplx = amp[a] * phase_factor;

        DataSetIndex index(pol, a, freq_hz);

        const vector<vector<double>> &Q_all = GetDataSet(index, lock);

        size_t n_ant_coeff = Q_all[0].size();
        vector<double> Ms1, Ns1, Ms2, Ns2;
        const vector<double> &modes_Type = m_Modes[0];
        const vector<double> &modes_M = m_Modes[1];
        const vector<double> &modes_N = m_Modes[2];

        int bUpdateNAccum = 0;

        // list of indexes where S=1 coefficients seat in array m_Modes ( and m_Modes_M and m_Modes_N )
        vector<int> s1_list;
        // idem for S=2
        vector<int> s2_list;

        for (size_t coeff = 0; coeff < n_ant_coeff; coeff++) {
            int mode_type = modes_Type[coeff];

            if (mode_type <= 1) {
                // s=1 modes :
                s1_list.push_back(coeff);

                Ms1.push_back(modes_M[coeff]);
                Ns1.push_back(modes_N[coeff]);
                if (modes_N[coeff] > Nmax) {
                    Nmax = modes_N[coeff];
                    bUpdateNAccum = 1;
                }
            } else {
                // s=2 modes :
                s2_list.push_back(coeff);

                Ms2.push_back(modes_M[coeff]);
                Ns2.push_back(modes_N[coeff]);
            }
        }

        if (bUpdateNAccum > 0) {
            coefs.N_accum = Ns1;
            coefs.M_accum = Ms1;
        }

        if (s1_list.size() != s2_list.size() || s2_list.size() != (n_ant_coeff / 2)) {
            throw std::runtime_error("Wrong number of coefficients for s1 and s2 condition " +
                                     std::to_string(s1_list.size()) + " = =" + std::to_string(s2_list.size()) +
                                     " == " + std::to_string(n_ant_coeff / 2) + " not satisfied");
        }

        vector<std::complex<double>> Q1, Q2;
        const vector<double> &Q_all_0 = Q_all[0];
        const vector<double> &Q_all_1 = Q_all[1];
        int my_len_half = (n_ant_coeff / 2);

        for (int i = 0; i < my_len_half; i++) {
            // calculate Q1:
            int s1_idx = s1_list[i];
            double s10_coeff = Q_all_0[s1_idx];
            double s11_coeff = Q_all_1[s1_idx];
            double arg = s11_coeff * deg2rad;
            complex<double> tmp(cos(arg), sin(arg));
            complex<double> q1_val = s10_coeff * tmp;
            Q1.push_back(q1_val);

            // calculate Q2:
            int s2_idx = s2_list[i];
            double s20_coeff = Q_all_0[s2_idx];
            double s21_coeff = Q_all_1[s2_idx];
            double arg2 = s21_coeff * deg2rad;
            complex<double> tmp2(cos(arg2), sin(arg2));
            complex<double> q2_val = s20_coeff * tmp2;
            Q2.push_back(q2_val);

            coefs.Q1_accum[i] = coefs.Q1_accum[i] + q1_val * Vcplx;
            coefs.Q2_accum[i] = coefs.Q2_accum[i] + q2_val * Vcplx;
        }
    }

    // Same as tragic python code:
    // MabsM=-M/np.abs(M)
    // MabsM[MabsM==np.NaN]=1 #for M=0, replace NaN with MabsM=1;
    // MabsM=(MabsM)**M
    for (size_t i = 0; i < coefs.M_accum.size(); i++) {
        int m = int(coefs.M_accum[i]);

        double m_abs_m = 1;
        if (m > 0) {
            if ((m % 2) != 0) {
                m_abs_m = -1;
            }
        }

        coefs.MabsM.push_back(m_abs_m);
    }

    return Nmax;
}

//------------------------------------------------------------------------------------------------------ maths functions
//and wrappers ---------------------------------------------------------------------------------------
// OUTPUT : returns list of Legendre polynomial values calculated up to order nmax :
int Beam2016Implementation::P1sin(int nmax, double theta, vector<double> &p1sin_out, vector<double> &p1_out) {
    int size = nmax * nmax + 2 * nmax;
    printf("size: %d\n", size);
    p1sin_out.resize(size);
    p1_out.resize(size);

    double sin_th, u;
    sincos(theta, &sin_th, &u);
    double delu = 1e-6;

    vector<double> P, Pm1, Pm_sin, Pu_mdelu, Pm_sin_merged, Pm1_merged;
    P.reserve(nmax + 1);
    Pm1.reserve(nmax + 1);
    Pm_sin.reserve(nmax + 1);
    Pu_mdelu.reserve(nmax + 1);
    Pm_sin_merged.reserve(nmax * 2 + 1);
    Pm1_merged.reserve(nmax * 2 + 1);

    // Create a look-up table for the legendre polynomials
    // Such that legendre_table[ m * nmax + (n-1) ] = legendre(n, m, u)
    vector<double> legendre_table(nmax * (nmax + 1));
    vector<double>::iterator legendre_iter = legendre_table.begin();
    for (int m = 0; m != nmax + 1; ++m) {
        double leg0 = boost::math::legendre_p(0, m, u);
        double leg1 = boost::math::legendre_p(1, m, u);
        *legendre_iter = leg1;
        ++legendre_iter;
        for (int n = 2; n != nmax + 1; ++n) {
            if (m < n)
                *legendre_iter = boost::math::legendre_next(n - 1, m, u, leg1, leg0);
            else if (m == n)
                *legendre_iter = boost::math::legendre_p(n, m, u);
            else
                *legendre_iter = 0.0;
            leg0 = leg1;
            leg1 = *legendre_iter;
            ++legendre_iter;
        }
    }
    printf("legendre table\n");
    for (int i = 0; i < nmax * (nmax + 1); i++) {
    // for (int i = 0; i < 50; i++) {
        printf("%2d: %f %f %f %f\n", i, legendre_table[4 * i], legendre_table[4 * i + 1], legendre_table[4 * i + 2],
               legendre_table[4 * i + 3]);
    }

    for (int n = 1; n <= nmax; n++) {
        P.resize(n + 1);
        // Assign P[m] to legendre(n, m, u)
        // This is equal to:
        // lpmv(P, n , u);
        for (size_t m = 0; m != size_t(n) + 1; ++m)
            P[m] = legendre_table[m * nmax + (n - 1)];

        // skip first 1 and build table Pm1 (P minus 1 )
        Pm1.assign(P.begin() + 1, P.end());
        Pm1.push_back(0);

        Pm_sin.assign(n + 1, 0.0);
        if (u == 1 || u == -1) {
            // In this case we take the easy approach and don't use
            // precalculated polynomials, since this path does not occur often.

            // TODO This path doesn't make sense.
            // I think Marcin should look at this; this path only occurs on polar positions so
            // is rare, but what is done here looks wrong: first calculate *all* polynomials,
            // then only use the pol for m=1. Pm_sin for indices 0 and >=2 are not calculated, this
            // seems not right.
            Pu_mdelu.resize(1);
            lpmv(Pu_mdelu, n, u - delu);

            // Pm_sin[1,0]=-(P[0]-Pu_mdelu[0])/delu #backward difference
            if (u == -1)
                Pm_sin[1] = -(Pu_mdelu[0] - P[0]) / delu; // #forward difference
            else
                Pm_sin[1] = -(P[0] - Pu_mdelu[0]) / delu;
        } else {
            for (size_t i = 0; i < P.size(); i++) {
                Pm_sin[i] = P[i] / sin_th;
            }
        }

        Pm_sin_merged.assign(Pm_sin.rbegin(), Pm_sin.rend() - 1);
        Pm_sin_merged.insert(Pm_sin_merged.end(), Pm_sin.begin(), Pm_sin.end());

        int ind_start = (n - 1) * (n - 1) + 2 * (n - 1); // #start index to populate
        int ind_stop = n * n + 2 * n;                    //#stop index to populate

        // P_sin[np.arange(ind_start,ind_stop)]=np.append(np.flipud(Pm_sin[1::,0]),Pm_sin)
        int modified = 0;
        for (int i = ind_start; i < ind_stop; i++) {
            p1sin_out[i] = (Pm_sin_merged[modified]);
            modified++;
        }

        // P1[np.arange(ind_start,ind_stop)]=np.append(np.flipud(Pm1[1::,0]),Pm1)
        Pm1_merged.assign(Pm1.rbegin(), Pm1.rend() - 1);
        Pm1_merged.insert(Pm1_merged.end(), Pm1.begin(), Pm1.end());
        modified = 0;
        for (int i = ind_start; i < ind_stop; i++) {
            p1_out[i] = Pm1_merged[modified];
            modified++;
        }
    }

    return nmax;
}

// Legendre polynomials :
void Beam2016Implementation::lpmv(vector<double> &output, int n, double x) {
    for (size_t order = 0; order < output.size(); order++) {
        double val = boost::math::legendre_p<double>(n, order, x);
        output[order] = val;
    }
}

//----------------------------------------------------------------------------------- HDF5 File interface and data
//structures for H5 data -----------------------------------------------------------------------------------
// This function goes thorugh all dataset names and records them info list of strings :
// Beam2016Implementation::m_obj_list
herr_t Beam2016Implementation::list_obj_iterate(hid_t loc_id, const char *name, const H5O_info_t *info,
                                                void *operator_data) {
    string szTmp;
    Beam2016Implementation *pBeamModelPtr = (Beam2016Implementation *)operator_data;

    if (pBeamModelPtr == nullptr)
        throw std::runtime_error("The pointer to  Beam2016Implementation class in "
                                 "Beam2016Implementation::list_obj_iterate must not be null");

    if (name[0] == '.') { /* Root group, do not print '.' */
    } else {
        switch (info->type) {
        case H5O_TYPE_GROUP:
        case H5O_TYPE_NAMED_DATATYPE:
        default:
            break;
        case H5O_TYPE_DATASET:
            szTmp = name;
            pBeamModelPtr->m_obj_list.push_back(szTmp);
            break;
        }
    }

    return 0;
}

const std::vector<std::vector<double>> &Beam2016Implementation::GetDataSet(const DataSetIndex &index,
                                                                           recursive_lock<std::mutex> &lock) {
    std::lock_guard<recursive_lock<std::mutex>> glock(lock);
    auto iter = _dataSetCache.find(index);
    if (iter == _dataSetCache.end()) {
        iter = _dataSetCache.emplace(index, std::vector<std::vector<double>>()).first;
        ReadDataSet(index.Name(), iter->second, *_h5File);
    }
    return iter->second;
}

// Read dataset_name from H5 file
void Beam2016Implementation::ReadDataSet(const std::string &dataset_name, vector<vector<double>> &out_vector,
                                         H5::H5File &h5File) {
    DataSet modes = h5File.openDataSet(dataset_name);
    DataSpace modes_dataspace = modes.getSpace();
    int rank = modes_dataspace.getSimpleExtentNdims();
    hsize_t dims_out[2];
    modes_dataspace.getSimpleExtentDims(dims_out, NULL);
    modes_dataspace.selectAll();

    ao::uvector<float> data(dims_out[0] * dims_out[1]);
    ao::uvector<float *> modes_data(dims_out[0]);
    for (size_t i = 0; i < dims_out[0]; i++)
        modes_data[i] = &data[i * dims_out[1]];

    DataSpace memspace(rank, dims_out);
    modes.read(data.data(), PredType::NATIVE_FLOAT, memspace, modes_dataspace);

    for (size_t i = 0; i != dims_out[0]; i++) {
        const float *startElement = &data[i * dims_out[1]];
        const float *endElement = startElement + dims_out[1];
        out_vector.emplace_back(startElement, endElement);
    }
}

// Interface to HDF5 file format and structures to store H5 data
// Functions for reading H5 file and its datasets :
// Read data from H5 file, file name is specified in the object constructor
void Beam2016Implementation::Read() {
    std::string h5_path;
    if (_searchPath.empty()) {
        string h5_test_path = DEFAULT_H5_FILE_PATH;
        h5_test_path += DEFAULT_H5_FILE;
        h5_path = System::FindPythonFilePath(h5_test_path);
    } else {
        boost::filesystem::path p = boost::filesystem::path(_searchPath) / DEFAULT_H5_FILE;
        if (!boost::filesystem::exists(p))
            throw std::runtime_error("Manually specified MWA directory did not contain H5 beam file: '" + p.string() +
                                     "' not found.");
        h5_path = p.string();
    }

    _h5File.reset(new H5File(h5_path.c_str(), H5F_ACC_RDONLY));

    // hid_t group_id = m_pH5File->getId();
    hid_t file_id = _h5File->getId();

    /* TODO :  not sure how to read attribute with the official HDF5 library ...
    if( H5Aexists( file_id, "VERSION" ) ){
                    char szVersion[128];
                    strcpy(szVersion,"TEST");
                    //H5std_string szVersion;

                    hid_t attr_id = H5Aopen(file_id, "VERSION", H5P_DEFAULT );
                    hid_t attr_type = H5Aget_type(attr_id);
                    herr_t err = H5Aread( attr_id, attr_type, (void*)szVersion );
                    printf("Ids = %d -> %d -> type = %d\n",file_id,attr_id,attr_type);
                    printf("Version of the %s file = %s (err = %d)\n",m_h5file.c_str(),szVersion,err);
    }else{
                    printf("ERROR : attribute version does not exist\n");
    }*/

    m_obj_list.clear();
    m_freq_list.clear();
    herr_t status = H5Ovisit(file_id, H5_INDEX_NAME, H5_ITER_NATIVE, list_obj_iterate, this);
    if (status < 0) {
        throw std::runtime_error("H5Ovisit returned with negative value which indicates a critical error");
    }

    int max_ant_idx = -1;
    for (size_t i = 0; i < m_obj_list.size(); i++) {
        const char *key = m_obj_list[i].c_str();
        if (strstr(key, "X1_")) {
            const char *szFreq = key + 3;
            m_freq_list.push_back(atol(szFreq));
        }

        if (key[0] == 'X') {
            int ant_idx = 0, freq_hz = 0;
            int scanf_ret = sscanf(key, "X%d_%d", &ant_idx, &freq_hz);
            if (scanf_ret == 2) {
                if (ant_idx > max_ant_idx) {
                    max_ant_idx = ant_idx;
                }
            }
        }
    }
    // number of antenna is read from the file
    if (max_ant_idx != N_ANT_COUNT) {
        throw std::runtime_error("Number of simulated antennae = " + std::to_string(max_ant_idx) +
                                 ", the code is currently implemented for " + std::to_string(N_ANT_COUNT));
    }

    std::sort(m_freq_list.begin(), m_freq_list.end());

    ReadDataSet("modes", m_Modes, *_h5File);
}

void JonesMatrix::zeros(vector<vector<JonesMatrix>> &jones, size_t x_size, size_t y_size) {
    vector<JonesMatrix> zero_vector(x_size, JonesMatrix(0, 0, 0, 0));
    jones.assign(y_size, zero_vector);
}
