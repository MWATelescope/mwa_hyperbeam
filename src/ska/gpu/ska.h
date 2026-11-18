// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// Portions of this file (the spherical-wave and associated Legendre code) are
// derived from ska-sdp-func (Copyright 2022 SKA Organisation; Copyright 2022
// Fred Dulwich, Andrew Ensor, Anthony Griffin, Seth Hall, Adam Campbell;
// Copyright 2022 University of Oxford), used under the BSD-3-Clause licence;
// see LICENSE-ska-sdp-func in the root of this repository.

// GPU code for the SKA-Low element and station beams. See src/ska/mod.rs for
// the conventions; everything here mirrors the CPU code.

#pragma once

#include "gpu_common.cuh"

/// The maximum spherical-wave degree supported on the GPU (the Legendre
/// workspace lives in registers/local memory).
#define SKA_MAX_L_MAX 64

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

typedef enum SKA_MODEL { SKA_ANALYTIC = 0, SKA_EEP_GRID = 1, SKA_EEP_SPH = 2 } SKA_MODEL;

/// Everything the SKA kernels need to know about the element model. All
/// pointers are device pointers.
typedef struct SkaGpuModel {
    /// Which element model.
    SKA_MODEL model;
    /// ska-low-analytic: 1 to apply the cos(theta) taper.
    int cos_taper;

    /// ska-low-eep-grid: the phi and theta axes (ascending).
    const FLOAT *phis;
    int num_phis;
    const FLOAT *thetas;
    int num_thetas;
    /// ska-low-eep-grid: one plane per frequency, already interpolated in
    /// frequency and with the sign convention applied. Each plane is num_phis
    /// x num_thetas Jones matrices (JONES).
    const void *planes;

    /// ska-low-eep-sph: the maximum degree, the number of coefficients per
    /// element (l_max * (l_max + 2)) and the number of elements.
    int l_max;
    int num_coeffs;
    int num_sph_elements;
    /// ska-low-eep-sph: coefficients, num_coeff_sets x num_sph_elements x
    /// num_coeffs x 4 complex values ([TE_X, TM_X, TE_Y, TM_Y]).
    const void *sph_coeffs;
    /// ska-low-eep-sph: for each frequency, the index of its coefficient set.
    const int *freq_to_coeff_set;
} SkaGpuModel;

/// Calculate SKA-Low beam responses.
///
/// Results are written to `d_results` (JONES), with shape num_tiles x
/// num_freqs x num_directions. A "tile" is either a single element (if
/// `d_positions` is NULL; `d_element_ids` then gives the element of each tile,
/// or element 0 for all if NULL) or a station of `num_station_elements`
/// elements (positions [tile][element][ENU] in metres and weights
/// [tile][element]). Station tiles are pointed at (`pointing_az`,
/// `pointing_za`). If `d_rotations` is not NULL, it gives each tile's rotation
/// about the local vertical [radians, the angle of its X dipoles from East
/// towards North]; elements are then evaluated at phi - rotation. If
/// `d_frames` is not NULL, it gives each tile's local frame (9 values per
/// tile, row major: the rows are the tile's East, North and Up unit vectors in
/// the frame of the directions; see `SkaStation::with_local_frame`), and the
/// directions, pointing and latitude are converted to it. If
/// `d_latitude_rad` is not NULL, the parallactic-angle correction is applied.
///
/// Directions outside the model's domain (e.g. outside an EEP grid) give zero
/// responses.
const char *gpu_ska_calc_jones(const SkaGpuModel *model, const FLOAT *d_azs, const FLOAT *d_zas,
                               const int num_directions, const FLOAT *d_freqs_hz, const int num_freqs,
                               const int num_tiles, const int *d_element_ids, const int num_station_elements,
                               const FLOAT *d_positions, const FLOAT *d_weights, const FLOAT *d_rotations,
                               const FLOAT *d_frames, const FLOAT pointing_az, const FLOAT pointing_za,
                               const FLOAT *d_latitude_rad, const int iau_order, void *d_results);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

// To allow bindgen to run on this file we have to hide a bunch of stuff behind
// a macro.
#ifndef BINDGEN

inline __device__ JONES ska_zero_jones() {
    return JONES{
        .j00 = MAKE_COMPLEX(0.0, 0.0),
        .j01 = MAKE_COMPLEX(0.0, 0.0),
        .j10 = MAKE_COMPLEX(0.0, 0.0),
        .j11 = MAKE_COMPLEX(0.0, 0.0),
    };
}

inline __device__ COMPLEX ska_conj(COMPLEX a) { return MAKE_COMPLEX(a.x, -a.y); }

// Negate a complex number. (Multiplying by a double literal like -1.0 can
// select HIP's own templated operators for single-precision complex types,
// which don't compile with ROCm 7; avoid mixed-precision scalar operations.)
inline __device__ COMPLEX ska_neg(COMPLEX a) { return MAKE_COMPLEX(-a.x, -a.y); }

inline __device__ void ska_add_scaled(JONES *acc, const JONES *j, COMPLEX s) {
    acc->j00 += j->j00 * s;
    acc->j01 += j->j01 * s;
    acc->j10 += j->j10 * s;
    acc->j11 += j->j11 * s;
}

// ---------------------------------------------------------------------------
// ska-low-analytic (OSKAR dipole).
// ---------------------------------------------------------------------------

inline __device__ void ska_dipole(FLOAT kl, FLOAT cos_kl, FLOAT phi, FLOAT sin_theta, FLOAT cos_theta,
                                  FLOAT *e_theta, FLOAT *e_phi) {
    FLOAT sin_phi, cos_phi;
    SINCOS(phi, &sin_phi, &cos_phi);
    const FLOAT denominator = 1.0 + cos_phi * cos_phi * (cos_theta * cos_theta - 1.0);
    if (denominator == 0.0) {
        *e_theta = 0.0;
        *e_phi = 0.0;
    } else {
        const FLOAT temp = (COS(kl * cos_phi * sin_theta) - cos_kl) / denominator;
        *e_theta = -cos_phi * cos_theta * temp;
        *e_phi = sin_phi * temp;
    }
}

inline __device__ JONES ska_analytic_response(int cos_taper, FLOAT freq_hz, FLOAT theta, FLOAT phi) {
    // Half-wavelength dipole: kL = (lambda / 2) * (pi f / c) = pi / 2.
    const FLOAT dipole_length_m = 0.5 * VEL_C / freq_hz;
    const FLOAT kl = dipole_length_m * (M_PI * freq_hz / VEL_C);
    const FLOAT cos_kl = COS(kl);
    FLOAT sin_theta, cos_theta;
    SINCOS(theta, &sin_theta, &cos_theta);
    FLOAT xt, xp, yt, yp;
    ska_dipole(kl, cos_kl, phi, sin_theta, cos_theta, &xt, &xp);
    ska_dipole(kl, cos_kl, phi + M_PI_2, sin_theta, cos_theta, &yt, &yp);
    const FLOAT taper = cos_taper ? cos_theta : 1.0;
    return JONES{
        .j00 = MAKE_COMPLEX(xt * taper, 0.0),
        .j01 = MAKE_COMPLEX(xp * taper, 0.0),
        .j10 = MAKE_COMPLEX(yt * taper, 0.0),
        .j11 = MAKE_COMPLEX(yp * taper, 0.0),
    };
}

// ---------------------------------------------------------------------------
// ska-low-eep-grid.
// ---------------------------------------------------------------------------

/// Find the grid interval containing `value` (EveryBeam's lower_bound - 1
/// semantics). Returns 0 if `value` is outside the axis.
inline __device__ int ska_locate(const FLOAT *values, int n, FLOAT value, int *index, FLOAT *weight) {
    if (!(value >= values[0] && value <= values[n - 1]))
        return 0;
    // First index with values[i] >= value.
    int lo = 0, hi = n;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (values[mid] < value)
            lo = mid + 1;
        else
            hi = mid;
    }
    int i = lo > 0 ? lo - 1 : 0;
    if (i > n - 2)
        i = n - 2;
    *index = i;
    *weight = (value - values[i]) / (values[i + 1] - values[i]);
    return 1;
}

inline __device__ int ska_grid_response(const SkaGpuModel *m, int i_freq, FLOAT theta, FLOAT phi, JONES *out) {
    // Wrap phi into [0, 2pi].
    if (phi < 0.0 || phi > M_2PI) {
        phi = phi - M_2PI * floor(phi / M_2PI);
    }
    int ip, it;
    FLOAT wp, wt;
    if (!ska_locate(m->phis, m->num_phis, phi, &ip, &wp))
        return 0;
    if (!ska_locate(m->thetas, m->num_thetas, theta, &it, &wt))
        return 0;
    const JONES *plane = (const JONES *)m->planes + (size_t)i_freq * m->num_phis * m->num_thetas;
    const JONES *c00 = &plane[ip * m->num_thetas + it];
    const JONES *c01 = c00 + 1;
    const JONES *c10 = &plane[(ip + 1) * m->num_thetas + it];
    const JONES *c11 = c10 + 1;
    const FLOAT w00 = (1.0 - wp) * (1.0 - wt), w01 = (1.0 - wp) * wt, w10 = wp * (1.0 - wt), w11 = wp * wt;
    *out = ska_zero_jones();
    ska_add_scaled(out, c00, MAKE_COMPLEX(w00, 0.0));
    ska_add_scaled(out, c01, MAKE_COMPLEX(w01, 0.0));
    ska_add_scaled(out, c10, MAKE_COMPLEX(w10, 0.0));
    ska_add_scaled(out, c11, MAKE_COMPLEX(w11, 0.0));
    return 1;
}

// ---------------------------------------------------------------------------
// ska-low-eep-sph (FEKO spherical waves).
// ---------------------------------------------------------------------------

/// Fully-normalised associated Legendre functions of degree n, orders 0..=n,
/// written to p[0..=n] (as ska-sdp-func's sdp_legendre_norm).
inline __device__ void ska_legendre_norm(int n, FLOAT cos_theta, FLOAT sin_theta, FLOAT *p) {
    const FLOAT tol = sqrt(1.17549435e-38); // sqrt(FLT_MIN)
    const FLOAT root_n_plus_half = SQRT((FLOAT)n + 0.5);
    const FLOAT twocot = sin_theta > 0.0 ? -2.0 * cos_theta / sin_theta : 0.0;
    FLOAT sn = 0.0;
    if (sin_theta != 0.0) {
        sn = pow(sin_theta, (FLOAT)n);
        if (n % 2 != 0)
            sn = -sn;
    }
    for (int m = 0; m <= n; ++m)
        p[m] = 0.0;

    // Underflow branch: used only when cos_theta is very close to 1.
    if (sin_theta > 0.0 && FABS(sn) <= tol) {
        const FLOAT tstart = 1.1920929e-7; // FLT_EPSILON
        const FLOAT v = 9.2 - log(tol) / (n * sin_theta);
        const FLOAT w = 1.0 / log(v);
        const FLOAT m1d = 1.0 + n * sin_theta * v * w * (1.0058 + w * (3.819 - w * 12.173));
        int m1 = (int)floor(m1d);
        if (m1 < 1)
            m1 = 1;
        if (m1 > n)
            m1 = n;
        const FLOAT sgn_m1 = ((m1 % 2) == 1) ? 1.0 : -1.0;
        const FLOAT sgn_n1 = (((n + 1) % 2) == 1) ? 1.0 : -1.0;
        p[m1 - 1] = (cos_theta < 0.0) ? (sgn_n1 * tstart) : (sgn_m1 * tstart);
        FLOAT sumsq = tol;
        for (int m = m1 - 2; m >= 0; --m) {
            const FLOAT a = (m + 1) * twocot * p[m + 1];
            const FLOAT b = SQRT((FLOAT)(n + m + 2)) * SQRT((FLOAT)(n - m - 1)) * p[m + 2];
            p[m] = (a - b) / (SQRT((FLOAT)(n + m + 1)) * SQRT((FLOAT)(n - m)));
            sumsq += p[m] * p[m];
        }
        const FLOAT scale = 1.0 / SQRT(2.0 * sumsq - p[0] * p[0]);
        for (int r = 0; r <= m1; ++r)
            p[r] *= scale;
    }

    // Non-underflow branch. (ska-sdp-func tests cos_theta != 1 here. In
    // single precision, cos(theta) rounds to exactly 1 within ~0.02 deg of the
    // pole, which would skip this branch and give wrong values; testing
    // sin(theta) instead is equivalent in double precision, as theta >= 1e-5.)
    if (sin_theta > 0.0 && FABS(sn) >= tol) {
        FLOAT c = 1.0;
        for (int k = 1; k <= n; ++k)
            c *= (1.0 - 1.0 / (2.0 * k));
        p[n] = SQRT(c) * sn;
        p[n - 1] = p[n] * twocot * (FLOAT)n / SQRT((FLOAT)(2 * n));
        for (int m = n - 2; m >= 0; --m) {
            const FLOAT a = (m + 1) * twocot * p[m + 1];
            const FLOAT b = SQRT((FLOAT)(n + m + 2)) * SQRT((FLOAT)(n - m - 1)) * p[m + 2];
            p[m] = (a - b) / (SQRT((FLOAT)(n + m + 1)) * SQRT((FLOAT)(n - m)));
        }
    }

    // Polar case.
    if (sin_theta == 0.0)
        p[0] = pow(cos_theta, (FLOAT)n);

    // Normalisation.
    for (int m = 0; m <= n; ++m) {
        const FLOAT sgn = ((m % 2) == 0) ? 1.0 : -1.0;
        p[m] *= root_n_plus_half * sgn;
    }
}

/// i^n
inline __device__ COMPLEX ska_ipow(int n) {
    switch (n & 3) {
    case 0:
        return MAKE_COMPLEX(1.0, 0.0);
    case 1:
        return MAKE_COMPLEX(0.0, 1.0);
    case 2:
        return MAKE_COMPLEX(-1.0, 0.0);
    default:
        return MAKE_COMPLEX(0.0, -1.0);
    }
}

/// The response of one element (EveryBeam convention, including the SKA-Low
/// conjugation and negation).
inline __device__ JONES ska_sph_response(const SkaGpuModel *m, int i_freq, int element_id, FLOAT theta,
                                         FLOAT phi) {
    const FLOAT sqrt_fac = SQRT(376.730313668 / M_2PI);
    const int coeff_set = m->freq_to_coeff_set[i_freq];
    const COMPLEX *coeffs = (const COMPLEX *)m->sph_coeffs +
                            ((size_t)coeff_set * m->num_sph_elements + element_id) * m->num_coeffs * 4;

    COMPLEX e_theta[2] = {MAKE_COMPLEX(0.0, 0.0), MAKE_COMPLEX(0.0, 0.0)};
    COMPLEX e_phi[2] = {MAKE_COMPLEX(0.0, 0.0), MAKE_COMPLEX(0.0, 0.0)};

    if (theta < 1e-5)
        theta = 1e-5;
    FLOAT sin_phi, cos_phi, sin_theta, cos_theta;
    SINCOS(phi, &sin_phi, &cos_phi);
    SINCOS(theta, &sin_theta, &cos_theta);

    FLOAT p[SKA_MAX_L_MAX + 2];
    for (int n = 1; n <= m->l_max; ++n) {
        ska_legendre_norm(n, cos_theta, sin_theta, p);
        p[n + 1] = 0.0;
        const FLOAT sqrt_n = SQRT((FLOAT)n * (n + 1));
        const COMPLEX i_n = ska_ipow(n);
        const COMPLEX i_np1 = ska_ipow(n + 1);

        FLOAT cos_m_phi = 1.0, sin_m_phi = 0.0;
        for (int abs_m = 0; abs_m <= n; ++abs_m) {
            const FLOAT cmn = SQRT((FLOAT)(n + abs_m + 1) * (n - abs_m));
            const FLOAT np_1 = cmn * p[abs_m + 1];
            const FLOAT np_ds = p[abs_m] / sin_theta;
            const FLOAT np_diff = np_ds * abs_m * cos_theta - np_1;

            const int repeat_max = (abs_m == 0) ? 1 : 2;
            for (int repeat = 0; repeat < repeat_max; ++repeat) {
                const int mm = (repeat == 0) ? abs_m : -abs_m;
                const FLOAT sign = (mm >= 0) ? 1.0 : ((mm & 1) ? -1.0 : 1.0);
                const FLOAT np_ds_m = np_ds * mm;
                const int row = n * (n + 1) + mm - 1;
                const FLOAT sin_m_phi_m = (mm >= 0) ? sin_m_phi : -sin_m_phi;
                const COMPLEX phi_const = MAKE_COMPLEX(cos_m_phi, sin_m_phi_m) * (sign / sqrt_n);
                for (int a = 0; a < 2; ++a) {
                    const COMPLEX te = coeffs[row * 4 + 2 * a];
                    const COMPLEX tm = coeffs[row * 4 + 2 * a + 1];
                    e_theta[a] += phi_const * i_n * CADD(te * -np_ds_m, tm * np_diff);
                    e_phi[a] += phi_const * i_np1 * CADD(tm * np_ds_m, te * -np_diff);
                }
            }

            // cos((m + 1) phi), sin((m + 1) phi). (phi_x == phi_y here.)
            const FLOAT c = cos_m_phi, s = sin_m_phi;
            cos_m_phi = cos_phi * c - sin_phi * s;
            sin_m_phi = sin_phi * c + cos_phi * s;
        }
    }

    // The SKA-Low convention: conjugate, and negate the Y dipole.
    return JONES{
        .j00 = ska_conj(e_theta[0] * sqrt_fac),
        .j01 = ska_conj(e_phi[0] * sqrt_fac),
        .j10 = ska_neg(ska_conj(e_theta[1] * sqrt_fac)),
        .j11 = ska_neg(ska_conj(e_phi[1] * sqrt_fac)),
    };
}

// ---------------------------------------------------------------------------
// Kernel.
// ---------------------------------------------------------------------------

/// The response of one element, in EveryBeam's convention. Returns 0 if the
/// direction is outside the model's domain.
inline __device__ int ska_element_response(const SkaGpuModel *m, int i_freq, FLOAT freq_hz, int element_id,
                                           FLOAT theta, FLOAT phi, JONES *out) {
    switch (m->model) {
    case SKA_ANALYTIC:
        *out = ska_analytic_response(m->cos_taper, freq_hz, theta, phi);
        return 1;
    case SKA_EEP_GRID:
        return ska_grid_response(m, i_freq, theta, phi, out);
    case SKA_EEP_SPH:
        *out = ska_sph_response(m, i_freq, element_id, theta, phi);
        return 1;
    }
    return 0;
}

// Apply the parallactic-angle correction (identical to the FEE beam's).
inline __device__ void ska_apply_pa_correction(JONES *jm, FLOAT pa, int iau_order) {
    FLOAT s_rot, c_rot;
    SINCOS(pa, &s_rot, &c_rot);
    if (iau_order == 1) {
        *jm = JONES{
            .j00 = CADD(jm->j10 * -c_rot, jm->j11 * s_rot),
            .j01 = CADD(jm->j10 * -s_rot, jm->j11 * -c_rot),
            .j10 = CADD(jm->j00 * -c_rot, jm->j01 * s_rot),
            .j11 = CADD(jm->j00 * -s_rot, jm->j01 * -c_rot),
        };
    } else {
        *jm = JONES{
            .j00 = CADD(jm->j00 * -s_rot, jm->j01 * -c_rot),
            .j01 = CADD(jm->j00 * -c_rot, jm->j01 * s_rot),
            .j10 = CADD(jm->j10 * -s_rot, jm->j11 * -c_rot),
            .j11 = CADD(jm->j10 * -c_rot, jm->j11 * s_rot),
        };
    }
}

// Convert a direction (az, za) to a tile's local frame `m` (row major; see
// to_local in src/ska/station.rs). `l` receives the local unit vector.
inline __device__ void ska_to_local(const FLOAT *m, FLOAT az, FLOAT za, FLOAT *local_az, FLOAT *local_za,
                                    FLOAT l[3]) {
    FLOAT s_az, c_az, s_za, c_za;
    SINCOS(az, &s_az, &c_az);
    SINCOS(za, &s_za, &c_za);
    const FLOAT v[3] = {s_za * s_az, s_za * c_az, c_za};
    for (int r = 0; r < 3; r++)
        l[r] = m[3 * r] * v[0] + m[3 * r + 1] * v[1] + m[3 * r + 2] * v[2];
    const FLOAT horizontal = SQRT(l[0] * l[0] + l[1] * l[1]);
    if (horizontal > (FLOAT)1e-12) {
        *local_az = ATAN2(l[0], l[1]);
    } else {
        // At the local zenith, keep the given azimuth's direction (it still
        // defines the polarisation basis).
        const FLOAT h0 = m[0] * s_az + m[1] * c_az;
        const FLOAT h1 = m[3] * s_az + m[4] * c_az;
        *local_az = ATAN2(h0, h1);
    }
    *local_za = ATAN2(horizontal, l[2]);
}

__global__ void ska_kernel(const SkaGpuModel model, const FLOAT *azs, const FLOAT *zas, const int num_directions,
                           const FLOAT *freqs_hz, const int num_freqs, const int num_tiles, const int *element_ids,
                           const int num_station_elements, const FLOAT *positions, const FLOAT *weights,
                           const FLOAT *rotations, const FLOAT *frames, const FLOAT pointing_az,
                           const FLOAT pointing_za, const FLOAT *latitude_rad, const int iau_order,
                           JONES *results) {
    for (int i_dir = blockIdx.x * blockDim.x + threadIdx.x; i_dir < num_directions;
         i_dir += gridDim.x * blockDim.x) {
        const FLOAT az = azs[i_dir];
        const FLOAT za = zas[i_dir];
        const FLOAT phi = M_PI_2 - az;

        // Direction and pointing unit vectors (East, North, Up), for stations.
        FLOAT s_az, c_az, s_za, c_za, s_paz, c_paz, s_pza, c_pza;
        SINCOS(az, &s_az, &c_az);
        SINCOS(za, &s_za, &c_za);
        SINCOS(pointing_az, &s_paz, &c_paz);
        SINCOS(pointing_za, &s_pza, &c_pza);
        const FLOAT ds[3] = {s_pza * s_paz - s_za * s_az, s_pza * c_paz - s_za * c_az, c_pza - c_za};

        // The parallactic angle, computed directly from (az, za) rather than
        // via (HA, Dec) so that it is well conditioned near the zenith in
        // single precision (see parallactic_angle in src/ska/mod.rs).
        FLOAT pa = 0.0;
        if (latitude_rad != NULL) {
            FLOAT s_lat, c_lat;
            SINCOS(*latitude_rad, &s_lat, &c_lat);
            pa = ATAN2(-s_az * c_lat, s_lat * s_za - c_lat * c_za * c_az);
        }

        for (int i_freq = 0; i_freq < num_freqs; i_freq++) {
            const FLOAT freq = freqs_hz[i_freq];
            const FLOAT k = -M_2PI * freq / VEL_C;

            // For models whose elements are all identical, the element
            // response only needs to be evaluated once per direction (for
            // unrotated tiles).
            JONES unrotated_element;
            int unrotated_ok = 0;
            if (model.model != SKA_EEP_SPH) {
                unrotated_ok = ska_element_response(&model, i_freq, freq, 0, za, phi, &unrotated_element);
            }

            for (int i_tile = 0; i_tile < num_tiles; i_tile++) {
                JONES j = ska_zero_jones();
                int ok = 1;
                // The direction, pointing and parallactic angle in the tile's
                // local frame (if it has one).
                FLOAT tile_za = za;
                FLOAT tile_base_phi = phi;
                FLOAT tile_ds[3] = {ds[0], ds[1], ds[2]};
                FLOAT tile_pa = pa;
                if (frames != NULL) {
                    const FLOAT *m = frames + (size_t)9 * i_tile;
                    FLOAT l_az, l_za, l[3], p_az, p_za, p[3];
                    ska_to_local(m, az, za, &l_az, &l_za, l);
                    ska_to_local(m, pointing_az, pointing_za, &p_az, &p_za, p);
                    tile_za = l_za;
                    tile_base_phi = M_PI_2 - l_az;
                    for (int r = 0; r < 3; r++)
                        tile_ds[r] = p[r] - l[r];
                    if (latitude_rad != NULL) {
                        // The celestial pole in the local frame.
                        FLOAT pole_az, pole_za, pole[3];
                        ska_to_local(m, 0.0, M_PI_2 - *latitude_rad, &pole_az, &pole_za, pole);
                        FLOAT s_daz, c_daz, s_lza, c_lza, s_plat, c_plat;
                        SINCOS(l_az - pole_az, &s_daz, &c_daz);
                        SINCOS(l_za, &s_lza, &c_lza);
                        SINCOS(M_PI_2 - pole_za, &s_plat, &c_plat);
                        tile_pa = ATAN2(-s_daz * c_plat, s_plat * s_lza - c_plat * c_lza * c_daz);
                    }
                }
                // The direction's phi in the tile's (rotated) frame.
                const FLOAT rotation = rotations != NULL ? rotations[i_tile] : 0.0;
                const FLOAT tile_phi = tile_base_phi - rotation;
                JONES shared_element = unrotated_element;
                int shared_ok = unrotated_ok;
                if (model.model != SKA_EEP_SPH && (rotation != 0.0 || frames != NULL)) {
                    shared_ok = ska_element_response(&model, i_freq, freq, 0, tile_za, tile_phi, &shared_element);
                }
                if (positions == NULL) {
                    // Single element.
                    if (model.model == SKA_EEP_SPH) {
                        const int element_id = element_ids != NULL ? element_ids[i_tile] : 0;
                        j = ska_sph_response(&model, i_freq, element_id, tile_za, tile_phi);
                    } else {
                        ok = shared_ok;
                        j = shared_element;
                    }
                } else {
                    // Station: sum of phased element responses.
                    const FLOAT *pos = positions + (size_t)i_tile * num_station_elements * 3;
                    const FLOAT *w = weights + (size_t)i_tile * num_station_elements;
                    FLOAT sum_w = 0.0;
                    COMPLEX af = MAKE_COMPLEX(0.0, 0.0);
                    for (int e = 0; e < num_station_elements; e++) {
                        sum_w += w[e];
                        if (w[e] == 0.0)
                            continue;
                        FLOAT s_ph, c_ph;
                        SINCOS(k * (tile_ds[0] * pos[3 * e] + tile_ds[1] * pos[3 * e + 1] +
                                    tile_ds[2] * pos[3 * e + 2]),
                               &s_ph, &c_ph);
                        const COMPLEX phasor = MAKE_COMPLEX(c_ph * w[e], s_ph * w[e]);
                        if (model.model == SKA_EEP_SPH) {
                            const JONES el = ska_sph_response(&model, i_freq, e, tile_za, tile_phi);
                            ska_add_scaled(&j, &el, phasor);
                        } else {
                            af += phasor;
                        }
                    }
                    if (model.model != SKA_EEP_SPH) {
                        ok = shared_ok;
                        ska_add_scaled(&j, &shared_element, af);
                    }
                    const FLOAT inv = sum_w != 0.0 ? 1.0 / sum_w : 0.0;
                    j.j00 *= inv;
                    j.j01 *= inv;
                    j.j10 *= inv;
                    j.j11 *= inv;
                }

                if (!ok) {
                    j = ska_zero_jones();
                } else {
                    // EveryBeam [X_theta, X_phi, Y_theta, Y_phi] -> hyperbeam
                    // [X_theta, X_az, Y_theta, Y_az].
                    j.j01 = ska_neg(j.j01);
                    j.j11 = ska_neg(j.j11);
                    if (latitude_rad != NULL)
                        ska_apply_pa_correction(&j, tile_pa, iau_order);
                }
                results[((size_t)i_tile * num_freqs + i_freq) * num_directions + i_dir] = j;
            }
        }
    }
}

#endif // BINDGEN
