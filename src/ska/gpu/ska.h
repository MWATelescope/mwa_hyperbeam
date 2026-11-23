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
#define SKA_MAX_L_MAX 510

/// The number of threads per block of the per-element-station kernel.
#define SKA_SPH_STATION_THREADS 128

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

    /// ska-low-eep-sph: the maximum degree of any coefficient set, and the
    /// number of elements.
    int l_max;
    int num_sph_elements;
    /// ska-low-eep-sph: coefficients, one set per coefficient file. Set `s`
    /// starts `coeff_set_offsets[s]` complex values in, has degree
    /// `coeff_set_l_max[s]` (so `l (l + 2)` coefficients per element), and is
    /// num_sph_elements x num_coeffs x 4 complex values ([TE_X, TM_X, TE_Y,
    /// TM_Y]).
    const void *sph_coeffs;
    const long long *coeff_set_offsets;
    const int *coeff_set_l_max;
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
/// [tile][element]; if `d_weights_y` is not NULL, the Y dipoles use those
/// weights instead). Station tiles are pointed at (`pointing_az`,
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
                               const FLOAT *d_positions, const FLOAT *d_weights, const FLOAT *d_weights_y,
                               const FLOAT *d_rotations, const FLOAT *d_frames, const FLOAT pointing_az,
                               const FLOAT pointing_za, const FLOAT *d_latitude_rad, const int iau_order,
                               void *d_results);

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
    const int l_max = m->coeff_set_l_max[coeff_set];
    const size_t num_coeffs = (size_t)l_max * (l_max + 2);
    const COMPLEX *coeffs = (const COMPLEX *)m->sph_coeffs + m->coeff_set_offsets[coeff_set] +
                            (size_t)element_id * num_coeffs * 4;

    COMPLEX e_theta[2] = {MAKE_COMPLEX(0.0, 0.0), MAKE_COMPLEX(0.0, 0.0)};
    COMPLEX e_phi[2] = {MAKE_COMPLEX(0.0, 0.0), MAKE_COMPLEX(0.0, 0.0)};

    if (theta < 1e-5)
        theta = 1e-5;
    FLOAT sin_phi, cos_phi, sin_theta, cos_theta;
    SINCOS(phi, &sin_phi, &cos_phi);
    SINCOS(theta, &sin_theta, &cos_theta);

    FLOAT p[SKA_MAX_L_MAX + 2];
    for (int n = 1; n <= l_max; ++n) {
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
                const size_t row = (size_t)n * (n + 1) + mm - 1;
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

// The direction, pointing and parallactic angle of a tile, in the tile's
// local frame (if it has one).
typedef struct SkaTileGeometry {
    // The direction's zenith angle and phi (East through North, before the
    // tile's rotation is applied).
    FLOAT za;
    FLOAT phi;
    // The pointing minus the direction (unit vectors, East, North, Up).
    FLOAT ds[3];
    // The parallactic angle (if a latitude is given).
    FLOAT pa;
} SkaTileGeometry;

inline __device__ SkaTileGeometry ska_tile_geometry(FLOAT az, FLOAT za, FLOAT pointing_az, FLOAT pointing_za,
                                                    const FLOAT *latitude_rad, const FLOAT *frame) {
    SkaTileGeometry g;
    if (frame == NULL) {
        FLOAT s_az, c_az, s_za, c_za, s_paz, c_paz, s_pza, c_pza;
        SINCOS(az, &s_az, &c_az);
        SINCOS(za, &s_za, &c_za);
        SINCOS(pointing_az, &s_paz, &c_paz);
        SINCOS(pointing_za, &s_pza, &c_pza);
        g.za = za;
        g.phi = M_PI_2 - az;
        g.ds[0] = s_pza * s_paz - s_za * s_az;
        g.ds[1] = s_pza * c_paz - s_za * c_az;
        g.ds[2] = c_pza - c_za;
        // The parallactic angle, computed directly from (az, za) rather than
        // via (HA, Dec) so that it is well conditioned near the zenith in
        // single precision (see parallactic_angle in src/ska/mod.rs).
        g.pa = 0.0;
        if (latitude_rad != NULL) {
            FLOAT s_lat, c_lat;
            SINCOS(*latitude_rad, &s_lat, &c_lat);
            g.pa = ATAN2(-s_az * c_lat, s_lat * s_za - c_lat * c_za * c_az);
        }
    } else {
        FLOAT l_az, l_za, l[3], p_az, p_za, p[3];
        ska_to_local(frame, az, za, &l_az, &l_za, l);
        ska_to_local(frame, pointing_az, pointing_za, &p_az, &p_za, p);
        g.za = l_za;
        g.phi = M_PI_2 - l_az;
        for (int r = 0; r < 3; r++)
            g.ds[r] = p[r] - l[r];
        g.pa = 0.0;
        if (latitude_rad != NULL) {
            // The celestial pole in the local frame.
            FLOAT pole_az, pole_za, pole[3];
            ska_to_local(frame, 0.0, M_PI_2 - *latitude_rad, &pole_az, &pole_za, pole);
            FLOAT s_daz, c_daz, s_lza, c_lza, s_plat, c_plat;
            SINCOS(l_az - pole_az, &s_daz, &c_daz);
            SINCOS(l_za, &s_lza, &c_lza);
            SINCOS(M_PI_2 - pole_za, &s_plat, &c_plat);
            g.pa = ATAN2(-s_daz * c_plat, s_plat * s_lza - c_plat * c_lza * c_daz);
        }
    }
    return g;
}

// Convert a response in EveryBeam's convention ([X_theta, X_phi, Y_theta,
// Y_phi]) to hyperbeam's ([X_theta, X_az, Y_theta, Y_az]), applying the
// parallactic-angle correction if there's a latitude.
inline __device__ void ska_finish_response(JONES *j, FLOAT pa, const FLOAT *latitude_rad, int iau_order) {
    j->j01 = ska_neg(j->j01);
    j->j11 = ska_neg(j->j11);
    if (latitude_rad != NULL)
        ska_apply_pa_correction(j, pa, iau_order);
}

// Scale the X row of a Jones matrix by `x` and the Y row by `y`.
inline __device__ void ska_scale_rows(JONES *j, COMPLEX x, COMPLEX y) {
    j->j00 = j->j00 * x;
    j->j01 = j->j01 * x;
    j->j10 = j->j10 * y;
    j->j11 = j->j11 * y;
}

// One thread per response: results are laid out as [tile][frequency][direction],
// and thread `i` calculates element `i` of that array (looping if there are
// more responses than threads in the grid). Parallelising over all three axes,
// rather than only over directions, keeps the GPU busy when there are few
// directions but many tiles and frequencies (e.g. a few hundred sky-model
// components for 68 stations and hundreds of fine channels), and the work per
// thread doesn't grow with the number of tiles or frequencies.
__global__ void ska_kernel(const SkaGpuModel model, const FLOAT *azs, const FLOAT *zas, const int num_directions,
                           const FLOAT *freqs_hz, const int num_freqs, const int num_tiles, const int *element_ids,
                           const int num_station_elements, const FLOAT *positions, const FLOAT *weights,
                           const FLOAT *weights_y, const FLOAT *rotations, const FLOAT *frames,
                           const FLOAT pointing_az, const FLOAT pointing_za, const FLOAT *latitude_rad,
                           const int iau_order, JONES *results) {
    const size_t num_responses = (size_t)num_tiles * num_freqs * num_directions;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < num_responses; i += stride) {
        const int i_dir = (int)(i % num_directions);
        const int i_freq = (int)((i / num_directions) % num_freqs);
        const int i_tile = (int)(i / ((size_t)num_directions * num_freqs));

        const FLOAT freq = freqs_hz[i_freq];
        const FLOAT k = -M_2PI * freq / VEL_C;
        const SkaTileGeometry g =
            ska_tile_geometry(azs[i_dir], zas[i_dir], pointing_az, pointing_za, latitude_rad,
                              frames == NULL ? NULL : frames + (size_t)9 * i_tile);
        // The direction's phi in the tile's (rotated) frame.
        const FLOAT rotation = rotations != NULL ? rotations[i_tile] : 0.0;
        const FLOAT tile_phi = g.phi - rotation;

        JONES j = ska_zero_jones();
        int ok = 1;
        if (model.model == SKA_EEP_SPH) {
            // Single element (stations of distinct elements are calculated by
            // ska_sph_station_kernel instead).
            const int element_id = element_ids != NULL ? element_ids[i_tile] : 0;
            j = ska_sph_response(&model, i_freq, element_id, g.za, tile_phi);
        } else {
            // The elements of these models are all identical, so there's one
            // element response for the tile.
            ok = ska_element_response(&model, i_freq, freq, 0, g.za, tile_phi, &j);
            if (positions != NULL) {
                // Station of identical elements: the element response times
                // the array factor (one for each dipole, as their weights may
                // differ).
                const FLOAT *pos = positions + (size_t)i_tile * num_station_elements * 3;
                const FLOAT *w_x = weights + (size_t)i_tile * num_station_elements;
                const FLOAT *w_y = weights_y != NULL ? weights_y + (size_t)i_tile * num_station_elements : w_x;
                FLOAT sum_w_x = 0.0, sum_w_y = 0.0;
                COMPLEX af_x = MAKE_COMPLEX(0.0, 0.0), af_y = MAKE_COMPLEX(0.0, 0.0);
                for (int e = 0; e < num_station_elements; e++) {
                    sum_w_x += w_x[e];
                    sum_w_y += w_y[e];
                    if (w_x[e] == 0.0 && w_y[e] == 0.0)
                        continue;
                    FLOAT s_ph, c_ph;
                    SINCOS(k * (g.ds[0] * pos[3 * e] + g.ds[1] * pos[3 * e + 1] + g.ds[2] * pos[3 * e + 2]), &s_ph,
                           &c_ph);
                    af_x += MAKE_COMPLEX(c_ph * w_x[e], s_ph * w_x[e]);
                    af_y += MAKE_COMPLEX(c_ph * w_y[e], s_ph * w_y[e]);
                }
                const FLOAT inv_x = sum_w_x != 0.0 ? 1.0 / sum_w_x : 0.0;
                const FLOAT inv_y = sum_w_y != 0.0 ? 1.0 / sum_w_y : 0.0;
                ska_scale_rows(&j, af_x * inv_x, af_y * inv_y);
            }
        }

        if (!ok) {
            j = ska_zero_jones();
        } else {
            ska_finish_response(&j, g.pa, latitude_rad, iau_order);
        }
        results[i] = j;
    }
}

// One thread's share of a station of distinct elements (ska-low-eep-sph):
// the weighted, phased sum over elements `i_thread`, `i_thread + num_threads`,
// ..., for the X and Y dipoles, and the sums of the weights.
inline __device__ void ska_sph_station_partial(const SkaGpuModel *model, const SkaTileGeometry *g, FLOAT tile_phi,
                                               FLOAT k, int i_freq, int num_station_elements, const FLOAT *pos,
                                               const FLOAT *w_x, const FLOAT *w_y, int i_thread, int num_threads,
                                               JONES *acc, FLOAT *sum_w_x, FLOAT *sum_w_y) {
    *acc = ska_zero_jones();
    *sum_w_x = 0.0;
    *sum_w_y = 0.0;
    for (int e = i_thread; e < num_station_elements; e += num_threads) {
        *sum_w_x += w_x[e];
        *sum_w_y += w_y[e];
        if (w_x[e] == 0.0 && w_y[e] == 0.0)
            continue;
        FLOAT s_ph, c_ph;
        SINCOS(k * (g->ds[0] * pos[3 * e] + g->ds[1] * pos[3 * e + 1] + g->ds[2] * pos[3 * e + 2]), &s_ph, &c_ph);
        JONES el = ska_sph_response(model, i_freq, e, g->za, tile_phi);
        ska_scale_rows(&el, MAKE_COMPLEX(c_ph * w_x[e], s_ph * w_x[e]), MAKE_COMPLEX(c_ph * w_y[e], s_ph * w_y[e]));
        acc->j00 += el.j00;
        acc->j01 += el.j01;
        acc->j10 += el.j10;
        acc->j11 += el.j11;
    }
}

// Combine the partial sums of ska_sph_station_partial into a station
// response, in hyperbeam's convention.
inline __device__ JONES ska_sph_station_finish(const JONES *partials, const FLOAT *sums_w_x, const FLOAT *sums_w_y,
                                               int num_partials, FLOAT pa, const FLOAT *latitude_rad,
                                               int iau_order) {
    JONES j = ska_zero_jones();
    FLOAT sum_w_x = 0.0, sum_w_y = 0.0;
    for (int t = 0; t < num_partials; t++) {
        j.j00 += partials[t].j00;
        j.j01 += partials[t].j01;
        j.j10 += partials[t].j10;
        j.j11 += partials[t].j11;
        sum_w_x += sums_w_x[t];
        sum_w_y += sums_w_y[t];
    }
    const FLOAT inv_x = sum_w_x != 0.0 ? 1.0 / sum_w_x : 0.0;
    const FLOAT inv_y = sum_w_y != 0.0 ? 1.0 / sum_w_y : 0.0;
    ska_scale_rows(&j, MAKE_COMPLEX(inv_x, 0.0), MAKE_COMPLEX(inv_y, 0.0));
    ska_finish_response(&j, pa, latitude_rad, iau_order);
    return j;
}

// Stations of distinct elements (ska-low-eep-sph). Each element's response is
// expensive (the coefficient files can have degrees in the hundreds), so each
// block handles one (direction, tile, frequency), with its threads sharing
// the elements.
__global__ void ska_sph_station_kernel(const SkaGpuModel model, const FLOAT *azs, const FLOAT *zas,
                                       const int num_directions, const FLOAT *freqs_hz, const int num_freqs,
                                       const int num_tiles, const int num_station_elements, const FLOAT *positions,
                                       const FLOAT *weights, const FLOAT *weights_y, const FLOAT *rotations,
                                       const FLOAT *frames, const FLOAT pointing_az, const FLOAT pointing_za,
                                       const FLOAT *latitude_rad, const int iau_order, JONES *results) {
    __shared__ JONES partials[SKA_SPH_STATION_THREADS];
    __shared__ FLOAT sums_w_x[SKA_SPH_STATION_THREADS];
    __shared__ FLOAT sums_w_y[SKA_SPH_STATION_THREADS];
    const size_t num_blocks = (size_t)num_directions * num_tiles * num_freqs;
    for (size_t i_block = blockIdx.x; i_block < num_blocks; i_block += gridDim.x) {
        const int i_dir = (int)(i_block % num_directions);
        const int i_freq = (int)((i_block / num_directions) % num_freqs);
        const int i_tile = (int)(i_block / ((size_t)num_directions * num_freqs));

        const SkaTileGeometry g =
            ska_tile_geometry(azs[i_dir], zas[i_dir], pointing_az, pointing_za, latitude_rad,
                              frames != NULL ? frames + (size_t)9 * i_tile : NULL);
        const FLOAT tile_phi = g.phi - (rotations != NULL ? rotations[i_tile] : 0.0);
        const FLOAT k = -M_2PI * freqs_hz[i_freq] / VEL_C;
        const FLOAT *pos = positions + (size_t)i_tile * num_station_elements * 3;
        const FLOAT *w_x = weights + (size_t)i_tile * num_station_elements;
        const FLOAT *w_y = weights_y != NULL ? weights_y + (size_t)i_tile * num_station_elements : w_x;
        ska_sph_station_partial(&model, &g, tile_phi, k, i_freq, num_station_elements, pos, w_x, w_y, threadIdx.x,
                                blockDim.x, &partials[threadIdx.x], &sums_w_x[threadIdx.x],
                                &sums_w_y[threadIdx.x]);
        __syncthreads();
        if (threadIdx.x == 0) {
            results[((size_t)i_tile * num_freqs + i_freq) * num_directions + i_dir] =
                ska_sph_station_finish(partials, sums_w_x, sums_w_y, blockDim.x, g.pa, latitude_rad, iau_order);
        }
        // Don't let the next iteration overwrite the partials before they're
        // summed.
        __syncthreads();
    }
}

#endif // BINDGEN
