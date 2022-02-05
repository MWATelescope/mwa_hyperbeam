#pragma once

#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#ifdef SINGLE
#define FLOAT  float
#define SINCOS sincosf
#define SIN    sinf
#define COS    cosf
#define FABS   fabsf
#define ATAN2  atan2f
#define SQRT   sqrtf
#else
#define FLOAT  double
#define SINCOS sincos
#define SIN    sin
#define COS    cos
#define FABS   fabs
#define ATAN2  atan2
#define SQRT   sqrt
#endif // SINGLE

// HIP-specific defines.
#if __HIPCC__
#define gpuMalloc             hipMalloc
#define gpuFree               hipFree
#define gpuMemcpy             hipMemcpy
#define gpuMemcpyHostToDevice hipMemcpyHostToDevice
#define gpuGetErrorString     hipGetErrorString
#define gpuGetLastError       hipGetLastError
#define gpuDeviceSynchronize  hipDeviceSynchronize
#define gpuError_t            hipError_t
#define gpuSuccess            hipSuccess

#ifdef SINGLE
#define CADD         hipCaddf
#define CSUB         hipCsubf
#define CMUL         hipCmulf
#define CDIV         hipCdivf
#define COMPLEX      hipFloatComplex
#define MAKE_COMPLEX make_hipFloatComplex
#else
#define CADD         hipCadd
#define CSUB         hipCsub
#define CMUL         hipCmul
#define CDIV         hipCdiv
#define COMPLEX      hipDoubleComplex
#define MAKE_COMPLEX make_hipDoubleComplex
#endif // SINGLE

// CUDA-specific defines.
#elif __CUDACC__
#define gpuMalloc             cudaMalloc
#define gpuFree               cudaFree
#define gpuMemcpy             cudaMemcpy
#define gpuMemcpyHostToDevice cudaMemcpyHostToDevice
#define gpuGetErrorString     cudaGetErrorString
#define gpuGetLastError       cudaGetLastError
#define gpuDeviceSynchronize  cudaDeviceSynchronize
#define gpuError_t            cudaError_t
#define gpuSuccess            cudaSuccess
#define warpSize              32

#ifdef SINGLE
#define CADD         cuCaddf
#define CSUB         cuCsubf
#define CMUL         cuCmulf
#define CDIV         cuCdivf
#define COMPLEX      cuFloatComplex
#define MAKE_COMPLEX make_cuFloatComplex
#else
#define CADD         cuCadd
#define CSUB         cuCsub
#define CMUL         cuCmul
#define CDIV         cuCdiv
#define COMPLEX      cuDoubleComplex
#define MAKE_COMPLEX make_cuDoubleComplex
#endif // SINGLE
#endif // __HIPCC_

#ifdef __CUDACC__
#include <cuComplex.h>
#elif __HIPCC__
#include <hip/hip_complex.h>
#include <hip/hip_runtime.h>
#endif

const FLOAT M_2PI = 2.0 * M_PI;
// The number of MWA bowties per tile.
const FLOAT NUM_DIPOLES = 16.0;
// MWA dipole separation [metres]
const FLOAT MWA_DPL_SEP = 1.100;
/// Speed of light [metres/second]
const FLOAT VEL_C = 299792458.0;

// To allow bindgen to run on this file we have to hide a bunch of stuff behind
// a macro.
#ifndef BINDGEN

/**
 * (HA, Dec.) coordinates. Both have units of radians.
 */
typedef struct HADec {
    /// Hour Angle [radians]
    FLOAT ha;
    /// Declination [radians]
    FLOAT dec;
} HADec;

/**
 * (Azimuth, Zenith Angle) coordinates. Both have units of radians.
 */
typedef struct AzZA {
    /// Azimuth [radians]
    FLOAT az;
    /// Zenith Angle [radians]
    FLOAT za;
} AzZA;

typedef struct JONES {
    COMPLEX j00;
    COMPLEX j01;
    COMPLEX j10;
    COMPLEX j11;
} JONES;

inline __device__ COMPLEX operator*(COMPLEX a, FLOAT b) {
    return MAKE_COMPLEX(a.x * b, a.y * b);
}

inline __device__ void operator*=(COMPLEX &a, FLOAT b) { 
    a.x *= b;
    a.y *= b;
}

inline __device__ void operator+=(COMPLEX &a, COMPLEX b) {
    a.x += b.x;
    a.y += b.y;
}

inline __device__ COMPLEX operator*(COMPLEX a, COMPLEX b) {
    return MAKE_COMPLEX(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

inline __device__ void operator*=(COMPLEX &a, COMPLEX b) {
    a = MAKE_COMPLEX(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// Convert a (azimuth, elevation) to HADec, given a location (latitude).
//
// This code is adapted from ERFA. The copyright notice associated with ERFA and
// the original code is at the bottom of this file.
inline __device__ HADec azel_to_hadec(FLOAT azimuth_rad, FLOAT elevation_rad, FLOAT latitude_rad) {
    /* Useful trig functions. */
    FLOAT sa, ca, se, ce, sp, cp;
    SINCOS(azimuth_rad, &sa, &ca);
    SINCOS(elevation_rad, &se, &ce);
    SINCOS(latitude_rad, &sp, &cp);

    /* HA,Dec unit vector. */
    FLOAT x = -ca * ce * sp + se * cp;
    FLOAT y = -sa * ce;
    FLOAT z = ca * ce * cp + se * sp;

    /* To spherical. */
    FLOAT r = SQRT(x * x + y * y);
    HADec hadec;
    hadec.ha = (r != 0.0) ? ATAN2(y, x) : 0.0;
    hadec.dec = ATAN2(z, r);

    return hadec;
}

// Convert a HADec to AzZA, given a location (latitude).
//
// This code is adapted from ERFA. The copyright notice associated with ERFA and
// the original code is at the bottom of this file.
inline __device__ AzZA hadec_to_azza(FLOAT hour_angle_rad, FLOAT dec_rad, FLOAT latitude_rad) {
    /* Useful trig functions. */
    FLOAT sh, ch, sd, cd, sp, cp;
    SINCOS(hour_angle_rad, &sh, &ch);
    SINCOS(dec_rad, &sd, &cd);
    SINCOS(latitude_rad, &sp, &cp);

    /* Az,Alt unit vector. */
    FLOAT x = -ch * cd * sp + sd * cp;
    FLOAT y = -sh * cd;
    FLOAT z = ch * cd * cp + sd * sp;

    /* To spherical. */
    FLOAT r = SQRT(x * x + y * y);
    FLOAT a = (r != 0.0) ? ATAN2(y, x) : 0.0;
    AzZA azza;
    azza.az = (a < 0.0) ? a + M_2PI : a;
    azza.za = M_PI_2 - ATAN2(z, r);

    return azza;
}

// Get the parallactic angle from a HADec position, given a location (latitude).
//
// This code is adapted from ERFA. The copyright notice associated with ERFA and
// the original code is at the bottom of this file.
inline static __device__ FLOAT get_parallactic_angle(HADec hadec, FLOAT latitude_rad) {
    FLOAT s_phi, c_phi, s_ha, c_ha, s_dec, c_dec, cqsz, sqsz;
    SINCOS(latitude_rad, &s_phi, &c_phi);
    SINCOS(hadec.ha, &s_ha, &c_ha);
    SINCOS(hadec.dec, &s_dec, &c_dec);

    sqsz = c_phi * s_ha;
    cqsz = s_phi * c_dec - c_phi * s_dec * c_ha;
    return ((sqsz != 0.0 || cqsz != 0.0) ? ATAN2(sqsz, cqsz) : 0.0);
}

#endif // BINDGEN

/*----------------------------------------------------------------------
**
**
**  Copyright (C) 2013-2021, NumFOCUS Foundation.
**  All rights reserved.
**
**  This library is derived, with permission, from the International
**  Astronomical Union's "Standards of Fundamental Astronomy" library,
**  available from http://www.iausofa.org.
**
**  The ERFA version is intended to retain identical functionality to
**  the SOFA library, but made distinct through different function and
**  file names, as set out in the SOFA license conditions.  The SOFA
**  original has a role as a reference standard for the IAU and IERS,
**  and consequently redistribution is permitted only in its unaltered
**  state.  The ERFA version is not subject to this restriction and
**  therefore can be included in distributions which do not support the
**  concept of "read only" software.
**
**  Although the intent is to replicate the SOFA API (other than
**  replacement of prefix names) and results (with the exception of
**  bugs;  any that are discovered will be fixed), SOFA is not
**  responsible for any errors found in this version of the library.
**
**  If you wish to acknowledge the SOFA heritage, please acknowledge
**  that you are using a library derived from SOFA, rather than SOFA
**  itself.
**
**
**  TERMS AND CONDITIONS
**
**  Redistribution and use in source and binary forms, with or without
**  modification, are permitted provided that the following conditions
**  are met:
**
**  1 Redistributions of source code must retain the above copyright
**    notice, this list of conditions and the following disclaimer.
**
**  2 Redistributions in binary form must reproduce the above copyright
**    notice, this list of conditions and the following disclaimer in
**    the documentation and/or other materials provided with the
**    distribution.
**
**  3 Neither the name of the Standards Of Fundamental Astronomy Board,
**    the International Astronomical Union nor the names of its
**    contributors may be used to endorse or promote products derived
**    from this software without specific prior written permission.
**
**  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
**  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
**  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
**  FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE
**  COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
**  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
**  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
**  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
**  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
**  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
**  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
**  POSSIBILITY OF SUCH DAMAGE.
**
*/
