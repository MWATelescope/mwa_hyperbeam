#ifndef __JONES_GPU__
#define __JONES_GPU__
#define NMAX 25
#include <cuComplex.h>
#include <complex.h>
#include <iostream>
#include "factorial.h"



struct JonesData {
  int nElements;
  cuDoubleComplex *q1_accum;
  size_t *q1_offsets;
  cuDoubleComplex *q2_accum;
  size_t *q2_offsets;
  double *m_accum;
  size_t *m_accum_offsets;
  double *n_accum;
  size_t *n_accum_offsets;
  double *mabsm;
  size_t *mabsm_offsets;
  double *nmax;
  JonesData copy_to_device(void);
  void destroy(void);
};


struct dJonesMatrix {
  cuDoubleComplex j00;
  cuDoubleComplex j01;
  cuDoubleComplex j10;
  cuDoubleComplex j11;
};


int jones_compute_gpu(int zenith_norm, double *az, double *za,
  JonesData jdatax,JonesData jdatay,int npos, dJonesMatrix* norm_jones, std::complex<double>* result);

#endif
