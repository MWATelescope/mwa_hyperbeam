#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex>
#include <cuda.h>
#include <cuComplex.h>
#include <iostream>
//#include <beam.h>
//#include <jones.h>
#include "factorial.h"
#include "legendre.h"
#include "jones_gpu.cuh"

#define print_complex_4(X)({\
  printf("00 = (%f, %f), 01 = (%f, %f), 10 = (%f, %f), 11 = (%f, %f)\n", (X).j00.x, (X).j00.y, (X).j01.x, (X).j01.y, (X).j10.x, (X).j10.y, (X).j11.x, (X).j11.y);\
})


#define FACTORIAL_MAX_CACHE_SIZE 100

#define CUDA_CHECK_ERROR(X)({\
  if((X) != cudaSuccess){\
    std::cerr << "CUDA ERROR " << (X) << ": " << cudaGetErrorString((X)) << " (" << __FILE__ << ":" << __LINE__ << std::endl;\
    exit(1);\
  }\
})

JonesData JonesData::copy_to_device(void){
  JonesData dev;
  dev.nElements = nElements;
  CUDA_CHECK_ERROR(cudaMalloc(&dev.nmax, sizeof(double) * nElements));
  CUDA_CHECK_ERROR(cudaMalloc(&dev.q1_accum, sizeof(cuDoubleComplex) * q1_offsets[nElements]));
  CUDA_CHECK_ERROR(cudaMalloc(&dev.q2_accum, sizeof(cuDoubleComplex) * q2_offsets[nElements]));
  CUDA_CHECK_ERROR(cudaMalloc(&dev.m_accum, sizeof(double) * m_accum_offsets[nElements]));
  CUDA_CHECK_ERROR(cudaMalloc(&dev.n_accum, sizeof(double) * n_accum_offsets[nElements]));
  CUDA_CHECK_ERROR(cudaMalloc(&dev.mabsm, sizeof(double) * mabsm_offsets[nElements]));
  CUDA_CHECK_ERROR(cudaMalloc(&dev.q1_offsets, sizeof(size_t) * (nElements + 1)));
  CUDA_CHECK_ERROR(cudaMalloc(&dev.q2_offsets, sizeof(size_t) * (nElements + 1)));
  CUDA_CHECK_ERROR(cudaMalloc(&dev.m_accum_offsets, sizeof(size_t) * (nElements + 1)));
  CUDA_CHECK_ERROR(cudaMalloc(&dev.n_accum_offsets, sizeof(size_t) * (nElements + 1)));
  CUDA_CHECK_ERROR(cudaMalloc(&dev.mabsm_offsets, sizeof(size_t) * (nElements + 1)));
  
  CUDA_CHECK_ERROR(cudaMemcpy(dev.nmax, nmax, sizeof(double) * nElements, cudaMemcpyHostToDevice));
  CUDA_CHECK_ERROR(cudaMemcpy(dev.q1_accum, q1_accum, sizeof(cuDoubleComplex)  * q1_offsets[nElements], cudaMemcpyHostToDevice));
  CUDA_CHECK_ERROR(cudaMemcpy(dev.q2_accum, q2_accum, sizeof(cuDoubleComplex)  * q2_offsets[nElements], cudaMemcpyHostToDevice));
  CUDA_CHECK_ERROR(cudaMemcpy(dev.m_accum, m_accum, sizeof(double) *  m_accum_offsets[nElements], cudaMemcpyHostToDevice));
  CUDA_CHECK_ERROR(cudaMemcpy(dev.n_accum, n_accum, sizeof(double) * n_accum_offsets[nElements], cudaMemcpyHostToDevice));
  CUDA_CHECK_ERROR(cudaMemcpy(dev.mabsm, mabsm, sizeof(double) *  mabsm_offsets[nElements], cudaMemcpyHostToDevice));
  CUDA_CHECK_ERROR(cudaMemcpy(dev.q1_offsets, q1_offsets, sizeof(size_t) * (nElements + 1), cudaMemcpyHostToDevice));
  CUDA_CHECK_ERROR(cudaMemcpy(dev.q2_offsets, q2_offsets, sizeof(size_t) * (nElements + 1), cudaMemcpyHostToDevice));
  CUDA_CHECK_ERROR(cudaMemcpy(dev.m_accum_offsets, m_accum_offsets, sizeof(size_t) * (nElements + 1), cudaMemcpyHostToDevice));
  CUDA_CHECK_ERROR(cudaMemcpy(dev.n_accum_offsets, n_accum_offsets, sizeof(size_t) * (nElements + 1), cudaMemcpyHostToDevice));
  CUDA_CHECK_ERROR(cudaMemcpy(dev.mabsm_offsets, mabsm_offsets, sizeof(size_t) * (nElements + 1), cudaMemcpyHostToDevice));
  
  return dev;
}

void JonesData::destroy(void){
    cudaFree(q1_accum);
    cudaFree(q2_accum);
    cudaFree(m_accum);
    cudaFree(n_accum);
    cudaFree(mabsm);
    cudaFree(nmax);
    cudaFree(q1_offsets);
    cudaFree(q2_offsets);
    cudaFree(m_accum_offsets);
    cudaFree(n_accum_offsets);
    cudaFree(mabsm_offsets);
}

/*################################################################*/
__device__ void lpmv_device(double *output, int n, double x )
{
   double p0,p1,p_tmp;
   p0=1;
   p1=x;
   if(n==0)
     output[0]=p0;
   else {
     unsigned l = 1;
     while(l < n)
     {
       p_tmp=p0;
       p0=p1;
       p1=p_tmp;
       p1 = ((2.0*(double)l+1)*x*p0-(double)l*p1)/((double)l+1); //legendre_next(n,0, x, p0, p1);
       ++l;
     }
     output[0]=p1;
   }
}

/*################################################################*/
__device__ int lidx_device(const int l, const int m){
    // summation series over l + m => (l*(l+1))/2 + m
    return ((l*(l+1))>>1) + m;
}

/*################################################################*/
__device__ void legendre_polynomials_device(double *legendre, const double x, const int P){
    //This factor is reuse 342210222sqrt(1 342210222 x^2)
    int l,m;
    const double factor = -sqrt(1.0-pow(x,2));

    // Init legendre
    legendre[lidx_device(0,0)] = 1.0;        // P_0,0(x) = 1
    // Easy values
    legendre[lidx_device(1,0)] = x;      // P_1,0(x) = x
    legendre[lidx_device(1,1)] = factor;     // P_1,1(x) = 342210222sqrt(1 342210222 x^2)

    for(l = 2; l <= P ; ++l ){
        for(m = 0; m < l - 1 ; ++m ){
            // P_l,m = (2l-1)*x*P_l-1,m - (l+m-1)*x*P_l-2,m / (l-k)
            legendre[lidx_device(l,m)] = ((double)(2*l-1) * x * legendre[lidx_device(l-1,m)] - (double)( l + m - 1 ) * legendre[lidx_device(l-2,m)] ) / (double)( l - m );
        }
        // P_l,l-1 = (2l-1)*x*P_l-1,l-1
        legendre[lidx_device(l,l-1)] = (double)(2*l-1) * x * legendre[lidx_device(l-1,l-1)];
        // P_l,l = (2l-1)*factor*P_l-1,l-1
        legendre[lidx_device(l,l)] = (double)(2*l-1) * factor * legendre[lidx_device(l-1,l-1)];
    }
}

/*################################################################*/
__device__ int jones_p1sin_device(int nmax, double theta, double *p1sin_out, int *p1sin_out_size, double *p1_out, int *p1_out_size) {
  int n,m;
  int size = nmax*nmax + 2*nmax;
  int ind_start,ind_stop;
  int modified;
  double sin_th, u;
  double delu=1e-6;
  double P[NMAX+1],Pm1[NMAX+1],Pm_sin[NMAX+1],Pu_mdelu[NMAX+1],Pm_sin_merged[NMAX*2+1],Pm1_merged[NMAX*2+1];
  double legendre_table[NMAX*(NMAX+1)],legendret[(((NMAX+2)*(NMAX+1))/2)];
   
  *p1sin_out_size=size;
  *p1_out_size=size;
  sincos(theta,&sin_th,&u);
  // Create a look-up table for the legendre polynomials
  // Such that legendre_table[ m * nmax + (n-1) ] = legendre(n, m, u)
  legendre_polynomials_device(legendret,u,nmax);
  for(n=1; n<=nmax; n++) {
    for(m=0; m!=n+1; ++m)
      legendre_table[m*nmax + (n-1)]=legendret[lidx_device(n,m)];
    for(m=n+1; m!=nmax+1; ++m)
      legendre_table[m*nmax + (n-1)]=0.0;
  }
  
  for(n=1;n<=nmax;n++) {
    int i;
    for(m=0;m!=n+1;++m) {
      P[m]=legendre_table[m*nmax+(n-1)];
    }
    memcpy(Pm1,&(P[1]),n*sizeof(double));
    Pm1[n]=0;
    for(i=0;i<n+1;i++)
      Pm_sin[i]=0.0;
    if(u==1 || u==-1) {
      // In this case we take the easy approach and don't use
      // precalculated polynomials, since this path does not occur often.

      //Pu_mdelu.resize(1);
      lpmv_device(Pu_mdelu, n, u-delu);
      // Pm_sin[1,0]=-(P[0]-Pu_mdelu[0])/delu #backward difference
      if( u == -1 )
        Pm_sin[1] = -(Pu_mdelu[0]-P[0])/delu; // #forward difference
      else
        Pm_sin[1] = -(P[0]-Pu_mdelu[0])/delu;
      //printf("Pm_sin[1]=%f %f %f\n",Pm_sin[1],delu,P[0]);
    } else {
      for(i=0;i<n+1;i++) {
        Pm_sin[i]=P[i]/sin_th;
      }
    }


    for(i=n;i>=0;i--)
      Pm_sin_merged[n-i]=Pm_sin[i];
    memcpy(&(Pm_sin_merged[n]),Pm_sin,(n+1)*sizeof(double));

    ind_start=(n-1)*(n-1)+2*(n-1); // #start index to populate
    ind_stop=n*n+2*n; //#stop index to populate

    modified=0;
    for(i=ind_start;i<ind_stop;i++) {
       p1sin_out[i]=Pm_sin_merged[modified];
       modified++;
    }

    for(i=n;i>0;i--)
      Pm1_merged[n-i]=Pm1[i];
    memcpy(&Pm1_merged[n],Pm1,(n+1)*sizeof(double));

    modified=0;
    for(i=ind_start;i<ind_stop;i++) {
       p1_out[i]=Pm1_merged[modified];
       modified++;
    }

  }

  return nmax;
}

/*################################################################*/
__device__ void jones_calc_sigmas_device(double phi, double theta,double Nmax,char pol,double **n_accum,int n_accum_size,double **m_accum,double **mabsm,double *dfcache,cuDoubleComplex *djpcache,cuDoubleComplex **q1_accum,cuDoubleComplex **q2_accum,dJonesMatrix *jm) {
  int i;
  int nmax=(int)Nmax;
  double cos_theta=cos(theta);
  double u=cos_theta;
  double P1sin_arr[NMAX*NMAX+2*NMAX],P1_arr[NMAX*NMAX+2*NMAX];
  int P1sin_arr_size,P1_arr_size;
  cuDoubleComplex sigma_P,sigma_T;
  sigma_P.x=0; sigma_P.y=0;
  sigma_T.x=0; sigma_T.y=0;
  
  
  jones_p1sin_device(nmax,theta,P1sin_arr,&P1sin_arr_size,P1_arr,&P1_arr_size);

 
  for(i=0;i<n_accum_size;i++) {
    double N=(*n_accum)[i];
    int n=(int)N;
    double M=(*m_accum)[i];
    double m_abs_m=(*mabsm)[i];
    double c_mn_sqr=(0.5*(2*N+1)*dfcache[(int)N-abs((int)M)]/dfcache[(int)N+abs((int)M)]);
    double c_mn = sqrt( c_mn_sqr );
    cuDoubleComplex ejm_phi=make_cuDoubleComplex(cos(M*phi),sin(M*phi));
    cuDoubleComplex phi_comp=cuCmul(ejm_phi, make_cuDoubleComplex(c_mn/(sqrt(N*(N+1)))*m_abs_m, 0));
    cuDoubleComplex j_power_n=djpcache[n];
    cuDoubleComplex s1=cuCmul(make_cuDoubleComplex(P1sin_arr[i]*fabs(M)*u,0),(*q2_accum)[i]);
    cuDoubleComplex s2=cuCmul(make_cuDoubleComplex(P1sin_arr[i]*M,0),(*q1_accum)[i]);
    cuDoubleComplex s3=cuCmul(make_cuDoubleComplex(P1_arr[i],0),(*q2_accum)[i]);
    cuDoubleComplex s4=cuCsub(s1,s2);
    cuDoubleComplex E_theta_mn=cuCmul(j_power_n,cuCadd(s4,s3));
    cuDoubleComplex j_power_np1=djpcache[n+1];
    cuDoubleComplex o1=cuCmul(make_cuDoubleComplex(P1sin_arr[i]*M,0),(*q2_accum)[i]);
    cuDoubleComplex o2=cuCmul(make_cuDoubleComplex(P1sin_arr[i]*fabs(M)*u,0),(*q1_accum)[i]);
    cuDoubleComplex o3=cuCmul(make_cuDoubleComplex(P1_arr[i],0),(*q1_accum)[i]);
    cuDoubleComplex o4=cuCsub(o1,o2);
    cuDoubleComplex E_phi_mn=cuCmul(j_power_np1,cuCsub(o4,o3));
    sigma_P=cuCadd(sigma_P,cuCmul(phi_comp,E_phi_mn));
    sigma_T=cuCadd(sigma_T,cuCmul(phi_comp,E_theta_mn)); 
  }

  if(pol=='x') {
    jm->j00=sigma_T;;
    jm->j01=cuCmul(make_cuDoubleComplex(-1,0),sigma_P);
  } else {
    jm->j10=sigma_T;
    jm->j11=cuCmul(make_cuDoubleComplex(-1,0),sigma_P);
  }

}


/*################################################################*/
__device__ dJonesMatrix jones_compute_direct_device(double az_rad, double za_rad,cuDoubleComplex *q1_accum_x, int q1_accum_size_x,cuDoubleComplex *q2_accum_x, int q2_accum_size_x,double *m_accum_x, int m_accum_size_x,double *n_accum_x, int n_accum_size_x,double *mabsm_x, int mabsm_size_x, double Nmax_x, cuDoubleComplex *q1_accum_y, int q1_accum_size_y,cuDoubleComplex *q2_accum_y, int q2_accum_size_y,double *m_accum_y, int m_accum_size_y,double *n_accum_y, int n_accum_size_y,double *mabsm_y, int mabsm_size_y, double Nmax_y,double *factorial_cache,cuDoubleComplex *j_power_cache) {

  //printf("az = %lf, za = %lf, n_accum_size_x = %d, n_accum_size_y = %d q1_accum_x = (%f, %f), m_accum_x= %lf, nmax = %lf\n", az_rad, za_rad, n_accum_size_x, n_accum_size_y, q1_accum_x[0].x, q1_accum_x[0].y, m_accum_x[0], Nmax_x);
  dJonesMatrix jm;
  double phi_rad=M_PI/2.00 - az_rad;

  jones_calc_sigmas_device(phi_rad,za_rad,Nmax_x,'x',&n_accum_x,n_accum_size_x,&m_accum_x,&mabsm_x,factorial_cache,j_power_cache,&q1_accum_x,&q2_accum_x,&jm);

  jones_calc_sigmas_device(phi_rad,za_rad,Nmax_y,'y',&n_accum_y,n_accum_size_y,&m_accum_y,&mabsm_y,factorial_cache,j_power_cache,&q1_accum_y,&q2_accum_y,&jm);


  return jm; 
}

__global__ void jones_compute_kernel(double *az_rad, double *za_rad, int npos, int zenith_norm, JonesData jonesx,
    JonesData jonesy, double *factorial_cache, cuDoubleComplex *j_power_cache, int *dev_jpcache_offsets, dJonesMatrix *norm_jones, dJonesMatrix *jm_out){
  int idx;
  dJonesMatrix jm;
  idx=blockDim.x * blockIdx.x + threadIdx.x;
  if(idx < npos * jonesx.nElements) {
    int pos = idx / jonesx.nElements;
    int channel = idx % jonesx.nElements;

    jm=jones_compute_direct_device(az_rad[pos],za_rad[pos],
      jonesx.q1_accum + jonesx.q1_offsets[channel], jonesx.q1_offsets[channel + 1] - jonesx.q1_offsets[channel],
      jonesx.q2_accum + jonesx.q2_offsets[channel], jonesx.q2_offsets[channel + 1] - jonesx.q2_offsets[channel],
      jonesx.m_accum + jonesx.m_accum_offsets[channel],jonesx.m_accum_offsets[channel+1]-jonesx.m_accum_offsets[channel],
      jonesx.n_accum + jonesx.n_accum_offsets[channel],jonesx.n_accum_offsets[channel+1]-jonesx.n_accum_offsets[channel],
      jonesx.mabsm + jonesx.mabsm_offsets[channel],jonesx.mabsm_offsets[channel+1]-jonesx.mabsm_offsets[channel],
      jonesx.nmax[channel],
      
      jonesy.q1_accum + jonesy.q1_offsets[channel], jonesy.q1_offsets[channel + 1] - jonesy.q1_offsets[channel],
      jonesy.q2_accum + jonesy.q2_offsets[channel], jonesy.q2_offsets[channel + 1] - jonesy.q2_offsets[channel],
      jonesy.m_accum + jonesy.m_accum_offsets[channel],jonesy.m_accum_offsets[channel+1]-jonesy.m_accum_offsets[channel],
      jonesy.n_accum + jonesy.n_accum_offsets[channel],jonesy.n_accum_offsets[channel+1]-jonesy.n_accum_offsets[channel],
      jonesy.mabsm + jonesy.mabsm_offsets[channel],jonesy.mabsm_offsets[channel+1]-jonesy.mabsm_offsets[channel],
      jonesy.nmax[channel],
      
      factorial_cache, j_power_cache + dev_jpcache_offsets[channel]);
    if(zenith_norm) {
      // we assume that freq_hz didn't change and that normalisation was already computed elsewhere
      //print_complex_4(jm);
      jm.j00 = cuCdiv(jm.j00,norm_jones[channel].j00);
      jm.j01 = cuCdiv(jm.j01,norm_jones[channel].j01);
      jm.j10 = cuCdiv(jm.j10,norm_jones[channel].j10);
      jm.j11 = cuCdiv(jm.j11,norm_jones[channel].j11);
    }
    jm_out[idx]=jm;
  }
}


void compute_powers(double maxn, std::complex<double> **jpcache, int *jpcache_size, int ch) {
  int i;
  std::complex<double> complex_j(0.0, 1.0);
  // initialise powers of j
  if(!(*jpcache)) {
    (*jpcache_size)=((int)maxn+20);
    (*jpcache)=(std::complex<double>*)malloc((*jpcache_size)*sizeof(std::complex<double>));
    (*jpcache)[0]=1;
    for(i=1;i<(*jpcache_size);i++)
      (*jpcache)[i]=(*jpcache)[i-1]*complex_j;
  } else if ((*jpcache_size)<((int)maxn+20)) {
    int jpcache_size_old=(*jpcache_size);
    (*jpcache_size)=((int)maxn+20);
    (*jpcache)=(std::complex<double>*)realloc((*jpcache),(*jpcache_size)*sizeof(std::complex<double>));
    (*jpcache)[0]=1;
    for(i=jpcache_size_old;i<(*jpcache_size);i++)
      (*jpcache)[i]=(*jpcache)[i-1]*complex_j;
  }
}


int jones_compute_gpu(int zenith_norm, double *az, double *ze,
    JonesData jdatax,JonesData jdatay,int npos, dJonesMatrix *norm_jones, std::complex<double>* result){

  //std::cerr << "Inside jones compute\n"<< std::endl;
  JonesData djonesx = jdatax.copy_to_device();
  JonesData djonesy = jdatay.copy_to_device();
  // copy position information
  double *dev_az;
  double *dev_ze;
  CUDA_CHECK_ERROR(cudaMalloc(&dev_az, sizeof(double) * npos));
  CUDA_CHECK_ERROR(cudaMalloc(&dev_ze, sizeof(double) * npos));
  CUDA_CHECK_ERROR(cudaMemcpy(dev_az, az, sizeof(double) * npos, cudaMemcpyHostToDevice));
  CUDA_CHECK_ERROR(cudaMemcpy(dev_ze, ze, sizeof(double) * npos, cudaMemcpyHostToDevice));
  cuDoubleComplex *djpcache;
  double *dfcache = nullptr, *fcache = nullptr;
  std::complex<double> **jpcache = new std::complex<double>*[jdatax.nElements];
  memset(jpcache, 0, sizeof(std::complex<double>*) * jdatax.nElements);
  int fcache_size;
  int *jpcache_size = new int[jdatax.nElements];
  factorial_cache_init(&fcache,&fcache_size, FACTORIAL_MAX_CACHE_SIZE);
  // std::cout << "Factorial: [";
  // for(int i = 0; i < fcache_size; i++) std::cout << fcache[i] << ", ";
  // std::cout << "]\n";
  CUDA_CHECK_ERROR(cudaMalloc(&dfcache,fcache_size*sizeof(double)));
  CUDA_CHECK_ERROR(cudaMemcpy(dfcache, fcache, fcache_size*sizeof(double), cudaMemcpyHostToDevice));
  
  int total_jpcache_size = 0;
  int *dev_jpcache_offsets;
  for(int i = 0; i < jdatax.nElements; i++){
    double maxn=(jdatax.nmax[i]>=jdatay.nmax[i]?jdatax.nmax[i]:jdatay.nmax[i]);
    compute_powers(maxn, jpcache + i, jpcache_size + i, i);
    total_jpcache_size += jpcache_size[i];
  }
  std::complex<double> *jpcache_unified = new std::complex<double>[total_jpcache_size];
  int *jpcache_offsets = new int[jdatax.nElements+1];
  jpcache_offsets[0] = 0;
  for(int i = 0; i < jdatax.nElements; i++){
    memcpy(jpcache_unified + jpcache_offsets[i], jpcache[i], sizeof(std::complex<double>) * jpcache_size[i]);
    jpcache_offsets[i+1] = jpcache_offsets[i] + jpcache_size[i];
  }
  // std::cout << "Powers: [";
  // for(int i = 0; i < jpcache_size; i++) std::cout << *(std::complex<double>*)&jpcache[i] << ", ";
  // std::cout << "]\n";
  CUDA_CHECK_ERROR(cudaMalloc(&djpcache,total_jpcache_size*sizeof(cuDoubleComplex)));
  CUDA_CHECK_ERROR(cudaMalloc(&dev_jpcache_offsets, sizeof(int) * (jdatax.nElements+1)));
  CUDA_CHECK_ERROR(cudaMemcpy(djpcache, jpcache_unified, total_jpcache_size*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
  CUDA_CHECK_ERROR(cudaMemcpy(dev_jpcache_offsets, jpcache_offsets,  sizeof(int) * (jdatax.nElements + 1), cudaMemcpyHostToDevice)); 
  for(int i = 0; i < jdatax.nElements; i++){
    delete[] jpcache[i]; 
  }
  delete[] jpcache;
  delete[] jpcache_size;
  delete[] jpcache_offsets;
  delete[] jpcache_unified;
  dJonesMatrix *dev_norm;
  CUDA_CHECK_ERROR(cudaMalloc(&dev_norm, sizeof(dJonesMatrix) * jdatax.nElements));
  CUDA_CHECK_ERROR(cudaMemcpy(dev_norm, norm_jones, sizeof(dJonesMatrix) * jdatax.nElements, cudaMemcpyHostToDevice));
  //continue here
  int nthreads = 128;
  int nblocks = (npos * jdatax.nElements + nthreads - 1) / nthreads; 
  // execute CUDA kernel
  cuDoubleComplex *output;
  CUDA_CHECK_ERROR(cudaMalloc(&output, sizeof(cuDoubleComplex) * 4 * npos * jdatax.nElements));
  jones_compute_kernel<<<nblocks,nthreads>>>(dev_az, dev_ze, npos, zenith_norm, djonesx, djonesy, dfcache, djpcache, dev_jpcache_offsets, dev_norm, (dJonesMatrix*) output);
  CUDA_CHECK_ERROR(cudaGetLastError());
  CUDA_CHECK_ERROR(cudaMemcpy(result, output,sizeof(cuDoubleComplex) * 4 * npos * jdatax.nElements, cudaMemcpyDeviceToHost));
  CUDA_CHECK_ERROR(cudaStreamSynchronize(0));
  cudaFree(output);
  cudaFree(djpcache);
  cudaFree(dfcache);
  cudaFree(dev_az);
  cudaFree(dev_ze);
  cudaFree(dev_norm);
  cudaFree(dev_jpcache_offsets);
  djonesx.destroy();
  djonesy.destroy();
  return 0;
}
