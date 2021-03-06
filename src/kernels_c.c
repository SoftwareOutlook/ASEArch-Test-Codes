/*
  variants for Jacobi smoother

  Lucian Anton, December 2014


Copyright (c) 2014, Science & Technology Facilities Council, UK
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer. 
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The views and conclusions contained in the software and documentation are those
of the authors and should not be interpreted as representing official policies, 
either expressed or implied, of the FreeBSD Project.

*/


#include "jacobi_c.h"
#include "comm_mpi_c.h"
#include "kernels_c.h"
#include "comm_mpi_c.h"
#include "utils_c.h"
#ifdef USE_CUDA
#include "gpu_laplace3d_wrapper.h"
#endif
#ifdef USE_OPENCL
#include"OpenCL/jacobi_opencl.h"
#endif
#ifdef _OPENACC
#include "openacc.h"
#endif

static const Real sixth=1.0/6.0;

static void Gold_laplace3d(int NX, int NY, int NZ, int nxShift, Real* u1, Real* u2);
static void Titanium_laplace3d(int NX, int NY, int NZ, int nxShift, Real* u1, Real* u2);
static void Blocked_laplace3d(int NX, int NY, int NZ, int nxShift, int BX, int BY, int BZ, Real* u1, Real* u2);
static void Wave_laplace3d(int NX, int NY, int NZ, int nxShift, int BX, int BY, int BZ, int iter_block, int threads_per_column, Real* u1, Real* u2);
static void Wave_diagonal_laplace3d(int NX, int NY, int NZ, int nxShift, int BX, int BY, int BZ, int iter_block, Real* u1, Real* u2);
static void cco_laplace3d(const struct grid_info_t *g, int iteration, double *tcomp, double *tcomm);

//! driver for the Jacobi kernels 
void laplace3d(const struct grid_info_t *g, double *tcomp, double *tcomm){
  // wrapper that controls which variant from above is to be excecuted
  // also does MPI communication is MPI is active

  int NX = g->nlx, NY = g->nly, NZ = g->nlz, nxShift;
  int BX = g->nb[0], BY = g->nb[1], BZ = g->nb[2];
  int compute_model = g->cmod_key, alg_key = g->alg_key;
  Real* tmp;
  double taux;
#ifdef USE_MPI
  double taux2;
#endif
  int  step; /*!< see abobe */
  //!if using GPU execution apply this function for cuda execution
  //!configuration parameters
#ifdef USE_CUDA
  int gridxy[4]; //for blocks and grid dims
#endif

  //! set shift for x direction, this is differs from NX if posix malign is used
  //! assumes that uOld > uNew
  if ( g->malign < 0) 
    nxShift = NX;
  else
    nxShift = (abs((int) (uNew - uOld))) / (NY*NZ); 

  *tcomm = 0.0;

  switch (compute_model)
    {
    case(CMODEL_OMP):
#ifdef USE_MPI
    case(CMODEL_MPIOMP):
#endif
      switch (alg_key)
	{
	case (ALG_BASELINE) :
	  taux = my_wtime();
	  for (step = 0; step < niter; ++step){
#ifdef USE_MPI
	    taux2 = my_wtime();
	    exchange_halos(g);
	    *tcomm += my_wtime() - taux2;
#endif
	    Gold_laplace3d(NX, NY, NZ, nxShift, uOld, uNew);
	    tmp = uNew; uNew = uOld; uOld = tmp;
	  } 
	  *tcomp = my_wtime() - taux;
	  break;
	case (ALG_BASELINE_OPT) :
	  taux = my_wtime();
	  for (step = 0; step < niter; ++step){  
#ifdef USE_MPI
	    taux2 = my_wtime();
	    exchange_halos(g);
	     *tcomm += my_wtime() - taux2;
#endif
	    Titanium_laplace3d(NX, NY, NZ, nxShift, uOld, uNew);
	    tmp = uNew; uNew = uOld; uOld = tmp;
	  }
	  *tcomp = my_wtime() - taux;
	  break;
	case (ALG_BLOCKED) :
	  taux = my_wtime();
	  for (step = 0; step < niter; ++step){   
#ifdef USE_MPI
	    taux2 =  my_wtime();
	    exchange_halos(g);
	    *tcomm += my_wtime() - taux2;
#endif
	    Blocked_laplace3d(NX, NY, NZ, nxShift, BX, BY, BZ, uOld, uNew);
	    tmp = uNew; uNew = uOld; uOld = tmp;
	  }
	  *tcomp = my_wtime() - taux;
	  break;
#ifdef USE_MPI
	case (ALG_CCO)  :  
	  for (step = 0; step < niter; ++step){
	    cco_laplace3d(g, niter, tcomp, tcomm);
	    tmp = uNew; uNew = uOld; uOld = tmp; 
	  }
	  break;
#endif
	case (ALG_WAVE) :
	  taux = my_wtime();
#ifdef USE_MPI
	  // Wave parallelism needs extended halos to accomodate the blocked wave evolution
	  //exchange_halos(g);
	  error_abort("MPI not implemented for wave parallelism", "");
#endif
	  Wave_laplace3d(NX, NY, NZ, nxShift, BX, BY, BZ, niter, g->threads_per_column, uOld, uNew);
	  if ( niter%2 == 1) {
	    tmp = uNew; uNew = uOld; uOld = tmp;
	  }
	  *tcomp = my_wtime() - taux;
	  break;
	case(ALG_WAVE_DIAGONAL) :
	  taux = my_wtime();
#ifdef USE_MPI
	  // Wave parallelism needs extended halos to accomodate the blocked wave evolution
	  //exchange_halos(g);
	  error_abort("MPI not implemented for wave diagonal parallelism","");
#endif
	  Wave_diagonal_laplace3d(NX, NY, NZ, nxShift, BX, BY, BZ, niter, uOld, uNew);
	  if ( niter%2 == 1) {
	    tmp = uNew; uNew = uOld; uOld = tmp;
	  }
	  *tcomp = my_wtime() - taux;
	  break;
	default :
	  error_abort("kernels_c.c: cannot find the specified algorithm for OpenMP language", "");
	}
      break;
#if defined(USE_CUDA)
    case(CMODEL_CUDA):
      switch(alg_key)
	{
	  float taux_comp, taux_comm;
	case ALG_CUDA_2D_BLK:
	case ALG_CUDA_3D_BLK:
	case ALG_CUDA_SHM :
	case ALG_CUDA_BANDWIDTH: 
	  calcGpuDims( alg_key, BX, BY, BZ, NX, NY, NZ, gridxy);
      //invoke GPU function
	  laplace3d_GPU(alg_key, uOld, NX, NY, NZ, gridxy, niter, &taux_comp, &taux_comm);
	  *tcomp = 0.001 * taux_comp; // CUDA timer works with ms
	  *tcomm = 0.001 * taux_comm;
	  break;
	default:
	  error_abort("kernels_c.c: cannot find the specified algorithm for CUDA laguage", ""); 
	}
      break;
#elif defined(USE_OPENCL)
    case (CMODEL_OPENCL):
      //OpenCL functionality
      switch(alg_key)
	{
	case(ALG_OPENCL_BASELINE):
	  /** \todo Generate timing detail for initialisation */
	  //Initialise the OpenCL context/program/kernel
	  taux = my_wtime();
	  OpenCL_Jacobi_CopyTo(uOld);
	  *tcomm=my_wtime()-taux;
	  //Call the OpenCL kernel
	  taux = my_wtime();
	  OpenCL_Jacobi_Iteration(niter);
	  *tcomp = my_wtime() - taux;
	  //Tidy up
	  taux=my_wtime();
	  OpenCL_Jacobi_CopyBack(uOld);
	  *tcomm+=my_wtime()-taux;
	  break;
	default:
	  error_abort("kernels_c.c: cannot find the specified algorithm for OpenCL laguage", "");
	}
      break;
#elif defined(_OPENACC)
    case (CMODEL_OPENACC):
      switch(alg_key)
	{
	case(ALG_OPENACC_BASELINE):
          taux = my_wtime();
#pragma acc enter data copyin(uOld[0:nxShift*NY*NZ]) create(uNew[0:nxShift*NY*NZ])
          *tcomm = my_wtime() - taux;
          Real* u1_device =acc_deviceptr(uOld);
          Real* u2_device =acc_deviceptr(uNew);
          taux = my_wtime();
          for (step = 0; step < niter; ++step){
#ifdef USE_MPI
            exchange_halos(g);
#endif
	    Gold_laplace3d(NX, NY, NZ, nxShift, u1_device, u2_device);
	    tmp = u2_device; u2_device = u1_device; u1_device = tmp;
          }
          *tcomp = my_wtime() - taux;
          uOld = acc_hostptr(u1_device);
          uNew = acc_hostptr(u2_device);
          taux = my_wtime();
#pragma acc exit data copyout(uOld[0:nxShift*NY*NZ]) delete(uNew)
          *tcomm += my_wtime() - taux;
          break;
	default:
          error_abort("kernels_c.c: cannot find the specified algorithm for OpenACC laguage", "");
	}
      break;
#endif
    default :
      error_abort("kernels_c.c: unkown compute model, try -cmodel help", "");
    }
}




////////////////////////////////////////////////////////////////////////////////
// Baseline version that describes the logic of Jacobi iteration in one set of loops.
// Howerver this version is far from optimal because of the if statement inside the
// inner loop and of the heavy algebra used to compute ind
static void Gold_laplace3d(int NX, int NY, int NZ, int nxShift, Real* u1, Real* u2) 
{
  int   i, j, k, ind;
  //Real sixth=1.0f/6.0f;  // predefining this improves performance more than 10%

#ifdef _OPENACC
#pragma acc data deviceptr(u1,u2)
#pragma acc kernels loop collapse(3) independent private(i,j,k,ind)
#else
#pragma omp parallel for schedule(static) default(none) shared(u1,u2,NX,NY,NZ,nxShift) private(i,j,k,ind)
#endif
  for (k=0; k<NZ; k++) {
    for (j=0; j<NY; j++) {
      for (i=0; i<NX; i++) {   // i loop innermost for sequential memory access
	ind = i + j*nxShift + k*nxShift*NY;

        if (i==0 || i==NX-1 || j==0 || j==NY-1|| k==0 || k==NZ-1) {
          u2[ind] = u1[ind];          // Dirichlet b.c.'s
        }
        else {
          u2[ind] = ( u1[ind-1    ] + u1[ind+1    ]
		      + u1[ind-nxShift   ] + u1[ind+nxShift   ]
		      + u1[ind-nxShift*NY] + u1[ind+nxShift*NY] ) * sixth;
        }
      }
    }
  }
}

// 1D loop that helps the compiler to vectorize
static void vec_oneD_loop(const int n, const Real *restrict uNorth, const Real *restrict uSouth, const Real *restrict uWest,const Real *restrict uEast, const Real *restrict uBottom, const Real *restrict uTop, Real *restrict w ){
  int i;
  //static Real inv6=1.0f/6.0f;
  //#pragma unroll(4)
  //extern int sscal;
#if 0
#ifdef __INTEL_COMPILER
#pragma ivdep
#endif
#ifdef __IBMC__
#pragma ibm independent_loop
#endif
#endif
  for (i=0; i < n; ++i)
    w[i] = sixth * (uNorth[i] + uSouth[i] + uWest[i] + uEast[i] + uBottom[i] + uTop[i]);
  //sscal(&n,&inv6,w);
}

// Version that does not initializes the boundaries in each iteration.
// It is enough to do it only once at initialisation stage.
// The inner loop index algebra uses less operations ( compare with Gold_laplace3d )
static void Titanium_laplace3d(int NX, int NY, int NZ, int nxShift, Real* u1, Real* u2) 
{
  int   i, j, k, ind, indmj, indpj, indmk, indpk, NXY;
  //Real sixth=1.0f/6.0f;  // predefining this improves performance more than 10%

  NXY = nxShift*NY;
#pragma omp parallel for default(none) shared(u1,u2,NX,NY,NZ,NXY, nxShift) private(i,j,k,ind,indmj,indpj,indmk,indpk) schedule(static) collapse(2)
  for (k=1; k<NZ-1; k++) {
    for (j=1; j<NY-1; j++) {
      ind = j*nxShift + k*NXY;
      indmj = ind - nxShift;
      indpj = ind + nxShift;
      indmk = ind - NXY;
      indpk = ind + NXY;
#ifdef USE_VEC1D
      vec_oneD_loop(NX-2, &u1[ind], & u1[ind+2], &u1[indmj+1], &u1[indpj+1], &u1[indmk+1],&u1[indpk+1], &u2[ind+1]);
#else
      for (i=1; i<NX-1; i++) {   // i loop innermost for sequential memory access
	u2[ind+i] = ( u1[ind+i-1] + u1[ind+i+1]
		      +   u1[indmj+i] + u1[indpj+i]
		      +   u1[indmk+i] + u1[indpk+i] ) * sixth;
	  
      }
#endif
    }
  }
}


static void Blocked_laplace3d(int NX, int NY, int NZ, int nxShift, int BX, int BY, int BZ, Real* u1, Real* u2) 
{
  int   i, j, k, ind, indmj, indpj, indmk, indpk, NXY,ii,jj,kk;

  NXY = nxShift*NY;
#pragma omp parallel default(none) shared(u1,u2,NX,NY,NZ,NXY, nxShift, BX,BY,BZ) private(i,j,k,ind,indmj,indpj,indmk,indpk,ii,jj,kk)
  {
#pragma omp for schedule(static,1) collapse(3)
    for (kk=1; kk<NZ-1; kk+=BZ)
      for (jj=1; jj<NY-1; jj+=BY)
	for(ii=1; ii<NX-1; ii+=BX)
	  for (k=kk; k<MIN(kk+BZ,NZ-1); k++) {
	    for (j=jj; j<MIN(jj+BY,NY-1); j++) {
	      ind = j*nxShift + k*NXY;
	      indmj = ind - nxShift;
	      indpj = ind + nxShift;
	      indmk = ind - NXY;
	      indpk = ind + NXY;
#ifdef USE_VEC1D
	      vec_oneD_loop( MIN(ii+BX,NX-1)-ii, &u1[ind+ii-1], &u1[ind+ii+1], &u1[indmj+ii], &u1[indpj+ii], &u1[indmk+ii], &u1[indpk+ii], &u2[ind+ii]);
#else
	      for (i=ii; i<MIN(ii+BX,NX-1); i++) {   // i loop innermost for sequential memory access
		u2[ind+i] = ( u1[ind+i-1] + u1[ind+i+1]
			      +   u1[indmj+i] + u1[indpj+i]
			      +   u1[indmk+i] + u1[indpk+i] ) * sixth;
		
	      }
#endif
	    }
	  }
  }
}

static void Wave_diagonal_laplace3d(int NX, int NY, int NZ, int nxShift, int BX, int BY, int BZ, int iter_block, Real* u1, Real* u2) 
// iterates trought the wave block along diagonals using
// omp for 
{
  int  NXY = nxShift*NY;

#pragma omp parallel default(none) shared(u1, u2, NX, NY, NZ, NXY, nxShift,BX, BY, BZ, iter_block)
  { 
    int iplane, wplane, iwave, left_y, right_y, iblock;
    int nwaves = iter_block;
    
    Real *unew, *uold, *tmp;
   
    int nby = (NY - 2) / BY;
    if ( (NY - 2) % BY != 0 ) ++nby;
    int nbz = (NZ - 2) / BZ;
    if ( (NZ - 2) % BZ != 0 ) ++nbz;
    
    for (iplane = 0; iplane <= (nby - 1 + nbz - 1) + 2 * (nwaves - 1); ++iplane){  
      unew = u2; uold = u1;
      for (iwave = 0; iwave < nwaves; ++iwave){
	
	wplane = iplane - 2 * iwave;
	
	if ( (wplane >= 0) && (wplane <= (nby - 1 + nbz - 1)) ){
	  
	  if ( wplane  < nbz ) 
	    left_y = 0;
	  else
	    left_y = wplane - nbz + 1;
	  
	  if ( wplane  < nby  )
	    right_y = wplane;
	  else
	    right_y = nby - 1;
	
	  //printf(" wave %d %d %d %d %d %d\n", iwave, left_y, right_y, nby, nbz, nwaves);

	  int ind, indmj, indpj, indmk, indpk, i, j, k, jj, kk;
#pragma omp for schedule(dynamic,1) nowait
	  for (iblock = left_y; iblock <= right_y; ++iblock){
	    jj = iblock * BY + 1;
	    kk = (wplane - iblock) * BZ + 1;
	    //printf(" wave jj kk %d %d\n", jj , kk);
	    for (k=kk; k<MIN(kk+BZ,NZ-1); k++) {
	      for (j=jj; j<MIN(jj+BY,NY-1); j++) {
		ind = j*nxShift + k*NXY;
		indmj = ind - nxShift;
		indpj = ind + nxShift;
		indmk = ind - NXY;
		indpk = ind + NXY;
#ifdef USE_VEC1D
		vec_oneD_loop(NX-2, &u1[ind], &uold[ind+2], &uold[indmj+1], &uold[indpj+1], &uold[indmk+1],&uold[indpk+1], &unew[ind+1]);
#else
		for (i = 1; i < NX-1; i++) {   
		  unew[ind+i] = ( uold[ind+i-1] + uold[ind+i+1]
				  +   uold[indmj+i] + uold[indpj+i]
				  +   uold[indmk+i] + uold[indpk+i] ) * sixth;
		  
		}
#endif
	      }
	    }
	  }
	}
	tmp = unew; unew = uold; uold = tmp; 
      }
#pragma omp barrier
    }
  }
}


static void Wave_laplace3d(int NX, int NY, int NZ, int nxShift, int BX, int BY, int BZ, int iter_block, int threads_per_column, Real* u1, Real* u2) 
// assigns set of threads columnwise accros the wave
//
/*
  --------------------------------
  |   |   |   |   |   |    | 
  4  |   |   |   |   |   |    |
  --------------------------------
  | t0|   |   |   |   |    |
  3  |   |   |   |   |   |    |
  --------------------------------
  |   | t2|   |   |   |    |
  2  |   |   |   |   |   |    |
  --------------------------------
  | t1|   | t4|   |   |    |
  1  |   |   |   |   |   |    |
  --------------------------------
  |   | t3|   | t6|   |    |
  0  |   |   |   |   |   |    |
  --------------------------------
  0   1   2   3   4    5

  Two waves with two threads per colomn	     

*/    
{
  int  NXY = nxShift*NY;
  int max_thread_column = nthreads/threads_per_column;

#pragma omp parallel default(none) shared(u1, u2, NX, NY, NZ, NXY, nxShift, BX, BY, BZ, iter_block, max_thread_column, threads_per_column)
  { 
    int iplane, left_y, right_y;
    int nwaves = iter_block;
    
    Real *unew, *uold;
   
    int nby = (NY - 2) / BY;
    if ( (NY - 2) % BY != 0 ) ++nby;
    int nbz = (NZ - 2) / BZ;
    if ( (NZ - 2) % BZ != 0 ) ++nbz;
    
    // the threads are partition in threads_per_column rows, max_thread_column columns
    // k is the fast direction
#ifdef _OPENMP
    int jth = omp_get_thread_num()/threads_per_column;
    int kth = omp_get_thread_num()%threads_per_column;
#else
    int jth = 0; int kth = 0;
#endif       
    // loop over the grid blocks in diagonal planes
    for (iplane = 0; iplane <= (nby - 1 + nbz - 1) + 2 * (nwaves - 1); ++iplane){  

      // set the left and right limits for wave
      // tricky here, its easy to loose some tiles
      // add more explanation
      if ( iplane - 2 * (nwaves - 1) < nbz ) 
	left_y = 0;
      else
	left_y = iplane - 2 * (nwaves - 1) - nbz + 1;
	  
      if ( iplane  < nby )
	right_y = iplane;
      else
	right_y = nby ;
      //  + 2 * (nwaves - 1)
      //printf(" wave %d %d %d %d %d %d\n", iplane, left_y, right_y, nby, nbz, nwaves);

      int ind, indmj, indpj, indmk, indpk, i, j, k, jj, kk;
      int jblock, kblock;
      
      // over the blocks belonging the waves at given iplane
      // first wave has the iplane index, subsequent ones are behind with stride 2 
      for (jblock = left_y + jth; jblock <= right_y; jblock += max_thread_column){
	for (kblock = iplane - jblock - 2 * kth; kblock >= MAX(0, iplane - jblock - 2 * (nwaves-1)); kblock -= 2 * threads_per_column ){
	 
	  // some blocks fall outside grid 
	  if (kblock <= nbz - 1 && jblock <= nby - 1){ 
	    // where to write the new values; get the index of the wave
	    if (((iplane - jblock - kblock)/2)%2 == 0){
	      unew = u2; uold = u1;}
	    else{ 
	      unew = u1; uold = u2;}
	    jj = jblock * BY + 1;
	    kk = kblock * BZ + 1;
	    //printf(" wave jj kk %d %d %d %d %d %d\n", jj , kk, iplane, nwaves, BY, BZ);
	    for (k = kk; k < MIN(kk+BZ, NZ-1); k++) {
	      for (j = jj; j < MIN(jj+BY, NY-1); j++) {
		ind = j*nxShift + k*NXY;
		indmj = ind - nxShift;
		indpj = ind + nxShift;
		indmk = ind - NXY;
		indpk = ind + NXY;
#ifdef USE_VEC1D
		vec_oneD_loop(NX-2, &uold[ind], &uold[ind+2], &uold[indmj+1], &uold[indpj+1], &uold[indmk+1],&uold[indpk+1], &unew[ind+1]);
#else
		for (i = 1; i < NX-1; i++) {   
		  unew[ind+i] = ( uold[ind+i-1] + uold[ind+i+1]
				  +   uold[indmj+i] + uold[indpj+i]
				  +   uold[indmk+i] + uold[indpk+i] ) * sixth;
		  
		}
#endif
	      }
	    }
	  }
	}
      }
#pragma omp barrier
    }
  }
}



/* MPI versions */
/*
  void blocked_laplace3d(int iteration, double *norm){

  Real * tmp;
  double x = 0.0;
  
  #pragma omp parallel if (nthreads > 1) default(none) shared(sx, ex, sy, ey, sz, ez, nthreads) reduction(+:x)
  {
  #ifdef USE_MPI
  post_recv();
  buffer_halo_transfers(OUT, &x, 0);
  exchange_halos();
  buffer_halo_transfers(IN, &x, 0);
  #pragma omp barrier
    
  #endif
  // explicit work placement equivalent to static schedule
  // could be done in initalisation with threadprivate variables 
  int tid = omp_get_thread_num();
  int s3, e3, dt, d1;
  int blk = (ez - sz + 1) / nthreads; 
  int r = (ez - sz + 1) % nthreads;
    
  if (tid < r ){ 
  dt = tid; d1 = 1;}
  else{
  dt = r; d1 = 0;}

  s3 = sz + tid * blk + dt;
  e3 = s3 - 1 + blk + d1 ;

  stencil_update(sx, ex, sy, ey, s3, e3, &x);
  }
  *norm = x;
  tmp = uNew; uNew = uOld; uOld = tmp;  
  }
*/

static void cco_laplace3d(const struct grid_info_t *g, int iteration, double *tcomp, double *tcomm){

  int sx = g->sx, ex = g->ex;
  int sy = g->sy, ey = g->ey;
  int sz = g->sz, ez = g->ez;
  double tp = 0.0, tc = 0.0; // time processing, time communication

#pragma omp parallel if (nthreads > 1) default(none) shared(g, sx, ex, sy, ey, sz, ez, nthreads, tp, tc) 
  {
#ifdef USE_MPI
    double t0 = my_wtime();
    post_recv(g);
    buffer_halo_transfers(g, OUT, NO_UPDATE);
    post_send(g);
    buffer_halo_transfers(g, IN, UPDATE);
    tc += my_wtime() - t0;
#endif
    // explicit work placement equivalent to static schedule
    // but only amongst threads 1 ... nthreads - 1 ( if nthreads > 1)
    // could be done in initalisation with thread private variables 
   
    int s3, e3, dt, d1, blk, r, tid;
    double tt = my_wtime();
    if ( nthreads > 1) {
#ifdef _OPENMP
      tid = omp_get_thread_num();
#else 
      tid = 0;
#endif
      if (tid > 0) {
	blk = (ez - 1 - (sz + 1) + 1) / (nthreads - 1); 
	r = (ez - 1 - (sz + 1) + 1) % (nthreads - 1);    
	if (tid - 1  < r ){ 
	  dt = tid - 1 ; d1 = 1;}
	else{
	  dt = r; d1 = 0;}
	s3 = sz + 1 + (tid - 1) * blk + dt;
	e3 = s3 - 1 + blk + d1;
      }
      else{
	s3 = 0; e3 = -1;
      }
    }
    else{
      s3 = sz+1; e3 = ez-1;
    } 
    
    stencil_update(g, sx+1, ex-1, sy+1, ey-1, s3, e3);
    //stencil_update(g, sx, ex, sy, ey, sz, ez);
    tt += my_wtime()-tt;
#pragma omp atomic
    tc += tt;
  }

  *tcomp = tp;
  *tcomm = tc / (double)(nthreads > 1 ? nthreads-1 : 1);
 
}

void stencil_update(const struct grid_info_t * g, int s1, int e1, int s2, int e2, int s3, int e3){
  
  int i, j, k, ijk, ijm1k, ijp1k, ijkm1, ijkp1;
  Real w;

  for (k = s3; k <= e3; ++k){
    for (j = s2; j <= e2; ++j){
      ijk   = uindex(g, s1, j, k);
      ijm1k = uindex(g, s1, j-1, k);
      ijp1k = uindex(g, s1, j+1, k);
      ijkm1 = uindex(g, s1, j, k-1);
      ijkp1 = uindex(g, s1, j, k+1);
      for (i = 0; i < e1 - s1 + 1; ++i){
	w = sixth *
	  (uOld[ijk + i - 1] + uOld[ijk + i + 1] +
	   uOld[ijm1k + i] + uOld[ijp1k + i] + 
	   uOld[ijkm1 + i] + uOld[ijkp1 + i]);
	uNew[ijk + i] = w;
      }
    }
  }  
}

int uindex(const struct grid_info_t *g, const int i, const int j, const int k){
  // Attention: this function works with global gird indices i, j, k

  int nxShift;
  
  if ( g->malign < 0) 
    nxShift = g->nlx;
  else
    nxShift = (abs((int) (uNew - uOld))) / (g->nly * g->nlz); 

 
  return (i - (g->sx - 1) + (j - (g->sy - 1)) * nxShift + (k - (g->sz - 1)) * nxShift * g->nly );
}

