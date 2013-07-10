// version of Jacobi smoother
///
// Lucian Anton, July 2013

#include "homb_c.h"
#include "functions_c.h"

static const Real sixth=1.0/6.0;

static void Gold_laplace3d(int NX, int NY, int NZ, Real* u1, Real* u2);

////////////////////////////////////////////////////////////////////////////////
// common version
static void Gold_laplace3d(int NX, int NY, int NZ, Real* u1, Real* u2) 
{
  int   i, j, k, ind;
  //Real sixth=1.0f/6.0f;  // predefining this improves performance more than 10%

  for (k=0; k<NZ; k++) {
    for (j=0; j<NY; j++) {
      for (i=0; i<NX; i++) {   // i loop innermost for sequential memory access
	ind = i + j*NX + k*NX*NY;

        if (i==0 || i==NX-1 || j==0 || j==NY-1|| k==0 || k==NZ-1) {
          u2[ind] = u1[ind];          // Dirichlet b.c.'s
        }
        else {
          u2[ind] = ( u1[ind-1    ] + u1[ind+1    ]
                    + u1[ind-NX   ] + u1[ind+NX   ]
                    + u1[ind-NX*NY] + u1[ind+NX*NY] ) * sixth;
        }
      }
    }
  }
}


void common_laplace3d(int iteration, double *norm){
  // wraper for Gold_laplace3d

  int NX = ngxyz[0], NY=ngxyz[1], NZ=ngxyz[2];
  Real* tmp;

  Gold_laplace3d(NX, NY, NZ, uOld, uNew);

  //compute the norm (squared)
  int i, j, k;
  *norm = 0.0;
  for (i=1; i<NX-1;++i)
    for (j=1; j<NY-1;++j)
      for (k=1; k<NZ-1;++k)
	*norm += uNew[i + j*NX + k*NX*NY]* uNew[i + j*NX + k*NX*NY];
  
   tmp = uNew; uNew = uOld; uOld = tmp; 

}

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


void cco_laplace3d(int iteration, double *norm){

  Real* tmp;
  double x = 0.0;
  

#pragma omp parallel if (nthreads > 1) default(none) shared(sx, ex, sy, ey, sz, ez, nthreads) reduction(+:x)
  {
#ifdef USE_MPI
    post_recv();
    buffer_halo_transfers(OUT, &x, 0);
    exchange_halos();
    buffer_halo_transfers(IN,  &x, 1);
#endif
    // explicit work placement equivalent to static schedule
    // but only amongst threads 1 ... nthreads - 1 ( if nthreads > 1)
    // could be done in initalisation with thread private variables 
   
    int s3, e3, dt, d1, blk, r, tid;
    if ( nthreads > 1) {
      tid = omp_get_thread_num();
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
      s3 = sz + 1; e3 = ez - 1;
    } 
    
    stencil_update(sx+1, ex-1, sy+1, ey-1, s3, e3, &x);

  }
  
  *norm = x;
  tmp = uNew; uNew = uOld; uOld = tmp;
  
}

void stencil_update(int s1, int e1, int s2, int e2, int s3, int e3, double * norm){
  
  int i, j, k, ijk, ijm1k, ijp1k, ijkm1, ijkp1;
  Real w;

  for (k = s3; k <= e3; ++k){
    for (j = s2; j <= e2; ++j){
      ijk = uindex(s1, j, k);
      ijm1k = uindex(s1, j-1, k);
      ijp1k = uindex(s1, j+1, k);
      ijkm1 = uindex(s1, j, k-1);
      ijkp1 = uindex(s1, j, k+1);
      for (i = 0; i < e1 - s1 + 1; ++i){
	w = sixth *
	  (uOld[ijk + i - 1] + uOld[ijk + i + 1] +
	   uOld[ijm1k + i] + uOld[ijp1k + i] + 
	   uOld[ijkm1 + i] + uOld[ijkp1 + i]);
	*norm = *norm + w*w;
	uNew[ijk + i] = w;
      }
    }
  }  
}

 inline int uindex(const int i, const int j, const int k){
  return (i - (sx - 1) + (j - (sy - 1)) * nlx + (k - (sz - 1)) * nlx * nly );
}

