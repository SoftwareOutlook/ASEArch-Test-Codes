// version of Jacobi smoother
///
// Lucian Anton, July 2013

#include "homb_c.h"
#include "functions_c.h"

static const Real sixth=1.0/6.0;

static void Gold_laplace3d(int NX, int NY, int NZ, Real* u1, Real* u2);
static void Titanium_laplace3d(int NX, int NY, int NZ, Real* u1, Real* u2);
static void Blocked_laplace3d(int NX, int NY, int NZ, int BX, int BY, int BZ, Real* u1, Real* u2);
//static void cco_laplace3d(int iteration, double *norm);
//static void baseline_laplace3d(int iteration);
//static void opt_baseline_laplace3d(int iteration);

////////////////////////////////////////////////////////////////////////////////
// Baseline version that describes the logic of Jacobi iteration in 1 set of loops.
// Howerver this version is far from optimal because of the if statement inside the
// inner loop and of the heavy algebra used to compute ind
static void Gold_laplace3d(int NX, int NY, int NZ, Real* u1, Real* u2) 
{
  int   i, j, k, ind;
  //Real sixth=1.0f/6.0f;  // predefining this improves performance more than 10%

#pragma omp parallel for schedule(static) default(none) shared(u1,u2,NX,NY,NZ) private(i,j,k,ind)
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


// Version that does not initializes the in each iteration boundaries. It is enough to do it only once 
// at initialisation stage.
// The inner loop index algebra uses less operations ( compare with Gold_laplace3d )
static void Titanium_laplace3d(int NX, int NY, int NZ, Real* u1, Real* u2) 
{
  int   i, j, k, ind, indmj, indpj, indmk, indpk, NXY;
  //Real sixth=1.0f/6.0f;  // predefining this improves performance more than 10%

  NXY = NX*NY;
#pragma omp parallel default(none) shared(u1,u2,NX,NY,NZ,NXY) private(i,j,k,ind,indmj,indpj,indmk,indpk)
  {
#pragma omp for schedule(static)
    for (k=1; k<NZ-1; k++) {
      for (j=1; j<NY-1; j++) {
	ind = j*NX + k*NXY;
	indmj = ind - NX;
	indpj = ind + NX;
	indmk = ind - NXY;
	indpk = ind + NXY;
	for (i=1; i<NX-1; i++) {   // i loop innermost for sequential memory access
          u2[ind+i] = ( u1[ind+i-1] + u1[ind+i+1]
			+   u1[indmj+i] + u1[indpj+i]
			+   u1[indmk+i] + u1[indpk+i] ) * sixth;
	  
	}
      }
    }
  }
}


static void Blocked_laplace3d(int NX, int NY, int NZ, int BX, int BY, int BZ, Real* u1, Real* u2) 
{
  int   i, j, k, ind, indmj, indpj, indmk, indpk, NXY,ii,jj,kk;
  //Real sixth=1.0f/6.0f;  // predefining this improves performance more than 10%

  NXY = NX*NY;
#pragma omp parallel default(none) shared(u1,u2,NX,NY,NZ,NXY,BX,BY,BZ) private(i,j,k,ind,indmj,indpj,indmk,indpk,ii,jj,kk)
  {
#pragma omp for schedule(static) 
    for (kk=1; kk<NZ-1; kk+=BZ)
      for (jj=1; jj<NY-1; jj+=BY)
	for(ii=1; ii<NX-1; ii+=BX)
	  for (k=kk; k<MIN(kk+BZ,NZ-1); k++) {
	    for (j=jj; j<MIN(jj+BY,NY-1); j++) {
	      ind = j*NX + k*NXY;
	      indmj = ind - NX;
	      indpj = ind + NX;
	      indmk = ind - NXY;
	      indpk = ind + NXY;
	      for (i=ii; i<MIN(ii+BX,NX-1); i++) {   // i loop innermost for sequential memory access
		u2[ind+i] = ( u1[ind+i-1] + u1[ind+i+1]
			      +   u1[indmj+i] + u1[indpj+i]
			      +   u1[indmk+i] + u1[indpk+i] ) * sixth;
		
	      }
	    }
	  }
  }
}

void laplace3d(const struct grid_info_t *g, int kernel_key, double *tstart, double *tend){
  // wraper for Gold_laplace3d

  int NX = g->ng[0], NY = g->ng[1], NZ = g->ng[2];
  int BX = g->nb[0], BY = g->nb[1], BZ = g->nb[2];
  Real* tmp;
  double norm;// needed by cco_laplace, to be removed later
  int iter; // see abobe
    switch (kernel_key)
      {
      case (BASELINE_KERNEL) :
	*tstart = my_wtime();
	Gold_laplace3d(NX, NY, NZ, uOld, uNew);
	*tend = my_wtime(); break;
      case (OPTBASE_KERNEL) :
	*tstart = my_wtime();
	Titanium_laplace3d(NX, NY, NZ, uOld, uNew);
	*tend = my_wtime(); break;
      case (BLOCKED_KERNEL)  :
	*tstart = my_wtime();
	Blocked_laplace3d(NX, NY, NZ, BX, BY, BZ, uOld, uNew);
	*tend = my_wtime(); break;
      case (CCO_KERNEL)      :  
	*tstart = my_wtime();
	//cco_laplace3d(iter, &norm); 
	*tend = my_wtime(); break;
      }
   
   tmp = uNew; uNew = uOld; uOld = tmp; 

}

/* MPI versions, to revise later
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
*/

void stencil_update(const struct grid_info_t * g, int s1, int e1, int s2, int e2, int s3, int e3, double * norm){
  
  int i, j, k, ijk, ijm1k, ijp1k, ijkm1, ijkp1;
  Real w;

  for (k = s3; k <= e3; ++k){
    for (j = s2; j <= e2; ++j){
      ijk = uindex(g, s1, j, k);
      ijm1k = uindex(g, s1, j-1, k);
      ijp1k = uindex(g, s1, j+1, k);
      ijkm1 = uindex(g, s1, j, k-1);
      ijkp1 = uindex(g, s1, j, k+1);
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

inline int uindex(const struct grid_info_t *g, const int i, const int j, const int k){
  return (i - (g->sx - 1) + (j - (g->sy - 1)) * g->nlx + (k - (g->sz - 1)) * g->nlx * g->nly );
}

