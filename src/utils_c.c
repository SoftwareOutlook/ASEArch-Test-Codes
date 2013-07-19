/*
This is the partof  DL_HOMB, a hybrid OpemMP MPI Benchmark for
3D Jacobi solver for Laplace equation.

Lucian Anton July 2013.

This code started from v 1.0 of HOMB
http://sourceforge.net/projects/homb/

Below is the original copyright and licence.



  Copyright 2009 Maxwell Lipford Hutchinson

  This file is part of HOMB.

  HOMB is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  HOMB is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with HOMB.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "homb_c.h"
#include "utils_c.h"
#include "kernels_c.h"
#include "comm_mpi_c.h"


// Number of iteration 
int niter;

// OpenMP threads number
int nthreads;

// solution array
Real  *udata, *uOld, *uNew;

// eigenvalue k vector (mode)
static Real kx=1.0, ky=1.0, kz=1.0;

// Output control
int pContext, testComputation; 
static int vOut, pHeader;


void initContext(int argc, char *argv[], struct grid_info_t * grid, int *kernel_key){
 
  /* "Logicals" for file output */
  vOut = 0;  pHeader = 1; pContext = 0; testComputation = 0;
  
  /* Defaults */
  grid->ng[0] = 33; grid->ng[1] = 33; grid->ng[2] = 33; //Global grid
  grid->nb[0] = grid->ng[0]; grid->nb[1] = grid->ng[1]; grid->nb[2] = grid->ng[2]; // block sizes 
  niter = 20;  nthreads = 1; grid->nproc = 1;
  grid->np[0] = 1; grid->np[1] = 1; grid->np[2] = 1;
  grid->cp[0] = 0; grid->cp[1] = 0; grid->cp[2] = 0;
  *kernel_key = BASELINE_KERNEL;
  int i;

  /* Cycle through command line args */
  i = 0;
  while ( i < argc - 1 ){
    ++i;
    //printf(" arg %d %d %s \n", i, argc, argv[i]);
    /* Look for grid sizes */
    if ( strcmp("-ng", argv[i]) == 0 ){
      sscanf(argv[++i],"%d",&(grid->ng[0]));
      sscanf(argv[++i],"%d",&(grid->ng[1]));
      sscanf(argv[++i],"%d",&(grid->ng[2]));
    }
    /* Look for computational block sizes */
    else if ( strcmp("-nb", argv[i]) == 0 ){
      sscanf(argv[++i],"%d", &(grid->nb[0]));
      sscanf(argv[++i],"%d", &(grid->nb[1]));
      sscanf(argv[++i],"%d", &(grid->nb[2]));
    }
    /* Look for MPI topology */  
    else if ( strcmp("-np", argv[i]) == 0 ){
      sscanf(argv[++i],"%d", &(grid->np[0]));
      sscanf(argv[++i],"%d", &(grid->np[1]));
      sscanf(argv[++i],"%d", &(grid->np[2]));
    }
    /* Look for number number of iterations*/  
    else if ( strcmp("-niter", argv[i]) == 0 ){
      sscanf(argv[++i],"%d",&niter);
    }
    /* Look for verbouse output */
    else if ( strcmp("-v", argv[i]) == 0 ){
      vOut = 1;
    }
    /* Look for "No Header" option */
    else if ( strcmp("-nh",argv[i]) == 0 ){
      pHeader = 0;
    }
    /* Look for "Print Context" option */
    else if ( strcmp("-pc",argv[i]) == 0 ){
      pContext = 1;
    }
    /* Look for kernel to use */
    else if ( strcmp("-model", argv[i]) == 0 ){
      ++i;
      if (strcmp("baseline",argv[i]) == 0)
	*kernel_key = BASELINE_KERNEL;
      else if ( strcmp("baseline-opt",argv[i]) == 0)
	*kernel_key = OPTBASE_KERNEL;
      else if (strcmp("blocked",argv[i]) == 0)
	*kernel_key = BLOCKED_KERNEL;
      else if (strcmp("cco",argv[i]) == 0)
	*kernel_key = CCO_KERNEL;
      else{
	printf("Wrong model specifier %s, try -help\n", argv[i]);
#ifdef USE_MPI
	MPI_Abort(MPI_COMM_WORLD,-1);
#else
	exit(-1);
#endif
      }
      
    }
    /* Look for eigenvalue test */
    else if ( strcmp("-t",argv[i] ) == 0){
      testComputation = 1;
      //printf(" i %d \n",i); 
    }
    /* Look for "verbose" standard out */
    else if ( strcmp("-help",argv[i]) == 0 ){
      printf("Usage: %s [-ng <grid-size-x> <grid-size-y> <grid-size-z> ] \
[ -nb <block-size-x> <block-size-y> <block-size-z>]			\
[-np <num-proc-x> <num-proc-y> <num-proc-z>]  -niter <num-iterations>	\
[-v] [-t] [-pc] [-model <model_name>] [-nh] [-help] \n", argv[0]);
#ifdef USE_MPI
      MPI_Finalize();
#endif
      exit(EXIT_SUCCESS);
    }
    else{
      /* basic test on flag values */
      if (argv[i][0] != '-' ){
	printf("Wrong option flag %s, try -help\n", argv[i]);
#ifdef USE_MPI
	MPI_Abort(MPI_COMM_WORLD,-1);
#else
	exit(-1);
#endif
      }
    }

  }
}

void setPEsParams(struct grid_info_t *g) {

#ifdef USE_MPI
  MPI_Comm_size(MPI_COMM_WORLD, &(g->nproc));
  if ( g->nproc != g->np[0] * g->np[1] * g->np[2]){
    fprintf(stderr, "MPI topology sizes %d %d %d does not fit with nproc %d \n", g->np[0], g->np[1], g->np[2], g->nproc);
    MPI_Abort(MPI_COMM_WORLD, -1);
  }
#else
  g->nproc = 1;
#endif

#ifdef USE_MPI
  MPI_Comm_rank(MPI_COMM_WORLD, &(g->myrank)); 
#else 
  g->myrank = 0;
#endif

  /* generate 3d topology */
#ifdef USE_MPI
  int periods[3] = {0, 0, 0}; 
  MPI_Cart_create(MPI_COMM_WORLD, 3, g->np, periods , 0, &(g->comm));
#endif

  compute_local_grid_ranges(g);

  /* Find number of threads per task */
  nthreads = 1;
#ifdef _OPENMP
  nthreads = omp_get_max_threads();
#endif
  //  #pragma omp parallel shared(nThreads)
  //{
  //  #pragma omp single
  //     nThreads = omp_get_num_threads();
  //}
}


void initialise_grid( const struct grid_info_t *g) {
 
  int i, j, k, ijk, n = 2 * g->nlx * g->nly * g->nlz; 
  
  udata = malloc(n * sizeof(Real));

  uOld = &udata[n/2]; uNew = &udata[0];
  // use openmp region for first touch policy for numa nodes; needs further refinement
#pragma omp parallel  for schedule(static) default (shared) private(i,j,k, ijk)
  for (k = g->sz - 1; k <= g->ez + 1; ++k)
    for (j = g->sy - 1; j <= g->ey + 1; ++j)
      for (i = g->sx - 1; i <= g->ex + 1; ++i){
	ijk = uindex(g,i,j,k);
	  uOld[ijk] = sin((PI * i * kx) / (g->ng[0] - 1)) * sin((PI * j * ky) / (g->ng[1] - 1)) * sin((PI * k * kz) / (g->ng[2] - 1));
	  uNew[ijk]=0.0;
      }
  
}


void printContext(const struct grid_info_t *g, int kernel_key){
  printf("Global grid sizes   : %d %d %d \n", g->ng[0], g->ng[1], g->ng[2]);
  if ( kernel_key == BLOCKED_KERNEL) 
    printf("Computational block : %d %d %d \n", g->nb[0], g->nb[1], g->nb[2]); 
#ifdef USE_MPI
  printf("MPI topology        : %d %d %d \n", g->np[0], g->np[1], g->np[2]);
#endif
  printf("Number of iterations: %d \n", niter);
  if (pHeader) 
    printf("Summary Standard Ouput with Header\n");
  else
    printf("Summary Standard Output without Header\n");
  
  if (vOut)
    printf( "Verbose Output \n");
 
#ifdef USE_MPI 
  if ( kernel_key == CCO_KERNEL )
    printf("Using computation-communication overlap \n");
#endif
}

void check_norm(const struct grid_info_t *g, int iter, double norm){
/* test ration of to consecutive norm agains the smoother eigenvalue for the chosen mode */

  double ln, gn, r, eig;
  static Real prev_gn;

  if (iter == 0){
    prev_gn = 1.0;
    if (g->myrank == ROOT){
      printf("Correctness check\n");
      printf("iteration, norm ratio, deviation from eigenvalue\n");
    }
  }
  ln = norm;
#ifdef USE_MPI
  MPI_Reduce(&ln, &gn, 1, MPI_DOUBLE, MPI_SUM, ROOT, MPI_COMM_WORLD);
#else
  gn = ln;
#endif
  if ( g->myrank == ROOT){
    r = sqrt(gn/prev_gn);
    eig = (cos(PI*kx/(g->ng[0]-1)) + cos(PI*ky/(g->ng[1]-1)) + cos(PI*kz/(g->ng[2]-1)))/3.0;
    if( iter > 0) 
      printf("%5d    %12.5e     %12.5e\n", iter, r, r - eig);
    prev_gn = gn;
  }
} 


double local_norm(const struct grid_info_t *g){
 //compute the L2 norm (squared)
  int i, j, k;
  int NX = g->nlx, NY = g->nly, NZ = g->nlz;
  double norm = 0.0;

  for (k=1; k<NZ-1;++k)
    for (j=1; j<NY-1;++j)
      for (i=1; i<NX-1;++i)
	norm += uOld[i+j*NX+k*NX*NY] * uOld[i+j*NX+k*NX*NY];

  return(norm);

}


void timeUpdate(double *times){

  /* Update Root's times matrix to include all times */

#ifdef MPI
  if ( myrank == ROOT )
    call MPI_Gather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, times, NITER, MPI_DOULBE, ROOT, MPI_COMM_WORLD);
  else
    call MPI_Gather(times, niter, MPI_DOUBLE, MPI_BOTTOM, 0, MPI_DATATYPE_NULL, ROOT, MPI_COMM_WORLD);
#endif
}
 

void statistics(const struct grid_info_t *g, double *times,  
                double *minTime, double *meanTime, double *maxTime,
                double *stdvTime, double *NstdvTime){

  int iPE, iter, shift, ii;
  int nproc = g->np[0] * g->np[1] * g->np[2];

  /* eliminate possible startup baias if niter is large enough */
  shift = ( niter > 20 ? 5 : 0);
    

  /* Compute mean, max, min of times */
  ii = 0;
    for (iPE = 0; iPE < nproc; iPE++){
      for (iter = shift; iter < niter; iter++){
	*meanTime += times[ii];
	*maxTime = MAX(*maxTime, times[ii]);
	*minTime = MIN(*minTime, times[ii]);
	++ii;
      }
      ii += shift;
    }
    *meanTime = *meanTime / (double) (niter - shift) / (double) nproc;

  /* Compute standard deviation of times */
    ii = 0;
    for (iPE = 0; iPE < nproc; iPE++){
      for (iter = shift; iter < niter; iter++){
	*stdvTime += (times[ii] - *meanTime) *
	  (times[ii] - *meanTime);
	++ii;
      }
      ii += shift;
    }
  *stdvTime = sqrt(*stdvTime / ((niter - shift)* nproc-1.0));

  /* Normalized standard deviation (stdv / mean) */
  *NstdvTime = *stdvTime / *meanTime;
}


void stdoutIO( const struct grid_info_t *g, const int kernel_key, const double *times,  
              double minTime, double meanTime, double maxTime, 
	       double NstdvTime, double norm){
  int iter, ii, i;

  if (pHeader){
    printf("# Last norm %22.15e\n",sqrt(norm));
#ifdef USE_MPI
    printf("#==================================================================================================================================#\n");
    printf("#\tNPx\tNPy\tNPz\tNThs\tNx\tNy\tNz\tNITER\tmeanTime \tmaxTime  \tminTime  \tNstdvTime  #\n");
    printf("#==================================================================================================================================#\n");
#else
    if ( kernel_key == BLOCKED_KERNEL){
      printf("#==================================================================================================================================#\n");
    printf("#\tNThs\tNx\tNy\tNz\tBx\tBy\tBz\tNITER\tmeanTime \tmaxTime  \tminTime  \tNstdvTime  #\n");
    printf("#==================================================================================================================================#\n");
    }
    else {
    printf("#==========================================================================================================#\n");
    printf("#\tNThs\tNx\tNy\tNz\tNITER\tmeanTime \tmaxTime  \tminTime  \tNstdvTime  #\n");
    printf("#==========================================================================================================#\n");
    }
#endif 
  }
#ifdef USE_MPI
  printf("\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%9.3e\t%9.3e\t%9.3e\t%9.3e\n",
         g->np[0], g->np[1], g->np[2], nthreads, g->ng[0], g->ng[1], g->ng[2],niter, meanTime, maxTime, minTime, NstdvTime);
#else
  if ( kernel_key == BLOCKED_KERNEL)
     printf("\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%9.3e\t%9.3e\t%9.3e\t%9.3e\n",
	     nthreads, g->ng[0], g->ng[1], g->ng[2], g->nb[0], g->nb[1], g->nb[2], niter, meanTime, maxTime, minTime, NstdvTime);
    else
      printf("\t%d\t%d\t%d\t%d\t%d\t%9.3e\t%9.3e\t%9.3e\t%9.3e\n",
	     nthreads, g->ng[0], g->ng[1], g->ng[2], niter, meanTime, maxTime, minTime, NstdvTime);
#endif
  
  /* Only if "Verbose Output" asked for */
  if (vOut){ 
    printf("\n"); printf("\n");
    printf("# Full Time Output (rows are times, cols are tasks)\n");
    ii = 0;
    int nproc = g->np[0] * g->np[1] * g->np[2];
    for (iter = 0; iter < niter; iter++){ 
      for (i = 0; i < nproc; i++){
        printf("%e \t",times[ii]);
	++ii;
      }
      printf("\n");
    }
  }
}

double my_wtime(){
  
#ifdef USE_MPI
  return (MPI_Wtime());
#else
#ifdef _OPENMP
  return (omp_get_wtime());
#else
  // needs a C standard timer
  printf("my_wtime not defined !!!\n");
  return(0.0);
#endif
#endif
}

