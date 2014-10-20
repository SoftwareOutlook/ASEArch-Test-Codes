/*
  This is part of JTC, a CUDA-OpenMP-MPI Benchmark for
  Jacobi solver applied to a 3D Laplace equation.

  Lucian Anton 
  March 2014.

  This file originates from v 1.0 of HOMB
  http://sourceforge.net/projects/homb/

  The original copyright and licence is below.

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

#include "jacobi_c.h"
#include "utils_c.h"
#include "kernels_c.h"
#include "comm_mpi_c.h"
#ifdef USE_GPU
#include "cutil_inline.h"
#endif

static void print_help(const struct grid_info_t *g, const char *s);

// Number of runs and iterations/run

int nruns, niter;

// OpenMP threads number
int nthreads;

// solution array
Real  *udata, *uOld, *uNew;

//GPU arrays
Real *d_u1, *d_u2, *d_foo;

// eigenvalue k vector (mode)
static Real kx=1.0, ky=1.0, kz=1.0;

// Output control
int pContext, testComputation; 
static int vOut, pHeader;


void initContext(int argc, char *argv[], struct grid_info_t * grid, int *kernel_key){
 
#ifdef USE_MPI
  MPI_Comm_rank(MPI_COMM_WORLD, &(grid->myrank)); 
#else 
  grid->myrank = 0;
#endif

  /* "Logicals" for file output */
  vOut = 0;  pHeader = 1; pContext = 0; testComputation = 0;
  
  /* Defaults */
  grid->ng[0] = 33; grid->ng[1] = 33; grid->ng[2] = 33; //Global grid
  grid->malign = -1;
  grid->nwaves = -1; 
  grid->gpuflag = 0;

  nruns = 5; niter = 1; nthreads = 1; grid->nproc = 1;
  grid->np[0] = 1; grid->np[1] = 1; grid->np[2] = 1;
  grid->cp[0] = 0; grid->cp[1] = 0; grid->cp[2] = 0;
  // if compiled with GPU flag use GPU kernel as default
  *kernel_key = BASELINE_KERNEL;

  int i;
  int have_blocks=0;

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
      have_blocks=1;
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
    /* Look for number number of iteration block (runs)*/  
    else if ( strcmp("-nruns", argv[i]) == 0 ){
      sscanf(argv[++i],"%d",&nruns);
    }
    /* Look for the size of iteration block */
    else if ( strcmp("-niter", argv[i]) == 0 ){
      int iaux;
      sscanf(argv[++i],"%d",&niter);
      // number of waves takes precedence over blk_iter
      //if ( ! niter_fixed )
      //niter = iaux;
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
    /* allocate aligned memory to help vectorization */
    else if (strcmp("-malign",argv[i]) == 0){
      sscanf(argv[++i],"%d", &(grid->malign));
    }
    /* Look for kernel to use */
    else if ( strcmp("-model", argv[i]) == 0 ){
      ++i;
      if (strcmp("baseline",argv[i]) == 0)
	*kernel_key = grid->key = BASELINE_KERNEL;
      else if ( strcmp("baseline-opt",argv[i]) == 0)
	*kernel_key = OPTBASE_KERNEL;
      else if (strcmp("blocked",argv[i]) == 0)
	*kernel_key = grid->key = BLOCKED_KERNEL;
      else if (strcmp("cco",argv[i]) == 0)
	*kernel_key = grid->key = CCO_KERNEL;
      else if (strcmp("wave",argv[i]) == 0 ){
	sscanf(argv[++i],"%d",&(grid->nwaves));
	//niter_fixed = 1;// prevent reseting by -niter flag
	sscanf(argv[++i],"%d",&(grid->threads_per_column));
	if ( grid->threads_per_column == 0 ) 
	  *kernel_key = grid->key = WAVE_DIAGONAL_KERNEL;
	else if ( grid->threads_per_column > 0)
	  *kernel_key = grid->key = WAVE_KERNEL;
	else
	  error_abort("wrong value for threads per column parameter", argv[i]);
      }
      //OpenACC switch - Mark Mawson 20/10/2014
      else if (strcmp("openacc",argv[i]) == 0){
#ifdef OPENACC
	*kernel_key = grid->key = OPENACC_KERNEL;

#else
	error_abort("GPU model specified without gpu compilation", "");
#endif
      }
      else if (strcmp("gpu-2d-blockgrid",argv[i]) == 0){
#ifdef USE_GPU
	*kernel_key = grid->key = GPU_BASE_KERNEL;
	grid->gpuflag = 1;
#else
	error_abort("GPU model specified without gpu compilation", "");
#endif
      }
      else if (strcmp("gpu-shm",argv[i]) == 0){
#ifdef USE_GPU
	*kernel_key = grid->key = GPU_SHM_KERNEL;
	grid->gpuflag = 1;
#else
	error_abort("GPU model specified without gpu compilation", "");
#endif
      }
      else if (strcmp("gpu-bandwidth",argv[i]) == 0){
#ifdef USE_GPU
	*kernel_key = grid->key = GPU_BANDWIDTH_KERNEL;
	grid->gpuflag = 1;
#else
	error_abort("GPU model specified without gpu compilation", "");
#endif
      }
      else if (strcmp("gpu-3d-blockgrid",argv[i]) == 0){
#ifdef USE_GPU
	*kernel_key = grid->key = GPU_MM_KERNEL;
	grid->gpuflag = 1;
#else
	error_abort("GPU model specified without gpu compilation", "");
#endif
      }

      /*
	else if (strcmp("blockedgpu",argv[i]) == 0){
	#ifdef USE_GPU
	*kernel_key = grid->key = BLOCKEDGPU_KERNEL;
	grid->gpuflag = 1;
	#else
	error_abort("GPU model specified without gpu compilation", "");
	#endif
	}
      */
      else if (strcmp("help",argv[i]) == 0) 
	print_help(grid, "model");
      else
	error_abort("Wrong model specifier, try -model help\n",argv[i]);
    }
    /* Look for eigenvalue test */
    else if ( strcmp("-t",argv[i] ) == 0){
      testComputation = 1;
      //printf(" i %d \n",i); 
    }
    else if ( strcmp("-version",argv[i] ) == 0){
      print_help(grid,"version");
      //printf(" i %d \n",i); 
    }
    /* Look for "verbose" standard out */
    else if ( strcmp("-help",argv[i]) == 0 )
      print_help(grid, "usage");
    else{
      /* basic test on flag values */
      //if (argv[i][0] != '-' )
      error_abort("Wrong command line argument, try -help\n", argv[i]);
      
    }
  }
  if (!have_blocks){
    // default blocks sizes are set to grid sizes if
    // not specified in command line
    // for GPU use a standard 32x4 block
    if (grid->gpuflag){
      grid->nb[0] = 32;
      grid->nb[1]= 4;
      grid->nb[2]= grid->ng[2];
    } else{
      grid->nb[0] = grid->ng[0]; grid->nb[1] = grid->ng[1]; grid->nb[2] = grid->ng[2];
    }
  }
}

void setPEsParams(struct grid_info_t *g, int kernel_key) {

 
#ifdef USE_MPI
  MPI_Comm_size(MPI_COMM_WORLD, &(g->nproc));
  if ( g->nproc != g->np[0] * g->np[1] * g->np[2]){
    fprintf(stderr, "MPI topology sizes %d %d %d does not fit with nproc %d \n", g->np[0], g->np[1], g->np[2], g->nproc);
    MPI_Abort(MPI_COMM_WORLD, -1);
  }
#else
  g->nproc = 1;
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

  /* sanity checkes for the time skewed algorithm */
  if (kernel_key == WAVE_KERNEL){ 
    char errmsg[255];
    if ( (nthreads % g->threads_per_column != 0) || 
	 (g->threads_per_column > nthreads)){     
      sprintf(errmsg,"nthreads must be a multiple of threads per column in wave model. nthreads : %d threads per column : %d \n", nthreads, g->threads_per_column);
      error_abort(errmsg,"");
    }
    if (g->threads_per_column > g->nwaves){
      sprintf(errmsg,"threads_per_column %d > number of waves %d !!! quitting ... \n",  g->threads_per_column,  g->nwaves);
      error_abort(errmsg,"");
    }
    if (niter%g->nwaves != 0){
      sprintf(errmsg,"number of iterations per run %d is not a multiple of number of waves %d !!! quitting ... \n", niter, g->nwaves);
      error_abort(errmsg,"");
    } 

  }
  //  #pragma omp parallel shared(nThreads)
  //{
  //  #pragma omp single
  //     nThreads = omp_get_num_threads();
  //}
}


void initialise_grid( const struct grid_info_t *g) {
 
  int i, j, k, ijk, r, n;
  size_t s;

  if (g->malign > 0){
    r = (g->nlx * sizeof(Real)) % g->malign;
    if ( r > 0 )
      s = g->nlx * sizeof(Real) - r + g->malign;
    else
      s = g->nlx * sizeof(Real);
    n = 2 * s / sizeof(Real) * g->nly * g->nlz;
    s = 2 * s * g->nly * g->nlz;
    //printf("maling %d \n", n);
    if (posix_memalign( (void **)&udata, (size_t) g->malign, s) != 0)
      error_abort("error posix_memalign","");  
  }   
  else{
    n = 2 * g->nlx * g->nly * g->nlz;
    s =  (size_t)n * sizeof(Real);
    udata = malloc(s);
    //printf("no maling %d \n", n*sizeof(Real));
  }

  uOld = &udata[n/2]; uNew = &udata[0];
  // use openmp region to initialise to enforce first touch policy for numa nodes; needs further refinement
#pragma omp parallel  for schedule(static) default (shared) private(i,j,k, ijk)
  for (k = g->sz - 1; k <= g->ez + 1; ++k)
    for (j = g->sy - 1; j <= g->ey + 1; ++j)
      for (i = g->sx - 1; i <= g->ex + 1; ++i){
	ijk = uindex(g,i,j,k);
	uOld[ijk] = sin((PI * i * kx) / (g->ng[0] - 1)) * sin((PI * j * ky) / (g->ng[1] - 1)) * sin((PI * k * kz) / (g->ng[2] - 1));
	uNew[ijk]=0.0;
      }
#ifdef USE_GPU
  /**
   * check if GPU model is requested in command arguments
   * if its requested then relevant GPU information is initialized by invoking
   * initialiseGPUData function
   */
    
  if(g->gpuflag==1)
    {
      initialiseGPUData(g->ng[0],g->ng[1],g->ng[2]);
    }
#endif
}


void printContext(const struct grid_info_t *g, int kernel_key){
  
  char kernel_name[20];
  switch (kernel_key)
    {
    case(BASELINE_KERNEL) :
      sprintf(kernel_name, "Gold baseline"); break;
    case(OPTBASE_KERNEL) :
      sprintf(kernel_name, "Titanium baseline"); break;
    case(BLOCKED_KERNEL) :
      sprintf(kernel_name, "Blocked"); break;
    case (CCO_KERNEL)  :
      sprintf(kernel_name, "MPI CCO"); break;
    case (WAVE_KERNEL) :
      sprintf(kernel_name, "Wave"); break;
    case (WAVE_DIAGONAL_KERNEL) :
      sprintf(kernel_name, "Wave diagonal"); break;  
    case(GPU_BASE_KERNEL) :
      sprintf(kernel_name, "GPU 2D block grid"); break;
    case(GPU_MM_KERNEL) :
      sprintf(kernel_name, "GPU 3D block grid"); break;
    case(GPU_SHM_KERNEL) :
      sprintf(kernel_name, "shared memory GPU"); break;
    case(GPU_BANDWIDTH_KERNEL) :
      sprintf(kernel_name, "GPU bandwidth"); break;
 case(OPENACC_KERNEL) :
      sprintf(kernel_name, "OpenACC"); break;
      //case(BASEGPU_SHM_KERNEL) :
      //sprintf(kernel_name, "Titanium SharedMem"); break;
      //case(BLOCKEDGPU_KERNEL) :
      //sprintf(kernel_name, "Blocked GPU"); break;
    }

  printf("\n This Jacobi Test Code v%s \n\n", JTC_VERSION);
  printf("Using %s kernel \n",kernel_name);
  printf("Global grid sizes   : %d %d %d \n", g->ng[0], g->ng[1], g->ng[2]);
  if ( (kernel_key == BLOCKED_KERNEL) && (kernel_key == WAVE_KERNEL) ) 
    printf("Computational block : %d %d %d \n", g->nb[0], g->nb[1], g->nb[2]);
  if  (kernel_key == WAVE_KERNEL)
    printf("Wave parallelism with %d threads per column \n", g->threads_per_column); 
#ifdef USE_VEC1D
  printf("\nUsing vector wraper for inner loop\n\n");  
#endif

#ifdef USE_MPI
  printf("MPI topology        : %d %d %d \n", g->np[0], g->np[1], g->np[2]);
#endif
  if ( niter <= 0 ) 
    error_abort("Non-positive value for iterations/run","");
  
  printf("Collecting over %d runs, %d iterations/run \n", nruns, niter);
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

#ifdef USE_GPU
  if(g->gpuflag==1) {
    int dev;
    struct cudaDeviceProp devProp;
    cudaSafeCall(cudaGetDevice(&dev));
    cudaSafeCall(cudaGetDeviceProperties(&devProp,dev));
    printf("\n Using CUDA device %d    : %s\n", dev, devProp.name);
    printf(" Compute capability     : %d%d\n", devProp.major, devProp.minor);
    printf(" Memory Clock Rate (KHz): %d\n", devProp.memoryClockRate);
    printf(" Memory Bus Width (bits): %d\n\n", devProp.memoryBusWidth);
  }
#endif

}


void check_norm(const struct grid_info_t *g, int irun, double norm){
  /* test ration of to consecutive norm agains the smoother eigenvalue for the chosen mode */



  double ln, gn, r, eig;
  static Real prev_gn;
  static int firsttime = 1;

  if (firsttime){
    firsttime = 0;
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
    if( irun > 0) 
      printf("%5d    %12.5e     %12.5e\n", irun, r, r - pow(eig, niter));
    prev_gn = gn;
  }
} 


double local_norm(const struct grid_info_t *g){
  //compute the L2 norm (squared)
  int i, j, k;
  int nxShift, NX = g->nlx, NY = g->nly, NZ = g->nlz;
  double norm = 0.0;
 
  if ( g->malign < 0) 
    nxShift = NX;
  else
    nxShift = (abs((int) (uNew - uOld))) / (NY*NZ); 

  for (k=1; k<NZ-1;++k)
    for (j=1; j<NY-1;++j)
      for (i=1; i<NX-1;++i)
	norm += uOld[i+j*nxShift+k*nxShift*NY] * uOld[i+j*nxShift+k*nxShift*NY];
  
  return(norm);

}


void timeUpdate(struct times_t *times){

  /* Update Root's times matrix to include all times */

#ifdef MPI
  if ( myrank == ROOT )
    call MPI_Gather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, times, NITER, MPI_DOULBE, ROOT, MPI_COMM_WORLD);
  else
    call MPI_Gather(times, niter, MPI_DOUBLE, MPI_BOTTOM, 0, MPI_DATATYPE_NULL, ROOT, MPI_COMM_WORLD);
#endif
}
 

void statistics(const struct grid_info_t *g, const struct times_t *times,  
                struct times_t *minTime, struct times_t *meanTime, struct times_t *maxTime){


  int iPE, irun, ii;
  int nproc = g->np[0] * g->np[1] * g->np[2];

  //for (ii =0; ii< niter; ++ii)
  //  printf("stats times %g \n",times[ii].comp);

  meanTime->comp = meanTime->comm = 0.0;
  maxTime->comp = maxTime->comm = -1.0;
  minTime->comp = minTime->comm = 1.e30;

  /* Compute mean, max, min of times */
  ii = 0;
  for (iPE = 0; iPE < nproc; iPE++){
    for (irun = 0; irun < nruns; irun++){
      meanTime->comp += times[ii].comp;
      maxTime->comp = MAX(maxTime->comp, times[ii].comp);
      minTime->comp = MIN(minTime->comp, times[ii].comp);
#if defined USE_GPU || defined OPENACC
      meanTime->comm += times[ii].comm;
      maxTime->comm = MAX(maxTime->comm, times[ii].comm);
      minTime->comm = MIN(minTime->comm, times[ii].comm);
#endif
      ++ii;
    }
  }
  meanTime->comp = meanTime->comp / (double) nruns / (double) nproc;
#if defined USE_GPU || defined OPENACC
  meanTime->comm = meanTime->comm / (double) nruns / (double) nproc;
#endif

    
  /* Compute standard deviation of times */
  /*
    ii = 0;
    for (iPE = 0; iPE < nproc; iPE++){

    for (iter = shift; iter < ntimes; iter++){

    *stdvTime += (times[ii] - *meanTime) *
    (times[ii] - *meanTime);
    ++ii;
    }
    ii += shift;
    }
    *stdvTime = sqrt(*stdvTime / ((ntimes - shift)* nproc-1.0));
    */
  /* Normalized standard deviation (stdv / mean) */
  //*NstdvTime = *stdvTime / *meanTime;
  /* normalise averages to 1 iteration step */
  meanTime->comp /= niter;
  maxTime->comp /= niter;
  minTime->comp /= niter;
}


void stdoutIO( const struct grid_info_t *g, const int kernel_key, const struct times_t *times,  
               const struct times_t *minTime,  const struct times_t *meanTime,  const struct times_t *maxTime, 
	       double norm){

  int gpu_header = (kernel_key == GPU_BASE_KERNEL) || (kernel_key == GPU_SHM_KERNEL) || (kernel_key == GPU_BANDWIDTH_KERNEL) || (kernel_key == GPU_MM_KERNEL) || (kernel_key == OPENACC_KERNEL);

  if (pHeader){
    printf("# Last norm %22.15e\n",sqrt(norm));
#ifdef USE_MPI
    printf("#==================================================================================================================================#\n");
    printf("#\tNPx\tNPy\tNPz\tNThs\tNx\tNy\tNz\tNITER \tminTime \tmeanTime \tmaxTime    #\n");
    printf("#==================================================================================================================================#\n");
#else
    if ( (kernel_key == BLOCKED_KERNEL) || (kernel_key == WAVE_KERNEL)){

      printf("#==================================================================================================================================#\n");
      printf("#\tNThs\tNx\tNy\tNz\tBx\tBy\tBz\tNITER\tminTime\tmeanTime \tmaxTime    #\n");
      printf("#==================================================================================================================================#\n");
    }
    else if ( gpu_header)
      {
	printf("#==================================================================================================================================================================#\n");
	printf("#\tNThs\tNx\tNy\tNz\tBx\tBy\tBz\tNITER\tminTime\t         meanTime\tmaxTime \tminCpyTime\tmeanCpyTime \tmaxCpyTime #\n");
	printf("#==================================================================================================================================================================#\n");
      }
    else {
      printf("#==========================================================================================================#\n");
      printf("#\tNThs\tNx\tNy\tNz\tNITER\tminTime    \tmeanTime \tmaxTime   \n ");
      printf("#==========================================================================================================#\n");
    }
#endif 
  }
#ifdef USE_MPI
  printf("\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%9.3e\t%9.3e\t%9.3e\n",
         g->np[0], g->np[1], g->np[2], nthreads, g->ng[0], g->ng[1], g->ng[2],blk_iter,
	 minTime->comp, meanTime->comp, maxTime->cop);
#else
  if ( (kernel_key == BLOCKED_KERNEL) || (kernel_key == WAVE_KERNEL))
    printf("\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%9.3e\t%9.3e\t%9.3e\n",
	   nthreads, g->ng[0], g->ng[1], g->ng[2], g->nb[0], g->nb[1], g->nb[2],
	   niter, minTime->comp, meanTime->comp, maxTime->comp);
  else if (gpu_header)
    printf("\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%9.3e\t%9.3e\t%9.3e\t%9.3e\t%9.3e\t%9.3e\n",
	   nthreads, g->ng[0], g->ng[1], g->ng[2], g->nb[0], g->nb[1], g->nb[2], niter, 
	   minTime->comp, meanTime->comp, maxTime->comp,minTime->comm, meanTime->comm,
	   maxTime->comm);
  else
    printf("\t%d\t%d\t%d\t%d\t%d\t%9.3e\t%9.3e\t%9.3e\n",
	   nthreads, g->ng[0], g->ng[1], g->ng[2], niter, minTime->comp,
	   meanTime->comp, maxTime->comp);

#endif
  /* Only if "Verbose Output" asked for */
  /*
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
  */
}

#ifndef USE_MPI
#ifndef _OPENMP
#include <sys/time.h>
#endif
#endif

double my_wtime(){
  
#ifdef USE_MPI
  return (MPI_Wtime());
#else
#ifdef _OPENMP
  return (omp_get_wtime());
#else
  // needs a C standard timer
  //printf("my_wtime not defined !!!\n");
  struct timeval t;
  if ( gettimeofday(&t, NULL) < 0){
    perror("get-time-of-day error");
    return(-1.0);
  } else
    return( (double) t.tv_sec+((double) t.tv_usec)*1.e-6);
#endif
#endif
}

void error_abort( const char *s1, const char *s2){
  fprintf(stderr,"\n ERROR: %s %s \n", s1, s2); 
#ifdef USE_MPI
  MPI_Abort(MPI_COMM_WORLD,-1);
#else
  exit(-1);
#endif
}

static void print_help( const struct grid_info_t *g, const char *s){

  if ( g->myrank == 0) {
    if ( strcmp(s, "usage") == 0 )
      printf("Usage: [-ng <grid-size-x> <grid-size-y> <grid-size-z> ] \
[ -nb <block-size-x> <block-size-y> <block-size-z>] \
 [-model <model_name> [num-waves] [threads-per-column]] \
[-niter <num-iterations>]  [-biter <iterations-block-size>] \
[-malign <memory-alignment> ] [-v] [-t] [-pc] [-nh] [-help] [-version] \n");
      
    else if (strcmp(s, "model") == 0)
      printf("possible values for model parameter: \n \
        baseline \n \
        baseline-opt\n \
        blocked\n \
        wave num-waves threads-per-column \n \
        gpu-2d-blockgrid\n \
        gpu-3d-blockgrid\n\n \
        gpu-bandwidth\n\n \
        Note for wave model: if threads-per-column == 0 diagonal wave kernel is used.\n");  
    else if (strcmp(s, "version") == 0)
      printf("%s \n",JTC_VERSION);
    else
      printf(" print_help: unknown help request");
  }
  
#ifdef USE_MPI
  MPI_Finalize();
#endif
  exit(EXIT_SUCCESS);    
  
}



/////////////////////////////////////////////////////////////////
/**
 * initialise all the GPU computation array which includes
 * initialising memories on CPU and GPU
 * further populate the GPU array with CPU array and eventually
 * copy the populated GPU array to GPU
 */
void initialiseGPUData(int NX,int NY,int NZ)
{
#ifdef USE_GPU
  cutilDeviceInit();
  cudaSafeCall(cudaMalloc((void **)&d_u1, sizeof(Real)*NX*NY*NZ));
  cudaSafeCall(cudaMalloc((void **)&d_u2, sizeof(Real)*NX*NY*NZ));
#endif
}

void freeDeviceMemory(){
#ifdef USE_GPU
  cudaSafeCall(cudaFree(d_u1));
  cudaSafeCall(cudaFree(d_u2));
#endif
}

