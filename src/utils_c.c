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
#ifdef USE_CUDA
#include "cutil_inline.h"
#endif

// constants needed to read the input options
#define LANGUAGE 777
#define ALGORITHM 999

// constants for help printing
#define H_USAGE 240
#define H_LANG 241
#define H_ALG 242
#define H_VERSION 243 

static void print_help(const struct grid_info_t *g, const int);
static void set_lang_ang_alg(struct grid_info_t *g, const char *optLang, const char *optAlg );
static void set_alg_omp(struct grid_info_t *g, const char * optAlg);
static void set_alg_cuda(struct grid_info_t *g, const char * optAlg);
static void set_g(struct grid_info_t *g, const int key, const int val, 
		   const char * name);

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
 
  nruns = 5; niter = 1; nthreads = 1; grid->nproc = 1;
  grid->np[0] = 1; grid->np[1] = 1; grid->np[2] = 1;
  grid->cp[0] = 0; grid->cp[1] = 0; grid->cp[2] = 0;
 

  int i;
  int have_blocks=0;
  char *optLang = NULL;
  char *optAlg = NULL;

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
    /* Look for number number of runs */  
    else if ( strcmp("-nruns", argv[i]) == 0 ){
      sscanf(argv[++i],"%d",&nruns);
    }
    /* Look for the run length */
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
    /* Look for language flag */
    else if ( strcmp("-lang", argv[i]) == 0 ){
      ++i;
      optLang = argv[i];
    }
    /* Look for algorithm to use */
    else if ( strcmp("-alg", argv[i]) == 0 ){
      ++i;
      optAlg = argv[i];
    }
    /* Look for eigenvalue test */
    else if ( strcmp("-t",argv[i] ) == 0){
      testComputation = 1;
      //printf(" i %d \n",i); 
    }
    else if ( strcmp("-version",argv[i] ) == 0){
      print_help(grid, H_VERSION);
      //printf(" i %d \n",i); 
    }
    /* Look for "verbose" standard out */
    else if ( strcmp("-help",argv[i]) == 0 )
      print_help(grid, H_USAGE);
    else{
      /* basic test on flag values */
      //if (argv[i][0] != '-' )
      error_abort("Wrong command line argument, try -help\n", argv[i]);
      
    }
  }
   
  set_lang_ang_alg(grid, optLang, optAlg);
 
  if (!have_blocks){
    // default blocks sizes are set to grid sizes if
    // not specified in command line
    // for GPU use a standard 32x4 block
    if (grid->lang_key == LANG_CUDA){
      grid->nb[0] = 32;
      grid->nb[1]= 4;
      grid->nb[2]= grid->ng[2];
    } else{ // default for OpenMP
      grid->nb[0] = grid->ng[0]; grid->nb[1] = grid->ng[1]; grid->nb[2] = grid->ng[2];
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
  if (g->alg_key == ALG_WAVE){ 
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
#ifdef USE_CUDA
  /**
   * check if CUDA algorithm is requested in command arguments
   * if its requested then relevant GPU information is initialized by invoking
   * initialiseGPUData function
   */
    
  if(g->lang_key == LANG_CUDA)
    {
      initialiseGPUData(g->ng[0],g->ng[1],g->ng[2]);
    }
#endif
}

//! print detailed info on runs parameters
void printContext(const struct grid_info_t *g){

  printf("\n This Jacobi Test Code v%s \n\n", JTC_VERSION);
  printf("Compiled with support for %s\n",g->lang_name);
  printf("Using algorithm %s\n", g->alg_name);
  printf("Global grid sizes   : %d %d %d \n", g->ng[0], g->ng[1], g->ng[2]);
  if ( (g->alg_key == ALG_BLOCKED) && (g->alg_key == ALG_WAVE) ) 
    printf("Grid block : %d %d %d \n", g->nb[0], g->nb[1], g->nb[2]);
  if  (g->alg_key == ALG_WAVE)
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
  if ( kernel_key == ALG_CCO )
    printf("Using MPI with computation-communication overlap \n");
#endif

#ifdef USE_CUDA
  switch(g->alg_key)
    {
      int dev;
      struct cudaDeviceProp devProp;
    case ALG_CUDA_3D_BLK :
    case ALG_CUDA_2D_BLK :
    case ALG_CUDA_SHM :
    case ALG_CUDA_BANDWIDTH :
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
#if defined USE_CUDA || USE_OPENCL
      meanTime->comm += times[ii].comm;
      maxTime->comm = MAX(maxTime->comm, times[ii].comm);
      minTime->comm = MIN(minTime->comm, times[ii].comm);
#endif
      ++ii;
    }
  }
  meanTime->comp = meanTime->comp / (double) nruns / (double) nproc;
#if defined USE_CUDA || defined USE_OPENCL
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


void stdoutIO( const struct grid_info_t *g, const struct times_t *times,  
               const struct times_t *minTime,  const struct times_t *meanTime,  
	       const struct times_t *maxTime, const double norm){

  int gpu_header = (g->alg_key == ALG_CUDA_2D_BLK) || (g->alg_key == ALG_CUDA_3D_BLK) || (g->alg_key == ALG_CUDA_SHM || (g->alg_key == ALG_CUDA_BANDWIDTH) || (g->alg_key == ALG_OPENCL_BASELINE) || (g->alg_key == ALG_OPENACC_BASELINE));

  if (pHeader){
    printf("# Last norm %22.15e\n",sqrt(norm));
#ifdef USE_MPI
    printf("#==================================================================================================================================#\n");
    printf("#\tNPx\tNPy\tNPz\tNThs\tNx\tNy\tNz\tNITER \tminTime \tmeanTime \tmaxTime    #\n");
    printf("#==================================================================================================================================#\n");
#else
    if ( (g->alg_key == ALG_BLOCKED) || (g->alg_key == ALG_WAVE)){

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
  if ( (g->alg_key == ALG_BLOCKED) || (g->alg_key == ALG_WAVE))
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

static void print_help( const struct grid_info_t *g, const int key){

  if ( g->myrank == 0) {
    switch (key)
      {
      case (H_LANG):
	printf("available languages:\n\
        openmp\n\
        cuda\n\
        opencl\n\
        openacc\n\n\
        Note:\n\
        A language needs to be acitvated at build time with preprocessor flags\n");
	break;
      case(H_ALG):
	printf("available algorithms: \n\
        baseline \n\
        baseline-opt (only OpenMP)\n\
        blocked (only OpenMP, CUDA) \n\
        wave num-waves threads-per-column (only OpenMP)\n\
        2d-blockgrid (only CUDA)\n\
        3d-blockgrid (only CUDA)\n\
        bandwidth (only CUDA)\n\n\
        Notes:\n\
        1) for wave algorithm, if threads-per-column == 0 diagonal wave kernel is used,\n\
        2) correctness test (i.e. -t flag) is irrelevant for bandwidth algorithm.\n");  
      break;
      case (H_VERSION):
	printf("%s \n",JTC_VERSION);
	break;
      case(H_USAGE):
	printf("Usage: [-ng <grid-size-x> <grid-size-y> <grid-size-z> ] \
[ -nb <block-size-x> <block-size-y> <block-size-z>] \
[-lang <language>] [-alg <algorithm> [num-waves] [threads-per-column]] \
[-nruns <number-of-runs>] [-niter <num-iterations-per-run>]  ] \
[-malign <memory-alignment> ] [-v] [-t] [-pc] [-nh] [-help] [-version] \n");
	break;
      default:
	printf("\nutils.c: print_help: unknown help request\n\n");
      }
  }
  
#ifdef USE_MPI
  MPI_Finalize();
#endif
  exit(EXIT_SUCCESS);    
  
}

static void set_lang_ang_alg(struct grid_info_t *g, const char *optLang, const char *optAlg ){
   

  //check if help is needed
  if ( optAlg != NULL) 
    if (strcmp("help",optAlg) == 0) 
      print_help(g, H_ALG);

  if(optLang != NULL){
    
    //check if help is needed
    if (strcmp("help",optLang) == 0) 
      print_help(g, H_LANG);

    if (strcmp("openmp",optLang) == 0){
      set_g(g, LANGUAGE, LANG_OMP, "OpenMP");
      set_alg_omp(g, optAlg);
    }
    else if (strcmp("cuda",optLang) == 0){
      set_g(g, LANGUAGE, LANG_OMP, "CUDA");
      set_alg_cuda(g, optAlg); 
    }
    else if (strcmp("opencl", optLang) == 0){
      set_g(g, LANGUAGE, LANG_OPENCL, "OpenCL");
      // need to write a set function if the number of algorithms increases for opencl anf openacc
      if ( (optAlg == NULL) || (strcmp("baseline",optAlg) == 0))
	set_g(g, ALGORITHM, ALG_OPENCL_BASELINE, "baseline");
      else
	error_abort(" set_lang_ang_alg: Wrong algorithm specifier for opencl language, try -alg help\n",optAlg);
    }
    else if (strcmp("openacc", optLang) == 0){
      set_g(g, LANGUAGE, LANG_OPENACC, "OpenACC");
      if ( (optAlg == NULL) || (strcmp("baseline",optAlg) == 0))
	set_g(g, ALGORITHM, ALG_OPENACC_BASELINE, "baseline");
      else
	error_abort(" set_lang_ang_alg: Wrong algorithm specifier for openacc language, try -alg help\n",optAlg);
    }
  }
  else {
    /* set default language and algorithm */
#if defined(USE_CUDA)
    set_g(g, LANGUAGE, LANG_CUDA, "CUDA");
    set_alg_cuda(g, optAlg);
#elif defined(USE_OPENCL)
    set_g(g, LANGUAGE, LANG_OPENCL, "OpenCL");
    set_alg_omp(g, optAlg);
#elif defined(USE_OPENACC)
    set_g(g, LANGUAGE, LANG_OPENACC, "OpenACC");
    set_alg_omp(g, optAlg);
#else
    set_g(g, LANGUAGE, LANG_OMP, "OpenMP");
    set_alg_omp(g, optAlg);
#endif
  }
 
// Sanity checks
#ifndef USE_CUDA
  if ( optLang == LANG_CUDA) error_abort("CUDA language selected without USE_CUDA preprocessor","");
#endif

#ifndef USE_OPENCL
  if ( optLang == LANG_OPENCL) error_abort("OpenCL language selected without USE_OPENCL preprocessor","");
#endif
     
#ifndef _OPENACC
  if ( optLang == LANG_OPENACC) error_abort("OpenACC language selected without accelerator compilation","");
#endif
  
}

static void set_alg_omp(struct grid_info_t *g, const char * optAlg){
  if (optAlg != NULL){
    if (strcmp("baseline", optAlg) == 0)
      set_g(g, ALGORITHM, ALG_BASELINE, "baseline");
    else if (strcmp("baseline-opt", optAlg) == 0)
      set_g(g, ALGORITHM, ALG_BASELINE_OPT, "baseline-opt");
    else if (strcmp("blocked",optAlg) == 0)
      set_g(g, ALGORITHM, ALG_BLOCKED, optAlg);
    else if (strcmp("cco",optAlg) == 0){
      error_abort("CCO algorithm not suported in this release", "");
    }
    else if (strcmp("wave",optAlg) == 0 ){
      error_abort("wave algorithm not suported in this release", "");
      /*
	sscanf(argv[++i],"%d",&(grid->nwaves));
	//niter_fixed = 1;// prevent reseting by -niter flag
	sscanf(argv[++i],"%d",&(grid->threads_per_column));
	if ( grid->threads_per_column == 0 ) 
	grid->alg_key = ALG_WAVE_DIAGONAL;
	else if ( grid->threads_per_column > 0)
	grid->alg_key = ALG_WAVE;
	else
	error_abort("wrong value for threads per column parameter", argv[i]);
      */
    }
    else
      error_abort(" set_alg_omp: Wrong algorithm specifier for openmp language, try -alg help\n",optAlg);
  }
else
	// default OpenMP algorithm (baseline is not a proper CPU implementation
	set_g(g, ALGORITHM, ALG_BASELINE_OPT, "baseline-opt");
}

static void set_alg_cuda(struct grid_info_t *g, const char * optAlg){

  if (optAlg != NULL){
    if ((strcmp("3d-blockgrid",optAlg) == 0) ||
	(strcmp("baseline",optAlg) == 0))
      set_g(g, ALGORITHM, ALG_CUDA_3D_BLK, optAlg);
    else if (strcmp("2d-blockgrid",optAlg) == 0)
      set_g(g, ALGORITHM, ALG_CUDA_2D_BLK, optAlg);
    else if (strcmp("gpu-shm",optAlg) == 0)
      set_g(g, ALGORITHM, ALG_CUDA_SHM, optAlg);
    else if (strcmp("bandwidth", optAlg) == 0)
      set_g(g, ALGORITHM, ALG_CUDA_BANDWIDTH, optAlg);
    else
      error_abort(" set_alg_cuda: Wrong algorithm specifier for cuda language, try -alg help\n",optAlg);
  }
  else 
    // default cuda algorithm
    set_g(g, ALGORITHM, ALG_CUDA_3D_BLK, "baseline");
}

static void set_g(struct grid_info_t *g, const int key, const int val,
                   const char * name){

  if (strlen(name)  >= MAXNAME) 
       error_abort("set_g name length larger than MAXNAME", name);
  
  switch(key)
    { 
    case(LANGUAGE):
      g->lang_key = val;
      snprintf(g->lang_name, MAXNAME, "%s", name);
      break;
    case(ALGORITHM):
      g->alg_key = val;
      snprintf(g->alg_name, MAXNAME, "%s", name);
      break;
    default :
      error_abort("unkwon key in set_g","");
    }
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
#ifdef USE_CUDA
  cutilDeviceInit();
  cudaSafeCall(cudaMalloc((void **)&d_u1, sizeof(Real)*NX*NY*NZ));
  cudaSafeCall(cudaMalloc((void **)&d_u2, sizeof(Real)*NX*NY*NZ));
#endif
}

void freeDeviceMemory(){
#ifdef USE_CUDA
  cudaSafeCall(cudaFree(d_u1));
  cudaSafeCall(cudaFree(d_u2));
#endif
}

