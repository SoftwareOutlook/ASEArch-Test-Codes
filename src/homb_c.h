/*
This is the source for HOMB, the Hybrid OpemMP MPI Benchmark, a Laplace Solver.
Data is distributed over MPI tasks and shared over OpenMP threads in a hybrid 
implementation.  When there is only one MPI task, the problem looks like a pure 
shared memory OpenMP implementation, and when there is only one OpenMP thread, 
the problem looks like a distributed memory MPI implimentation.  This is useful 
for testing Hybrid OpenMP/MPI performance on shared memory (NUMA) and multicore 
machines.
*/
/*
  Copyright 2009 Maxwell Lipford Hutchinson

  This file is part of HOMB.

  PGAF is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  PGAF is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with PGAF.  If not, see <http://www.gnu.org/licenses/>.
*/

/* Include Files */ 
#ifdef USE_MPI
#include <mpi.h>
#endif
#include <omp.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <float.h>
#include <sys/types.h>

#define PI 3.14159265358979323846
#define MAX(x,y) (((x) > (y))? x : y)
#define MIN(x,y) (((x) > (y))? y : x)

#ifdef USE_DOUBLE_PRECISION
typedef double Real;
  #ifdef USE_MPI
    #define REAL_MPI MPI_DOUBLE
  #endif
#else
typedef float Real;
  #ifdef USE_MPI
    #define REAL_MPI MPI_FLOAT
  #endif
#endif

// struc to keep grid data
struct grid_info_t
{
  int ng[3], nb[3]; // global grid and computational blocks
  int sx, ex, sy, ey, sz, ez, nlx, nly, nlz; // start end indices for local grids ( used with MPI runs)
  int np[3]; // MPI topology
}; 

#define ROOT 0

  /* Global variables */
  // number of iterations
int niter, myrank, nproc;
// keys of the available kernels
#define BASELINE_KERNEL 0
#define OPTBASE_KERNEL  1
#define BLOCKED_KERNEL  2
#define CCO_KERNEL      3 

// run info
int testComputation, pContext;

/*********** Functions ***********/
void laplace3d(const struct grid_info_t *grid, int kernel_key , double *tstart, double *tend);
void initContext( int argc, char *argv[],  struct grid_info_t * grid, int *kernel_key);
void setPEsParams(struct grid_info_t *grid);
void initialise_grid(const struct grid_info_t *grid);
void printContext(const struct grid_info_t *grid);
void check_norm(const struct grid_info_t *g, int iter, double norm);
void timeUpdate(double *times);
void statistics(double *times,  
                double *minTime, double *meanTime, double *maxTime,
                double *stdvTime, double *NstdvTime);
void stdoutIO( const struct grid_info_t *grid, const int kernel_key, const double *times,  
              double minTime, double meanTime, double maxTime, 
	       double NstdvTime, double norm);
double my_wtime();
double local_norm(const struct grid_info_t *grid);




