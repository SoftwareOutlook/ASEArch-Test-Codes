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

#define ROOT 0

  /* Global variables */
  // number of iterations
int niter, myrank, nproc;
// kernel to be used
int kernel_key;
#define BASELINE_KERNEL 0
#define OPTBASE_KERNEL  1
#define BLOCKED_KERNEL  2
#define CCO_KERNEL      3 

// run info
int testComputation, pContext;

/*********** Functions ***********/
void blocked_laplace3d(int iteration, double *norm);
void cco_laplace3d(int iteration, double *norm);
void baseline_laplace3d(int iteration);
void opt_baseline_laplace3d(int iteration);
void initContext( int argc, char *argv[]);
void setPEsParams(void);
void initial_field(void);
void printContext(void);
void check_norm(int iter, double norm);
void timeUpdate(double *times);
void statistics(double *times,  
                double *minTime, double *meanTime, double *maxTime,
                double *stdvTime, double *NstdvTime);
void stdoutIO( double *times,  
              double minTime, double meanTime, double maxTime, 
	       double NstdvTime, double norm);
double my_wtime();
double local_norm();




