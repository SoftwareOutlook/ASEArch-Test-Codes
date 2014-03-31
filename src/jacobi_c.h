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

#define JTC_VERSION "1.0.0b (27th of March 2014)"

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
  int threads_per_column; // number of threads per column in wave algorithm
  int sx, ex, sy, ey, sz, ez, nlx, nly, nlz; // start end indices for local grids ( used with MPI runs)
  int nproc, myrank; // MPI rank
  int np[3]; // MPI topology size
  int cp[3]; // grid coordinated of this ranks
  int key;	// kernel identifier
  int malign; // allocate aligned memory to help vectorization
  int nwaves; // number of waves used in time skwed algorithm
#ifdef USE_GPU
  int gpuflag; //gpu flag
  float kernelTimer;
  float memoryTimer;
  int memoryCtr;
#endif
#ifdef USE_MPI
  MPI_Comm comm;
#endif 
}; 

struct times_t
{
  double comp, comm;
  // comm is used for GPU <--> host timmings
};

// MPI root or proc 0
#define ROOT 0

// keys of the available kernels
#define BASELINE_KERNEL 0
#define OPTBASE_KERNEL  1
#define BLOCKED_KERNEL  2
#define CCO_KERNEL      3 
#define WAVE_KERNEL     4
#define WAVE_DIAGONAL_KERNEL	5
#define GPU_BASE_KERNEL   6
#define GPU_SHM_KERNEL   7
#define GPU_BANDWIDTH_KERNEL   8
#define GPU_MM_KERNEL 9
