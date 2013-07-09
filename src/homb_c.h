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
#define ROOT 0
#define IN -1
#define OUT 1
#define NORTH 0
#define SOUTH 1
#define WEST 2
#define EAST 3
#define BOTTOM 4
#define TOP 5

/* Global variables */
int niter, nproc, myrank, pContext;
// Computation communication overlap; relevan only for MPI case
int use_cco;

// print the difference between norm ration and egenvalue ( it must be small)
int testComputation;

/*********** Functions ***********/

void jacobi_smoother(int iteration, double *norm);
void jacobi_smoother_cco(int iteration, double *norm);
#ifdef USE_MPI
void post_recv(void);
void exchange_halos(void);
void buffer_halo_transfers(int dir, double *norm, int update);
void transfer_data(const int dir, int side);
#endif
void stencil_update( int s1, int e1, int s2, int e2, int s3, int e3, double * x);
inline int uindex(const int i, const int j, const int k);
void initContext( int argc, char *argv[]);
void setPEsParams(void);
void compute_local_grid_ranges( void );
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




