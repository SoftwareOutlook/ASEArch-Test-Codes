/*
This is the C driver for Jacobi Test Code (JTC) , a hybrid CUDA-OpenMP-MPI benchmark for
Jacobi solver applied to a 3D Laplace equation.

Iteration starts from a Jacobi iterator eigenvalue, boundary conditions are set to 0.

Lucian Anton
March 2014.

This code started from v 1.0 of HOMB
http://sourceforge.net/projects/homb/

Below is the original copyright and licence.

*/
/*
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

int main(int argc, char *argv[]) {
  int irun, kernel_key;

  /* grid info */
  struct grid_info_t grid;

   /* L2 norms */
  double gnorm, norm;

  /* Timing measurements */
  struct times_t meanTime, minTime, maxTime;
  struct times_t *times;
  
  double compTime, commTime;

#ifdef USE_MPI
  /* MPI thread safety level parameters */
  int requested_mpi_safety = MPI_THREAD_FUNNELED, provided_mpi_safety;

 /* Initialize MPI, check the thread safety level */
  MPI_Init_thread(&argc, &argv, requested_mpi_safety, &provided_mpi_safety);
#endif

  /* Initialize Global Context */ 
  initContext(argc, argv, &grid, &kernel_key);

  /* Get task/thread information */
  setPEsParams(&grid, kernel_key);

#ifdef USE_MPI
  if ( (grid.myrank == 0) && (requested_mpi_safety != provided_mpi_safety) ) {
    printf( " Warning, MPI thread safety requested level is not equal with provided \n");
    printf( " requested %d \n ", requested_mpi_safety);
    printf( " provided  %d \n ", provided_mpi_safety);
  }
#endif

  if ( grid.myrank == ROOT)
    times = malloc(nruns * grid.nproc * sizeof(struct times_t));
  else
    times = malloc(nruns * sizeof(struct times_t));
  
  initialise_grid(&grid);

  if (grid.myrank == ROOT && pContext)
    printContext(&grid, kernel_key);

  /* Solve */
  for (irun = 0; irun < nruns; ++irun){
    
    laplace3d(&grid, kernel_key, &compTime, &commTime);

    times[irun].comp = compTime;
#ifdef USE_GPU
    times[irun].comm = commTime;
#endif

    if (testComputation) {
      norm = local_norm(&grid);
      check_norm(&grid, irun, norm);
    }


  /* Gather iteration runtimes to ROOT's matrix */
    timeUpdate(times);
  }//end for loop

  /* Run statistics on times (Root only) */
  if (grid.myrank == ROOT) 
    statistics(&grid, times, &minTime, &meanTime,
               &maxTime);
  
  //compute the final global norm, useful for quick validation
  if (!testComputation) norm = local_norm(&grid);
#ifdef USE_MPI
  MPI_Reduce(&norm, &gnorm, 1, MPI_DOUBLE,
		  MPI_SUM, ROOT, MPI_COMM_WORLD);
#else
  gnorm = norm;
#endif

  /* Output */
  if (grid.myrank == ROOT) 
     stdoutIO(&grid, kernel_key, times, &minTime, &meanTime, &maxTime, gnorm);
  
  /* MPI Finalize */
#ifdef USE_MPI
  MPI_Finalize();
#endif
  
#ifdef USE_GPU
  if(grid.gpuflag==1)
	  freeDeviceMemory();
#endif
  return(EXIT_SUCCESS);

}


