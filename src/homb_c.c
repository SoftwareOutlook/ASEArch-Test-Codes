/*
This is the C driver for DL_HOMB, a hybrid OpemMP MPI Benchmark for
3D Jacobi solver for Laplace equation.

Iteration starts from a Jacobi iterator eigenvalue, boundary conditions are set to 0.

Lucian Anton July 2013.

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

#include "homb_c.h"


int main(int argc, char *argv[]) {
  int iter;

   /* L2 norms */
  double norm, gnorm; 

  /* Timing measurements */
  double startTime, endTime;
  double meanTime = 0., minTime = 1.e100, maxTime = 0.;
  double stdvTime = 0., NstdvTime = 0., *times;


#ifdef USE_MPI
  /* MPI thread safety level parameters */
  int requested_mpi_safety = MPI_THREAD_FUNNELED, provided_mpi_safety;

 /* Initialize MPI, check the thread safety level */
  MPI_Init_thread(&argc, &argv, requested_mpi_safety, &provided_mpi_safety);
#endif

  /* Initialize Global Context */ 
  initContext(argc, argv);

  /* Get task/thread information */
  setPEsParams();
#ifdef USE_MPI
  if ( (myrank == 0) && (requested_mpi_safety != provided_mpi_safety) ) {
    printf( " Warning, MPI thread safety requested level is not equal with provided \n");
    printf( " requested %d \n ", requested_mpi_safety);
    printf( " provided  %d \n ", provided_mpi_safety);
  }
#endif

  if ( myrank == ROOT)
    times = malloc(niter * nproc * sizeof(double));
  else
     times = malloc(niter * sizeof(double));
  

  initial_field();

  if (myrank == ROOT && pContext)
    printContext();

  /* Solve */
  for (iter = 0; iter < niter; ++iter){
    
    startTime = my_wtime();

    switch (kernel_key)
      {
      case (COMMON_KERNEL) :  common_laplace3d(iter, &norm); break;
      case (BLOCKED_KERNEL):  blocked_laplace3d(iter, &norm); break;
      case (CCO_KERNEL):      cco_laplace3d(iter, &norm); break;
      }

    /* End timing */
    endTime = my_wtime();

    /* check the convergence progess */
//    if ( myPE == ROOT )
//      printf("max dt %e \n", dtg);

    /* Store Time */
    times[iter] = endTime-startTime;

    if (testComputation) 
      check_norm(iter, norm);


  /* Gather iteration runtimes to ROOT's matrix */
  timeUpdate(times);

  }

  /* Run statistics on times (Root only) */
  if (myrank == ROOT) 
    statistics(times, &minTime, &meanTime,
               &maxTime, &stdvTime, &NstdvTime);
  
  //compute the final global norm, useful for quick validation
#ifdef USE_MPI
  MPI_Reduce(&norm, &gnorm, 1, MPI_DOUBLE,
		  MPI_SUM, ROOT, MPI_COMM_WORLD);
#else
  gnorm = norm;
#endif

  /* Output */
  if (myrank == ROOT) 
    stdoutIO(times, minTime, meanTime, maxTime, NstdvTime, gnorm);
  
  /* MPI Finalize */
#ifdef USE_MPI
  MPI_Finalize();
#endif
  
  return(EXIT_SUCCESS);

}


