DL_HOMB: hybrid (MPI+OpenMP) benchmark for 3D Jacobi solver
  based on HOMB ( http://sourceforge.net/projects/homb/ )

Basic idea of the algorithm:

	- Iterates the 6 points stencil on a cuboid grid.
	- Initial value is a eigenvalue of the Jacobi smoother
               sin(pi*kx*x/L)*sin(pi*ky*y/L)sin(pi*kz*z/L)

The runtime for each iteration is collected outside OpenMP region. 
At the end of the run the average, minimum and maximum values of
the runtimes are printed together with the values of the grid size,
 MPI topology and the number of OpenMP thread used.
        
Build: Make can be customised with the help of platforms/*inc files.
       Use platforms/gcc.inc as a template for a quick test on local
       machine. 
       platforms/cray-xe6.inc shows a customisation for a
       system with mutiple compilers accesible via module.



Usage description:

The following flags can be used to set the grid sized and other run parameters:

-ng <nx> <ny> <nz>       set the global gris sizes
                         

-np <px> <py> <pz>       set the MPI Cartesian topology
                         the global  grid is partition approximately equal
                         amongst the tasks ( i.e. the reminder  n<d> % p<d> is 
                         spread over the first ranks in each direction)

-niter <n>               set the number of smoother iterations ( default 20)

-t                       prints the diference between the norm ratio of two consecutive
			 iterations and smoother eigenvalue (it must be very small).

-nh                      output without header

-model <name>            selects one of the implemented Jacobi versions.
                         <name> can be one of the following:
                         common  : uses Gold_laplace3d
			 blocked : uses the blocked version
                         cco     : uses a blocked version that keeps the master thread
                                   for MPI communication and domain faces, the other threads
				   iterates over the inside points. 
                         Default is common.

-pc                      prints various setting of the run

-v                       Prints all collected timings for all MPI tasks.
                         Default output prints only the average, min and
                         max value of the timings and the normalised standard deviation.

   



