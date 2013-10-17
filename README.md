DL_HOMB: hybrid (MPI+OpenMP, or GPU) benchmark for 3D Jacobi solver
  based on HOMB ( http://sourceforge.net/projects/homb/ )

Basic idea of the algorithm:

	- Iterates the 6 points stencil on a cuboid grid.
	- Initial value is a eigenvalue of the Jacobi smoother
               sin(pi*kx*x/L)*sin(pi*ky*y/L)sin(pi*kz*z/L)

The code does a number o runs each over a given number of iterations.
The time taken for each run is collected; the program outputs the minimum, the average and
the maximum values per iteration over the runs. The output contains also the
values of the grid size, block size ( if loops are blocked), MPI topology
and the number of OpenMP thread used.
        
Build: Make can be customised with the help of platforms/*inc files.
       Use platforms/gcc.inc as a template for a quick test on local
       machine. 
       platforms/cray-xe6.inc shows a customisation for a
       system with multiple compilers accesible via module environment.



Usage:

The following flags can be used to set the grid sized and other run parameters:

-ng <nx> <ny> <nz>       set the global gris sizes

-nb <bx> <by> <bz>       set the computational block size, relevant only for blocked model.
                         Notes: 1) no sanity checks tests are done, you are on your own.
                                2) for blocked model the OpeNMP parallelism is done over
                                   computational blocks. One must ensure that there
                                   enough work for all threads by setting suitable 
                                   block sizes.
                                   The basic rule for block size should be 

                                     bx*by*bz = MIN(cache_size, nx*ny*nz/Nthreads) 
 
                                   However if nx*ny*nx/Nthreads is less that a few cache lines 
                                   probable is better to leave some threads unused rather than having 
                                   them fighting over the cache lines.
                                3) For GPU runs <bx> and <by> are the dimension of the block threads,
                                   <bz> is not used but it must be provided.
                         
-nruns <n>               set the number of smoother runs ( default 5)
-niter <b>               set the number of iteration per run.
                         Note: if wave model is activated niter is fixed to num-wave(see below) 
                               and this flag is ignored. 

-t                       prints the diference between the norm ratio of two consecutive
			 runs and smoother eigenvalue (it must be very small, theoretically is 0).

-nh                      output without header (useful when collecting large data sets for plots)

-model <name>            selects one of the implemented Jacobi versions.
                         <name> can be one of the following:

                         baseline: uses Gold_laplace3d

			 baseline-opt: used Titanium_laplace3d which is Gold_laplace3d with
                                       basic loop optimisations ( hoisted if, faster index algebra)
                  
			 blocked : uses the blocked loops version

                         wave num-waves threads-per-column : time skewed blocks
                                 -num-waves is the number of waves used in one grid swap
                                 -threads-per-column is the number of threads used on each columns 
                                  (< number of waves)
                                 NOTE: the wave is applied in yz plane, the blocks have local
                                       domain length in x direction.      

                         gpu-baseline : uses basic CUDA implementation of Jacobi solver
                                        
                         gpu-shm : uses shared memory to store plane xy in a block of threads  
                          
                         NOTES: 1) For GPU runs transfer time between device and host is also implemented      
                                2) Default model is baseline.

-pc                      prints information on run parameters.

-malign <n>              use posix_memalign to allocate working array with an address alignment of <n> bytes. 
                         It may help vectorisation on certain systems.
                         Default allocation is done with malloc.

-np <px> <py> <pz>       set the MPI Cartesian topology
                         the global  grid is partition approximately equal
                         amongst the tasks ( i.e. the reminder  n<d> % p<d> is 
                         spread over the first ranks in each direction)



   



