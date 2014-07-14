Jacobi Test Code (JTC): OpenMP and GPU benchmark for 3D Jacobi solver
Version: 1.0.3b


Basic idea of the algorithm:

	- Iterates the 6 points stencil on a cuboid grid.
	- The initial grid state is an eigenvector of the Jacobi smoother
               sin(pi*kx*x/L)*sin(pi*ky*y/L)sin(pi*kz*z/L)

The code does a number of runs each over a given number of iterations.
The average time per iteration is collected for each run and
the minimum, the average and the maximum values over the
runs are reported. The output contains also the values of the grid sizes, block sizes
(if the loops are blocked), and the number of used OpenMP threads.
        
Build: The Makefile is customised with the help of platforms/*inc files.
       Use platforms/gcc.inc as a template for a quick test on local machine. 
       platforms/cray-xe6.inc shows a customisation for a
       system with multiple compilers accesible via module environment.
       Select the desired platform by passing a value to PLATFORM variable (without .inc suffix)


Usage:

The following flags can be used to set the grid sized and other run parameters:

-ng <nx> <ny> <nz>       set the global gris sizes

-nb <bx> <by> <bz>       set the computational block size, relevant for blocked model and GPU kernels.
                         Notes: 1) only a few sanity checks tests are done, you are on your own.
                                2) for the blocked algorithm the OpenMP parallelism is done over
                                   computational blocks, i.e. a block is updated by one threads. 
				   One must ensure that there is
                                   enough work for all threads by setting suitable 
                                   block sizes.
                                   The basic rule for block size should be 

                                     bx*by*bz = MIN(cache_size, nx*ny*nz/Nthreads) 
 
                                   However if nx*ny*nx/Nthreads is less that a few cache lines 
                                   probable is better to leave some threads unused rather than having 
                                   them fighting over the cache lines.
                                3) For GPU runs <bx> and <by> are the dimension of the block of threads,
                                   <bz> is irrelevant in this version, the block of threads are 2D or 3D 
				   with size 1 in z direction.
				4) Default sizes for GPU block of threads: 32x4x1
                         
-nruns <n>               set the number of smoother runs (default 5)
-niter <b>               set the number of iteration per run (default 1).
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

                         gpu-2d-blockgrid : uses 2d CUDA grids, each block of threads loops in z direction
                                        
                         gpu-3d-blockgrid : uses 3d CUDA grids, the block of threads have size 1 in z
                                            direction, hence grid size is <nz>.  
			 gpu-bandwidth : measures the time for the simple update u[i] = const * v[i]
                                         useful to measure the effective GPU bandwidth.
			                 Note: -t is meaningless in this case
                          
                         NOTES: 1) GPU runs report also the transfer time between device and host,
                                2) Default model is baseline.

-pc                      prints information on run parameters at the beginning of calculation.

-malign <n>     use posix_memalign to allocate working array with an address alignment of <n> bytes. 
                         It may help vectorisation on certain systems.
                         Default allocation is done with malloc.

-help                   prints short description of command line arguments
-version                prints version


Run script:

For systematic data collection of execution times over a set of grid
sizes the script utils/run_spectra.sh is provided.  The script adjusts
the number of iterations function grid size such that the timings are
meaningful for small grids and run time is not unnecessarily large for
large grids. The timing is collected in an output file suitable for
gnuplot.

Try `sh <path>/utils/run_spectra.sh -help` for more details.

   



