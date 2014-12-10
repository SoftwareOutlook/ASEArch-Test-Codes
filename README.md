Jacobi Test Code (JTC): MPI, OpenMP, CUDA,  OpenCL and OpenACC benchmark for 3D Jacobi solver
Version: 1.1.0


Introduction:
=============

* Iterates the 7 points stencil on a cuboid grid.
* The initial grid state is an eigenvector of the Jacobi smoother
    `sin(pi*kx*x/L)*sin(pi*ky*y/L)sin(pi*kz*z/L)`. This is useful for testing.

The executable does a number of runs,  each of them over a given 
number of iterations.
The average time per iteration is collected for each run and
the minimum, the average and the maximum values over the
runs are reported. The output contains also the values of the grid sizes, 
block sizes (if the loops are blocked), the number of used OpenMP threads and
the MPI topology (if MPI is enabled).
        
Build:
=====

The Makefile is customised with the help of
platforms/<name>.inc.  Select the desired platform by passing <name>
to PLATFORM variable (without .inc suffix).

It should be fairly easy to adapt one of the existing platforms file
for your local setup.  The file platforms/cray-xe6.inc shows a
customisation for a system with multiple compilers accesible via
the module environment.


Usage:
=====

The following flags can be used to set the grid sized and other run parameters:

-ng nx ny nz       set the global grid sizes

-nb bx by bz       set the computational block size - relevant for blocked algorithms
                         in OpenMP and CUDA
             Notes:
	     ------
             * For CUDA runs <bx> and <by> are the dimension of the block of threads,
               <bz> is irrelevant in this version, the block of threads are 2D (i.e. bz = nz) or 3D 
               with size 1 in z direction.
             * Default sizes for CUDA block of threads are: 32x4x1
                         
-nruns __n__   number of runs (default 5)
-niter __b__   number of full iteration (swaps) over the grid per run (default 1).
           Note: if wave algorithm is used niter is fixed to num-wave(see below)
                 and this flag is ignored. 

-t test the correctness; if enabled the executable collects the norms
	of the vectors (grid functions) at the end of each run and prints
	the diference between the ratio of the norms of two consecutive
	runs and the Jacobi smoother eigenvalue to the power niter. This
	quantity should be small (machine precision), theoretically is
	0). This works because the initial vector on the grid is an
	eigenvector of the Jacobi smoother. Warning: Numerical artefacts
	may creep in for large number of iterations and small grids.

-cmodel  __name__
     selects the compute model implementation.
   __name__ can take one of the following values:
     openmp
     mpi+omp
     cuda
     openacc
     opencl
     help

     Notes:
     ------
     OpenMP compute module is available for any build,
	 the default language depends on the build options.
     OpenMP can be disabled if make variable USE_OPENMP is unset.
	 
     __help__ prints the available compute models and quits.

-alg __name__
       selects an algorithm for the Jacobi iteration. Each compute model has at least
	   one algorithm, named baseline. Obviously, baseline is not the same algorithm
	   for all compute models.

	  Other algoritms:
	  ---------------
          baseline-opt - used in kernel Titanium_laplace3d which is Gold_laplace3d with
                        basic loop optimisations ( hoisted if, faster index algebra). Implentend for MPI+OpenMP and OpenMP.
                  
          blocked - uses the blocked loops version, implemented for MPI+OpenMP and OpenMP.

         cco - implemented in MPI+OpenMP, uses master thread for halo exchage. Note: the Jacobi loops don't use vec1D function to help vectorisation in this version.

          wave num-waves threads-per-column - time skewed blocks
	      Notes:
	      ------
              num-waves is the number of waves used in one grid swap
              threads-per-column is the number of threads used on each columns (< number of waves)
                                 if the wave is applied in yz plane, the blocks have local
                                 domain length in x direction.

          2d-blockgrid - uses 2d CUDA grids, each block of threads loops in z direction

          3d-blockgrid - uses 3d CUDA grids, the block of threads have size 1 in z
                         direction, hence grid size is nz in z. ( this also CUDA baseline)
          gpu-bandwidth - measures the time for the simple update u[i] = const * v[i]
                         useful to measure the effective GPU bandwidth.
                         Note: -t is meaningless in this case

          help : prints the list of available algorithms.

          Notes:
	  -----
	  1. GPU runs report also the transfer time between device and host,
          2. Default model is baseline.
          3. OpenCL uses an algorithm similar to the CUDA 3d-blockgrid model.
             If the DEVICE flag is not set then it takes on the type
             CL_DEVICE_TYPE_DEFAULT. See your OpenCL documentation
             for valid values for DEVICE.                           	
				
-pc       prints information on run parameters at the beginning of a calculation.

-nh       output without the table header (useful when collecting large data sets for plots)

-malign __n__    use `posix_memalign` to allocate the working array with an address alignment
             of __n__ bytes. 
             It may help vectorisation on certain systems. Default allocation is done with malloc.

-help        prints a short description of command line arguments
-version     prints version


Run script:

For systematic data collection of execution times over a set of grid
sizes the script `utils/run_spectra.sh` is provided.  The script adjusts
the number of iterations function grid size such that the timings are
meaningful for small grids and run time is not unnecessarily large for
large grids. The timing is collected in an output file suitable for
gnuplot.

Try `sh <path>/utils/run_spectra.sh -help` for more details.

   



