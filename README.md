Jacobi Test Code (JTC): MPI, OpenMP, CUDA,  OpenCL and OpenACC benchmark for 3D Jacobi solver
Version: 1.0.5b (8th December 2014)


Introduction:
=============
* This application provides several implementation of the
   Jacobi algorithm which iterate the 7 points stencil on a cuboid grid.

* The initial grid state is an eigenvector of the Jacobi smoother
    `sin(pi*kx*x/L)*sin(pi*ky*y/L)sin(pi*kz*z/L)`. This is useful for testing.

* The numerical kernel is applied to the grid for a selected number of
  iterations.  Timing data is collected over several runs, each with
  the same number of iterations in order to avoid hardware transient
  effects.

* The average time per iteration is collected for each run and
  the minimum, the average and the maximum values over the runs are
  reported. The output contains also the values of the grid sizes, block
  sizes (if the loops are blocked), the number of used OpenMP threads
  and the MPI topology (if MPI is enabled).
        
Build:
=====
The Makefile variable are customised with the help of
platforms/<name>.inc include files.  Select the desired platform by
passing <name> to PLATFORM variable (without .inc suffix).

`make list_platforms` produces a list of available platforms.

The following make variable control the build:

`USE_OPENMP`   : if set to 'yes' enable OpenMP, if set to 'no' the build is for serial execution
`USE_MPI`          : if defined the enables the MPI build, the default compute model is set to
                             mpi+omp (see below)
`USE_CUDA`      : if defined compilations for CUDA, the default compute model is set to cuda
`USE_OPENACC`: if defined compiles with OpenACC, the default compute model is set to openacc
`USE_OPENCL`   : if defined compiles with OpenCL, the default compute model is set to opencl

`USE_DOUBLE_PRECISION` : if defined enables double precision.

`USE_VEC1D` : if defined enable the use of the optimised form of the inner loop in Jacobi update rule.
                         It is relevant only for MPI and OpenMP builds.
BUILD              : if set to `opt` builds with optimisation compiler flags, if set to `debug` builds with
                         debug compiler flags.

It should be fairly easy to adapt one of the existing platforms file
for another platform. For example the file platforms/cray-xc30.inc
shows a customisation for a system with multiple compilers accesible
via the module environment; platforms/mac_gpu.inc is for laptop with
CUDA programable GPU.

Note: the Fortran build is not active in this release.



Usage:
=====
The following command line options can be used to set the  run parameters:

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
	0. This works because the initial vector on the grid is an
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
     OpenMP compute model is available for any build,
	 but the default depends on the build options. E. g. if the
	 compilation flags enable CUDA compute model the default
     kernel is the CUDA version of Jacobi.
     OpenMP can be disabled if make variable USE_OPENMP is unset.
	 
     __help__ prints the available compute models and quits.

-alg __name__
       selects an algorithm for the Jacobi iteration. Each compute model has at least
	   one algorithm, named baseline. Obviously, baseline is not the same algorithm
	   for all compute models.

	  Other algoritms:
	  ---------------
          baseline-opt - used in kernel Titanium_laplace3d which is Gold_laplace3d with
                        basic loop optimisations ( hoisted if, faster index algebra). Implemented for MPI+OpenMP and OpenMP.
                  
          blocked - uses the blocked loops version, implemented for MPI+OpenMP and OpenMP.

         cco - implemented in MPI+OpenMP, uses master thread for halo exchange.
		          __Note__: the Jacobi loops don't use vec1D function to help vectorisation in this version.

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
				
-pc       prints information on the run parameters.

-nh       output timings without the table header (this useful when collecting large data sets for plots)

-malign __n__  use `posix_memalign` to allocate the working array with an address alignment
             of __n__ bytes. 
             It may help vectorisation on certain systems. Default allocation is done with malloc.

-help        prints a short description of command line arguments
-version   prints version


Run script:
==========
The script `utils/performance_profile.sh` is provided for systematic data
collection of execution times over a set of grid sizes.  The script
adjusts the number of iterations function grid size such that the
timings are meaningful for small grids and run time is not
unnecessarily large for large grids. The format of timings data
suitable for gnuplot.

Try `sh <path>/utils/run_spectra.sh -help` for more details.

Job submission scripts:
======================
In the folder utils there are several sample files for job submit scripts 
suitable for ARCHER PBS and Blue Wonder LSF schedulers.
   



