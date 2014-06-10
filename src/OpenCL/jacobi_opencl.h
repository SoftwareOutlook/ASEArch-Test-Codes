/** \mainpage
 * This is the OpenCL version of JTC extra stuff
 *\section Introduction
 *The rapid development of parallel hardware in the last decade requires
 a constant evaluation of the parallel algoritms used in large scale scientific
 applications . Iterative kernels are
 essential components of iterative solvers which are the preferred 
 technique in a variety of large scale problems.
 Jacobi iteration for the second order discretisation of the Laplacian \f$ 3D\f$ operator:
 \f{equation}{
 u^{(new)}_{i,j,k}=\frac{1}{6}(u^{(old)}_{i-1,j,k}+
 u^{(old)}_{i+1,j,k}+u^{(old)}_{i,j-1,k}+ u^{(old)}_{i,j+1,k}+u^{(old)}_{i,j,k-1}+u^{(old)}_{i,j,k+1})
 \ ,
 \f}
 *is the one of the simplest, yet not trivial, example of iterative
 kernel. In its simple form it contains the features relevant to the
 performance for a large class of iterators: i) stranded memory access
 and ii) low number of floating point operations per memory reference.
\section Results
\image latex OPENCLRESULTS.pdf "My application" width=10cm
 */

/** @file
 *
 * Copyright (C) 2014 Mark Mawson
 *
 * Author: Mark Mawson <mark.mawson@stfc.ac.uk>
 * \todo Include header files to grab typename Real
 */

#include "../"
#ifndef OPENCL_JACOBI 
#define  OPENCL_JACOBI

/** OpenCLInstance - A struct containing the device ide, context, command queue, program, kernel and problem size for OpenCL
 */
struct OpenCLInstance{
  cl_device_id     device_id;     /**< Compute device id */
  cl_context       context;       /**< Compute context */
  cl_command_queue commands;      /**< Compute command queue */
  cl_program       program;       /**< Compute program */
  cl_kernel        jacobi_ocl;    /**< Compute kernel */
    
  cl_mem d_u1;                     /**< Device memory used for the input unknown 1 vector */
  cl_mem d_u2;                     /**< Device memory used for the input  unknown 2 vector */

  unsigned int xDim, yDim, zDim; /**< Grid dimensions */
};  


/** OpenCLJacobi -- Initialise the OpenCL runtime and perform memcopies
 * @param Nx - The x size of the domain.
 * @param Ny - The y size of the domain.
 * @param Nz - The z size of the domain.
 * @param unknown - The initial conditions
 */
void OpenCL_Jacobi(int Nx, int Ny, int Nx, Real *unknown);


/** OpenCL_Jacobi_Iteration --
 * @param maxIters - The maximum number of iterations to perform
 * @param convegenceIters - the number of iteration between convergence checks
 */
void OpenCL_Jacobi_Iteration(int maxIters, int convergenceIters);

#endif 
