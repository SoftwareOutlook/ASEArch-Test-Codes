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

#ifndef OPENCL_JACOBI 
#define  OPENCL_JACOBI
#include "../jacobi_c.h"


/** OpenCLJacobi -- Initialise the OpenCL runtime and perform memcopies
 * @param Nx - The x size of the domain.
 * @param Ny - The y size of the domain.
 * @param Nz - The z size of the domain.
 * @param unknown - The initial conditions
 */
int OpenCL_Jacobi(int Nx, int Ny, int Nz, Real *unknown);


/** OpenCL_Jacobi_Iteration --
 * @param maxIters - The maximum number of iterations to perform
 * @param compTime - An accumulated count of the time spent in computational kernels
 * @param commTime - An accumulated count of the time spent performing OpenCL communication
 */
int  OpenCL_Jacobi_Iteration(int maxIters, double *compTime, double *commTime, Real *unknown);

#endif 
