/** \mainpage
 * This is the OpenCL version of JTC
 */

/** @file
 *
 * Copyright (C) 2014 Mark Mawson
 *
 * Author: Mark Mawson <mark.mawson@stfc.ac.uk>
 * \todo Include header files to grab typename Real
 */

#include "../"
#ifndef OPENCL 
#define  OPENCL

/** OpenCLJacobi -- Perform MaxIters jacobi iterations whilst checking convergence every ConvergeIters 
 * @param cldevicetype - The type of OpenCL device to use. 
 * @param Nx - The x size of the domain.
 * @param Ny - The y size of the domain.
 * @param Nz - The z size of the domain.
 * @param maxIters - The maximum number of iterations to perform
 * @param convergeIters- The number of iterations between convergence checks
 * @param tolerance - The tolereance at which convergence is deemed to have been reached
 * @param unknown - The initialised unknown array on the host
 * @param unknown1 - The first unknown array cl buffer?
 * @param unknown2 - The second unknown cl buffer?
 * @param residual - The residual of the solution in a cl buffer?
 * @param solution - The correct solution in a cl buffer?
 * \todo Do the cl buffers need to be passed back?????
 */
void OpenCL_Jacobi(const int Nx,const int Ny,const int Nx,const int maxIters,const int convergenceIters,const Real tolerance, Real *unknown);

#endif 
