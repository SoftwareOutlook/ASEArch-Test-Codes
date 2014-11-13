

/** @file
 *
 * Copyright (C) 2014 Mark Mawson
 *
 * Author: Mark Mawson <mark.mawson@stfc.ac.uk>

 */

#ifndef OPENCL_JACOBI 
#define  OPENCL_JACOBI
#include "../jacobi_c.h"


/** OpenCLJacobi -- Initialise the OpenCL runtime and perform memcopies.
 * @param Nx - The x size of the domain.
 * @param Ny - The y size of the domain.
 * @param Nz - The z size of the domain.
 * @param unknown - The initial conditions.
 */
int OpenCL_Jacobi(int Nx, int Ny, int Nz, Real * unknown);


/** OpenCL_Jacobi_Iteration -- Perform an iteration. 
 * @param maxIters - The maximum number of iterations to perform.
 */
int  OpenCL_Jacobi_Iteration(int maxIters);

/** OpenCL_Jacobi_Tidy -- Copy the result back into the array "unknown" and tidy up OpenCL memory
 * @param unknown - The host array to copy the results into.
 */
int OpenCL_Jacobi_Tidy();

int OpenCL_Jacobi_CopyTo(Real *unknown);
int OpenCL_Jacobi_CopyBack(Real * unknown);

#endif 
