/** @file jacobi_opencl.c --- 
 *
 * Copyright (C) 2014 Mark Mawson
 *
 * Author: Mark Mawson <mark.mawson@stfc.ac.uk>
 * 
 */
#include "jacobi_opencl.h"
//#include "../jacobi_c.h"

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#include <unistd.h>
#else
#include <CL/cl.h>
#endif

//pick up device type from compiler command line or from 
//the default type
#ifndef DEVICE
#define DEVICE CL_DEVICE_TYPE_DEFAULT
#endif

extern int output_device_info(cl_device_id );
char* err_code (cl_int);

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



/** jacobi_relaxation_ocl --
 * @param Nx - The x size of the domain
 * @param Ny - The y size of the domain
 * @param Nz - The z size of the domain
 * @param d_u1 - The input array
 * @param d_u2 - The output array 
 */
const char *KernelSource = "\n" \
  "__kernel void  jacobi_relaxation_ocl(const int Nx, \n"	\
  "				     const int Ny, \n"		\
  "				     const int Nz, \n"			\
  "				     global const Real* restrict d_u1, \n" \
  "				     global Real* restrict d_u2){ \n"	\
  "  const Real sixth=1.0/6.0; \n"					\
  "  //Thread Indices \n"						\
  "  const int x =get_global_id(0); \n" \
  "  const int y =get_global_id(1); \n" \
  "  const int z =get_global_id(2); \n" \
  "  const int loc=z*Ny*Nx+y*Nx+x; \n" \
  "   if(x!=0&&x!=(Nx-1)&&y!=0&&y!=(Ny-1)&&z!=0&&z!=(Nz-1)){ \n" \
  "    d_u2[loc]= sixth*(d_u1[loc-Ny*Nx]+d_u1[loc+Nx*Ny]+d_u1[loc-Nx]+d_u1[loc+Nx]+d_u1[loc-1]+d_u1[loc+1]); \n" \
  " }  \n" \
"} \n" \
"\n";

static struct OpenCLInstance OCLInst;


//OpenCL setup code
int OpenCL_Jacobi(int Nx, int Ny, int Nz, Real *unknown){
  int err; //An error tracking integer
  int i;// A generic counting variable

  //Initialise an opencl instance
  OCLInst.xDim=Nx;
  OCLInst.yDim=Ny;
  OCLInst.zDim=Nz;

  // Set up platform and GPU device
  cl_uint numPlatforms;

  // Find number of platforms
  err = clGetPlatformIDs(0, NULL, &numPlatforms);
  if (err != CL_SUCCESS || numPlatforms <= 0)
    {
      printf("Error: Failed to find a platform!\n%s\n",err_code(err));
      return EXIT_FAILURE;
    }

  // Get all platforms
  cl_platform_id Platform[numPlatforms];
  err = clGetPlatformIDs(numPlatforms, Platform, NULL);
  if (err != CL_SUCCESS || numPlatforms <= 0)
    {
      printf("Error: Failed to get the platform!\n%s\n",err_code(err));
      return EXIT_FAILURE;
    }

  // Secure a GPU
  for (i = 0; i < numPlatforms; i++)
    {
      err = clGetDeviceIDs(Platform[i], DEVICE, 1, &OCLInst.device_id, NULL);
      if (err == CL_SUCCESS)
        {
	  break;
        }
    }

  if (OCLInst.device_id == NULL)
    {
      printf("Error: Failed to create a device group!\n%s\n",err_code(err));
      return EXIT_FAILURE;
    }

  err = output_device_info(OCLInst.device_id);
  
  // Create a compute context 
  OCLInst.context = clCreateContext(0, 1, &OCLInst.device_id, NULL, NULL, &err);
  if (!OCLInst.context)
    {
      printf("Error: Failed to create a compute context!\n%s\n", err_code(err));
      return EXIT_FAILURE;
    }

  // Create a command queue
  OCLInst.commands = clCreateCommandQueue(OCLInst.context, OCLInst.device_id, 0, &err);
  if (!OCLInst.commands)
    {
      printf("Error: Failed to create a command commands!\n%s\n", err_code(err));
      return EXIT_FAILURE;
    }


  // Create the compute program from the source buffer
  OCLInst.program = clCreateProgramWithSource(OCLInst.context, 1, (const char **) &KernelSource, NULL, &err);
  if (!OCLInst.program)
    {
      printf("Error: Failed to create compute program!\n%s\n", err_code(err));
      return EXIT_FAILURE;
    }

  // Build the program  
  err = clBuildProgram(OCLInst.program, 0, NULL, NULL, NULL, NULL);
  if (err != CL_SUCCESS)
    {
      size_t len;
      char buffer[2048];

      printf("Error: Failed to build program executable!\n%s\n", err_code(err));
      clGetProgramBuildInfo(OCLInst.program, OCLInst.device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
      printf("%s\n", buffer);
      return EXIT_FAILURE;
    }

  // Create the compute kernel from the program 
  OCLInst.jacobi_ocl = clCreateKernel(OCLInst.program, "jacobi_relaxation_ocl", &err);
  if (!OCLInst.jacobi_ocl || err != CL_SUCCESS)
    {
      printf("Error: Failed to create compute kernel!\n%s\n", err_code(err));
      return EXIT_FAILURE;
    }

  // Create the input (u1, u2) arrays in device memory
  OCLInst.d_u1  = clCreateBuffer(OCLInst.context,  CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,  sizeof(Real) *OCLInst.xDim*OCLInst.yDim*OCLInst.zDim, unknown, NULL);
  OCLInst.d_u2  = clCreateBuffer(OCLInst.context,  CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,  sizeof(Real) *OCLInst.xDim*OCLInst.yDim*OCLInst.zDim, unknown, NULL);
  if (!OCLInst.d_u1 || !OCLInst.d_u2)
    {
      printf("Error: Failed to allocate device memory!\n");
      exit(1);
    }    
    
  // Write u1 and u2 vectors into compute device memory 
  err = clEnqueueWriteBuffer(OCLInst.commands, OCLInst.d_u1, CL_TRUE, 0, sizeof(Real) * OCLInst.xDim*OCLInst.yDim*OCLInst.zDim, unknown, 0, NULL, NULL);
  if (err != CL_SUCCESS)
    {
      printf("Error: Failed to write the d_u1 to opencl buffer!\n%s\n", err_code(err));
      exit(1);
    }

  err = clEnqueueWriteBuffer(OCLInst.commands, OCLInst.d_u2, CL_TRUE, 0, sizeof(Real) * OCLInst.xDim*OCLInst.yDim*OCLInst.zDim, unknown, 0, NULL, NULL);
  if (err != CL_SUCCESS)
    {
      printf("Error: Failed to write d_u2 to opencl buffer!\n%s\n", err_code(err));
      exit(1);
    }
  return 0;
}
//////////////////////////////END "CONSTRUCTOR" HERE/////////////////////////////////////"





//OpenCL execution code
/** \todo Add comms timing for argument setting and cleanup, then comp timing for kernel execution */
int OpenCL_Jacobi_Iteration(int maxIters, double *compTime, double *commTime, Real *unknown){
  int err;//Error checking integer
  int i;//Generic loop counting variable
  size_t global[3]={(size_t)OCLInst.xDim, (size_t)OCLInst.yDim, (size_t)OCLInst.zDim};//Setting the glboal kernel size

  // Set the arguments to our compute kernel
  err  = clSetKernelArg(OCLInst.jacobi_ocl, 0, sizeof(int), &OCLInst.xDim);
  err |= clSetKernelArg(OCLInst.jacobi_ocl, 1, sizeof(int), &OCLInst.yDim);
  err |= clSetKernelArg(OCLInst.jacobi_ocl, 2, sizeof(int), &OCLInst.zDim);

  
  if (err != CL_SUCCESS)
    {
      printf("Error: Failed to set kernel arguments!\n");
      exit(1);
    }
  //Checkl for an even number of iterations and caution otherwise
  if(maxIters%2==1) printf("Your number of iterations is not even, the OpenCL implementation will perform 1 more iteration than specified.\n");
  for(i=0;i<maxIters/2;++i){
    //Allcoate the buffers to the kernel/swap the buffers bacl
    err |= clSetKernelArg(OCLInst.jacobi_ocl, 3, sizeof(cl_mem), &OCLInst.d_u1);
    err |= clSetKernelArg(OCLInst.jacobi_ocl, 4, sizeof(cl_mem), &OCLInst.d_u2);
    // Execute the kernel
    // letting the OpenCL runtime choose the work-group size
   
    err = clEnqueueNDRangeKernel(OCLInst.commands, OCLInst.jacobi_ocl, 3, NULL, global, NULL, 0, NULL, NULL);
    if (err)
      {
	printf("Error: Failed to execute kernel!\n%s\n", err_code(err));
	return EXIT_FAILURE;
      }
    // Wait for the commands to complete
    clFinish(OCLInst.commands);
    //Swap the buffers around
    err |= clSetKernelArg(OCLInst.jacobi_ocl, 3, sizeof(cl_mem), &OCLInst.d_u1);
    err |= clSetKernelArg(OCLInst.jacobi_ocl, 4, sizeof(cl_mem), &OCLInst.d_u2);
    //Execute the kernel a second time
    err = clEnqueueNDRangeKernel(OCLInst.commands, OCLInst.jacobi_ocl, 3, NULL, global, NULL, 0, NULL, NULL);
    if (err)
      {
	printf("Error: Failed to execute kernel!\n%s\n", err_code(err));
	return EXIT_FAILURE;
      }
    // Wait for the commands to complete
    clFinish(OCLInst.commands);
  }



  // Read back the results from the compute device
  /** \todo Fix this memory transfer */
  err = clEnqueueReadBuffer(OCLInst.commands,OCLInst.d_u1, CL_TRUE, 0, sizeof(Real) *  OCLInst.xDim*OCLInst.yDim*OCLInst.zDim, unknown, 0, NULL, NULL );  
  if (err != CL_SUCCESS)
    {
      printf("Error: Failed to read output array!\n%s\n", err_code(err));
      exit(1);
    }
    


    
  // cleanup then shutdown
  clReleaseMemObject(OCLInst.d_u1);
  clReleaseMemObject(OCLInst.d_u2);
  clReleaseProgram(OCLInst.program);
  clReleaseKernel(OCLInst.jacobi_ocl);
  clReleaseCommandQueue(OCLInst.commands);
  clReleaseContext(OCLInst.context);
			  
  return 0;
}


