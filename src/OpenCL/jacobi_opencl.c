/** @file jacobi_opencl.c --- 
 *
 * Copyright (C) 2014 Mark Mawson
 *
 * Author: Mark Mawson <mark.mawson@stfc.ac.uk>
 * 
 */
#include "jacobi_opencl.h"
#include "../jacobi_c.h"

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

static struct OpenCLInstance OCLInst;


//OpenCL setup code
void OpenCL_Jacobi(int Nx, int Ny, int Nx, Real *unknown){

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
  if (!context)
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

  char *source;
  const char *sourceFile = "jacobi_relaxation_ocl.cl";
  // This function reads in the source code of the program
  source = readSource(sourceFile);
  // Create the compute program from the source buffer
  OCLInst.program = OCLInst.clCreateProgramWithSource(OCLInst.context, 1, (const char **) &source, NULL, &err);
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
  OCLInst.d_u1  = clCreateBuffer(context,  CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_POINTER,  sizeof(Real) * count, unknown, NULL);
  OCLInst.d_u2  = clCreateBuffer(context,  CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_POINTER,  sizeof(Real) * count, unknown, NULL);
  if (!OCLInst.d_u1 || !OCLInst.d_u2)
    {
      printf("Error: Failed to allocate device memory!\n");
      exit(1);
    }    
    
  // Write u1 and u2 vectors into compute device memory 
  err = clEnqueueWriteBuffer(OCLInst.commands, OCLInst.d_u1, CL_TRUE, 0, sizeof(Real) * count, unknown, 0, NULL, NULL);
  if (err != CL_SUCCESS)
    {
      printf("Error: Failed to write the d_u1 to opencl buffer!\n%s\n", err_code(err));
      exit(1);
    }

  err = clEnqueueWriteBuffer(OCLInst.commands, OCLInst.d_u2, CL_TRUE, 0, sizeof(Real) * count, unknown, 0, NULL, NULL);
  if (err != CL_SUCCESS)
    {
      printf("Error: Failed to write d_u2 to opencl buffer!\n%s\n", err_code(err));
      exit(1);
    }
}
//////////////////////////////END "CONSTRUCTOR" HERE/////////////////////////////////////"





//OpenCL execution code
void OpenCL_Jacobi_Iteration(int maxIters, int convergenceIters){

  // Set the arguments to our compute kernel
  err  = clSetKernelArg(OCLInst.jacobi_ocl, 0, sizeof(int), OCLInst.xDim);
  err |= clSetKernelArg(OCLInst.jacobi_ocl, 1, sizeof(int), OCLInst.yDim);
  err |= clSetKernelArg(OCLInst.jacobi_ocl, 2, sizeof(int), OCLInst.zDim);
  err |= clSetKernelArg(OCLInst.jacobi_ocl, 3, sizeof(cl_mem), &OCLInst.d_u1);
  err |= clSetKernelArg(OCLInst.jacobi_ocl, 4, sizeof(cl_mem), &OCLInst.d_u2);
  
  if (err != CL_SUCCESS)
    {
      printf("Error: Failed to set kernel arguments!\n");
      exit(1);
    }
  // Execute the kernel over the entire range of our 1d input data set
  // letting the OpenCL runtime choose the work-group size
  global = {OCLInst.xDim, OCLInst.yDim, OCLInst.zDim};
  err = clEnqueueNDRangeKernel(OCLInst.commands, OCLInst.jacobi_ocl, 3, NULL, &global, NULL, 0, NULL, NULL);
  if (err)
    {
      printf("Error: Failed to execute kernel!\n%s\n", err_code(err));
      return EXIT_FAILURE;
    }

  // Wait for the commands to complete before stopping the timer
  clFinish(OCLInst.commands);


  // Read back the results from the compute device
  /** \todo Fix this memory transfer */
  err = clEnqueueReadBuffer(OCLInst.commands,OCLInst.d_u1, CL_TRUE, 0, sizeof(Real) * count, unknown, 0, NULL, NULL );  
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

}


