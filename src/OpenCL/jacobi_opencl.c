/** @file jacobi_opencl.c --- 
 *
 * Copyright (C) 2014 Mark Mawson
 *
 * Author: Mark Mawson <mark.mawson@stfc.ac.uk>
 * \todo Include header files to grab typename Real
 */
#include "jacobi_opencl.h"

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

void OpenCL_Jacobi(int Nx, int Ny, int Nx, int maxIters,Real tolerance, Real *unknown){
  cl_device_id     device_id;     // compute device id 
  cl_context       context;       // compute context
  cl_command_queue commands;      // compute command queue
  cl_program       program;       // compute program
  cl_kernel        jacobi_ocl;       // compute kernel
    
  cl_mem d_u1;                     // device memory used for the input unknown 1 vector
  cl_mem d_u2;                     // device memory used for the input  unknown 2 vector

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
      err = clGetDeviceIDs(Platform[i], DEVICE, 1, &device_id, NULL);
      if (err == CL_SUCCESS)
        {
	  break;
        }
    }

  if (device_id == NULL)
    {
      printf("Error: Failed to create a device group!\n%s\n",err_code(err));
      return EXIT_FAILURE;
    }

  err = output_device_info(device_id);
  
  // Create a compute context 
  context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
  if (!context)
    {
      printf("Error: Failed to create a compute context!\n%s\n", err_code(err));
      return EXIT_FAILURE;
    }

  // Create a command queue
  commands = clCreateCommandQueue(context, device_id, 0, &err);
  if (!commands)
    {
      printf("Error: Failed to create a command commands!\n%s\n", err_code(err));
      return EXIT_FAILURE;
    }

   char *source;
   const char *sourceFile = "jacobi_relaxation_ocl.cl";
   // This function reads in the source code of the program
   source = readSource(sourceFile);
  // Create the compute program from the source buffer
  program = clCreateProgramWithSource(context, 1, (const char **) &source, NULL, &err);
  if (!program)
    {
      printf("Error: Failed to create compute program!\n%s\n", err_code(err));
      return EXIT_FAILURE;
    }

  // Build the program  
  err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
  if (err != CL_SUCCESS)
    {
      size_t len;
      char buffer[2048];

      printf("Error: Failed to build program executable!\n%s\n", err_code(err));
      clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
      printf("%s\n", buffer);
      return EXIT_FAILURE;
    }

  // Create the compute kernel from the program 
  jacobi_ocl = clCreateKernel(program, "vadd", &err);
  if (!jacobi_ocl || err != CL_SUCCESS)
    {
      printf("Error: Failed to create compute kernel!\n%s\n", err_code(err));
      return EXIT_FAILURE;
    }

  // Create the input (u1, u2) arrays in device memory
  d_u1  = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(Real) * count, NULL, NULL);
  d_u2  = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(Real) * count, NULL, NULL);
  if (!d_u1 || !d_u2)
    {
      printf("Error: Failed to allocate device memory!\n");
      exit(1);
    }    
    
  // Write u1 and u2 vectors into compute device memory 
  err = clEnqueueWriteBuffer(commands, d_u1, CL_TRUE, 0, sizeof(Real) * count, unknown, 0, NULL, NULL);
  if (err != CL_SUCCESS)
    {
      printf("Error: Failed to write the d_u1 to opencl buffer!\n%s\n", err_code(err));
      exit(1);
    }

  err = clEnqueueWriteBuffer(commands, d_u2, CL_TRUE, 0, sizeof(Real) * count, unknown, 0, NULL, NULL);
  if (err != CL_SUCCESS)
    {
      printf("Error: Failed to write d_u2 to opencl buffer!\n%s\n", err_code(err));
      exit(1);
    }
	
  // Set the arguments to our compute kernel
  err  = clSetKernelArg(jacobi_ocl, 0, sizeof(int), Nx);
  err |= clSetKernelArg(jacobi_ocl, 1, sizeof(int), Ny);
  err |= clSetKernelArg(jacobi_ocl, 2, sizeof(int), Nz);
  err |= clSetKernelArg(jacobi_ocl, 3, sizeof(cl_mem), &d_u1);
  err |= clSetKernelArg(jacobi_ocl, 4, sizeof(cl_mem), &d_u2);
  
  if (err != CL_SUCCESS)
    {
      printf("Error: Failed to set kernel arguments!\n");
      exit(1);
    }
  // Execute the kernel over the entire range of our 1d input data set
  // letting the OpenCL runtime choose the work-group size
  global = {Nx,Ny,Nz};
  err = clEnqueueNDRangeKernel(commands, jacobi_ocl, 3, NULL, &global, NULL, 0, NULL, NULL);
  if (err)
    {
      printf("Error: Failed to execute kernel!\n%s\n", err_code(err));
      return EXIT_FAILURE;
    }

  // Wait for the commands to complete before stopping the timer
  clFinish(commands);


  // Read back the results from the compute device
  /** \todo Fix this memory transfer */
  err = clEnqueueReadBuffer( commands, d_u1, CL_TRUE, 0, sizeof(Real) * count, unknown, 0, NULL, NULL );  
  if (err != CL_SUCCESS)
    {
      printf("Error: Failed to read output array!\n%s\n", err_code(err));
      exit(1);
    }
    


    
  // cleanup then shutdown
  clReleaseMemObject(d_u1);
  clReleaseMemObject(d_u2);
  clReleaseProgram(program);
  clReleaseKernel(jacobi_ocl);
  clReleaseCommandQueue(commands);
  clReleaseContext(context);

}


