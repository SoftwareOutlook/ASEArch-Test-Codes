/*
This is part of Jacobi Test Code (JTC) , a hybrid CUDA-OpenMP-MPI benchmark for
Jacobi solver applied to a 3D Laplace equation.

This file contains CUDA kernels and auxiliary functions

Contributions from Mike Giles, Saif Mula and Mark Mawson

Lucian Anton
March 2014.

Copyright (c) 2014, Science & Technology Facilities Council, UK
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer. 
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The views and conclusions contained in the software and documentation are those
of the authors and should not be interpreted as representing official policies, 
either expressed or implied, of the FreeBSD Project.


*/

#include<stdio.h>
#include "cutil_inline.h"

// thread blocks and grid parameters
static int BlockX, BlockY, BlockZ, GridX, GridY, GridZ;


/** 
 * this notifies the compiler that the definitions are present in external file
 */
extern "C" {
#include "jacobi_c.h"
#include "gpu_laplace3d_wrapper.h"
}

__global__ void kernel_laplace3d_baseline(int NX, int NY, int NZ, const Real* __restrict__ d_u1,
                                                            Real* __restrict__ d_u2)
{
  int  i, j, k, indg, IOFF, JOFF, KOFF, interior, active;
   Real u2, sixth=1.0/6.0;

  //
  // define global indices and array offsets
  //

  i    = threadIdx.x + blockIdx.x*blockDim.x;
  j    = threadIdx.y + blockIdx.y*blockDim.y;
  indg = i + j*NX;

  IOFF = 1;
  JOFF = NX;
  KOFF = NX*NY;

  interior = i >  0 && i< NX-1 && j> 0 && j<NY-1;
  active   = i >= 0 && i<= NX-1 && j>= 0 && j <= NY-1;
  
  if ( active){
    d_u2[indg] = d_u1[indg];
    indg += KOFF;
    
    for (k=1; k<NZ-1; k++) {
      if (interior) {
	u2 = ( d_u1[indg-IOFF] + d_u1[indg+IOFF]
	       + d_u1[indg-JOFF] + d_u1[indg+JOFF]
	       + d_u1[indg-KOFF] + d_u1[indg+KOFF] ) * sixth;
      }
      else {
	u2 =  d_u1[indg];  // Dirichlet b.c.'s
      }                    // the active flags selects only boundary points
      d_u2[indg] = u2;
      indg += KOFF;
    }
    d_u2[indg] = d_u1[indg];
  }
  
}

//
// Notes:one thread per node in the 2D block;
// after initialisation it marches in the k-direction
//

extern Real *d_u1, *d_u2;


__global__ void kernel_laplace3d_MarkMawson(int Nx, int Ny, int Nz, Real *d_u1, Real *d_u2)
{
  //int   i, j, k, bz, ks, ke, indg, active, IOFF, JOFF, KOFF;
  Real sixth=1.0/6.0;

	//Thread Indices
  int x = blockIdx.x*blockDim.x+threadIdx.x;
  int y = blockIdx.y*blockDim.y+threadIdx.y;
  int z = blockIdx.z*blockDim.z+threadIdx.z;

  if(x<(Nx)&&y<(Ny)&&z<(Nz)){
    if(x!=0&&x!=(Nx-1)&&y!=0&&y!=(Ny-1)&&z!=0&&z!=(Nz-1)){
      d_u2[z*Ny*Nx+y*Nx+x]= sixth*(d_u1[(z-1)*Ny*Nx+(y  )*Nx+(x  )]
				  +d_u1[(z+1)*Ny*Nx+(y  )*Nx+(x  )]
				  +d_u1[(z  )*Ny*Nx+(y-1)*Nx+(x  )]
				  +d_u1[(z  )*Ny*Nx+(y+1)*Nx+(x  )]
				  +d_u1[(z  )*Ny*Nx+(y  )*Nx+(x-1)]
				  +d_u1[(z  )*Ny*Nx+(y  )*Nx+(x+1)]);
    }else{
      d_u2[z*Ny*Nx+y*Nx+x]=d_u1[z*Ny*Nx+y*Nx+x];
    }
  }	
}


__global__ void kernel_laplace3d_shm(int NX, int NY, int NZ, Real *d_u1, Real *d_u2)
{
  extern __shared__ Real plane[];
  int indg, active, halo, indp, IOFF, JOFF, KOFF;
  Real u2, sixth=1.0/6.0;

  //
  // define global indices and array offsets
  //

  int i = blockIdx.x*(blockDim.x-2)+threadIdx.x-1;
  int j = blockIdx.y*(blockDim.y-2)+threadIdx.y-1;
  int k = blockIdx.z*blockDim.z+threadIdx.z; 

  indg = i + j*NX + k*NX*NY;
  indp = threadIdx.x + blockDim.x * threadIdx.y;

  IOFF = 1;
  JOFF = blockDim.x;//for plane layout
  KOFF = NX*NY;

  active = i>=0 && i<=NX-1 && j>=0 && j<=NY-1;

  halo = threadIdx.x == 0 || threadIdx.x == blockDim.x - 1 ||
    threadIdx.y == 0 || threadIdx.y == blockDim.y - 1;
  
  // populate plane with first layer
     if(active)
        plane[indp] = d_u1[indg]; 
     __syncthreads();

    if (active ) {
      if (i==0 || i==NX-1 || j==0 || j==NY-1 || k==0 || k==NZ-1) {
        u2 = d_u1[indg];  // Dirichlet b.c.'s
      }
      else {
        if (!halo)
          u2 = ( plane[indp-IOFF] + plane[indp+IOFF]
               + plane[indp-JOFF] + plane[indp+JOFF]
               + d_u1[indg-KOFF] + d_u1[indg+KOFF] ) * sixth;
        /*
         u2 = ( d_u1[indg-IOFF] + d_u1[indg+IOFF]
             + d_u1[indg-JOFF] + d_u1[indg+JOFF]
             + d_u1[indg-KOFF] + d_u1[indg+KOFF] ) * sixth;
        */
      }
      if (!halo)
        d_u2[indg] = u2;
      }
  }

//This kernel can be used for a quick extimate of the bandwidth 
__global__ void kernel_BandWidth(int NX, int NY, int NZ, Real *d_u1, Real *d_u2)
{
  Real sixth=1.0/6.0;

  //
  // define global indices and array offsets
  //
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int j = blockIdx.y*blockDim.y + threadIdx.y;
  int k = blockIdx.z*blockDim.z + threadIdx.z;
  
  // WARNING: no checks for the interior, grid sizes need to multiple of blocks

  int indg = i + j * NY + k * NX * NY;
  d_u2[indg] = d_u1[indg] * sixth;
}



/**
 * this function is used to just check if an CUDA enabled GPU device is present in
 * the system and also to check it's working status.
 */
extern "C"
void CudaDeviceInit(){
	int cudaDevCnt;
	cudaGetDeviceCount(&cudaDevCnt);
	if (cudaDevCnt==0){
		printf("No CUDA device found, exiting ...\n");
		exit(-1);
	}
	else{
		printf("Number of cuda devices found %d\n",cudaDevCnt);
	}
	cudaFree(0);
	#ifdef __cplusplus
		cudaDeviceProp devProp;
	#else
		struct cudaDeviceProp devProp;
	#endif
	int dev;
	cudaGetDevice(&dev);
	cudaGetDeviceProperties(&devProp,dev);
	printf("Using CUDA device %d: %s\n\n",dev,devProp.name);
	//set cache config to l1 cache
//	cudaError_t error = cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
//	printf("set Cacheconfig error %d\n",error);
}
/////////////////////////////////////////////////////////////////
/**
 * this function copies the updated GPU array to GPU
 */
extern "C"
void setInitialData(float* dev,float* host,int NX,int NY,int NZ,float* memoryTimer,int* memoryCtr){
	printf("Setting up initial Data ...\n");
	cudaEvent_t start,stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
  //start the timer
	cudaEventRecord(start,0);
	cudaMemcpy(dev,host, sizeof(float)*NX*NY*NZ,
                           cudaMemcpyHostToDevice);
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime,start,stop);
	*memoryTimer += elapsedTime;
        *memoryCtr += 1;
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

}
////////////////////////////////////////////////////////////////

/**
 * this function invokes the Laplace3d GPU function which
 * executes the functionality on GPU
 */
extern "C"
void laplace3d_GPU(const int kernel_key, Real* uOld, int NX,int NY,int NZ,const int* gridparams, int iter_block, float *compTime,  float *commTime)
{
  float taux;
  Real *aux;
  int iter;
  size_t shmsize;
  int threadperblock = 4;//for shared memory blocksize which is currently static
  
 
  //event timer
  cudaEvent_t compStart, compStop, commStart, commStop;
  
  cudaSafeCall(cudaEventCreate(&commStart));
  cudaSafeCall(cudaEventCreate(&commStop));
  cudaSafeCall(cudaEventCreate(&compStart));
  cudaSafeCall(cudaEventCreate(&compStop));
  //start the timer
  *commTime = 0.0;

  cudaEventRecord(commStart,0);
  cudaSafeCall(cudaMemcpy(d_u1, uOld, sizeof(Real)*NX*NY*NZ,
			  cudaMemcpyHostToDevice));
  cudaEventRecord(commStop,0);
  cudaEventSynchronize(commStop);
  cudaEventElapsedTime(&taux, commStart, commStop);
  *commTime += taux;

  dim3 dimGrid(GridX,GridY,GridZ) ;
  dim3 dimBlock(BlockX, BlockY,BlockZ) ;
					
  cudaEventRecord(compStart,0);
  
  switch(kernel_key)
    {
    case(GPU_BASE_KERNEL):
      for (iter = 0; iter < iter_block; ++iter){
	kernel_laplace3d_baseline<<<dimGrid, dimBlock>>>(NX, NY, NZ, d_u1, d_u2);
	cudaSafeCall(cudaPeekAtLastError());    	
	aux=d_u1; d_u1=d_u2; d_u2=aux;
      }
      break;
      case(GPU_MM_KERNEL):
	for (iter = 0; iter < iter_block; ++iter){
	  kernel_laplace3d_MarkMawson<<<dimGrid, dimBlock>>>(NX, NY, NZ, d_u1, d_u2);
	  cudaSafeCall(cudaPeekAtLastError());    	
	  aux=d_u1; d_u1=d_u2; d_u2=aux;
      }
      break;
    case(GPU_SHM_KERNEL):
      shmsize=BlockX*BlockY*sizeof(Real);
      for (iter = 0; iter < iter_block; ++iter){
	kernel_laplace3d_shm<<<dimGrid, dimBlock, shmsize>>>(NX, NY, NZ, d_u1, d_u2);
	cudaSafeCall(cudaPeekAtLastError());
	aux=d_u1; d_u1=d_u2; d_u2=aux;
      }
      break;
    case(GPU_BANDWIDTH_KERNEL):
      for (iter = 0; iter < iter_block; ++iter){
        kernel_BandWidth<<<dimGrid, dimBlock>>>(NX, NY, NZ, d_u1, d_u2);
	cudaSafeCall(cudaPeekAtLastError());
        aux=d_u1; d_u1=d_u2; d_u2=aux;
      }
      break;
    }
  
  cudaEventRecord(compStop,0);
  cudaEventSynchronize(compStop);  
  cudaEventElapsedTime(compTime,compStart,compStop);
  
  cudaEventRecord(commStart,0);
  // Becase of the above swap d_u1 points to the last iteration data
  cudaSafeCall(cudaMemcpy(uOld, d_u1, sizeof(Real)*NX*NY*NZ,
			 cudaMemcpyDeviceToHost));
  cudaEventRecord(commStop, 0);
  cudaEventSynchronize(commStop);
  cudaEventElapsedTime(&taux, commStart, commStop);
  *commTime += taux;
 
  cudaSafeCall(cudaEventDestroy(commStart));
  cudaSafeCall(cudaEventDestroy(commStop));
  cudaSafeCall(cudaEventDestroy(compStart));
  cudaSafeCall(cudaEventDestroy(compStop));
}
/**
 * --function: getUpdatedArray
 * this function downloads gpu array from GPU and populates the data
 * from GPU array to CPU array
 */
extern "C"
void getUpdatedArray(float* host,float* dev,int NX,int NY,int NZ,float* memoryTimer,int* memoryCtr)
{
	cudaEvent_t start,stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
  //start the timer
	cudaEventRecord(start,0);
	cudaMemcpy(host, dev, sizeof(float)*NX*NY*NZ,
	                           cudaMemcpyDeviceToHost);
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime,start,stop);
	*memoryTimer += elapsedTime;
	*memoryCtr += 1;
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

}

extern "C"
void calcGpuDims(int kernel_key, int blockXsize, int blockYsize, int blockZsize,int NX,int NY, int NZ, int* gridsize)
{
  // set threads block sizes and grid sizes.
  // used 2 dimensions
  // 0, 1 -> grid x, y
  // 2,3 -> block x, y

  switch (kernel_key)
    {
    case(GPU_BASE_KERNEL):
      GridX = 1 + (NX-1)/blockXsize;
      GridY = (1 + (NY-1)/blockYsize); //* (1 + (NZ-1) / blockZsize);
      GridZ = 1;
      BlockX = blockXsize;
      BlockY = blockYsize;
      BlockZ = 1;
      break;
    case (GPU_SHM_KERNEL) :
      GridX = 1 + (NX-1)/blockXsize; 
      GridY = 1 + (NY-1)/blockYsize;
      GridZ = NZ;
      BlockX = blockXsize + 2; // halo
      BlockY = blockYsize + 2;
      BlockZ = 1;
      break;
    case(GPU_BANDWIDTH_KERNEL):
    case(GPU_MM_KERNEL):
      GridX = 1 + (NX-1)/blockXsize;
      GridY = 1 + (NY-1)/blockYsize; //* (1 + (NZ-1) / blockZsize);
      GridZ = NZ;
      BlockX = blockXsize;
      BlockY = blockYsize;
      BlockZ = 1;
      break;
   
    default:
      fprintf(stderr,"unkwon gpu kernel in calcGpuDims, quitting ...!");
      exit (1);
    }
  // Needs to test if the blocks and grid sizes are meaningful 
}

 
