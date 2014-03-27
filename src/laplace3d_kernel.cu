/*
This is part of Jacobi Test Code (JTC) , a hybrid CUDA-OpenMP-MPI benchmark for
Jacobi solver applied to a 3D Laplace equation.

This file contains CUDA kernels and auxiliary functions

Contributions from Mike Giles, Saif Mula and Mark Mawson

Lucian Anton
March 2014.
*/

#include<stdio.h>
#include "cutil_inline.h"


//#include<cuda.h>
//#include<cuda_runtime.h>

/** 
 * this notifies the compiler that the definitions are present in external file
 */
extern "C" {
#include "jacobi_c.h"
#include "gpu_laplace3d_wrapper.h"
}

//
// Notes:one thread per node in the 2D block;
// after initialisation it marches in the k-direction
//

extern Real *d_u1, *d_u2;

__global__ void kernel_laplace3d(int NX, int NY, int NZ, Real *d_u1, Real *d_u2)
{
  int   i, j, k, bz, ks, ke, indg, active, IOFF, JOFF, KOFF;
  Real u2, sixth=1.0/6.0;

  //
  // define global indices and array offsets
  //

  // number of blocks that cover the grid in y dir
  int nby =  1 + (NY-1) / blockDim.y;
  // thickness in z direction
  bz = 1 + (NZ-1) / (1 + (gridDim.y - 1) / nby); 

  i    = threadIdx.x + blockIdx.x * blockDim.x;
  j    = threadIdx.y + (blockIdx.y % nby) * blockDim.y;
  ks   =  (blockIdx.y / nby) * bz;
  //j    = threadIdx.y + blockIdx.y * blockDim.y;
  indg = i + j*NX + ks*NX*NY;

  IOFF = 1;
  JOFF = NX;
  KOFF = NX*NY;

  active = i>=0 && i<=NX-1 && j>=0 && j<=NY-1;

  ke = ( ks+bz > NZ ? NZ : ks+bz);
  for (k=ks; k<ke; k++) {

    if (active) {
      if (i==0 || i==NX-1 || j==0 || j==NY-1 || k==0 || k==NZ-1) {
        u2 = d_u1[indg];  // Dirichlet b.c.'s
      }
      else {
        u2 = ( d_u1[indg-IOFF] + d_u1[indg+IOFF]
             + d_u1[indg-JOFF] + d_u1[indg+JOFF]
             + d_u1[indg-KOFF] + d_u1[indg+KOFF] ) * sixth;
      }
      d_u2[indg] = u2;

      indg += KOFF;
    }
  }
}


__global__ void kernel_laplace3d_shm(int NX, int NY, int NZ, Real *d_u1, Real *d_u2)
{
  extern __shared__ Real plane[];
  int   i, j, k, indg, active, halo, indp, IOFF, JOFF, KOFF;
  Real u2, sixth=1.0/6.0;

  //
  // define global indices and array offsets
  //

  i    = threadIdx.x - 1 + blockIdx.x * (blockDim.x - 2);
  j    = threadIdx.y - 1 + blockIdx.y * (blockDim.y - 2);
  indg = i + j*NX;
  indp = threadIdx.x + blockDim.x * threadIdx.y;

  IOFF = 1;
  JOFF = blockDim.x;//for plane layout
  KOFF = NX*NY;

  active = i>=0 && i<=NX-1 && j>=0 && j<=NY-1;

  halo = threadIdx.x == 0 || threadIdx.x == blockDim.x - 1 ||
    threadIdx.y == 0 || threadIdx.y == blockDim.y - 1;

// populate plane with first layer
//  if(active)
//     plane[indp] = d_u1[indg+KOFF];  
//  __syncthreads();	

  for (k=0; k<NZ; k++) {
     __syncthreads();
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

      indg += KOFF;
    }
  }
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
  
  dim3 dimGrid(gridparams[0],gridparams[1]);
  dim3 dimBlock(gridparams[2],gridparams[3]);
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

  cudaEventRecord(compStart,0);
  
  switch(kernel_key)
    {
    case(GPUBASE_KERNEL):
      for (iter = 0; iter < iter_block; ++iter){
	kernel_laplace3d<<<dimGrid, dimBlock>>>(NX, NY, NZ, d_u1, d_u2);
    	  aux=d_u1; d_u1=d_u2; d_u2=aux;
      }
      break;
    case(GPUSHM_KERNEL):
      shmsize=gridparams[2]*gridparams[3]*sizeof(Real);
      for (iter = 0; iter < iter_block; ++iter){
	kernel_laplace3d_shm<<<dimGrid, dimBlock, shmsize>>>(NX, NY, NZ, d_u1, d_u2);
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
    case (GPUSHM_KERNEL) :
      gridsize[2] = blockXsize + 2; // halo
      gridsize[3] = blockYsize + 2;
      gridsize[0] = 1 + (NX-1)/blockXsize;
      gridsize[1] = 1 + (NY-1)/blockYsize;
      break;
    case(GPUBASE_KERNEL):
      gridsize[0] = 1 + (NX-1)/blockXsize;
      gridsize[1] = (1 + (NY-1)/blockYsize) * (1 + (NZ-1) / blockZsize);
      gridsize[2] = blockXsize;
      gridsize[3] = blockYsize;
      break;
    default:
      fprintf(stderr,"unkwon gpu kernel in calcGpuDims, quitting ...!");
      exit (1);
    }
  // Needs to test if the blocks and grid sizes are meaningful 
}

 
