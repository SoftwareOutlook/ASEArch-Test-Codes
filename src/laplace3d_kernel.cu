#include<stdio.h>
#include "cutil_inline.h"


//#include<cuda.h>
//#include<cuda_runtime.h>

/** 
 * this notifies the compiler that the definitions are present in external file
 */
extern "C" {
#include "homb_c.h"
#include "gpu_laplace3d_wrapper.h"
}

//
// Notes:one thread per node in the 2D block;
// after initialisation it marches in the k-direction
//

/**
 * preprocessor directives for GPU values, it will be use
 * while invoking GPU invocation of laplace3d
 * these values simple specify the block sizes 
 */
#define BLOCK_X 32
#define BLOCK_Y 4
#define SH3D 8
#define U2SIZE 4
#define SHMEM 16

extern Real *d_u1, *d_u2;

__global__ void kernel_laplace3d(int NX, int NY, int NZ, Real *d_u1, Real *d_u2)
{
  int   i, j, k, indg, active, IOFF, JOFF, KOFF;
  Real u2, sixth=1.0/6.0;

  //
  // define global indices and array offsets
  //

  i    = threadIdx.x + blockIdx.x * blockDim.x;
  j    = threadIdx.y + blockIdx.y * blockDim.y;
  indg = i + j*NX;

  IOFF = 1;
  JOFF = NX;
  KOFF = NX*NY;

  active = i>=0 && i<=NX-1 && j>=0 && j<=NY-1;

  for (k=0; k<NZ; k++) {

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


//////////////////////////////////////////////////////////////////////////////////////////////

/**
 * this kernel is GPU implementation of Titanium laplace 3d
 * this kernel eliminated branch conditions and utilises local memory caching on GPU
 */

__global__ void kernel_Titanium_laplace3d(int NX, int NY, int NZ, Real *d_u1, Real *d_u2)
{
	int ii,jj,kk,index,mjoff,pjoff,mkoff,pkoff,incx,incy,NXY;

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	incx = gridDim.x * blockDim.x;
	incy = gridDim.x * blockDim.y;
	NXY = NX * NY;
	Real sixth=1.0/6.0;

	for(kk=i+1;kk<NZ-1;kk+=incx)
	{
		for(jj=j+1;jj<NY-1;jj+=incy)
		{
			index = jj*NX + kk * NX * NY;
			mjoff = index - NX;
			pjoff = index + NX;
			mkoff = index - NXY;
			pkoff = index + NXY;

			for(ii=1;ii<NX-1;ii++)
			{
				int indoff = index + ii;
				d_u2[index+ii] = (d_u1[indoff-1] + d_u1[indoff+1]+d_u1[mjoff+ii]+d_u1[pjoff+ii]+d_u1[mkoff+ii]+d_u1[pkoff+ii]) * sixth;
			}
		}
	}
}
////////////////////////////////////////////////////////////////////////////////

/**
 * this kernel is shared memory optimized version of laplace 3d
 */

__global__ void kernel_Titanium_shmem(int NX, int NY, int NZ, Real *d_u1, Real *d_u2)
{

	extern __shared__ float shu1[];
	__shared__ float shArr3d[SH3D];
	__shared__ float shu2[U2SIZE];


	unsigned int si,sj,tidx,tidy,kk,jj,i,j,ind,index,halfbd,t,t1,t2,tidoffset,mkoff,pkoff;
	unsigned int NXY = NX * NY;
	unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int dimx = gridDim.x * blockDim.x;
	unsigned int dimy = gridDim.y * blockDim.y;
	tidx = threadIdx.x;
	tidy = threadIdx.y;
	Real sixth = 1.0/6.0;

	//copy data from global memory to shared memory
	// here we have copied the array in 2D form into shared memory shu1
	// and further the 3d access values into shared memory shArr3d
	//*NOTE: add ones	
	for(si=x;si<NX;si+=dimx)
	{
		for(sj=y;sj<NY;sj+=dimy)
		{
			ind = si*blockDim.x+sj;
			shu1[tidx*blockDim.x+tidy] = d_u1[ind];
		}
	}


	__syncthreads();

	for(kk=x+1;kk<NZ-1;kk+=dimx){
		for(jj=y+1;jj<NY-1;jj+=dimy){
			ind = jj*NX + kk * NXY;
			mkoff = ind - NXY;
			pkoff = ind + NXY;
			
			for(i=threadIdx.x+1;i<blockDim.x-1;i+=blockDim.x){
				for(j=threadIdx.y+1;j<blockDim.y-1;j+=blockDim.y){
					t1 = j-1;
					t = j && 1;
					t1 += t;
					t2 = t1 + 1;
//					printf("mkoff %d %d ij %d %d\n",mkoff,pkoff,i,j);
					shArr3d[threadIdx.x*blockDim.x+t1] = d_u1[mkoff+j];
					shArr3d[threadIdx.x*blockDim.x+t2] = d_u1[pkoff+j];
				}
			}
		}
	}
	__syncthreads();


	//begin processing the laplace formula on shared memory data
	// copied from global memory in code block above.
	halfbd = blockDim.x/2;
	for(kk=threadIdx.x+1;kk<blockDim.x-1;kk+=blockDim.x){
		unsigned int bd = blockDim.x;
        index = kk*bd;
		unsigned int mjoff = index - bd;
		unsigned int pjoff = index + bd;
		tidoffset = 2 * (threadIdx.x && 1);

		for(jj=threadIdx.y+1;jj<blockDim.y-1;jj+=blockDim.y){
			t = jj - 1;
			t2 = t && jj;
			t += t2;
			t1 = t2 + 1;
			unsigned int indoff = index + jj;

			shu2[threadIdx.x*halfbd+threadIdx.y] = (shu1[indoff-1] + shu1[indoff+1] +
					shu1[mjoff+jj] + shu1[pjoff+jj] +
					shArr3d[tidoffset*halfbd+t] + shArr3d[tidoffset*halfbd+t1]) * sixth;
		}//end of jj
	}//end of kk

	__syncthreads();
	//now copy the updated shared memory data to global memory

	for(kk=threadIdx.x+1;kk<blockDim.x-1;kk+=blockDim.x){
		for(jj=threadIdx.y+1;jj<blockDim.y-1;jj+=blockDim.y){
			index = (x+1)*blockDim.y+(y+1);
			d_u2[index] = shu2[kk*blockDim.y+jj];
		}
	}

}
///////////////////////////////////////////////////////////////////////////////

/**
 * BLOCKED LAPLACE3D GPU implementation
 * This function is GPU implemenation of equivalent CPU implementation of blocked
 * laplace3d
 */
/*

__global__ void kernel_blocked_laplace3d(float* dev1, float* dev2, int NX,int NY,int NZ,int BX,int BY,int BZ)
{
	unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int kk,jj,ii,k,j,i;
	int ind,offju,offjd,offku,offkd;
	int NXY = NX * NY;
	unsigned int incr = blockDim.x * gridDim.x;
	float sixth = (float) 1.0f/6.0f;

	for(kk=x+1;kk<NZ-1;kk+=(1+BZ)){
		for(jj=1;jj<NY-1;jj+=BY){
			for(ii=1;ii<NX-1;ii+=BX){
				for(k=kk; k<min(kk+BZ,NZ-1); k++){
					for(j=jj; j<min(jj+BY,NY-1); j++){
						ind = j*NX + k*NXY;
						offju = ind - NX;
						offjd = ind + NX;
						offku = ind - NXY;				
						offkd = ind - NXY;	
						
						for(i=ii;i<min(ii+BX,NX-1);i++){
							dev2[ind+i] = (dev1[ind+i-1] + dev1[ind+i+1]
									+ dev1[offju+i] + dev1[offjd+i]
									+ dev1[offku+i] + dev1[offkd+i]) * sixth;
						}			
					}//j ends
				}//k ends
			}//end of ii
		}//end of jj loop
	}//end of kk loop
}
*/
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
      /*
      int gridx = (NX + threadperblock - 1)/threadperblock;
      int gridy = (NY + threadperblock - 1)/threadperblock;
      dim3 dimBlock(threadperblock,threadperblock);
      dim3 dimGrid(gridx,gridy);
      int sharedsize = threadperblock*threadperblock * sizeof(Real);
      */
      shmsize=gridparams[2]*gridparams[3]*sizeof(Real);
      for (iter = 0; iter < iter_block; ++iter){
	kernel_laplace3d_shm<<<dimGrid, dimBlock, shmsize>>>(NX, NY, NZ, d_u1, d_u2);
	  //kernel_Titanium_shmem<<<dimGrid, dimBlock,sharedsize>>>(NX, NY, NZ, d_u1, d_u2);
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
////////////////////////////////////////////////////////////////////

extern "C"
void GPU_Titanium_Laplace3d(Real* dev1, Real* dev2, int NX,int NY,int NZ,const int* gridparams,float* kernelTimer)
{
	/*int tpb = 4;
	int gridx = (NX + tpb -1)/tpb;
	int gridy = (NY + tpb -1)/tpb;*/
	dim3 dimGrid(gridparams[0],gridparams[1]);
	dim3 dimBlock(gridparams[2],gridparams[2]);
	//event timer
	cudaEvent_t start,stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
  //start the timer
	cudaEventRecord(start,0);
	kernel_Titanium_laplace3d<<<dimGrid,dimBlock>>>(NX,NY,NZ,dev1,dev2);
	cudaEventRecord(stop,0);
	//cudaDeviceSynchronize();
	cudaEventSynchronize(stop);
	float elapsedTime = 0.0f;
	cudaEventElapsedTime(&elapsedTime,start,stop);
	*kernelTimer += elapsedTime;
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
}
////////////////////////////////////////////////////////////////////

extern "C"
void GPU_Titanium_Shmem(Real* dev1, Real* dev2, int NX, int NY, int NZ,float* kernelTimer)
{
	int threadperblock = 4;
	int gridx = (NX + threadperblock - 1)/threadperblock;
	int gridy = (NY + threadperblock - 1)/threadperblock;
	dim3 dimBlock(threadperblock,threadperblock);
	dim3 dimGrid(gridx,gridy);
	int sharedsize = threadperblock*threadperblock * sizeof(float);

	cudaEvent_t start,stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
  //start the timer
	cudaEventRecord(start,0);
	kernel_Titanium_shmem<<<dimGrid,dimBlock,sharedsize>>>(NX,NY,NZ,dev1,dev2);
	cudaEventRecord(stop,0);
	//cudaDeviceSynchronize();
	cudaEventSynchronize(stop);
	float elapsedTime = 0.0f;
	cudaEventElapsedTime(&elapsedTime,start,stop);
	*kernelTimer += elapsedTime;
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
}
///////////////////////////////////////////////////////////////

/*
extern "C"
void GPU_Blocked_Laplace3d(float* dev1, float* dev2,int NX,int NY,int NZ,int BX,int BY,int BZ){
//	int tb = 1;
//	int gridsize = (NX + tb - 1)/tb;
//	kernel_blocked_laplace3d<<<1,tb>>>(dev1,dev2,NX,NY,NZ,2,2,2);
//	cudaDeviceSynchronize();
}
*/
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
void calcGpuDims(int blockXsize,int blockYsize,int* gridsize,int NX,int NY, int kernel_key)
{
  if (kernel_key == GPUSHM_KERNEL){
    gridsize[2] = blockXsize + 2; // halo
    gridsize[3] = blockYsize + 2;
    gridsize[0] = 1 + (NX-1)/blockXsize;
    gridsize[1] = 1 + (NY-1)/blockYsize;

  }
  else{
	gridsize[0] = 1 + (NX-1)/blockXsize;
	gridsize[1] = 1 + (NY-1)/blockYsize;
	gridsize[2] = blockXsize;
	gridsize[3] = blockYsize;
  }
}

 
