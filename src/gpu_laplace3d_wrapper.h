/**
 * this file is a wrapper file which defines the functions to be invoked on the GPU
 * this file is used to elimate the linkage error while compiling using two different compilers
 * this file is created to avoid the name mangling problems occurring while compilation
 * since we are using gcc for all the programs except cuda files with *.cu extention
 * this problem could have been fixed by simply using g++ compiler instead of C however
 * due to code implementation in C it made obvious to use GCC
 */
#ifndef LAPLACE3D_GPU_h
#define LAPLACE3D_GPU_h

void laplace3d_GPU(const int kernel_key, Real* uOld,int NX,int NY,int NZ,const int* gridparams, int iteration_block, float *compTime, float *commTime);
void CudaDeviceInit();
void setInitialData(float* device,float* host,int NX,int NY,int NZ,float* memoryTimer,int* memoryCtr);
void getUpdatedArray(float* host,float* dev,int NX,int NY,int NZ,float* memoryTimer,int* memoryCtr);
void GPU_Titanium_Laplace3d(Real* dev1, Real* dev2,int NX, int NY, int NZ,const int* gridparams,float* kernelTimer);
void GPU_Titanium_Shmem(Real* dev1, Real* dev2, int NX, int NY, int NZ,float* kernelTimer);
void GPU_Blocked_Laplace3d(Real* dev1, Real* dev2,int NX,int NY,int NZ,int BX,int BY,int BZ);
void calcGpuDims(int blockXsize,int blockYsize,int* gridsize,int NX,int NY, int kernel_key);
#endif
