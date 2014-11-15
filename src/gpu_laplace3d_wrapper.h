/**
 * this file is a wrapper file which defines the functions to be invoked on the GPU
 * this file is used to elimate the linkage error while compiling using two different compilers
 * this file is created to avoid the name mangling problems occurring while compilation
 * since we are using gcc for all the programs except cuda files with *.cu extention
 * this problem could have been fixed by simply using g++ compiler instead of C however
 * due to code implementation in C it made obvious to use GCC

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
#ifndef LAPLACE3D_CUDA_h
#define LAPLACE3D_CUDA_h

void laplace3d_GPU(const int kernel_key, Real* uOld,int NX,int NY,int NZ,const int* gridparams, int iteration_block, float *compTime, float *commTime);
void CudaDeviceInit();
void setInitialData(float* device,float* host,int NX,int NY,int NZ,float* memoryTimer,int* memoryCtr);
void getUpdatedArray(float* host,float* dev,int NX,int NY,int NZ,float* memoryTimer,int* memoryCtr);
void GPU_Titanium_Laplace3d(Real* dev1, Real* dev2,int NX, int NY, int NZ,const int* gridparams,float* kernelTimer);
void GPU_Titanium_Shmem(Real* dev1, Real* dev2, int NX, int NY, int NZ,float* kernelTimer);
void GPU_Blocked_Laplace3d(Real* dev1, Real* dev2,int NX,int NY,int NZ,int BX,int BY,int BZ);
void calcGpuDims(int kernel_key, int blockXsize, int blockYsize, int blockZsize, int NX, int NY, int NZ, int* gridsize);
#endif
