/*
This is part of Jacobi Test Code (JTC) , a hybrid CUDA-OpenMP-MPI benchmark for
Jacobi solver applied to a 3D Laplace equation.

Lucian Anton
December 2014.

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

#ifndef JACOBI_C
#define JACOBI_C

#define JTC_VERSION "1.1.1 (16th of March 2015)"

/* Include Files */ 
#ifdef USE_MPI
#include <mpi.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <float.h>
#include <sys/types.h>

#define PI 3.14159265358979323846
#define MAX(x,y) (((x) > (y))? x : y)
#define MIN(x,y) (((x) > (y))? y : x)

#ifdef USE_DOUBLE_PRECISION
typedef double Real;
  #ifdef USE_MPI
    #define REAL_MPI MPI_DOUBLE
  #endif
#else
typedef float Real;
  #ifdef USE_MPI
    #define REAL_MPI MPI_FLOAT
  #endif
#endif

#define MAXNAME 31

// struc to keep grid data
struct grid_info_t
{
  int ng[3], nb[3]; // global grid and computational blocks
  int threads_per_column; // number of threads per column in wave algorithm
  int sx, ex, sy, ey, sz, ez, nlx, nly, nlz; // start end indices for local grids ( used with MPI runs)
  int nproc, myrank; // MPI rank
  int np[3]; // MPI topology size
  int cp[3]; // grid coordinated of this ranks
  int cmod_key;	// language identifier
  char cmod_name[MAXNAME+1]; // language name
  int alg_key; // algorithm indentifier
  char alg_name[MAXNAME+1]; // algorithm name
  int malign; // allocate aligned memory to help vectorization
  int nwaves; // number of waves used in time skwed algorithm
#ifdef USE_MPI
  MPI_Comm comm;
#endif 
}; 

struct times_t
{
  double comp, comm;
  // comm is used for GPU <--> host timmings
};

// MPI root or proc 0
#define ROOT 0

//! keys for available languages
#define CMODEL_OMP 100
#define CMODEL_CUDA 200
#define CMODEL_OPENCL 300
#define CMODEL_OPENACC 400
#define CMODEL_MPIOMP 500


//! keys for available algoritms 
//! not all algorithms are available to every language
//! OpenMP kernels
#define ALG_BASELINE 101
#define ALG_BASELINE_OPT  100
#define ALG_BLOCKED  120
#define ALG_CCO      130 
#define ALG_WAVE     140
#define ALG_WAVE_DIAGONAL  141
//! CUDA kernels
#define ALG_CUDA_2D_BLK   201
#define ALG_CUDA_3D_BLK 202
#define ALG_CUDA_SHM   210
#define ALG_CUDA_BANDWIDTH   299
//! OpenCL kernels
#define ALG_OPENCL_BASELINE 300
//! OpenACC kernels
#define ALG_OPENACC_BASELINE 400

//! MPI specific kernels
#define ALG_MPIOMP_CCO 510
#endif
// ifndef JACOBI_C
