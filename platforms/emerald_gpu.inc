#============================================================
# Makefile configuration for GNU compiler, MPICH2 + OpenMP  
#    
# Lucian Anton 
# created: 08/13
#============================================================

LANG := C

BUILD := opt

COMP := nvcc

USE_OPENMP := yes
USE_CUDA := yes
CUDAO=kernels_cuda.o

# upgrade arch flag for emerald keples cards
# needs cuda/6.0.37 or newer for kepler

CUDA_ARCH := -arch=sm_20
ifdef CUDA_KEPLER
  CUDA_ARCH := -arch=sm_35
endif

ifdef USE_MPI
  CC  := mpicc
  F90 := mpif90
  CPPFLAGS := -DUSE_MPI
else
  CC  := gcc
  F90 := gfortran
endif

# this is for CPU
ifdef USE_VEC1D
  CPPFLAGS += -DUSE_VEC1D
endif

NVCC := $(CUDA_HOME)/bin/nvcc
CPPFLAGS += -DUSE_CUDA
INC	:= -I$(CUDA_HOME)/include

LIB	:= -L$(CUDA_HOME)/lib64 -lcuda -lstdc++ -lcudart

ifdef USE_DOUBLE_PRECISION
  CPPFLAGS += -DUSE_DOUBLE_PRECISION
  PNVFLAGS += -DUSE_DOUBLE_PRECISION	
endif

CFLAGS_opt = -O3  
CFLAGS_debug = -g -Wall -Wuninitialized -O1 
CFLAGS := $(CPPFLAGS) $(CFLAGS_$(BUILD)) -std=c99
#CFLAGS := $(CPPFLAGS) $(CFLAGS_debug) 
NVCCFLAGS := $(PNVFLAGS)  $(GPU_ARCH) --ptxas-options=-v --use_fast_math

FFLAGS_opt := -O3
FFLAGS_debug := -g -fbounds-check -Wall -fbacktrace -ffpe-trap=invalid,zero,overflow -fsignaling-nans -fdump-core 
FFLAGS := $(CPPFLAGS) $(FFLAGS_$(BUILD))

ifdef USE_OPENMP
  OMPFLAGS := -fopenmp 
endif
