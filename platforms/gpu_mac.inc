#============================================================
# Makefile configuration for GNU compiler, MPICH2 + OpenMP  
#    
# Lucian Anton 
# created: 08/13
#============================================================

LANG := C

COMP=gcc

BUILD=opt

USE_OPENMP=yes
USE_GPU=yes
CUDAO=laplace3d_kernel.o

ifdef USE_MPI
  CC  := mpicc
  F90 := mpif90
  CPPFLAGS := -DUSE_MPI
else
  CC  := gcc
  F90 := gfortran
endif

NVCC := nvcc
CPPFLAGS := -DUSE_GPU
INC	:= -I/usr/local/cuda/include 
LIB	:= -L/usr/local/cuda/lib -lcuda -lcudart -lstdc++

ifdef USE_DOUBLE_PRECISION
  CPPFLAGS += -DUSE_DOUBLE_PRECISION
  PNVFLAGS += -DUSE_DOUBLE_PRECISION
endif

CFLAGS_opt = -O3  
CFLAGS_debug = -g -Wall -Wuninitialized -O1 
CFLAGS := $(CPPFLAGS) $(CFLAGS_$(BUILD)) -std=c99
#CFLAGS := $(CPPFLAGS) $(CFLAGS_debug) 
NVCCFLAGS := $(PNVFLAGS) --ptxas-options=-v --use_fast_math
#-arch=sm_30

FFLAGS_opt := -O3
FFLAGS_debug := -g -fbounds-check -Wall -fbacktrace -ffpe-trap=invalid,zero,overflow -fsignaling-nans -fdump-core 
FFLAGS := $(CPPFLAGS) $(FFLAGS_$(BUILD))

ifdef USE_OPENMP
  OMPFLAGS := -fopenmp 
endif
