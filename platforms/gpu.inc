#============================================================
# Makefile configuration for GNU compiler, MPICH2 + OpenMP  
#    
# Lucian Anton 
# created: 08/13
#============================================================

LANG := C

COMP=gcc


USE_OPENMP=yes

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
INC	:= -I/usr/local/cuda-5.0/include 
LIB	:= -L/usr/local/cuda-5.0/lib64 -lcuda -lcudart

ifdef USE_DOUBLE_PRECISION
  CPPFLAGS += -DUSE_DOUBLE_PRECISION
endif

CFLAGS_opt = -O3  
CFLAGS_debug = -g -Wall -Wuninitialized -O1 
CFLAGS := $(CPPFLAGS) $(CFLAGS_$(BUILD)) 
#CFLAGS := $(CPPFLAGS) $(CFLAGS_debug) 
NVCCFLAGS := -arch=sm_30 --ptxas-options=-v --use_fast_math

FFLAGS_opt := -O3
FFLAGS_debug := -g -fbounds-check -Wall -fbacktrace -ffpe-trap=invalid,zero,overflow -fsignaling-nans -fdump-core 
FFLAGS := $(CPPFLAGS) $(FFLAGS_$(BUILD))

ifdef USE_OPENMP
  OMPFLAGS := -fopenmp 
endif
