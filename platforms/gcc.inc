
#============================================================
# Makefile configuration for GNU compiler, MPICH2 + OpenMP  
#    
# Lucian Anton 
# created: 08/13
#============================================================

LANG := C

COMP=gcc

USE_OPENMP=yes
BUILD=opt
##USE_FORTRAN=no

ifdef USE_MPI
  CC  := mpicc
  F90 := mpif90
  CPPFLAGS := -DUSE_MPI
else
  CC  := gcc
  F90 := gfortran
endif

ifdef USE_CUDA
	CPPFLAGS += -DUSE_CUDA
	INC	:= -I$(CUDADIR)/include 
	LIB	:= -L$(CUDADIR)/lib64 -lcuda -lcudart
	CUDAO := laplace3d_kernel.o
	NVCCFLAGS_opt := -O3 -gencode arch=compute_30,code=sm_30 
	NVCCFLAGS_debug := -g -G -O0 -gencode arch=compute_30,code=sm_30 
	NVCCFLAGS := $(NVCCFLAGS_$(BUILD))
	#-keep --ptxas-options=-v
endif

ifdef USE_DOUBLE_PRECISION
  CPPFLAGS += -DUSE_DOUBLE_PRECISION
endif

USE_VEC1D := on
ifdef USE_VEC1D
  CPPFLAGS += -DUSE_VEC1D
endif

##USE_FORTRAN := off

##ifdef USE_FORTRAN
##  CPPFLAGS += -DUSE_FORTRAN
##endif

CFLAGS_opt = -O3
CFLAGS_debug = -gdwarf-3 -Wall -Wuninitialized -O1
CFLAGS := $(CPPFLAGS) $(CFLAGS_$(BUILD)) -std=c99 

FFLAGS_opt := -O3
FFLAGS_debug := -gdwarf-3 -fbounds-check -Wall -fbacktrace -ffpe-trap=invalid,zero,overflow -fsignaling-nans -fdump-core 
FFLAGS := $(CPPFLAGS) $(FFLAGS_$(BUILD))

ifdef USE_OPENMP
  OMPFLAGS := -fopenmp
endif

ifeq ($(LANG),C)
  LIB := -lm -lgfortran
endif

#ifdef USE_FORTRAN
#  LIB += -lgfortran
#endif
