#============================================================
# Makefile configuration for GNU compiler, MPICH2 + OpenMP only 
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
  FMPI = -DUSE_MPI
else
  CC  := gcc
  F90 := gfortran
endif

CFLAGS_opt = -O3
CFLAGS_debug = -g -Wall -Wuninitialized -O1
CFLAGS := $(FMPI) $(CFLAGS_$(BUILD)) 

FFLAGS_opt := -O3
FFLAGS_debug := -g -fbounds-check -Wall -fbacktrace -ffpe-trap=invalid,zero,overflow -fsignaling-nans -fdump-core 
FFLAGS := $(FFLAGS_$(BUILD))

ifdef USE_OPENMP
  OMPFLAGS := -fopenmp
endif