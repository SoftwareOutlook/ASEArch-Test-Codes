#============================================================
# Makefile configuration macros for cygwin + GNU compiler
#    
# Lucian Anton 
# created: 6/06/11
#============================================================

CC = mpicc
MPIFLAGS = 
OMPFLAGS = -fopenmp
OPTFLAGS = -O3

F90 := mpif90
OMPFLAGS := -fopenmp
FFLAGS_opt := -O3
FFLAGS_debug := -g -fbounds-check -Wall -fbacktrace -ffpe-trap=invalid,zero,overflow -fsignaling-nans -fdump-core 
FFLAGS := $(FFLAGS_$(BUILD))