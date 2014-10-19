#============================================================
# Makefile variable for  intel mpi
#    
# Lucian Anton 
# created: 5/06/12
#============================================================

COMP := intel
LANG = C

F90 := ifort
CC := icc 

ifdef USE_MIC
  FMIC = -mmic
endif

ifdef USE_MPI
  FMPI=-DUSE_MPI
endif

ifdef USE_DOUBLE_PRECISION
  DOUBLE=-DUSE_DOUBLE_PRECISION
endif

USE_VEC1D := on

ifdef USE_VEC1D
  VEC1D := -DUSE_VEC1D
endif

USE_OPENMP = yes

FFLAGS_intel_opt   := -O3 $(FMIC) vec-report6 -opt-streaming-stores auto 
FFLAGS_intel_debug  :=  -g -C  $(FMIC)  -fpe0 -check noarg_temp_created

CFLAGS_intel_opt := $(VEC1D) $(FMPI) $(DOUBLE) -O3 -restrict -align -vec-report6 $(FMIC) -xHost -opt-streaming-stores never
CFLAGS_intel_debug := $(VEC1D) $(FMPI) $(DOUBLE) -g -restrict $(FMIC)  

MPIFLAGS         :=
OMPFLAGS_intel_yes := -openmp
OMPFLAGS_intel_no  := 

FFLAGS = $(FFLAGS_$(COMP)_$(BUILD)) 
CFLAGS := $(CFLAGS_$(COMP)_$(BUILD))
OMPFLAGS = $(OMPFLAGS_$(COMP)_$(USE_OPENMP))
