#============================================================
# Makefile settings for IDataPlex Blue Wonder (Daresbury) 
#    
# Lucian Anton 
# created: 5/06/12
#============================================================

CC := module add intel/comp && icc 

LANG = C
BUILD = opt
USE_OPENMP = yes 

ifdef USE_MPI
  FMPI = -DUSE_MPI
endif

ifdef USE_DOUBLE_PRECISION
  DOUBLE = -DUSE_DOUBLE_PRECISION
endif

USE_VEC1D := on
ifdef USE_VEC1D
  VEC1D = -DUSE_VEC1D
endif

#FC = module add intel/comp intel/mpi && mpiifort

COMP := intel

USE_OPENMP = yes

FFLAGS_intel_opt   := -O3 $(FMIC) -opt-streaming-stores auto 
FFLAGS_intel_debug  :=  -g -C  $(FMIC)  -fpe0 -check noarg_temp_created

CFLAGS_intel_opt := $(VEC1D) $(FMPI) $(DOUBLE) -O3 -xHost -restrict -align -vec-report6 $(FMIC) -opt-streaming-stores auto
CFLAGS_intel_debug := $(VEC1D) $(FMPI) $(DOUBLE) -g $(FMIC)  

MPIFLAGS         :=
OMPFLAGS_intel_yes := -openmp
OMPFLAGS_intel_no  := 

FFLAGS = $(FFLAGS_$(COMP)_$(BUILD)) 
CFLAGS := $(CFLAGS_$(COMP)_$(BUILD))
OMPFLAGS = $(OMPFLAGS_$(COMP)_$(USE_OPENMP))
