#============================================================
# Makefile variable for  intel mpi
#    
# Lucian Anton 
# created: 5/06/12
#============================================================

F90 := source /opt/intel/composerxe/bin/compilervars.sh intel64 && \
      source /opt/intel/impi/4.1.0/intel64/bin/mpivars.sh && mpiifort

CC := source /opt/intel/composerxe/bin/compilervars.sh intel64 && \
      source /opt/intel/impi/4.1.0/intel64/bin/mpivars.sh && icc

LANG = C

ifdef USE_MIC
  FMIC = -mmic
endif

ifdef USE_MPI
  FMPI=-DUSE_MPI
endif

ifdef USE_DOUBLE_PRECISION
  DOUBLE=-DUSE_DOUBLE_PRECISION
endif

ifdef USE_VEC1D
  VEC1D = -DUSE_VEC1D
endif

#FC = module add intel/comp intel/mpi && mpiifort

COMP := intel

USE_OPENMP = yes

FFLAGS_intel_opt   := -O3 $(FMIC) -opt-streaming-stores auto 
FFLAGS_intel_debug  :=  -g -C  $(FMIC)  -fpe0 -check noarg_temp_created

CFLAGS_intel_opt := $(VEC1D) $(FMPI) $(DOUBLE) -O3 -xHost -align -vec-report6 $(FMIC)
CFLAGS_intel_debug := $(VEC1D) $(FMPI) $(DOUBLE) -g $(FMIC)  

MPIFLAGS         :=
OMPFLAGS_intel_yes := -openmp
OMPFLAGS_intel_no  := 

FFLAGS = $(FFLAGS_$(COMP)_$(BUILD)) 
CFLAGS := $(CFLAGS_$(COMP)_$(BUILD))
OMPFLAGS = $(OMPFLAGS_$(COMP)_$(USE_OPENMP))
