#============================================================
# Makefile variable for  intel mpi
#    
# Lucian Anton 
# created: 5/06/12
#============================================================

F90 := source /opt/intel/composerxe/bin/compilervars.sh intel64 && \
      source /opt/intel/impi/4.1.0/intel64/bin/mpivars.sh && mpiifort

CC := source /opt/intel/composerxe/bin/compilervars.sh intel64 && \
      source /opt/intel/impi/4.1.0/intel64/bin/mpivars.sh && mpiicc


ifdef USE_MIC
  FMIC = -mmic
endif

ifdef USE_MPI
  FMPI=-DUSE_MPI
endif

ifdef USE_DOUBLE_PRECISION
  DOUBLE=-DUSE_DOUBLE_PRECISION
endif

#FC = module add intel/comp intel/mpi && mpiifort

COMP := intel

USE_OPENMP = yes

FFLAGS_intel_opt   := -O3 $(FMIC) -opt-streaming-stores auto
FFLAGS_intel_debug  :=  -g -C  $(FMIC)  -fpe0 -check noarg_temp_created

CFLAGS_intel_opt := $(FMPI) $(DOUBLE) -O3 $(FMIC)
CFLAGS_intel_debug := $(FMPI) $(DOUBLE) -g $(FMIC)  

MPIFLAGS         :=
OMPFLAGS_intel_yes := -openmp
OMPFLAGS_intel_no  := 

FFLAGS = $(FFLAGS_$(COMP)_$(BUILD)) 
CFLAGS := $(CFLAGS_$(COMP)_$(BUILD))
OMPFLAGS = $(OMPFLAGS_$(COMP)_$(USE_OPENMP))
