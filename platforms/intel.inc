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

LANG = C

ifdef USE_MIC
  FMIC = -mmic 
# -no-opt-prefetch
endif

ifdef USE_MPI
  FMPI=-DUSE_MPI
endif

ifdef USE_DOUBLE_PRECISION
  DOUBLE=-DUSE_DOUBLE_PRECISION
endif

USE_VEC1D := on
ifdef USE_VEC1D
  FVEC1D := -DUSE_VEC1D
endif

#FC = module add intel/comp intel/mpi && mpiifort

COMP := intel

USE_OPENMP = yes

FFLAGS_intel_opt   := -O3 $(FMIC) -opt-streaming-stores auto 
FFLAGS_intel_debug  :=  -g -C  $(FMIC)  -fpe0 -check noarg_temp_created

CFLAGS_intel_opt := $(FVEC1D) $(FMPI) $(DOUBLE) -O3 -restrict -align -vec-report6 $(FMIC)
CFLAGS_intel_debug := $(FVEC1D) $(FMPI) $(DOUBLE) -g -restrict $(FMIC)  

MPIFLAGS         :=
OMPFLAGS_intel_yes := -openmp
OMPFLAGS_intel_no  := 

FFLAGS = $(FFLAGS_$(COMP)_$(BUILD)) 
CFLAGS := $(CFLAGS_$(COMP)_$(BUILD))
OMPFLAGS = $(OMPFLAGS_$(COMP)_$(USE_OPENMP))
