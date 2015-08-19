#============================================================
# Makefile variable for  intel mpi
#    
# Lucian Anton 
# created: 5/06/12
#============================================================

#F90 := source /opt/intel/composerxe/bin/compilervars.sh intel64 && \
#      source /opt/intel/impi/4.1.0/intel64/bin/mpivars.sh && mpiifort

#CC := source /opt/intel/composerxe/bin/compilervars.sh intel64 && \
#      source /opt/intel/impi/4.1.0/intel64/bin/mpivars.sh && mpiicc

F90 = ifort
CC = icc

LANG = C

USE_OPENMP=yes
BUILD=opt

ifdef USE_MIC
  FMIC = -mmic 
# -no-opt-prefetch
endif

ifdef USE_MPI
  CPPFLAGS := -DUSE_MPI
  CC := mpiicc
  F90 := mpiifort
endif

ifdef USE_DOUBLE_PRECISION
  CPPFLAGS += -DUSE_DOUBLE_PRECISION
endif

#ifdef USE_FORTRAN
#  FORTRAN := -DUSE_FORTRAN
#  LIB = -lifcore -limf
#endif

USE_VEC1D := on
ifdef USE_VEC1D
  CPPFLAGS += -DUSE_VEC1D
endif

#FC = module add intel/comp intel/mpi && mpiifort

COMP := intel

#USE_OPENMP = yes

FFLAGS_opt   := -O3 $(FMIC) -opt-streaming-stores auto
FFLAGS_debug  :=  -g -C $(FMIC)  -fpe0 -check noarg_temp_created
FFLAGS := $(CPPFLAGS) $(FFLAGS_$(BUILD))

CFLAGS_opt := -O3 $(FMIC) -restrict -align -vec-report6 -free
CFLAGS_debug :=  -g -restrict $(FMIC)
CFLAGS := $(CPPFLAGS) $(CFLAGS_$(BUILD))

##MPIFLAGS_intel_0  :=
##MPIFLAGS_intel_1 := -mt_mpi
ifdef USE_OPENMP
  OMPFLAGS := -openmp
endif
#OMPFLAGS_intel_no  := 

##FFLAGS = $(FFLAGS_$(COMP)_$(BUILD)) 
##CFLAGS := $(CFLAGS_$(COMP)_$(BUILD))
##OMPFLAGS = $(OMPFLAGS_$(COMP)_$(USE_OPENMP))
##MPIFLAGS = $(MPIFLAGS_$(COMP)_$(USE_MPI))

ifeq ($(LANG), C)
  LIB := -lifcore -limf
endif

