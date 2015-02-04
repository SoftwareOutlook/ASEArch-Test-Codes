#============================================================
# Makefile configuration macros for Cray XC30, Cray Intel and GNU compiler
#    
# Lucian Anton 
# created: 5/11/14
#============================================================
#
# Note: the Fortran flags are relicts from older versions.
#       The Fortran code may be upgraded.
#

CC  := cc
F90 := ftn

LANG := C
COMP := nonsense

ifeq ($(PE_ENV),INTEL)
   COMP := intel
endif

ifeq ($(PE_ENV),GNU)
   COMP := gnu
endif

ifeq ($(PE_ENV),CRAY)
   COMP := cray
endif

USE_VEC1D := on

ifdef USE_VEC1D
  VEC1D := -DUSE_VEC1D
endif

ifdef USE_MPI
  FMPI := -DUSE_MPI
endif

USE_DOUBLE_PRECISION := on

ifdef USE_DOUBLE_PRECISION
  DOUBLE = -DUSE_DOUBLE_PRECISION
endif

USE_OPENMP := yes


# GNU compiler
FFLAGS_gnu_opt   := -O3 -fopt-info
FFLAGS_gnu_debug := -g -fbounds-check -Wall -fbacktrace -ffpe-trap=invalid,zero,overflow -fsignaling-nans -fdump-core 
CFLAGS_gnu_opt   := $(VEC1D) $(DOUBLE) $(FMPI) -O3 -std=c99 -fopt-info
CFLAGS_gnu_debug := $(VEC1D) $(DOUBLE) $(FMPI) -g -Wall -fsignaling-nans -std=c99
OMPFLAGS_gnu_yes := -fopenmp
OMPFLAGS_gnu_no  :=

# Intel flags
FFLAGS_intel_opt   := -fast
FFLAGS_intel_debug := -g 
CFLAGS_intel_opt   := $(VEC1D) $(DOUBLE) $(FMPI) -O2 -xHost -restrict -align -vec-report6  -opt-streaming-stores auto
CFLAGS_intel_debug := $(VEC1D) $(DOUBLE) $(FMPI) -g 
OMPFLAGS_intel_yes := -openmp
OMPFLAGS_intel_no  :=



# Cray flags
FFLAGS_cray_opt   := -rm 
FFLAGS_cray_debug := -g
CFLAGS_cray_opt   :=  $(VEC1D) $(DOUBLE) $(FMPI) -h report=fisv
CFLAGS_cray_debug :=  $(VEC1D) $(DOUBLE) $(FMPI) -g
OMPFLAGS_cray_yes := 
OMPFLAGS_cray_no  := -h noomp

FFLAGS       := $(FFLAGS_$(COMP)_$(BUILD))
CFLAGS       := $(CFLAGS_$(COMP)_$(BUILD))	     
OMPFLAGS     := $(OMPFLAGS_$(COMP)_$(USE_OPENMP))
