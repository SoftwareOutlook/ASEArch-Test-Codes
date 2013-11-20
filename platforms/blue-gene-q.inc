F90  := mpif90
CC  := mpicc

LANG := C
COMP := ibm
BUILD := opt
USE_OPENMP := yes

ifeq ($(strip $(COMP)),ibm)
  MODULE_PRG := ibmmpi
	
  ifdef USE_MPI  
    ifeq ($(strip $(LANG)),F90)
      FMPI := -WF,-DMPI
    else
      FMPI := -DMPI
    endif
  endif

  ifdef USE_DOUBLE_PRECISION
    DOUBLE := -DUSE_DOUBLE_PRECISION
    ifeq ($(strip $(LANG)),F90)
      DOUBLE := -WF, $(DOUBLE)
    endif
  endif

  ifdef USE_VEC1D
    VEC1D := -DUSE_VEC1D
    ifeq ($(strip $(LANG)),F90) 
      VEC1D := -WF,-DUSE_VEC1D
    endif
  endif

else ifeq ($(strip $(COMP)),gcc)
  MODULE_PRG := gccmpi

  ifdef USE_MPI
    FMPI := -DMPI
  endif

  ifdef USE_DOUBLE_PRECISION
    DOUBLE := -DUSE_DOUBLE_PRECISION
  endif

  ifdef USE_VEC1D
    VEC1D = -DUSE_VEC1D
  endif
endif


FLAGS_ibm_debug := -g -C  
FLAGS_ibm_opt   := -O3  -qstrict
CLAGS_ibm_debug := -g -qnoopt   
CLAGS_ibm_opt   := -O3 -qhot -qreport
OMPFLAGS_ibm_yes := -qsmp=omp
OMPFLAGS_ibm_no :=

FLAGS_gcc_opt :=  -O3 -std=c99
FLAGS_gcc_debug := -g -fcheck=all -finit-real=snan -Wall -Wtabs -Wno-unused-fbacktrace -ffpe-trap=invalid,zero,overflow -fsignaling-nans -fdump-core -std=c99
CLAGS_gcc_opt = -O3 -std=c99 
CLAGS_gcc_debug = -g -Wall -Wuninitialized -O1 -std=c99
OMPFLAGS_gcc_yes  := -fopenmp
OMPFLAGS_gcc_no   :=

FFLAGS := $(FMPI) $(DOUBLE) $(VEC1D) $(FLAGS_$(COMP)_$(BUILD)) 
CFLAGS := $(FMPI) $(DOUBLE) $(VEC1D) $(CLAGS_$(COMP)_$(BUILD)) 
OMPFLAGS := $(OMPFLAGS_$(COMP)_$(USE_OPENMP))

LD_FLAGS := 

F90 := module unload ibmmpi && module unload gccmpi && module load $(MODULE_PRG) && $(F90)
CC := module unload ibmmpi && module unload gccmpi && module load $(MODULE_PRG) && $(CC)

ifeq ($(LANG),C)
  ifeq ($(COMP),gcc)
    LIB := -lm
  endif
endif

