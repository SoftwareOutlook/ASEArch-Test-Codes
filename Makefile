#==============================================
# Makefile for HOMB
# Max Hutchinson (mhutchin@psc.edu)
# created: 7/26/08
# modified by Lucian Anton June 2013
#==============================================

JTC_C = jacobi_c
JTC_F90 = jacobi_f90

PLATFORM ?= dummy
PLATLIST = $(basename $(notdir $(wildcard platforms/*.inc )))
#$(info INFO: available platforms $(PLATLIST)

# check if the define platform is available
ifeq (,$(filter $(PLATLIST),$(PLATFORM)))
    $(error ERROR: unknown value for PLATFORM $(PLATFORM), try make list_platforms)
endif
#$(info $(PLATLIST))

include platforms/$(PLATFORM).inc


# default target and name base
JTC := $(JTC_$(LANG))

# Build the executable name
EXE := $(JTC)_$(COMP)_$(BUILD).exe

ifdef USE_VEC1D
  EXE := $(basename $(EXE))_vec1d.exe
endif

ifdef USE_CUDA
  EXE := $(basename $(EXE))_cuda.exe 
endif

ifdef USE_OPENCL
  EXE := $(basename $(EXE))_opencl.exe 
endif

ifdef USE_OPENACC
  EXE := $(basename $(EXE))_openacc.exe 
endif

ifdef USE_MPI
  EXE := $(basename $(EXE))_mpi.exe 
endif

ifdef USE_DOUBLE_PRECISION
  EXE := $(basename $(EXE))_dp.exe
endif

OBJ := kernels_c.o comm_mpi_c.o utils_c.o jacobi_c.o $(CUDAO) $(OPENCLO)

default: $(JTC_C)
all: $(JTC_C)

$(JTC_C): checkplatform checkbuild $(OBJ)
	$(CC) $(MPIFLAGS) $(OMPFLAGS) $(CFLAGS) -o $(EXE) $(OBJ) $(LIB) 

# Fortran build is inactive in this version
$(JTC_F90) : homb_f90.o functions_f90.o 
	$(F90) $(MPIFLAGS) $(FFLAGS) $(OMPFLAGS) -o $(EXE) $^

clean:
	rm -f *.mod *.o  
vclean:
	rm -f *.mod *.o *.exe

%_f90.o : src/%_f90.f90
	$(F90) -c -o $@ $(OMPFLAGS) $(FFLAGS) $< 

%_c.o : src/%_c.c
	$(CC) -c -o $@ $(OMPFLAGS) $(CFLAGS) $(INC) $<

ifdef USE_CUDA
%_cuda.o : src/%_cuda.cu
	$(NVCC) -c -o $@ $(NVCCFLAGS) $<
endif

ifdef USE_OPENCL
%cl.o : src/OpenCL/%cl.c
	$(CC) -c -o $@ $(OMPFLAGS) $(CFLAGS) $(INC) $<
endif

checkplatform :
	@if [ $(PLATFORM) = dummy ] ; then echo "PLATFORM needs to be provided; try make list_platforms"; exit 1 ; fi

checkbuild :
	@if [ ! "$(BUILD)" = opt -a ! "$(BUILD)" = debug ] ; then echo  "ERROR: unknown value for BUILD, try 'opt' or 'debug'"; exit 1 ; fi

list_platforms :
	@echo $(filter-out dummy,$(PLATLIST)) 

homb_f90.o : functions_f90.o
utils_c.o : src/utils_c.c src/jacobi_c.h src/utils_c.h src/kernels_c.h src/comm_mpi_c.h 
kernels_c.o : src/kernels_c.c src/jacobi_c.h src/kernels_c.h src/utils_c.h src/comm_mpi_c.h 
comm_mpi_c.o : src/comm_mpi_c.c src/jacobi_c.h src/comm_mpi_c.h 
jacobi_c.o : src/jacobi_c.c src/jacobi_c.h src/utils_c.h src/kernels_c.h
ifdef USE_CUDA
utils_c.o : src/cutil_inline.h
kernels_cuda.o : src/kernels_cuda.cu src/gpu_laplace3d_wrapper.h src/cutil_inline.h
endif 


ifdef USE_OPENCL
  kernels_c.o : src/OpenCL/jacobi_opencl.h
  device_info_cl.o : src/OpenCL/device_info_cl.c 
  err_code_cl.o: src/OpenCL/err_code_cl.c
  jacobi_opencl.o : src/OpenCL/jacobi_opencl.c src/OpenCL/jacobi_opencl.h src/OpenCL/err_code_cl.c src/OpenCL/device_info_cl.c 
endif
