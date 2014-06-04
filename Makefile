#==============================================
# Makefile for HOMB
# Max Hutchinson (mhutchin@psc.edu)
# created: 7/26/08
# modified by Lucian Anton June 2013
#==============================================

JTC_C = jacobi_c
JTC_F90 = jacobi_f90


ifndef PLATFORM
   $(error PLATFORM needes to be defined)
endif

include platforms/$(PLATFORM).inc

ifndef BUILD
   $(error ERROR: BUILD needes to be defined)
else
   ifeq (,$(filter opt debug,$(BUILD)))
     $(error ERROR: unknown value for BUILD, try 'opt' or 'debug')
   endif
endif

# default target and name base
JTC := $(JTC_$(LANG))

# Build the executable name
EXE := $(JTC)_$(COMP)_$(BUILD).exe

ifdef USE_VEC1D
  EXE := $(basename $(EXE))_vec1d.exe
else
  ifndef USE_GPU
    $(warning WARNING: using non-vectorised inner loop in kernels_c.c) 
  endif
endif

ifdef USE_GPU
  EXE := $(basename $(EXE))_gpu.exe 
endif

ifdef USE_DOUBLE_PRECISION
  EXE := $(basename $(EXE))_dp.exe
endif


default: $(JTC)

all: $(JTC)



$(JTC_C):  kernels_c.o comm_mpi_c.o utils_c.o jacobi_c.o $(CUDAO)
	$(CC) $(MPIFLAGS) $(OMPFLAGS) $(CFLAGS) -o $(EXE) $^ $(LIB) 

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
ifdef USE_GPU
%_cuda.o : src/%_cuda.cu
	$(NVCC) -c -o $@ $(NVCCFLAGS) $<
endif
homb_f90.o : functions_f90.o

utils_c.o : src/utils_c.c src/jacobi_c.h src/utils_c.h src/kernels_c.h src/comm_mpi_c.h 
kernels_c.o : src/kernels_c.c src/jacobi_c.h src/kernels_c.h src/utils_c.h src/comm_mpi_c.h
comm_mpi_c.o : src/comm_mpi_c.c src/jacobi_c.h src/comm_mpi_c.h 
jacobi_c.o : src/jacobi_c.c src/jacobi_c.h src/utils_c.h src/kernels_c.h
ifdef USE_GPU
utils_c.o : src/cutil_inline.h
kernels_cuda.o : src/kernels_cuda.cu src/gpu_laplace3d_wrapper.h src/cutil_inline.h
endif 
