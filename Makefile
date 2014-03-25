#==============================================
# Makefile for HOMB
# Max Hutchinson (mhutchin@psc.edu)
# created: 7/26/08
# modified by Lucian Anton June 2013
#==============================================

HOMB_C = homb_c
HOMB_F90 = homb_f90


ifndef PLATFORM
   $(error PLATFORM needes to be defined)
endif


include platforms/$(PLATFORM).inc

HOMB := $(HOMB_$(LANG))

EXE := $(HOMB)_$(COMP)_$(BUILD).exe

ifdef USE_VEC1D
  EXE := $(HOMB)_$(COMP)_$(BUILD)_vec1d.exe 
endif

ifdef USE_GPU
  EXE := $(HOMB)_$(COMP)_$(BUILD)_gpu.exe 
endif

default: $(HOMB)

all: $(HOMB)



$(HOMB_C):  kernels_c.o comm_mpi_c.o utils_c.o jacobi_c.o $(CUDAO)
	$(CC) $(MPIFLAGS) $(OMPFLAGS) $(CFLAGS) -o $(EXE) $^ $(LIB) 


$(HOMB_F90) : homb_f90.o functions_f90.o 
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
%_kernel.o : src/%_kernel.cu
	$(NVCC) -c -o $@ $(NVCCFLAGS) $<
endif
homb_f90.o : functions_f90.o

utils_c.o : src/utils_c.c src/jacobi_c.h src/utils_c.h src/kernels_c.h src/comm_mpi_c.h 
kernels_c.o : src/kernels_c.c src/jacobi_c.h src/kernels_c.h src/utils_c.h src/comm_mpi_c.h
comm_mpi_c.o : src/comm_mpi_c.c src/jacobi_c.h src/comm_mpi_c.h 
jacobi_c.o : src/jacobi_c.c src/jacobi_c.h src/utils_c.h src/kernels_c.h
ifdef USE_GPU
laplace3d_kernel.o : src/laplace3d_kernel.cu src/gpu_laplace3d_wrapper.h src/cutil_inline.h
endif 
