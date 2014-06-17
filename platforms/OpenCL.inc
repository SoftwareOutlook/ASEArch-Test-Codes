#============================================================
# Makefile variable for  OpenCL
# Lucian Anton 
# created: 5/06/12
#============================================================


CC := icc

LANG = C



#need to add a correction here for double precision, for now it is deprecated
#ifdef USE_DOUBLE_PRECISION
#  DOUBLE=-DUSE_DOUBLE_PRECISION
#endif




COMP := OpenCL

USE_OPENMP=yes
USE_OPENCL=yes

OMPFLAGS= -openmp
CFLAGS_OpenCL_opt := $(DOUBLE) -O3  -lOpenCL -DUSE_OPENCL -restrict -std=c99
CFLAGS_OpenCL_debug := $(DOUBLE) -g -lOpenCL -DUSE_OPENCL



CFLAGS := $(CFLAGS_$(COMP)_$(BUILD))

