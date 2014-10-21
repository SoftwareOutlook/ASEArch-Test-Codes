#============================================================
# Makefile variable for  OpenACC
#    
# Mark Mawson 
# created: 20/10/2014
#============================================================


CC := pgcc

LANG = C

USE_ACC := 1

ifdef USE_MPI
  FMPI=-DUSE_MPI
endif

ifdef USE_DOUBLE_PRECISION
  DOUBLE=-DUSE_DOUBLE_PRECISION
endif


COMP=pgi
BUILD=opt

# PGI 

CFLAGS_pgi_opt := -Minfo $(FMPI) $(DOUBLE) -ta=tesla:kepler,fastmath  $(FMPI) $(DOUBLE) -fast 

CFLAGS_pgi_debug := -g -Mbounds -traceback -Minfo $(FMPI) $(DOUBLE) -DOPENACC -ta=tesla:kepler


# the flags are:
CFLAGS =  $(CFLAGS_$(COMP)_$(BUILD))
