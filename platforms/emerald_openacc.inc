#============================================================
# Makefile variable for  OpenACC
#    
# Mark Mawson 
# created: 20/10/2014
#============================================================


CC := pgcc

LANG = C

ifdef USE_MPI
  FMPI=-DUSE_MPI
endif

ifdef USE_DOUBLE_PRECISION
  DOUBLE=-DUSE_DOUBLE_PRECISION
endif


COMP=pgi
BUILD=opt

# PGI 

CFLAGS_pgi_opt := $(FMPI) $(DOUBLE) -ta=tesla:kepler,fastmath  $(FMPI) $(DOUBLE) -fast 

CFLAGS_pgi_debug := -g -Mbounds -traceback $(FMPI) $(DOUBLE) -ta=tesla:kepler,fastmath


# the flags are:
CFLAGS =  $(CFLAGS_$(COMP)_$(BUILD))

# needed for executable name
USE_OPENACC=1