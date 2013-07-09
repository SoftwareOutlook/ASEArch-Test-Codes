#============================================================
# Makefile configuration macros for Cray XE6, PGI and GNU compiler
#    
# Lucian Anton 
# created: 6/06/11
#============================================================

CC  := cc
F90 := ftn

COMP := nonsense

ifeq ($(PE_ENV),PGI)
   COMP := pgi
endif

ifeq ($(PE_ENV),GNU)
   COMP := gnu
endif

ifeq ($(PE_ENV),CRAY)
   COMP := cray
endif

MPIFLAGS     :=
OMPFLAGS_gnu     := -fopenmp
FFLAGS_gnu_opt   := -O3
FFLAGS_gnu_debug := -g -fbounds-check -Wall -fbacktrace -ffpe-trap=invalid,zero,overflow -fsignaling-nans -fdump-core 
CFLAGS_gnu_opt   := -O3
CFLAGS_gnu_debug := -g -fbounds-check -Wall -fsignaling-nans  

# PGI flags

OMPFLAGS_pgi     := -mp
FFLAGS_pgi_opt   := -fast
FFLAGS_pgi_debug := -g 
CFLAGS_pgi_opt   := -fast
CFLAGS_pgi_debug := -g 

# Cray flags
OMPFLAGS_cray     := 
FFLAGS_cray_opt   := -rm 
FFLAGS_cray_debug := -g
CFLAGS_cray   := 
CFLAGS_cray_debug := -g

FFLAGS       := $(FFLAGS_$(COMP)_$(VERSION))
CFLAGS       := $(CFLAGS_$(COMP)_$(VERSION))	     
OMPFLAGS     := $(OMPFLAGS_$(COMP))
