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

default: $(HOMB)

all: $(HOMB)

$(HOMB_C): functions_c.o homb_c.o
	$(CC) $(MPIFLAGS) $(OMPFLAGS) $(CFLAGS) -o $(EXE)  $^

$(HOMB_F90) : homb_f90.o functions_f90.o
	$(F90) $(MPIFLAGS) $(FFLAGS) $(OMPFLAGS) -o $(EXE) $^ 

clean:
	rm -f *.mod *.o
vclean:
	rm -f *.mod *.o *.exe

%_f90.o : src/%_f90.f90
	$(F90) -c -o $@ $(OMPFLAGS) $(FFLAGS) $< 

%_c.o : src/%_c.c
	$(CC) -c -o$@ $(OMPFLAGS) $(CFLAGS) $<

homb_f90.o : functions_f90.o

functions_c.o : src/functions_c.c src/homb_c.h
homb_c.o : src/homb_c.c src/homb_c.h functions_c.o 
