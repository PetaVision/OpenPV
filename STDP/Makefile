# Path to PetaVision workspace goes here 
PVDIR = ../PetaVision
SRCDIR = $(PVDIR)/src
LIBDIR = $(PVDIR)/lib
GDALDIR = /usr/local

CPP = mpic++
CC  = mpicc

INCLUDES = -I$(PVDIR) -I$(GDALDIR)/include

CFLAGS = -g -Wall -std=c99 $(INCLUDES)
CPPFLAGS = -g -Wall $(INCLUDES)

LIBS = -lpv -L$(GDALDIR)/lib -lgdal

EXES = stdp

.SUFFIXES: .cpp .c .o

.cpp.o:
	$(CPP) $(CPPFLAGS) -c $(@:.o=.cpp)

.c.o:
	$(CC) $(CFLAGS) -c $(@:.o=.c)

all: $(EXES)

stdp: stdp.o Gratings.o
	$(CPP) -o $@ $+ -L$(LIBDIR) $(LIBS)

clean: force
	rm -f $(EXES) *.o

force:
