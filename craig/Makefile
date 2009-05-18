# Path to PetaVision workspace goes here 
PVDIR = ../PetaVision
SRCDIR = $(PVDIR)/src
LIBDIR = $(PVDIR)/lib

CPP = g++
CC  = gcc

MPI_DIR = $(SRCDIR)/arch/mpi
MPI_INC = -I$(MPI_DIR)

CFLAGS = -g -Wall -std=c99 -Wall -I$(PVDIR) -I$(MPI_DIR)
CPPFLAGS = -g -Wall -I$(PVDIR) -I$(MPI_DIR)

LIBS = -lpv -pthread

OBJS = LinearPostConnProbe.o Retina1D.o

EXES = stdp

.SUFFIXES: .cpp .c .o

.cpp.o:
	$(CPP) $(CPPFLAGS) -c $(@:.o=.cpp)

.c.o:
	$(CC) $(CFLAGS) -c $(@:.o=.c)

all: $(EXES)

stdp: stdp.o $(OBJS) $(LIBDIR)/libpv.a
	$(CPP) -o $@ $< $(OBJS) -L$(LIBDIR) $(LIBS)

clean: force
	rm -f $(EXES) $(OBJS) *.o

force:
