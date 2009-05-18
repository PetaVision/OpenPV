# Path to PetaVision workspace goes here 
PVDIR = ../pv
SRCDIR = $(PVDIR)/src
BUILDDIR = $(SRCDIR)/build

CPP = g++
CC  = gcc

MPI_DIR = $(SRCDIR)/arch/mpi
MPI_INC = -I$(MPI_DIR)

CFLAGS = -g -Wall -std=c99 -Wall -I$(PVDIR) -I$(MPI_DIR)
CPPFLAGS = -g -Wall -I$(PVDIR) -I$(MPI_DIR)

LIBS = -lpv -pthread

EXES = pv_ca

.SUFFIXES: .cpp .c .o

.cpp.o:
	$(CPP) $(CPPFLAGS) -c $(@:.o=.cpp)

.c.o:
	$(CC) $(CFLAGS) -c $(@:.o=.c)

all: $(EXES)

pv_ca: pv_ca.o # $(BUILDDIR)/libpv.a
	$(CPP) -o $@ $< -L$(BUILDDIR) $(LIBS)

clean: force
	rm -f $(EXES) *.o

force:
