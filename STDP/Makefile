# Path to PetaVision workspace goes here 
PVDIR = ../PetaVision
SRCDIR = $(PVDIR)/src
LIBDIR = $(PVDIR)/lib

CPP = mpic++
CC  = mpicc

CFLAGS = -g -Wall -std=c99 -I$(PVDIR) -I/usr/local/include
CPPFLAGS = -g -Wall -I$(PVDIR) -I/usr/local/include

LIBS = -lpv -lgdal

EXES = stdp

.SUFFIXES: .cpp .c .o

.cpp.o:
	$(CPP) $(CPPFLAGS) -c $(@:.o=.cpp)

.c.o:
	$(CC) $(CFLAGS) -c $(@:.o=.c)

all: $(EXES)

stdp: stdp.o
	$(CPP) -o $@ $< -L$(LIBDIR) $(LIBS)

clean: force
	rm -f $(EXES) *.o

force:
