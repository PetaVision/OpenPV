# Path to PetaVision workspace goes here 
PVDIR = ../PetaVision
SRCDIR = $(PVDIR)/src
LIBDIR = $(PVDIR)/lib
MPIDIR = /usr/local/openmpi/bin
BUILDDIR = Debug

TARGET = stdp

#CPP = g++
#CC  = gcc
CPP = $(MPIDIR)/mpic++
CC  = $(MPIDIR)/mpicc

MPI_DIR = $(SRCDIR)/arch/mpi
MPI_INC = -I$(MPI_DIR)

CFLAGS = -g -Wall -std=c99 -Wall -I$(PVDIR) -I$(MPI_DIR)
CPPFLAGS = -g -Wall -I$(PVDIR) -I$(MPI_DIR)

LIBS = -lpv -lpthread -framework OpenGL -framework GLUT

#OBJS = LinearPostConnProbe.o Retina1D.o

EXES = $(BUILDDIR)/$(TARGET)

.SUFFIXES: .cpp .c .o

.cpp.o:
	$(CPP) $(CPPFLAGS) -c $(@:.o=.cpp)

.c.o:
	$(CC) $(CFLAGS) -c $(@:.o=.c)

all: $(EXES)

$(BUILDDIR)/$(TARGET): $(BUILDDIR)/$(TARGET).o $(OBJS) $(LIBDIR)/libpv.a
	$(CPP) -o $@ $< $(OBJS) -L$(LIBDIR) $(LIBS)

$(BUILDDIR)/$(TARGET).o: $(TARGET).cpp
	$(CPP) $(CPPFLAGS) -c $(TARGET).cpp -o $(BUILDDIR)/$(TARGET).o

clean: force
	rm -f $(EXES) $(OBJS) *.o

force:
