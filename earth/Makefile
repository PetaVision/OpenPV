# Path to PetaVision workspace goes here 
PVDIR = ../PetaVision
SRCDIR = .
BUILDDIR = $(SRCDIR)/Debug
LIBDIR = $(PVDIR)/lib

CPP = g++
CC  = gcc

CFLAGS = -g -Wall -std=c99 -Wall -I$(PVDIR)
CPPFLAGS = -g -Wall -I$(PVDIR)

LIBS = -lpv -framework GLUT -framework OpenGL

EXES = pv

.SUFFIXES: .cpp .c .o

.cpp.o:
	$(CPP) $(CPPFLAGS) -c $(@:.o=.cpp)

.c.o:
	$(CC) $(CFLAGS) -c $(@:.o=.c)

all: $(EXES)

earth: earth-stdp.o
	$(CPP) -o $@ $< -L$(LIBDIR) $(LIBS)

clean: force
	rm -f $(EXES) *.o

force:
