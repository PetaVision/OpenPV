# Path to PetaVision workspace goes here 
PVDIR = ../PVII
SRCDIR = $(PVDIR)/src
LIBDIR = $(PVDIR)/lib
MPIDIR =
BUILDDIR = Debug

INCDIR := $(PVDIR)
PVLIB  := -L$(PVDIR)/lib

# flags
#
dbg  := 0
mpi  := 0
gdal := 1

include $(PVDIR)/src/include/clmake.mk

TARGET = stdp

OBJS = $(BUILDDIR)/Patterns.o
EXES = $(BUILDDIR)/$(TARGET)

.SUFFIXES: .cpp .c .o

.cpp.o:
	$(CPP) $(CPPFLAGS) -c $(@:.o=.cpp)

.c.o:
	$(CC) $(CFLAGS) -c $(@:.o=.c)

all: $(OBJS) $(EXES)

$(BUILDDIR)/$(TARGET): $(BUILDDIR)/$(TARGET).o $(OBJS)
	$(CPP) -o $@ $< $(OBJS) -L$(LIBDIR) $(LIBS)

$(BUILDDIR)/$(TARGET).o: $(TARGET).cpp
	$(CPP) $(CPPFLAGS) -c $(TARGET).cpp -o $(BUILDDIR)/$(TARGET).o

$(BUILDDIR)/Patterns.o: Patterns.cpp
	$(CPP) $(CPPFLAGS) -c Patterns.cpp -o $(BUILDDIR)/Patterns.o

clean: force
	rm -f $(EXES) $(OBJS) *.o

force:
