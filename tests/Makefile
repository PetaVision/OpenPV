SRCDIR = ../src
BUILDDIR = build
LIBPV = ../lib/libpv.a

INPARAMS = input/params.test

# flags
#
dbg    := 1
mpi    := 1
gdal   := 1
opencl := 0

# detect OS                                                                               
OSUPPER = $(shell uname -s 2>/dev/null | tr [:lower:] [:upper:])
OSLOWER = $(shell uname -s 2>/dev/null | tr [:upper:] [:lower:])
# 'linux' is output for Linux system, 'darwin' for OS X                                   
DARWIN = $(strip $(findstring DARWIN, $(OSUPPER)))

# detect architecture
OSARCH= $(shell uname -m)

INCDIR  ?= .

ifeq "$(DARWIN)" "DARWIN"
   GDALINCLDIR  := /opt/local/include
   GDALLIBDIR   := /opt/local/lib
   MPICC        := mpicc-openmpi-mp
   MPICPP       := mpicxx-openmpi-mp
   MPIRUN       := mpiexec-openmpi-mp
else
   SDKROOTDIR   := /opt/NVIDIA_GPU_Computing_SDK
   LIBDIR       := $(SDKROOTDIR)/shared/lib/
   SHAREDDIR    := $(SDKROOTDIR)/shared/
   SHAREDLIBDIR := $(SDKROOTDIR)/shared/lib/linux
   OCLROOTDIR   := $(SDKROOTDIR)/OpenCL/
   OCLCOMMONDIR := $(OCLROOTDIR)/common/
   OCLLIBDIR    := $(OCLCOMMONDIR)/lib
   GDALINCLDIR  := /usr/include/gdal
   GDALLIBDIR   := /usr/lib64
   MPICC        := mpicc
   MPICPP       := mpic++
   MPIRUN       := mpirun
endif

# Compilers
ifeq ($(mpi),1)
   CC  := $(MPICC)
   CPP := $(MPICPP)
else
   CC  := cc
   CPP := c++
endif
LINK   := $(CPP) -fPIC

# Includes
ifeq "$(DARWIN)" "DARWIN"
   INCLUDES += -I$(INCDIR)
else
   INCLUDES += -I$(INCDIR) -I$(OCLCOMMONDIR)/inc -I$(SHAREDDIR)/inc
endif
ifeq ($(gdal),1)
  INCLUDES += -I$(GDALINCLDIR)
endif

# Warning flags
CWARN_FLAGS := -Wall

# architecture flag for nvcc and gcc compilers build
LIB_ARCH := $(OSARCH)

ifeq "$(DARWIN)" ""
   ARCH_FLAGS += -m64
else
   ARCH_FLAGS += -m64
endif

# Compiler-specific flags
CFLAGS   := $(CWARN_FLAGS) $(ARCH_FLAGS)
LINK     += $(ARCH_FLAGS)

# Common flags
ifeq "$(DARWIN)" "DARWIN"
   COMMONFLAGS += -DMAC
else
   COMMONFLAGS += -DUNIX
endif

# Debug/release configuration
ifeq ($(dbg),1)
   COMMONFLAGS += -g
else
   COMMONFLAGS += -O3
   CFLAGS      += -fno-strict-aliasing
endif

# Libs
ifneq "$(DARWIN)" ""
   LIBS := $(LIBPV) -framework OpenCL
else
   LIBS := $(LIBPV) -L${OCLLIBDIR} -L$(LIBDIR) -L$(SHAREDDIR)/lib/$(OSLOWER) 
   ifeq ($(opencl),1)
      LIBS += -lOpenCL
   endif
endif
ifeq ($(gdal),1)
   LIBS += -L$(GDALLIBDIR) -lgdal
endif

# Add common flags
CFLAGS   += $(INCLUDES) $(COMMONFLAGS) -DHAS_MAIN=1
CPPFLAGS := $(CFLAGS)
CFLAGS   += -std=c99

TESTS = test_datatypes

CL_TESTS = test_cl_random test_cl_lif test_cl_all

MPI_TESTS = #test_read_header test_image_io test_weights_io

.SUFFIXES: .cpp .c .o

.cpp.o:
	$(CPP) $(CPPFLAGS) -c $(@:.o=.cpp)

.c.o:
	$(CC) $(CFLAGS) -c $(@:.o=.c)

$(BUILDDIR)/%.o: %.cpp
	@echo 'Building file: $<'
	$(CPP) $(CPPFLAGS) -c -o"$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

$(BUILDDIR)/%.o: %.c
	@echo 'Building file: $<'
	$(CC) $(CFLAGS) -c -o"$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

all: make-in-subdirs $(TESTS) $(MPI_TESTS)

check: runtests

test_read_header: $(BUILDDIR)/test_read_header.o $(PVLIB)
	$(CPP) -o $@ $< $(LIBS)

test_image_io: $(BUILDDIR)/test_image_io.o $(PVLIB)
	$(CPP) -o $@ $< $(LIBS)

test_cl_random: $(BUILDDIR)/test_cl_random.o $(PVLIB)
	$(CPP) -o $@ $< $(LIBS)

test_cl_lif: $(BUILDDIR)/test_cl_lif.o $(PVLIB)
	$(CPP) -o $@ $< $(LIBS)
	
test_cl_all: $(BUILDDIR)/test_cl_all.o $(PVLIB)
	$(CPP) -o $@ $< $(LIBS)

test_weights_io: $(BUILDDIR)/test_weights_io.o $(PVLIB)
	$(CPP) -o $@ $< $(LIBS)

test_decode_msg: $(BUILDDIR)/test_decode_msg.o $(PVLIB)
	$(CC) -o $@ $< $(LIBS)

test_datatypes: $(BUILDDIR)/test_datatypes.o $(PVLIB)
	$(CPP) -o $@ $< $(LIBS)

runMPItests: make-in-subdirs $(MPI_TESTS)
	@rm -rf output/*.pvp
#	@echo "Running test_read_header with 1 node ..."
#	@./test_read_header -n 1 -p $(INPARAMS)
#	@echo "Running test_read_header with 2 nodes ..."
#	@openmpirun -np 2 test_read_header -n 1 -p $(INPARAMS)
#	@echo "Running test_read_header with 4 nodes ..."
#	@openmpirun -np 4 test_read_header -n 1 -p $(INPARAMS)
#	@rm -rf output/*.pvp
#	@rm -rf output/*.tif
#	@echo "Running test_image_io with 1 node ..."
#	@./test_image_io -n 1 -p $(INPARAMS)
#	@echo "Running test_image_io with 2 nodes ..."
#	@openmpirun -np 2 test_image_io -n 1 -p $(INPARAMS)
#	@echo "Running test_image_io with 4 nodes ..."
#	@openmpirun -np 4 test_image_io -n 1 -p $(INPARAMS)
#	@echo "Running test_image_io with 6 nodes ..."
#	@openmpirun -np 6 test_image_io -n 1 -p $(INPARAMS)
#	@echo "Running test_weights_io with 1 node ..."
#	@./test_weights_io -n 1 -p $(INPARAMS)
#	@echo "Running test_weights_io with 2 nodes ..."
#	@openmpirun -np 2 test_weights_io -n 1 -p $(INPARAMS)
#	@echo "Running test_weights_io with 4 nodes ..."
#	@openmpirun -np 4 test_weights_io -n 1 -p $(INPARAMS)
#	@echo "Running test_weights_io with 6 nodes ..."
#	@openmpirun -np 6 test_weights_io -n 1 -p $(INPARAMS)

runtests: make-in-subdirs $(TESTS)
	@rm -rf output/*.pvp

make-in-subdirs: force
	@for subdir in $(SUBDIRS); do \
	    if [ -d $$subdir -a -f $$subdir/Makefile ] ; then \
	        echo "making in $$subdir" ; \
	        (cd $$subdir && $(MAKE) all) ; \
	    fi \
	done

clean: clean-in-subdirs
	rm -f *.o *.bin output/*.pvp
	rm -f $(TESTS) $(CL_TESTS) $(MPI_TESTS) *.o *.bin output/*.pvp
	rm -f $(BUILDDIR)/*.o

cleanall: clean
	rm -f $(MPI_OBJ)
	rm -rf output

clean-in-subdirs: force
	@for subdir in $(SUBDIRS); do \
	    if [ -d $$subdir -a -f $$subdir/Makefile ] ; then \
	        echo "cleaning in $$subdir" ; \
	        (cd $$subdir && $(MAKE) clean) ; \
	    fi \
	done

force:
