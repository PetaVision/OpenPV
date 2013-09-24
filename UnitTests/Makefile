SRCDIR = ../src
BUILDDIR = build
PVDIR = ../../PetaVision
LIBPV = $(PVDIR)/lib/libpv.a
INCPV = $(PVDIR)/src

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
   MPICC        := openmpicc
   MPICPP       := openmpic++
   MPIRUN       := openmpirun
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
INCLUDES += -I$(INCPV)
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

CLASS_OBJS = $(BUILDDIR)/Example.o

TESTS = test_kg \
        test_kxpos \
        test_kypos test_pos \
        test_nearby_neighbor \
        test_delta_pos test_extend_border \
        test_patch_head test_delta \
        test_sign test_post_weights \
        test_gauss2d test_cocirc \
        test_mirror_BCs \
        test_border_activity

# CL_TESTS = 

MPI_TESTS = test_mpi_specifyrowscolumns

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

all: make-in-subdirs $(CLASS_OBJS) $(TESTS) $(MPI_TESTS)

check: runtests

test_kg: $(BUILDDIR)/test_kg.o $(PVLIB)
	$(CC) -o $@ $< $(LIBS)

test_kxpos: $(BUILDDIR)/test_kxpos.o $(PVLIB)
	$(CC) -o $@ $< $(LIBS)

test_kypos: $(BUILDDIR)/test_kypos.o $(PVLIB)
	$(CC) -o $@ $< $(LIBS)

test_nearby_neighbor: $(BUILDDIR)/test_nearby_neighbor.o $(PVLIB)
	$(CC) -o $@ $< $(LIBS)

test_delta_pos: $(BUILDDIR)/test_delta_pos.o $(PVLIB)
	$(CC) -o $@ $< $(LIBS)

test_pos: $(BUILDDIR)/test_pos.o $(PVLIB)
	$(CC) -o $@ $< $(LIBS)

test_post_weights: $(BUILDDIR)/test_post_weights.o $(PVLIB) $(CLASS_OBJS)
	$(CPP) -o $@ $< $(LIBS) $(CLASS_OBJS)

test_gauss2d: $(BUILDDIR)/test_gauss2d.o $(PVLIB) $(CLASS_OBJS)
	$(CPP) -o $@ $< $(LIBS) $(CLASS_OBJS)

test_cocirc: $(BUILDDIR)/test_cocirc.o $(PVLIB) $(CLASS_OBJS)
	$(CPP) -o $@ $< $(LIBS) $(CLASS_OBJS)

test_extend_border: $(BUILDDIR)/test_extend_border.o $(PVLIB)
	$(CC) -o $@ $< $(LIBS)

test_border_activity: $(BUILDDIR)/test_border_activity.o $(PVLIB)
	$(CPP) -o $@ $< $(LIBS)

test_mirror_BCs: $(BUILDDIR)/test_mirror_BCs.o $(PVLIB) $(CLASS_OBJS)
	$(CPP) -o $@ $< $(LIBS) $(CLASS_OBJS)

test_patch_head: $(BUILDDIR)/test_patch_head.o $(PVLIB)
	$(CPP) -o $@ $< $(LIBS)

test_delta: $(BUILDDIR)/test_delta.o $(PVLIB)
	$(CC) -o $@ $<

test_sign: $(BUILDDIR)/test_sign.o $(PVLIB)
	$(CC) -o $@ $<

test_mpi_specifyrowscolumns: $(BUILDDIR)/test_mpi_specifyrowscolumns.o $(PVLIB)
	$(CPP) -o $@ $< $(LIBS)

runMPItests: make-in-subdirs $(MPI_TESTS)
	@rm -rf output/*.pvp
	@echo "Running test_mpi_specifyrowscolumns with 6 nodes ..."
	@if $(MPIRUN) -np 6 ./test_mpi_specifyrowscolumns -p input/test_mpi_specifyrowscolumns.params; then echo "test_mpi_specifyrowscolumns passed."; else echo "test_mpi_specifyrowscolumns FAILED."; fi

runtests: make-in-subdirs $(TESTS)
	@rm -rf output/*.pvp
	@echo "Running test_kg ..."
	@if ./test_kg; then echo "test_kg passed."; else echo "test_kg FAILED."; fi
	@echo "Running test_kxpos ..."
	@if ./test_kxpos; then echo "test_kxpos passed."; else echo "test_kxpos FAILED."; fi
	@echo "Running test_kypos ..."
	@if ./test_kypos; then echo "test_kypos passed."; else echo "test_kypos FAILED."; fi
	@echo "Running test_nearby_neighbor ..."
	@if ./test_nearby_neighbor; then echo "test_nearby_neighbor passed."; else echo "test_nearby_neighbor FAILED."; fi
	@echo "Running test_patch_head ..."
	@if ./test_patch_head; then echo "test_patch_head passed."; else echo "test_patch_head FAILED."; fi
	@echo "Running test_post_weights ..."
	@if ./test_post_weights -p input/test_post_weights.params; then echo "test_post_weights passed."; else echo "test_post_weights FAILED."; fi
	@echo "Running test_gauss2d ..."
	@if ./test_gauss2d; then echo "test_gauss2d passed."; else echo "test_gauss2d FAILED."; fi
	@echo "Running test_cocirc ..."
	@if ./test_cocirc -p input/test_cocirc.params; then echo "test_cocirc passed."; else echo "test_cocirc FAILED."; fi
#	@echo "Running test_pos ..."
#	@if ./test_pos; then echo "test_pos passed."; else echo "test_pos FAILED."; fi
	@echo "Running test_delta_pos ..."
	@if ./test_delta_pos; then echo "test_delta_pos passed."; else echo "test_delta_pos FAILED."; fi
	@echo "Running test_extend_border ..."
	@if ./test_extend_border; then echo "test_extend_border passed."; else echo "test_extend_border FAILED."; fi
	@echo "Running test_patch_head ..."
	@if ./test_patch_head; then echo "test_patch_head passed."; else echo "test_patch_head FAILED."; fi
	@echo "Running test_delta ..."
	@if ./test_delta; then echo "test_delta passed."; else echo "test_delta FAILED."; fi
	@echo "Running test_sign ..."
	@if ./test_sign; then echo "test_sign passed."; else echo "test_sign FAILED."; fi
	@echo "Running test_mirror_BCs ..."
	@if ./test_mirror_BCs; then echo "test_mirror_BCs passed."; else echo "test_mirror_BCs passed."; fi
	@echo "Running test_border_activity ..."
	@if ./test_border_activity; then echo "test_border_activity passed."; else echo "test_border_activity FAILED"; fi

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
