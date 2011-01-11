SRCDIR = ../src
BUILDDIR = build
GDALDIR = /usr/local

include $(SRCDIR)/include/sources.mk

# flags
#
dbg := 0
mpi := 0

# detect OS                                                                               
OSUPPER = $(shell uname -s 2>/dev/null | tr [:lower:] [:upper:])
OSLOWER = $(shell uname -s 2>/dev/null | tr [:upper:] [:lower:])
# 'linux' is output for Linux system, 'darwin' for OS X                                   
DARWIN = $(strip $(findstring DARWIN, $(OSUPPER)))

# detect architecture
OSARCH= $(shell uname -m)

INCDIR  ?= .

ifeq ($(DARWIN),)
   SDKROOTDIR   := /opt/local/NVIDIA_GPU_Computing_SDK
   LIBDIR       := $(SDKROOTDIR)/shared/lib/
   SHAREDDIR    := $(SDKROOTDIR)/shared/
   SHAREDLIBDIR := $(SDKROOTDIR)/shared/lib/linux
   OCLROOTDIR   := $(SDKROOTDIR)/OpenCL/
   OCLCOMMONDIR := $(OCLROOTDIR)/common/
   OCLLIBDIR    := $(OCLCOMMONDIR)/lib
endif

# Compilers
ifeq ($(mpi),1)
   CC  := mpicc
   CPP := mpic++
else
   CC  := cc
   CPP := c++
endif
LINK   := $(CPP) -fPIC

# Includes
ifeq ($(DARWIN),1)
   INCLUDES += -I$(INCDIR)
else
   INCLUDES += -I$(INCDIR) -I$(OCLCOMMONDIR)/inc -I$(SHAREDDIR)/inc
endif
INCLUDES += -I$(GDALDIR)/include

# Warning flags
CWARN_FLAGS := -W

# architecture flag for nvcc and gcc compilers build
LIB_ARCH := $(OSARCH)

ifeq ($(DARWIN),)
   ARCH_FLAGS += -m64
else
   ARCH_FLAGS += -m64
endif

# Compiler-specific flags
CFLAGS   := $(CWARN_FLAGS) $(ARCH_FLAGS)
LINK     += $(ARCH_FLAGS)

# Common flags
COMMONFLAGS += -DUNIX

# Add Mac Flags
ifeq ($(DARWIN),1)
   COMMONFLAGS += -DMAC
endif

# Debug/release configuration
ifeq ($(dbg),1)
   COMMONFLAGS += -g
else
   COMMONFLAGS += -O3
   CFLAGS      += -fno-strict-aliasing
endif

# Libs
ifneq ($(DARWIN),)
   LIBS := -framework OpenCL
else
   LIBS := -L${OCLLIBDIR} -L$(LIBDIR) -L$(SHAREDDIR)/lib/$(OSLOWER) 
   LIBS += -lOpenCL
endif
LIBS += -L $(GDALDIR)/lib -lgdal

# Add common flags
CFLAGS   += $(INCLUDES) $(COMMONFLAGS) -DHAS_MAIN=1
CPPFLAGS := $(CFLAGS)
CFLAGS   += -std=c99

MAKE = make

%.o: %.cpp
	@echo 'Building file: $<'
	$(CPP) -c $(CPPFLAGS) -o $@ $<
	@echo 'Finished building: $<'
	@echo ' '

all: make-in-subdirs $(BUILDDIR)/libpv.a $(EXES)

pv_example: pv_example.o $(DEPENDS) $(OBJS) $(MPI_OBJ)
	$(CPP) -o $@ pv_example.o $(OBJS) $(MPI_OBJ) -lm

pv_cocirc: pv_cocirc.o $(DEPENDS) $(OBJS) $(MPI_OBJ)
	$(CPP) -o $@ $< $(OBJS) $(MPI_OBJ) -lm

pv_gabor_cocirc: pv_gabor_cocirc.o $(DEPENDS) $(OBJS) $(MPI_OBJ)
	$(CPP) -o $@ $< $(OBJS) $(MPI_OBJ) -lm

pv_: $(notdir $(CPPSRCS:.cpp=.o)) $(notdir $(CSRCS:.c=.o))
	$(CPP) -o $@ $+ -lm

$(MPI_OBJ): $(MPI_DIR)/mpi.h $(MPI_DIR)/mpi.c
	$(CC) -c -o $@ $(CFLAGS) $(MPI_DIR)/mpi.c

libpv.so: $(notdir $(CPPSRCS:.cpp=.o)) $(notdir $(CSRCS:.c=.o))
	$(CPP) -shared -Wl,-soname,$@ -o $@ $+ -lc

$(BUILDDIR)/libpv.a: $(DEPENDS) $(OBJS) $(MPI_OBJ)
	ar rcs $(BUILDDIR)/libpv.a $+
	ar rcs ../lib/libpv.a $+

make-in-subdirs: force
	@for subdir in $(SUBDIRS); do \
	    if [ -d $$subdir -a -f $$subdir/Makefile ] ; then \
	        echo "making in $$subdir" ; \
	        (cd $$subdir && $(MAKE) all) ; \
	    fi \
	done

clean: clean-in-subdirs
	rm -f $(BUILDDIR)/*.o $(BUILDDIR)/pv $(BUILDDIR)/*.P $(BUILDDIR)/tags
	rm -f $(MPI_OBJ)
	rm -f ./output/*.ps ./output/*.bin ./output/*.txt
	rm -f $(BUILDDIR)/libpv.a $(BUILDDIR)/libpv.so
	rm -f $(EXES)

clean-in-subdirs: force
	@for subdir in $(SUBDIRS); do \
	    if [ -d $$subdir -a -f $$subdir/Makefile ] ; then \
	        echo "cleaning in $$subdir" ; \
	        (cd $$subdir && $(MAKE) clean) ; \
	    fi \
	done

force:

tags:
	ctags `find . -name "*.[ch]"` `find . -name "*.[ch]pp"`

include $(SRCDIR)/include/depend.mk
