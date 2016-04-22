# detect OS                                                                               
OSUPPER = $(shell uname -s 2>/dev/null | tr [:lower:] [:upper:])
OSLOWER = $(shell uname -s 2>/dev/null | tr [:upper:] [:lower:])
# 'linux' is output for Linux system, 'darwin' for OS X                                   
DARWIN = $(strip $(findstring DARWIN, $(OSUPPER)))

# detect architecture
OSARCH= $(shell uname -m)

INCDIR  ?= .

ifeq ($(DARWIN),)
   SDKROOTDIR   := /opt/NVIDIA_GPU_Computing_SDK
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
ifeq ($(gdal),1)
  INCLUDES += -I$(GDALDIR)/include
endif

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
ifneq ($(DARWIN),1)
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
ifneq ($(DARWIN),)
   LIBS := $(LIBPV) -lpv -framework OpenCL
else
   LIBS := $(LIBPV) -lpv -L${OCLLIBDIR} -L$(LIBDIR) -L$(SHAREDDIR)/lib/$(OSLOWER) 
   LIBS += -lOpenCL
endif
ifeq ($(gdal),1)
   LIBS += -L$(GDALDIR)/lib -lgdal
endif

# Add common flags
CFLAGS   += $(INCLUDES) $(COMMONFLAGS) -DHAS_MAIN=1
CPPFLAGS := $(CFLAGS)
CFLAGS   += -std=c99
