#SUBDIRS = $(BUILDDIR)/ppu \
#          $(BUILDDIR)/spu \
#          $(BUILDDIR)/pthreads

HEADERS = $(SRCDIR)/arch/opencl/CLBuffer.hpp \
          $(SRCDIR)/arch/opencl/CLDevice.hpp \
          $(SRCDIR)/arch/opencl/CLKernel.hpp \
          $(SRCDIR)/include/pv_arch.h \
          $(SRCDIR)/include/pv_common.h \
          $(SRCDIR)/columns/HyPerCol.hpp \
          $(SRCDIR)/columns/HyPerColRunDelegate.hpp \
          $(SRCDIR)/connections/HyPerConn.hpp \
          $(SRCDIR)/connections/GaborConn.hpp \
          $(SRCDIR)/connections/CocircConn.hpp \
          $(SRCDIR)/connections/KernelConn.hpp \
          $(SRCDIR)/connections/PoolConn.hpp \
          $(SRCDIR)/connections/RandomConn.hpp \
          $(SRCDIR)/connections/RuleConn.hpp \
          $(SRCDIR)/connections/PVConnection.h \
          $(SRCDIR)/layers/Gratings.hpp \
          $(SRCDIR)/layers/Image.hpp \
          $(SRCDIR)/layers/LayerDataInterface.hpp \
          $(SRCDIR)/layers/LIF.hpp \
          $(SRCDIR)/layers/Movie.hpp \
          $(SRCDIR)/layers/PVLayer.h \
          $(SRCDIR)/io/ConnectionProbe.hpp \
          $(SRCDIR)/io/LayerProbe.hpp \
          $(SRCDIR)/io/LinearActivityProbe.hpp \
          $(SRCDIR)/io/PointProbe.hpp \
          $(SRCDIR)/io/PostConnProbe.hpp \
          $(SRCDIR)/io/StatsProbe.hpp \
          $(SRCDIR)/io/PVParams.hpp \
          $(SRCDIR)/io/fileio.hpp \
          $(SRCDIR)/io/imageio.hpp \
          $(SRCDIR)/io/tiff.h \
          $(SRCDIR)/utils/conversions.h \
          $(SRCDIR)/utils/cl_random.h \
          $(SRCDIR)/utils/rng.h \
          $(SRCDIR)/utils/Timer.hpp

CPPSRCS = $(SRCDIR)/arch/CLBuffer.cpp \
          $(SRCDIR)/arch/CLDevice.cpp \
          $(SRCDIR)/arch/CLKernel.cpp \
          $(SRCDIR)/columns/HyPerCol.cpp \
          $(SRCDIR)/columns/HyPerColRunDelegate.cpp \
          $(SRCDIR)/columns/Communicator.cpp \
          $(SRCDIR)/columns/InterColComm.cpp \
          $(SRCDIR)/columns/DataStore.cpp \
          $(SRCDIR)/connections/HyPerConn.cpp \
          $(SRCDIR)/connections/GaborConn.cpp \
          $(SRCDIR)/connections/CocircConn.cpp \
          $(SRCDIR)/connections/InhibConn.cpp \
          $(SRCDIR)/connections/KernelConn.cpp \
          $(SRCDIR)/connections/PoolConn.cpp \
          $(SRCDIR)/connections/RandomConn.cpp \
          $(SRCDIR)/connections/RuleConn.cpp \
          $(SRCDIR)/layers/HyPerLayer.cpp \
          $(SRCDIR)/layers/Gratings.cpp \
          $(SRCDIR)/layers/Image.cpp \
          $(SRCDIR)/layers/LayerDataInterface.cpp \
          $(SRCDIR)/layers/Movie.cpp \
          $(SRCDIR)/layers/Retina.cpp \
          $(SRCDIR)/layers/LIF.cpp \
          $(SRCDIR)/io/ConnectionProbe.cpp \
          $(SRCDIR)/io/LayerProbe.cpp \
          $(SRCDIR)/io/LinearActivityProbe.cpp \
          $(SRCDIR)/io/PointProbe.cpp \
          $(SRCDIR)/io/PostConnProbe.cpp \
          $(SRCDIR)/io/StatsProbe.cpp \
          $(SRCDIR)/io/PVParams.cpp \
          $(SRCDIR)/io/fileio.cpp \
          $(SRCDIR)/io/imageio.cpp \
          $(SRCDIR)/io/parser/param_parser.cpp

CPPOBJS = $(BUILDDIR)/CLBuffer.o \
          $(BUILDDIR)/CLDevice.o \
          $(BUILDDIR)/CLKernel.o \
          $(BUILDDIR)/HyPerCol.o \
          $(BUILDDIR)/HyPerColRunDelegate.o \
          $(BUILDDIR)/Communicator.o \
          $(BUILDDIR)/InterColComm.o \
          $(BUILDDIR)/DataStore.o \
          $(BUILDDIR)/HyPerLayer.o \
          $(BUILDDIR)/Gratings.o \
          $(BUILDDIR)/Image.o \
          $(BUILDDIR)/LayerDataInterface.o \
          $(BUILDDIR)/Movie.o \
          $(BUILDDIR)/HyPerConn.o \
          $(BUILDDIR)/GaborConn.o \
          $(BUILDDIR)/CocircConn.o \
          $(BUILDDIR)/InhibConn.o \
          $(BUILDDIR)/KernelConn.o \
          $(BUILDDIR)/PoolConn.o \
          $(BUILDDIR)/RandomConn.o \
          $(BUILDDIR)/RuleConn.o \
          $(BUILDDIR)/Retina.o \
          $(BUILDDIR)/LIF.o \
          $(BUILDDIR)/ConnectionProbe.o \
          $(BUILDDIR)/LayerProbe.o \
          $(BUILDDIR)/LinearActivityProbe.o \
          $(BUILDDIR)/PointProbe.o \
          $(BUILDDIR)/PostConnProbe.o \
          $(BUILDDIR)/StatsProbe.o \
          $(BUILDDIR)/Timer.o \
          $(BUILDDIR)/PVParams.o \
          $(BUILDDIR)/fileio.o \
          $(BUILDDIR)/imageio.o \
          $(BUILDDIR)/param_parser.o

CSRCS   = $(SRCDIR)/connections/PVConnection.c \
          $(SRCDIR)/layers/fileread.c \
          $(SRCDIR)/layers/PVLayer.c \
          $(SRCDIR)/io/io.c \
          $(SRCDIR)/io/tiff.c \
          $(SRCDIR)/io/parser/param_lexer.c \
          $(SRCDIR)/utils/box_muller.c \
          $(SRCDIR)/utils/cl_random.c \
          $(SRCDIR)/utils/conversions.c

COBJS   = $(BUILDDIR)/PVConnection.o \
          $(BUILDDIR)/fileread.o \
          $(BUILDDIR)/PVLayer.o \
          $(BUILDDIR)/io.o \
          $(BUILDDIR)/tiff.o \
          $(BUILDDIR)/param_lexer.o \
          $(BUILDDIR)/box_muller.o \
          $(BUILDDIR)/cl_random.o \
          $(BUILDDIR)/conversions.o

SRCS = $(CPPSRCS) $(CSRCS)

OBJS = $(CPPOBJS) $(COBJS)
