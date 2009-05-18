#SUBDIRS = $(BUILDDIR)/ppu \
#          $(BUILDDIR)/spu \
#          $(BUILDDIR)/pthreads

SUBDIRS = $(BUILDDIR)/pthreads

HEADERS = $(SRCDIR)/include/pv_arch.h \
          $(SRCDIR)/include/pv_common.h \
          $(SRCDIR)/columns/HyPerCol.hpp \
          $(SRCDIR)/connections/HyPerConn.hpp \
          $(SRCDIR)/connections/GaborConn.hpp \
          $(SRCDIR)/connections/CocircConn.hpp \
          $(SRCDIR)/connections/LongRangeConn.hpp \
          $(SRCDIR)/connections/PoolConn.hpp \
          $(SRCDIR)/connections/RandomConn.hpp \
          $(SRCDIR)/connections/RuleConn.hpp \
          $(SRCDIR)/connections/PVConnection.h \
          $(SRCDIR)/layers/elementals.h \
          $(SRCDIR)/layers/LIF2.h \
          $(SRCDIR)/layers/PVLayer.h \
          $(SRCDIR)/io/PVLayerProbe.hpp \
          $(SRCDIR)/io/LinearActivityProbe.hpp \
          $(SRCDIR)/io/PointProbe.hpp \
          $(SRCDIR)/io/StatsProbe.hpp \
          $(SRCDIR)/io/PVParams.hpp \
          $(SRCDIR)/io/tiff.h

CPPSRCS = $(SRCDIR)/columns/HyPerCol.cpp \
          $(SRCDIR)/columns/InterColComm.cpp \
          $(SRCDIR)/columns/DataStore.cpp \
          $(SRCDIR)/connections/HyPerConn.cpp \
          $(SRCDIR)/connections/GaborConn.cpp \
          $(SRCDIR)/connections/CocircConn.cpp \
          $(SRCDIR)/connections/InhibConn.cpp \
          $(SRCDIR)/connections/LongRangeConn.cpp \
          $(SRCDIR)/connections/PoolConn.cpp \
          $(SRCDIR)/connections/RandomConn.cpp \
          $(SRCDIR)/connections/RuleConn.cpp \
          $(SRCDIR)/layers/Example.cpp \
          $(SRCDIR)/layers/HyPerLayer.cpp \
          $(SRCDIR)/layers/Retina.cpp \
          $(SRCDIR)/layers/LGN.cpp \
          $(SRCDIR)/layers/V1.cpp \
          $(SRCDIR)/io/PVLayerProbe.cpp \
          $(SRCDIR)/io/LinearActivityProbe.cpp \
          $(SRCDIR)/io/PointProbe.cpp \
          $(SRCDIR)/io/StatsProbe.cpp \
          $(SRCDIR)/io/PVParams.cpp \
          $(SRCDIR)/io/parser/param_parser.cpp

CPPOBJS = $(BUILDDIR)/HyPerCol.o \
          $(BUILDDIR)/InterColComm.o \
          $(BUILDDIR)/DataStore.o \
          $(BUILDDIR)/HyPerLayer.o \
          $(BUILDDIR)/HyPerConn.o \
          $(BUILDDIR)/GaborConn.o \
          $(BUILDDIR)/CocircConn.o \
          $(BUILDDIR)/InhibConn.o \
          $(BUILDDIR)/LongRangeConn.o \
          $(BUILDDIR)/PoolConn.o \
          $(BUILDDIR)/RandomConn.o \
          $(BUILDDIR)/RuleConn.o \
          $(BUILDDIR)/Retina.o \
          $(BUILDDIR)/Example.o \
          $(BUILDDIR)/LGN.o \
          $(BUILDDIR)/V1.o \
          $(BUILDDIR)/PVLayerProbe.o \
          $(BUILDDIR)/LinearActivityProbe.o \
          $(BUILDDIR)/PointProbe.o \
          $(BUILDDIR)/StatsProbe.o \
          $(BUILDDIR)/PVParams.o \
          $(BUILDDIR)/param_parser.o

CSRCS   = $(SRCDIR)/connections/PVConnection.c \
          $(SRCDIR)/layers/fileread.c \
          $(SRCDIR)/layers/LIF2.c \
          $(SRCDIR)/layers/PVLayer.c \
          $(SRCDIR)/io/io.c \
          $(SRCDIR)/io/tiff.c \
          $(SRCDIR)/io/parser/param_lexer.c

COBJS   = $(BUILDDIR)/PVConnection.o \
          $(BUILDDIR)/fileread.o \
          $(BUILDDIR)/LIF2.o \
          $(BUILDDIR)/PVLayer.o \
          $(BUILDDIR)/io.o \
          $(BUILDDIR)/tiff.o \
          $(BUILDDIR)/param_lexer.o \
          $(BUILDDIR)/pthreads/pv_thread.o

SRCS = $(CPPSRCS) $(CSRCS)

OBJS = $(CPPOBJS) $(COBJS)
