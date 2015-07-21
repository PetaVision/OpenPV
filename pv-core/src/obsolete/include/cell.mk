XLC_TOP = /opt/ibmcmp/xlc/cbe/9.0
SDK_TOP = /opt/cell/sdk
SDK_BIN = /opt/cell/toolchain/bin

# PPU artifacts
PPU_C_SRCS = $(SRCDIR)/arch/rr/ppu/pv_ppu.c
PPU_C_OBJS = $(BUILDDIR)/ppu/pv_ppu.o
PPU_EXE = $(BUILDDIR)/pv_ppu

#SPU artifacts
SPU_C_SRCS = $(SRCDIR)/arch/rr/spu/pv_spu.c
SPU_C_OBJS = $(BUILDDIR)/spu/pv_spu.o
SPU_EMBED  = $(BUILDDIR)/spu/pv_spu-embed64.o
SPU_LIB    = $(BUILDDIR)/spu/lib_pv_spu.a
SPU_EXE    = $(BUILDDIR)/pv_spu

SPU_CC  = $(XLC_TOP)/bin/spuxlc
PPU_CC  = $(XLC_TOP)/bin/ppuxlc
PPU_CPP = $(XLC_TOP)/bin/ppuxlc++

PPU_EMBED_FLAGS = -m64
PPU_EMBEDSPU    = $(SDK_BIN)/ppu-embedspu

PPU_AR_FLAGS = -qcs
PPU_AR       = $(SDK_BIN)/ppu-ar

OPT = -g
SPU_INC = -I. -I../ -I $(SDK_TOP)/usr/spu/include 
PPU_INC = -I. -I $(SDK_TOP)/usr/include

PPU_VEC_FLAGS = -qaltivec -qenablevmx

SPU_CFLAGS = -qcpluscmt -M -ma $(SPU_INC) $(OPT)
PPU_CFLAGS = -q64 -qcpluscmt -M -ma $(PPU_INC) $(PPU_VEC_FLAGS) $(OPT)

SPU_LDFLAGS = -Wl,-N
PPU_LDFLAGS = -L$(SDK_TOP)/usr/lib64 -R$(SDK_TOP)/usr/lib64 -q64 -Wl,-m,elf64ppc
PPU_LIBS = $(SPU_LIB) -lspe2

#
# Example output from make
# changed optimization and -I include path
#
#/opt/ibmcmp/xlc/cbe/9.0/bin/spuxlc -qcpluscmt -M -ma -I. -I../ -I /opt/cell/sdk/usr/spu/include  -g -c pv_spu.c
#/opt/ibmcmp/xlc/cbe/9.0/bin/spuxlc -qcpluscmt -M -ma -I. -I../ -I /opt/cell/sdk/usr/spu/include  -g -c pv_spu.c

#/opt/ibmcmp/xlc/cbe/9.0/bin/spuxlc -o pv_spu  pv_spu.o -Wl,-N     

#/opt/cell/toolchain/bin/ppu-embedspu -m64 pv_spu pv_spu pv_spu-embed64.o

#/opt/cell/toolchain/bin/ppu-ar -qcs lib_pv_spu.a pv_spu-embed64.o

#/opt/ibmcmp/xlc/cbe/9.0/bin/ppuxlc -q64 -qcpluscmt -M -ma -I. -I /opt/cell/sdk/usr/include -qaltivec -qenablevmx -g -c pv_ppu.c

#/opt/ibmcmp/xlc/cbe/9.0/bin/ppuxlc -o pv_ppu pv_ppu.o -L/users/rasmussn/Codes/sdk/sysroot/opt/cell/sdk/usr/lib64 -R/opt/cell/sdk/usr/lib64  -q64 -Wl,-m,elf64ppc spu/lib_pv_spu.a -lspe2
#/opt/ibmcmp/xlc/cbe/9.0/bin/ppuxlc -o pv_ppu pv_ppu.o  lib_pv_spu.a -lspe2

