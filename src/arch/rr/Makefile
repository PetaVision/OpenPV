SPU_APP = pv_spu
PPU_APP = pv_ppu

XLC_TOP = /opt/ibmcmp/xlc/cbe/9.0
SDK_TOP = /opt/cell/sdk
SDK_BIN = /opt/cell/toolchain/bin

SPU_CC	= $(XLC_TOP)/bin/spuxlc
PPU_CC	= $(XLC_TOP)/bin/ppuxlc
PPU_CPP = ppu-g++

PPU_EMBEDSPU = $(SDK_BIN)/ppu-embedspu
PPU_AR = $(SDK_BIN)/ppu-ar

OPT = -g
SPU_INC = -I. -I../ -I $(SDK_TOP)/usr/spu/include
PPU_INC = -I. -I $(SDK_TOP)/usr/include

PPU_VEC_FLAGS = -qaltivec -qenablevmx
GCC_ARCH_FLAG = -m64
XLC_ARCH_FLAG = -q64
PPU_ARCH_FLAG = $(GCC_ARCH_FLAG)

SPU_CFLAGS = -qcpluscmt -M -ma $(SPU_INC) $(OPT)
PPU_CFLAGS = -q64 -qcpluscmt -M -ma $(PPU_INC) $(PPU_VEC_FLAGS) $(OPT)
PPU_CCFLAGS = $(PPU_ARCH_FLAG)

PPU_LDFLAGS = -L$(SDK_TOP)/usr/lib64 -R$(SDK_TOP)/usr/lib64 $(PPU_ARCH_FLAG) -Wl,-m,elf64ppc
PPU_LIBS = lib_pv_spu.a -lspe2

SPU_BLD = $(SPU_APP) $(SPU_APP)-embed64.o lib_$(SPU_APP).a
PPU_BLD = pv # $(PPU_APP)

# Each subdirectory must supply rules for building sources it contributes
%.o: %.cpp
	@echo 'Building file: $@'
	@echo 'Invoking: GCC C++ Compiler'
	$(PPU_CPP) $(PPU_CCFLAGS) -c $<
	@echo 'Finished building: $@'
	@echo ' '

all: $(SPU_BLD) $(PPU_BLD)

pv: pv.o pv_ppu.o lib_pv_spu.a
	$(PPU_CPP) -o $@ $< pv_ppu.o $(PPU_LDFLAGS) $(PPU_LIBS)

pv_spu.o: pv_spu.c
	$(SPU_CC) $(SPU_CFLAGS) $(INCLUDES) -c $<

pv_spu: pv_spu.o
	$(SPU_CC) -o $@ $< -Wl,-N

pv_spu-embed64.o: pv_spu
	$(PPU_EMBEDSPU) -m64 $< $< $@

lib_pv_spu.a: pv_spu-embed64.o
	$(PPU_AR) -qcs $@ $<

pv_ppu.o: pv_ppu.c
	$(PPU_CC) $(PPU_CFLAGS) -c $<

pv_ppu: pv_ppu.o lib_pv_spu.a
	$(PPU_CC) -o $@ $< $(PPU_LDFLAGS) $(PPU_LIBS)

clean:
	rm -f *.o *.a *.d $(SPU_BLD) $(PPU_BLD)

#
# output from make
#
#/opt/ibmcmp/xlc/cbe/9.0/bin/spuxlc -qcpluscmt -M -ma -I. -I../ -I /opt/cell/sdk/usr/spu/include  -g	  -c pv_spu.c
#/opt/ibmcmp/xlc/cbe/9.0/bin/spuxlc -o pv_spu pv_spu.o  -Wl,-N	
#/opt/cell/toolchain/bin/ppu-embedspu -m64 pv_spu pv_spu pv_spu-embed64.o
#/opt/cell/toolchain/bin/ppu-ar -qcs lib_pv_spu.a pv_spu-embed64.o
#/opt/ibmcmp/xlc/cbe/9.0/bin/ppuxlc -q64 -qcpluscmt -M -ma -I. -I /opt/cell/sdk/usr/include -qaltivec -qenablevmx -g -c pv_ppu.c
#/opt/ibmcmp/xlc/cbe/9.0/bin/ppuxlc -o pv_ppu pv_ppu.o -L/opt/cell/sdk/usr/lib64 -R/opt/cell/sdk/usr/lib64 -q64 -Wl,-m,elf64ppc lib_pv_spu.a -lspe2
