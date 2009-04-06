SRCDIR = ../..
BUILDDIR = $(SRCDIR)/build

include $(SRCDIR)/include/cell.mk

$(BUILDDIR)/ppu/%.o: $(SRCDIR)/arch/rr/ppu/%.c
	@echo 'Building file: $<'
	$(PPU_CC) $(PPU_CFLAGS) -c -o"$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

all: $(PPU_EXE)

$(BUILDDIR)/pv_ppu: $(PPU_C_OBJS) $(SPU_LIB)
	$(PPU_CC) -o"$@" $(PPU_C_OBJS) $(PPU_LDFLAGS) $(PPU_LIBS)

clean:
	rm -f *.o $(BUILDDIR)/pv_ppu
