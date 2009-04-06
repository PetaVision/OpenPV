SRCDIR = ../..
BUILDDIR = $(SRCDIR)/build

include $(SRCDIR)/include/cell.mk

$(BUILDDIR)/spu/%.o: $(SRCDIR)/arch/rr/spu/%.c
	@echo 'Building file: $<'
	$(SPU_CC) $(SPU_CFLAGS) -c -o"$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

all: $(SPU_EXE) $(SPU_LIB)

$(BUILDDIR)/pv_spu: $(SPU_C_OBJS)
	$(SPU_CC) -o"$@" "$<" $(SPU_LDFLAGS)

$(SPU_EMBED): $(SPU_EXE)
	$(PPU_EMBEDSPU) $(PPU_EMBED_FLAGS) "$<" "$<" "$@"

$(SPU_LIB): $(SPU_EMBED)
	$(PPU_AR) $(PPU_AR_FLAGS) "$@" "$<"

clean:
	rm -f *.o $(BUILDDIR)/pv_spu $(SPU_EMBED) $(SPU_LIB)
