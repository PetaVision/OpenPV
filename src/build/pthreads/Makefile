SRCDIR = ../../../src
BUILDDIR = ..

$(BUILDDIR)/pthreads/%.o: $(SRCDIR)/arch/pthreads/%.c
	@echo 'Building file: $<'
	$(CC) -c -o"$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

all: $(BUILDDIR)/pthreads/pv_thread.o

clean:
	rm -f *.o
