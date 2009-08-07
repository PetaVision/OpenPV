SRCDIR = ../../../src
BUILDDIR = ..

$(BUILDDIR)/pthreads/%.o: $(SRCDIR)/arch/pthreads/%.c
	@echo 'Building file: $<'
	$(CC) -c -o"$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

#all: $(BUILDDIR)/pthreads/pv_thread.o

all:
	@echo 'No longer building with pthreads'

clean:
	rm -f *.o
