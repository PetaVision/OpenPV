SRCDIR = ../src
BUILDDIR = build

include $(SRCDIR)/include/sources.mk

#MPI_DIR = arch/mpi
#MPI_INC = -I./$(MPI_DIR)
#MPI_OBJ = $(MPI_DIR)/mpi.o

CC = mpicc
CPP = mpic++
#CC = gcc
#CPP = g++

MAKE = make

# VEC_FLAGS = -ftree-vectorize -ftree-vectorizer-verbose=2
CFLAGS = -g -std=c99 -Wall -I./ $(MPI_INC) $(VEC_FLAGS) -DHAS_MAIN=1
CPPFLAGS = -g -Wall -I./ $(MPI_INC) $(VEC_FLAGS) -DHAS_MAIN=1

EXES = pv_example

%.o: %.cpp
	@echo 'Building file: $<'
	$(CPP) -c $(CPPFLAGS) -o $@ $<
	@echo 'Finished building: $<'
	@echo ' '

#all: libpv.a $(EXES)

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
