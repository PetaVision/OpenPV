PVDIR = ../..  # Path to PetaVision workspace goes here
SRCDIR = $(PVDIR)/src
BUILDDIR = $(SRCDIR)/../Debug/src

include $(SRCDIR)/include/sources.mk

CPP = g++
CC  = gcc

MPI_DIR = $(BUILDDIR)/arch/mpi
MPI_INC = -I$(MPI_DIR)
MPI_OBJ = $(MPI_DIR)/mpi.o

CFLAGS = -g -Wall -std=c99 -Wall $(MPI_INC) -I$(SRCDIR)/io
CPPFLAGS = -g -Wall $(MPI_INC)

LIBS = -lm -lpthread

DEPENDS = $(HEADERS)

EXES = circles circle_pixels create_input create_weights bin_to_tiff \
       expand_tiff test_read_tiff test_read_binary # parse_params

.SUFFIXES: .cpp .c .o

.cpp.o:
	$(CPP) $(CPPFLAGS) -c $(@:.o=.cpp)

.c.o:
	$(CC) $(CFLAGS) -c $(@:.o=.c)

all: make-in-subdirs $(EXES)

circles: circles.o
	$(CC) $(CFLAGS) -o $@ $< -lm

create_input: create_input.o
	$(CC) $(CFLAGS) -o $@ $<

circle_pixels: circle_pixels.o 
	$(CC) $(CFLAGS) -o $@ $< -lm

create_weights: create_weights.o $(DEPENDS) $(OBJS) $(MPI_OBJ)
	$(CPP) $(CPPFLAGS) -o $@ $< $(OBJS) $(MPI_OBJ) $(LIBS)

bin_to_tiff: bin_to_tiff.o tiff.o
	$(CC) $(CFLAGS) -o $@ $< tiff.o

expand_tiff: expand_tiff.o tiff.o
	$(CC) $(CFLAGS) -o $@ $< tiff.o

test_read_tiff: test_read_tiff.o tiff.o
	$(CC) $(CFLAGS) -o $@ $< tiff.o

test_read_binary: test_read_binary.o $(BUILDDIR)/io/io.o $(BUILDDIR)/io/tiff.o
	$(CC) $(CFLAGS) -o $@ $< $(BUILDDIR)/io/io.o $(BUILDDIR)/io/tiff.o $(MPI_OBJ) $(LIBS)

parse_params: param_parser.o param_lexer.o PVParams.o
	$(CPP) $(CPPFLAGS) -o $@ param_parser.o param_lexer.o PVParams.o

circles.o: circles.c ../include/pv.h
	$(CC) $(CFLAGS) -c circles.c

circle_pixels.o: circle_pixels.c ../include/pv.h
	$(CC) $(CFLAGS) -c circle_pixels.c

input/%.bin : input/%.tif
	convert $< -define quantum:format=floating-point -depth 32 gray:$@

input/%.bin : input/%.tiff
	convert $< -define quantum:format=floating-point -depth 32 gray:$@

$(MPI_OBJ): $(MPI_DIR)/mpi.h $(MPI_DIR)/mpi.c
	$(CC) $(CFLAGS) -c $(MPI_DIR)/mpi.c -o $@
	
make-in-subdirs: force
	@for subdir in $(SUBDIRS); do \
	    if [ -d $$subdir -a -f $$subdir/Makefile ] ; then \
	        echo "making in $$subdir" ; \
	        (cd $$subdir && $(MAKE) all) ; \
	    fi \
	done
	
clean: clean-in-subdirs
	rm -f param_parser.cpp param_parser.hpp param_lexer.c *.o $(EXES)

cleanall: clean
	rm -f $(MPI_OBJ)
	rm -f ./input/*.bin
	
clean-in-subdirs: force
	@for subdir in $(SUBDIRS); do \
	    if [ -d $$subdir -a -f $$subdir/Makefile ] ; then \
	        echo "cleaning in $$subdir" ; \
	        (cd $$subdir && $(MAKE) clean) ; \
	    fi \
	done

force:



include $(SRCDIR)/include/depend.mk
