PVDIR = ../../..
SRCDIR = ../../../src

CPP = g++
CC  = gcc

MPI_DIR = $(SRCDIR)/arch/mpi
MPI_INC = -I$(MPI_DIR)

CFLAGS = -g -Wall -std=c99 -Wall -I$(PVDIR) $(MPI_INC)
CPPFLAGS = -g -Wall -I$(PVDIR) $(MPI_INC)

LIBS = -L../../../lib -lpv -lm -lpthread

DEPENDS = $(HEADERS)

EXES = weights circles circle_pixels create_input test_read_tiff \
       bin_to_tiff expand_tiff # create_weights parse_params test_read_binary

EXES = weights

.SUFFIXES: .cpp .c .o

%.o: %.cpp
	@echo 'Building file: $<'
	$(CPP) $(CPPFLAGS) -c -o"$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

%.o: %.c
	@echo 'Building file: $<'
	$(CC) $(CFLAGS) -c -o"$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

all: $(EXES)

weights: weights.o
	$(CPP) -o $@ $< $(LIBS)

circles: circles.o
	$(CC) -o $@ $< $(LIBS)

create_input: create_input.o
	$(CC) -o $@ $< $(LIBS)

circle_pixels: circle_pixels.o 
	$(CC) -o $@ $< $(LIBS)

create_weights: create_weights.o $(DEPENDS) $(OBJS) $(MPI_OBJ)
	$(CPP) -o $@ $< $(LIBS)

bin_to_tiff: bin_to_tiff.o
	$(CC) -o $@ $< $(LIBS)

expand_tiff: expand_tiff.o
	$(CC) -o $@ $< $(LIBS)

test_read_tiff: test_read_tiff.o
	$(CC) -o $@ $< $(LIBS)

test_read_binary: test_read_binary.o
	$(CC) -o $@ $< $(LIBS)

parse_params: param_parser.o param_lexer.o PVParams.o
	$(CPP) -o $@ param_parser.o param_lexer.o PVParams.o

input/%.bin : input/%.tif
	convert $< -define quantum:format=floating-point -depth 32 gray:$@

input/%.bin : input/%.tiff
	convert $< -define quantum:format=floating-point -depth 32 gray:$@

clean:
	rm -f param_parser.cpp param_parser.hpp param_lexer.c *.o $(EXES)

force:
