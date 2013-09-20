SRCDIR = ../../src
BUILDDIR = .

include $(SRCDIR)/include/sources.mk

INPARAMS = $(SRCDIR)/io/input/inparams.txt

CPP = mpic++
CC  = mpicc
#CPP = c++
#CC  = cc
FC = ifort

#MPI_DIR = ../../src/arch/mpi
#MPI_INC = -I$(MPI_DIR)
#MPI_OBJ = $(MPI_DIR)/mpi.o

VEC_FLAGS = -ftree-vectorize -ftree-vectorizer-verbose=3

CFLAGS = -O3 -std=c99 -Wall $(VEC_FLAGS) $(MPI_INC)
CPPFLAGS = -O3 -Wall $(VEC_FLAGS) $(MPI_INC)

DEPENDS = $(HEADERS) clock.h

LIBS = ../../src/build/libpv.a clock.o

TESTS = time_accumulate time_recv_input time_accumulate_f time_mpi

.SUFFIXES: .cpp .c .o

.cpp.o:
	$(CPP) $(CPPFLAGS) -c $(@:.o=.cpp)

.c.o:
	$(CC) $(CFLAGS) -c $(@:.o=.c)

all: $(TESTS)

time_accumulate_f: time_accumulate_f.f90 clock.o
	$(FC) -o $@ $@.f90 clock.o

time_accumulate: time_accumulate.o $(DEPENDS) $(OBJS) $(MPI_OBJ) clock.o
	$(CPP) -o $@ $@.o $(OBJS) $(MPI_OBJ) clock.o

time_recv_input: time_recv_input.o $(DEPENDS) $(OBJS) $(MPI_OBJ) clock.o
	$(CPP) -o $@ $@.o $(OBJS) $(MPI_OBJ) clock.o

time_mpi: $(BUILDDIR)/time_mpi.o $(LIBS)
	$(CPP) -o $@ $< $(LIBS)

$(MPI_OBJ): $(MPI_DIR)/mpi.h $(MPI_DIR)/mpi.c
	$(CC) $(CFLAGS) -c $(MPI_DIR)/mpi.c -o $@

runtests: $(TESTS)
	@mkdir output
	@echo "Running time_accumulate ..."
	@time_accumulate
	@echo "Running time_recv_input ..."
	@time_recv_input -n 1 -p $(INPARAMS)

clean:
	rm -rf $(TESTS) *.o *.bin

cleanall: clean
	rm -f $(MPI_OBJ)
	rm -rf output

include $(SRCDIR)/include/depend.mk