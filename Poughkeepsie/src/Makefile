#MPI_INC = -I./mpi
#MPI_OBJ = mpi/mpi.o

CC = mpicc

CFLAGS = -g -I./ $(MPI_INC) -ftree-vectorize -ftree-vectorizer-verbose=2

all: pv

pv: pv.o pv_init.o pv_ppu.o pv_spu.o pv_output.o mpi_harness.o
	$(CC) -o pv pv.o pv_ppu.o pv_spu.o pv_init.o pv_output.o mpi_harness.o $(MPI_OBJ) -lm

pv.o: pv.c pv.h
	$(CC) $(CFLAGS) -c pv.c

pv_ppu.o: pv_ppu.c pv.h
	$(CC) $(CFLAGS) -c pv_ppu.c

pv_spu.o: pv_spu.c pv.h
	$(CC) $(CFLAGS) -c pv_spu.c

pv_output.o: pv_output.c pv.h
	$(CC) $(CFLAGS) -c pv_output.c

pv_init.o: pv_init.c pv.h
	$(CC) $(CFLAGS) -c pv_init.c

arch/mpi/mpi_harness.o: arch/mpi/mpi_harness.c pv.h
	$(CC) $(CFLAGS) -c arch/mpi/mpi_harness.c

clean:
	rm -f *.o pv
