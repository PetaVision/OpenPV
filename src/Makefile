MPI_INC = -I./arch/mpi
MPI_OBJ = mpi.o

#CC = mpicc
CC = cc

# VEC_FLAGS = -ftree-vectorize -ftree-vectorizer-verbose=2
CFLAGS = -g -I./ $(MPI_INC) $(VEC_FLAGS)

all: pv

pv: pv.o PVHyperCol.o PVLayer.o retina.o zucker.o neighbor.o pv_output.o mpi_harness.o $(MPI_OBJ)
	$(CC) -o pv pv.o PVHyperCol.o PVLayer.o retina.o zucker.o neighbor.o pv_output.o mpi_harness.o $(MPI_OBJ) -lm

pv.o: pv.c pv.h
	$(CC) $(CFLAGS) -c pv.c

PVHyperCol.o: columns/PVHyperCol.c pv.h columns/PVHyperCol.h layers/PVLayer.h
	$(CC) $(CFLAGS) -c columns/PVHyperCol.c

neighbor.o: columns/neighbor.c pv.h columns/PVHyperCol.h layers/PVLayer.h
	$(CC) $(CFLAGS) -c columns/neighbor.c

PVLayer.o: layers/PVLayer.c pv.h columns/PVHyperCol.h layers/PVLayer.h
	$(CC) $(CFLAGS) -c layers/PVLayer.c

retina.o: layers/retina.c layers/retina.h pv.h columns/PVHyperCol.h layers/PVLayer.h
	$(CC) $(CFLAGS) -c layers/retina.c

zucker.o: layers/zucker.c layers/zucker.h pv.h columns/PVHyperCol.h layers/PVLayer.h
	$(CC) $(CFLAGS) -c layers/zucker.c

pv_output.o: io/pv_output.c pv.h columns/PVHyperCol.h layers/PVLayer.h
	$(CC) $(CFLAGS) -c io/pv_output.c

mpi_harness.o: arch/mpi/mpi_harness.c pv.h
	$(CC) $(CFLAGS) -c arch/mpi/mpi_harness.c

mpi.o: arch/mpi/mpi.c
	$(CC) $(CFLAGS) -c arch/mpi/mpi.c

clean:
	rm -f *.o pv
	rm -f ./output/*.ps ./output/*.bin
