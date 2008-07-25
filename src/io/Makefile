MPI_INC = -I./arch/mpi
#MPI_OBJ = mpi.o

#CC = mpicc
CC = cc

CCFLAGS = -g -I./ $(MPI_INC) 
all: circles

circles: circles.o #mpi_harness.o $(MPI_OBJ)
	$(CC) -o circles circles.o  -lm

circles.o: circles.c ../pv.h
	$(CC) $(CCFLAGS) -c circles.c

#$(MPI_OBJ): ../arch/mpi/mpi.c ../arch/mpi/mpi.h
#      $(CC) $(CCFLAGS) -c ../arch/mpi/mpi.c

#mpi_harness.o: ../arch/mpi/mpi_harness.c ../arch/mpi/mpi.h
#	$(CC) $(CCFLAGS) -c ../arch/mpi/mpi_harness.c
clean:
	rm -f *.o circles
#	rm -f ./input/*.bin