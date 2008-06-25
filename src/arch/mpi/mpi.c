#include "mpi.h"
#include <time.h>

/* TODO - add a timer to MPI_Wtime */

int MPI_Init(int *argc, char ***argv)
  {
    return 0;
  }

int MPI_Finalize()
  {
    return 0;
  }

int MPI_Comm_rank(MPI_Comm comm, int* rank)
  {
    *rank = 0;
    return 0;
  }

int MPI_Comm_size(MPI_Comm comm, int* size)
  {
    *size = 1;
    return 0;
  }

int MPI_Gather(void *sendbuf, int sendcount, MPI_Datatype sendtype,
    void *recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm)
  {
    return 0;
  }

int MPI_Irecv(void *buf, int count, MPI_Datatype datatype, int source, int tag,
    MPI_Comm comm, MPI_Request *request)
  {
    return 0;
  }

int MPI_Send(void *buf, int count, MPI_Datatype datatype, int dest, int tag,
    MPI_Comm comm)
  {
    return 0;
  }

int MPI_Waitall(int count, MPI_Request array_of_requests[],
    MPI_Status array_of_statuses[])
  {
    return 0;
  }

int MPI_Waitany(int count, MPI_Request array_of_requests[], int *index,
    MPI_Status *status)
  {
    return 0;
  }

double MPI_Wtime()
  {
    return clock() / CLOCKS_PER_SEC;
  }

double MPI_Wtick()
  {
    return CLOCKS_PER_SEC;
  }
