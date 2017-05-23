#include <cMakeHeader.h>

#ifdef PV_USE_MPI
#include <mpi.h>
#else // PV_USE_MPI

// stubs for
#ifndef MPI_H_
#define MPI_H_

typedef void *voidptr;

#define MPI_Request int
#define MPI_Datatype int
#define MPI_Request int
#define MPI_Status int
#define MPI_Comm voidptr
#define MPI_Op int

#define MPI_COMM_WORLD ((voidptr)1)
#define MPI_BYTE 1
#define MPI_CHAR 1
#define MPI_UNSIGNED_CHAR 1
#define MPI_INT (sizeof(int))
#define MPI_UNSIGNED (sizeof(unsigned int))
#define MPI_LONG (sizeof(long))
#define MPI_FLOAT (sizeof(float))
#define MPI_DOUBLE (sizeof(double))
#define MPI_STATUS_IGNORE 0
#define MPI_STATUSES_IGNORE NULL
#define MPI_IN_PLACE ((voidptr)1)
#define MPI_MAX 1
#define MPI_MIN 2
#define MPI_SUM 3
#define MPI_PROD 4
#define MPI_LAND 5
#define MPI_BAND 6
#define MPI_LOR 7
#define MPI_BOR 8
#define MPI_LXOR 9
#define MPI_BXOR 10
#define MPI_MAXLOC 11
#define MPI_MINLOC 12
#define MPI_REPLACE 13

#ifdef __cplusplus
extern "C" {
#endif

int MPI_Initialized(int *flag);
int MPI_Init(int *argc, char ***argv);
int MPI_Finalize();

int MPI_Barrier(MPI_Comm comm);

int MPI_Bcast(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm);

int MPI_Iallreduce(
      void *sendbuf,
      void *recvbuf,
      int count,
      MPI_Datatype datatype,
      MPI_Op op,
      MPI_Comm comm,
      MPI_Request *request);

int MPI_Allreduce(
      void *sendbuf,
      void *recvbuf,
      int count,
      MPI_Datatype datatype,
      MPI_Op op,
      MPI_Comm comm);

int MPI_Reduce(
      void *sendbuf,
      void *recvbuf,
      int count,
      MPI_Datatype datatype,
      MPI_Op op,
      int root,
      MPI_Comm comm);

int MPI_Comm_rank(MPI_Comm comm, int *rank);
int MPI_Comm_size(MPI_Comm comm, int *size);

int MPI_Irecv(
      void *buf,
      int count,
      MPI_Datatype datatype,
      int source,
      int tag,
      MPI_Comm comm,
      MPI_Request *request);
int MPI_Recv(
      void *buf,
      int count,
      MPI_Datatype datatype,
      int source,
      int tag,
      MPI_Comm comm,
      MPI_Status *status);
int MPI_Isend(
      void *buf,
      int count,
      MPI_Datatype datatype,
      int dest,
      int tag,
      MPI_Comm comm,
      MPI_Request *request);
int MPI_Send(void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm);
int MPI_Testall(int count, MPI_Request *reqs, int *flag, MPI_Status *stats);
int MPI_Waitall(int count, MPI_Request array_of_requests[], MPI_Status array_of_statuses[]);

int MPI_Comm_split(MPI_Comm comm, int color, int key, MPI_Comm *newcomm);

double MPI_Wtime();

#ifdef __cplusplus
}
#endif

#endif /* MPI_H_ */

#endif // PV_USE_MPI
