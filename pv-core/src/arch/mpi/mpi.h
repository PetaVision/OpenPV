//#include "../../include/cMakeHeader.h"
#include <cMakeHeader.h>

#ifdef PV_USE_MPI
#include <mpi.h>
#else // PV_USE_MPI

// stubs for 
#ifndef MPI_H_
#define MPI_H_

typedef void * voidptr;

#define MPI_Request  int
#define MPI_Datatype int
#define MPI_Request  int
#define MPI_Status   int
#define MPI_Comm     voidptr
#define MPI_Op       int

#define MPI_COMM_WORLD     ((voidptr) 1)
#define MPI_BYTE           1
#define MPI_CHAR           1
#define MPI_UNSIGNED_CHAR  1
#define MPI_INT            (sizeof(int))
#define MPI_UNSIGNED       (sizeof(unsigned int))
#define MPI_LONG           (sizeof(long))
#define MPI_FLOAT          (sizeof(float))
#define MPI_DOUBLE         (sizeof(double))
#define MPI_STATUS_IGNORE  0
#define MPI_IN_PLACE       ((voidptr) 1)
#define MPI_MAX            0
#define MPI_MIN            1
#define MPI_SUM            2

#ifdef __cplusplus
extern "C"
{
#endif

int MPI_Initialized(int * flag);
int MPI_Init(int * argc, char *** argv);
int MPI_Finalize();

int MPI_Barrier(MPI_Comm comm);

int MPI_Bcast (void * buffer, int count, MPI_Datatype datatype, int root,
               MPI_Comm comm);

int MPI_Allreduce(void * sendbuf, void * recvbuf, int count, MPI_Datatype datatype,
                  MPI_Op op, MPI_Comm comm);

int MPI_Reduce(void * sendbuf, void * recvbuf, int count, MPI_Datatype datatype,
               MPI_Op op, int root, MPI_Comm comm);

int MPI_Comm_rank(MPI_Comm comm, int * rank);
int MPI_Comm_size(MPI_Comm comm, int * size);

int MPI_Irecv( void * buf, int count, MPI_Datatype datatype, int source,
               int tag, MPI_Comm comm, MPI_Request *request );
int MPI_Recv(void * buf, int count, MPI_Datatype datatype, int source, int tag,
              MPI_Comm comm, MPI_Status *status);
int MPI_Isend(void * buf, int count, MPI_Datatype datatype, int dest, int tag,
              MPI_Comm comm, MPI_Request * request);
int MPI_Send( void * buf, int count, MPI_Datatype datatype, int dest,
              int tag, MPI_Comm comm );
int MPI_Waitall(
        int count,
        MPI_Request array_of_requests[],
        MPI_Status array_of_statuses[] );

double MPI_Wtime();

#ifdef __cplusplus
}
#endif

#endif /* MPI_H_ */

#endif // PV_USE_MPI


