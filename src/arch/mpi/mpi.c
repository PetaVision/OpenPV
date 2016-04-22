#include "mpi.h"

#ifndef PV_USE_MPI

#include <assert.h>
#include <string.h>
#include <time.h>

int pvmpiInitialized = 0;

/**
 * A stub for MPI_Initialized() when PV_USE_MPI is off.  It
 * returns the value of the flag that MPI_Init turns on and MPI_Finalize turns off.
 */
 int MPI_Initialized(int * flag)
 {
    *flag = pvmpiInitialized;
    return 0;
 }

/**
 * A stub for MPI_Init() when PV_USE_MPI is off.  It sets the global variable
 * pvmpiInitialized to true.
 */
int MPI_Init(int * argc, char *** argv)
{
   pvmpiInitialized = 1;
   return 0;
}

/**
 * A stub for MPI_Finalize() when PV_USE_MPI is off.  It sets the global variable
 * pvmpiInitialized to false.
 */
int MPI_Finalize()
{
   pvmpiInitialized = 0;
   return 0;
}

/**
 * A stub for MPI_Barrier() when PV_USE_MPI is off.
 */

int MPI_Barrier(MPI_Comm comm)
{
   return 0;
}

/**
 * A stub for MPI_Bcast when PV_USE_MPI is off */

int MPI_Bcast (void * buffer, int count, MPI_Datatype datatype, int root,
               MPI_Comm comm)
{
   return 0;
}

/**
 * A stub for MPI_Allreduce when PV_USE_MPI is off.  Copies recvbuf into sendbuf
 * (if MPI_IN_PLACE is used or if sendbuf==recvbuf, returns immediately).
 */
int MPI_Allreduce(void * sendbuf, void * recvbuf, int count, MPI_Datatype datatype,
                  MPI_Op op, MPI_Comm comm)
{
   if (sendbuf != MPI_IN_PLACE && sendbuf != recvbuf) {
      memmove(recvbuf, sendbuf, count*datatype);
   }
   return 0;
}

/**
 * A stub for MPI_Allreduce when PV_USE_MPI is off.  Copies recvbuf into sendbuf
 * (if MPI_IN_PLACE is used or if sendbuf==recvbuf, returns immediately).
 * The root argument is not read.
 */
int MPI_Reduce(void * sendbuf, void * recvbuf, int count, MPI_Datatype datatype,
                  MPI_Op op, int root, MPI_Comm comm)
{
   if (sendbuf != MPI_IN_PLACE && sendbuf != recvbuf) {
      memmove(sendbuf, recvbuf, count*datatype);
   }
   return 0;
}

/**
 * A stub for MPI_Comm_rank when PV_USE_MPI is off.  The resulting rank is always zero.
 */

int MPI_Comm_rank(MPI_Comm comm, int* rank)
{
   *rank = 0;
   return 0;
}

/**
 * A stub for MPI_Comm_size when PV_USE_MPI is off.  The resulting size is always one.
 */

int MPI_Comm_size(MPI_Comm comm, int* size)
{
   *size = 1;
   return 0;
}

/**
 * A stub for MPI_Irecv when PV_USE_MPI is off.  Returns immediately.
 * Note that if the code calls this function when PV_USE_MPI is false,
 * the receive buffer does not get written to.  A better implementation of MPI_Irecv
 * would need to be implemented if that's ever something we find convenient to do.
 */

int MPI_Irecv(void * buf, int count, MPI_Datatype datatype, int source, int tag,
              MPI_Comm comm, MPI_Request *request)
{
   return 0;
}

/**
 * A stub for MPI_Recv when PV_USE_MPI is off.  Returns immediately.
 * Note that if the code calls this function when PV_USE_MPI is false,
 * the receive buffer does not get written to.  A better implementation of MPI_Recv
 * would need to be implemented if that's ever something we find convenient to do.
 */

int MPI_Recv(void * buf, int count, MPI_Datatype datatype, int source, int tag,
              MPI_Comm comm, MPI_Status *status)
{
   return 0;
}

/**
 * A stub for MPI_Isend when PV_USE_MPI is off.  Returns immediately.
 * Note that if the code calls this function when PV_USE_MPI is false,
 * the send buffer does not get read.  A better implementation of MPI_Isend
 * would need to be implemented if that's ever something we find convenient to do.
 */

int MPI_Isend(void * buf, int count, MPI_Datatype datatype, int dest, int tag,
              MPI_Comm comm, MPI_Request * request)
{
   return 0;
}

/**
 * A stub for MPI_Send when PV_USE_MPI is off.  Returns immediately.
 * Note that if the code calls this function when PV_USE_MPI is false,
 * the send buffer does not get read.  A better implementation of MPI_Send
 * would need to be implemented if that's ever something we find convenient to do.
 */
int MPI_Send(void * buf, int count, MPI_Datatype datatype, int dest, int tag,
             MPI_Comm comm)
{
   return 0;
}

/**
 * A stub for MPI_Waitall when PV_USE_MPI is off.  Returns immediately.
 */
int MPI_Waitall(int count, MPI_Request array_of_requests[],
                MPI_Status array_of_statuses[])
{
   return 0;
}

/**
 * A stub for MPI_Wtime when PV_USE_MPI is off.  Returns the time in seconds
 * since the calling process started.
 */
double MPI_Wtime()
{
   return clock() / CLOCKS_PER_SEC;
}

#else

void pv_mpi_noop() { }

#endif // PV_USE_MPI
