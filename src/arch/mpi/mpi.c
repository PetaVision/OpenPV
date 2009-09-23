#ifdef PV_USE_LOCAL_MPI

#include "mpi.h"
#include <assert.h>
#include <string.h>
#include <time.h>

/* TODO - add a timer to MPI_Wtime */

/* the following items are for a very special form recv/send (recv immediately followed by send) */
char * PV_MPI_recvBuf;
int    PV_MPI_recvCount;

/**
 * @argc
 * @argv
 */

int MPI_Init(int * argc, char *** argv)
{
   return 0;
}

int MPI_Finalize()
{
   return 0;
}

/**
 * @comm
 */

int MPI_Barrier(MPI_Comm comm)
{
   return 0;
}

/**
 * @buffer
 * @count
 * @datatype
 * @root
 * @comm
 */

int MPI_Bcast (void * buffer, int count, MPI_Datatype datatype, int root,
               MPI_Comm comm)
{
   return 0;
}

/**
 * @comm
 * @rank
 */

int MPI_Comm_rank(MPI_Comm comm, int* rank)
{
   *rank = 0;
   return 0;
}

/**
 * @comm
 * @size
 */

int MPI_Comm_size(MPI_Comm comm, int* size)
{
   *size = 1;
   return 0;
}

/**
 * @sendbuf
 * @sendcount
 * @sendtype
 * @recvbuf
 * @recvcount
 * @recvtype
 * @root
 * @comm
 */

int MPI_Gather(void * sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf,
      int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm)
{
   // should be able to just copy from sendbuf to recvbuf as MPI size is 1
   memcpy(recvbuf, sendbuf, sendtype*sendcount);
   return 0;
}

/**
 * @buf
 * @count
 * @datatype
 * @source
 * @tag
 * @comm
 * @request
 */

int MPI_Irecv(void * buf, int count, MPI_Datatype datatype, int source, int tag,
              MPI_Comm comm, MPI_Request *request)
{
   assert(datatype == MPI_CHAR);
   PV_MPI_recvBuf = buf;
   PV_MPI_recvCount = count;
   return 0;
}

/**
 * @buf
 * @count
 * @datatype
 * @dest
 * @tag
 * @comm
 */

int MPI_Send(void * buf, int count, MPI_Datatype datatype, int dest, int tag,
             MPI_Comm comm)
{
   /* WARNING, this only works for MPI_Irecv immediately followed by MPI_Send */
   assert(datatype == MPI_CHAR);
   assert(count == PV_MPI_recvCount);
   memcpy(PV_MPI_recvBuf, buf, count);

   return 0;
}

/**
 * @count
 * @array_of_requests
 * @array_of_statuses
 */

int MPI_Waitall(int count, MPI_Request array_of_requests[],
                MPI_Status array_of_statuses[])
{
   return 0;
}

/**
 * @count
 * @array_of_requests
 * @index
 * @status
 */

int MPI_Waitany(int count, MPI_Request array_of_requests[], int * index,
      MPI_Status * status)
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

#endif // PV_USE_LOCAL_MPI
