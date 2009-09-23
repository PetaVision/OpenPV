#ifdef PV_USE_LOCAL_MPI

#define MPI_Request  int
#define MPI_Datatype int
#define MPI_Request  int
#define MPI_Status   int
#define MPI_Comm     int

#define MPI_COMM_WORLD     1
#define MPI_CHAR           1
#define MPI_FLOAT          4
#define MPI_STATUS_IGNORE  0

#ifdef __cplusplus
extern "C"
{
#endif

int MPI_Init(int * argc, char *** argv);
int MPI_Finalize();

int MPI_Barrier(MPI_Comm comm);

int MPI_Bcast (void * buffer, int count, MPI_Datatype datatype, int root,
               MPI_Comm comm);

int MPI_Comm_rank(MPI_Comm comm, int * rank);
int MPI_Comm_size(MPI_Comm comm, int * size);

int MPI_Gather(void * sendbuf, int sendcount, MPI_Datatype sendtype,
               void * recvbuf, int recvcount, MPI_Datatype recvtype,
               int root, MPI_Comm comm);

int MPI_Irecv( void * buf, int count, MPI_Datatype datatype, int source,
               int tag, MPI_Comm comm, MPI_Request *request );
int MPI_Send( void * buf, int count, MPI_Datatype datatype, int dest,
              int tag, MPI_Comm comm );
int MPI_Waitall(
        int count,
        MPI_Request array_of_requests[],
        MPI_Status array_of_statuses[] );
int MPI_Waitany(
        int count,
        MPI_Request array_of_requests[],
        int * index,
        MPI_Status * status );

double MPI_Wtime();
double MPI_Wtick();

#ifdef __cplusplus
}
#endif

#endif // PV_USE_LOCAL_MPI


