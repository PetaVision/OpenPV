#ifndef MPI_STUBS_H_
#define MPI_STUBS_H_

#define MPI_Request  int
#define MPI_Datatype int
#define MPI_Request  int
#define MPI_Status   int
#define MPI_Comm     int
#define MPI_Group    int

#define MPI_COMM_WORLD     1
#define MPI_CHAR           1
#define MPI_INT            2
#define MPI_FLOAT          4

#define MPI_STATUS_IGNORE   0
#define MPI_STATUSES_IGNORE 0

#ifdef __cplusplus
extern "C"
{
#endif

inline int MPI_Init(int * argc, char *** argv)  {return 0;}
inline int MPI_Finalize()                       {return 0;}

inline int MPI_Barrier(MPI_Comm comm)           {return 0;}

int MPI_Bcast (void * buffer, int count, MPI_Datatype datatype, int root,
               MPI_Comm comm);

inline
int MPI_Comm_rank(MPI_Comm comm, int * rank)   {*rank=0; return 0;}
inline
int MPI_Comm_size(MPI_Comm comm, int * size)   {*size=1; return 0;}

int MPI_Comm_dup(MPI_Comm comm, MPI_Comm * newcomm);
int MPI_Comm_group(MPI_Comm comm, MPI_Group * group);
int MPI_Comm_create(MPI_Comm comm, MPI_Group group, MPI_Comm * newcomm);

int MPI_Gather(void * sendbuf, int sendcount, MPI_Datatype sendtype,
               void * recvbuf, int recvcount, MPI_Datatype recvtype,
               int root, MPI_Comm comm);

int MPI_Group_excl(MPI_Group group, int n, int * ranks, MPI_Group * newgroup);

int MPI_Irecv(void * buf, int count, MPI_Datatype datatype, int source,
              int tag, MPI_Comm comm, MPI_Request *request );
int MPI_Recv(void * buf, int count, MPI_Datatype datatype, int source, 
             int tag, MPI_Comm comm, MPI_Status * status);
int MPI_Send(void * buf, int count, MPI_Datatype datatype, int dest,
             int tag, MPI_Comm comm );

int MPI_Type_commit(MPI_Datatype * type);
int MPI_Type_vector(int count, int blocklength, int stride, 
                    MPI_Datatype oldtype, MPI_Datatype * newtype);

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

#endif // MPI_STUBS_H_
