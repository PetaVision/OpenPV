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

double MPI_Wtime();
double MPI_Wtick();

#ifdef __cplusplus
}
#endif

#endif // MPI_STUBS_H_
