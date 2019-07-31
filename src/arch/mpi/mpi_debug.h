#include <cMakeHeader.h>
#include <mpi.h>

#ifndef SKIP_MPI_WRAP
#define MPI_Init(a,b) do { \
        DEBUG_MPI_Init((a), (b), __FILE__, __LINE__); \
    } while (0)

#define MPI_Finalize() do { \
        DEBUG_MPI_Finalize(__FILE__, __LINE__); \
    } while (0)

#define MPI_Barrier(a) do { \
        DEBUG_MPI_Barrier((a), __FILE__, __LINE__); \
    } while (0)

#define MPI_Bcast(a,b,c,d,e) do { \
        DEBUG_MPI_Bcast((a),(b),(c),(d),(e), __FILE__, __LINE__); \
    } while (0)

#define MPI_Iallreduce(a,b,c,d,e,f,g) do { \
        DEBUG_MPI_Iallreduce((a),(b),(c),(d),(e),(f),(g), __FILE__, __LINE__); \
    } while (0)

#define MPI_Allreduce(a,b,c,d,e,f) do { \
        DEBUG_MPI_Allreduce((a),(b),(c),(d),(e),(f), __FILE__, __LINE__); \
    } while (0)

#define MPI_Reduce(a,b,c,d,e,f,g) do { \
        DEBUG_MPI_Reduce((a),(b),(c),(d),(e),(f),(g), __FILE__, __LINE__); \
    } while (0)

#define MPI_Irecv(a,b,c,d,e,f,g) do { \
        DEBUG_MPI_Irecv((a),(b),(c),(d),(e),(f),(g), __FILE__, __LINE__); \
    } while (0)

#define MPI_Recv(a,b,c,d,e,f,g) do { \
        DEBUG_MPI_Recv((a),(b),(c),(d),(e),(f),(g), __FILE__, __LINE__); \
    } while (0)

#define MPI_Isend(a,b,c,d,e,f,g) do { \
        DEBUG_MPI_Isend((a),(b),(c),(d),(e),(f),(g), __FILE__, __LINE__); \
    } while (0)

#define MPI_Send(a,b,c,d,e,f) do { \
        DEBUG_MPI_Send((a),(b),(c),(d),(e),(f), __FILE__, __LINE__); \
    } while (0)

#define MPI_Testall(a,b,c,d) do { \
        DEBUG_MPI_Testall((a),(b),(c),(d), __FILE__, __LINE__); \
    } while (0)

#define MPI_Waitall(a,b,c) do { \
        DEBUG_MPI_Waitall((a),(b),(c), __FILE__, __LINE__); \
    } while (0)
#endif

#ifndef MPI_DEBUG_H_
#define MPI_DEBUG_H_

#ifdef __cplusplus
extern "C" {
#endif

int DEBUG_MPI_Init(int *argc, char ***argv, const char *file, const int line);

int DEBUG_MPI_Finalize(const char *file, const int line);

int DEBUG_MPI_Barrier(MPI_Comm comm, const char *file, const int line);

int DEBUG_MPI_Bcast(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm,
        const char *file, const int line);

int DEBUG_MPI_Iallreduce(
      void *sendbuf,
      void *recvbuf,
      int count,
      MPI_Datatype datatype,
      MPI_Op op,
      MPI_Comm comm,
      MPI_Request *request,
      const char *file, const int line);

int DEBUG_MPI_Allreduce(
      void *sendbuf,
      void *recvbuf,
      int count,
      MPI_Datatype datatype,
      MPI_Op op,
      MPI_Comm comm,
      const char *file, const int line);

int DEBUG_MPI_Reduce(
      void *sendbuf,
      void *recvbuf,
      int count,
      MPI_Datatype datatype,
      MPI_Op op,
      int root,
      MPI_Comm comm,
      const char *file, const int line);

int DEBUG_MPI_Irecv(
      void *buf,
      int count,
      MPI_Datatype datatype,
      int source,
      int tag,
      MPI_Comm comm,
      MPI_Request *request,
      const char *file, const int line);

int DEBUG_MPI_Recv(
      void *buf,
      int count,
      MPI_Datatype datatype,
      int source,
      int tag,
      MPI_Comm comm,
      MPI_Status *status,
      const char *file, const int line);

int DEBUG_MPI_Isend(
      void *buf,
      int count,
      MPI_Datatype datatype,
      int dest,
      int tag,
      MPI_Comm comm,
      MPI_Request *request,
      const char *file, const int line);

int DEBUG_MPI_Send(void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm,
      const char *file, const int line);
int DEBUG_MPI_Testall(int count, MPI_Request *reqs, int *flag, MPI_Status *stats,
      const char *file, const int line);
int DEBUG_MPI_Waitall(int count, MPI_Request array_of_requests[], MPI_Status array_of_statuses[],
      const char *file, const int line);


#ifdef __cplusplus
}
#endif

#endif /* MPI_DEBUG_H_ */
