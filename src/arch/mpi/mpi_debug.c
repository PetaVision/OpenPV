#define SKIP_MPI_WRAP
#include "mpi_debug.h"
#undef  SKIP_MPI_WRAP

#include <mpi.h>

#include <string.h>
#include <stdio.h>
#include <time.h>

static FILE *mpi_log = NULL;
static int  rank;

static long int debug_mpi_log_call(const char *func, const char *file, const int line) {
    long int t = time(NULL);
    if (mpi_log != NULL) {
        fprintf(mpi_log, "[%ld] %-16s %s:%d\n",
                t, func, file, line);
    }
    return t;
}

static void debug_mpi_log_arg_com(const char *arg, MPI_Comm comm) {
    if (mpi_log != NULL) {
        char name[256];
        int len = 0;
        MPI_Comm_get_name(comm, name, &len);
        name[len] = '\0';
        fprintf(mpi_log, "%16s = %s\n", arg, name);
    }
}

static void debug_mpi_log_arg_typ(const char *arg, MPI_Datatype type) {
    if (mpi_log != NULL) {
        char name[256];
        int len = 0;
        MPI_Type_get_name(type, name, &len);
        name[len] = '\0';
        fprintf(mpi_log, "%16s = %s\n", arg, name);
    }
}

static void debug_mpi_log_arg_ptr(const char *arg, const void *value) {
    if (mpi_log != NULL) {
        fprintf(mpi_log, "%16s = 0x%016lX\n", arg, (unsigned long int)value);
    }
}

static void debug_mpi_log_arg_int(const char *arg, const int value) {
    if (mpi_log != NULL) {
        fprintf(mpi_log, "%16s = %d\n", arg, value);
    }
}

static void debug_mpi_log_arg_str(const char *arg, const char *str) {
    if (mpi_log != NULL) {
        fprintf(mpi_log, "%16s = %s\n", arg, str);
    }
}


int DEBUG_MPI_Init(int *argc, char ***argv, const char *file, const int line) {
    char fname[256];
    int ret = MPI_Init(argc, argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    sprintf(fname, "mpi_rank_%d.log", rank);
    mpi_log = fopen(fname, "w");

    debug_mpi_log_call("MPI_Init", file, line);
    return ret;
}

int DEBUG_MPI_Finalize(const char *file, const int line) {
    int ret = MPI_Finalize();
    debug_mpi_log_call("MPI_Finalize", file, line);
    fclose(mpi_log);
    return ret;
}

int DEBUG_MPI_Barrier(MPI_Comm comm, const char *file, const int line) {
    int ret = MPI_Barrier(comm);
    debug_mpi_log_call("MPI_Barrier", file, line);
    debug_mpi_log_arg_com("comm", comm);
    return ret;
}

int DEBUG_MPI_Bcast(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm,
        const char *file, const int line) {
    static int counter = 0;
    int ret = MPI_Bcast(buffer, count, datatype, root, comm);
    long int t = debug_mpi_log_call("MPI_Bcast", file, line);
    debug_mpi_log_arg_ptr("buffer", buffer);
    debug_mpi_log_arg_int("count", count);
    debug_mpi_log_arg_typ("datatype", datatype);
    debug_mpi_log_arg_int("root", (int)root);
    debug_mpi_log_arg_com("comm", comm);

    char fname[256];
    int dtsize = 1;
    sprintf(fname, "%ld-%d-Bcast-%d.bin", t, rank, counter++);
    FILE *data = fopen(fname, "wb");
    MPI_Type_size(datatype, &dtsize);
    fwrite(buffer, dtsize, count, data); 
    fclose(data);

    return ret;
}

int DEBUG_MPI_Iallreduce(
      void *sendbuf,
      void *recvbuf,
      int count,
      MPI_Datatype datatype,
      MPI_Op op,
      MPI_Comm comm,
      MPI_Request *request,
      const char *file, const int line) {
    static int counter = 0;
    int ret = MPI_Iallreduce(sendbuf, recvbuf, count, datatype, op, comm, request);
    long int t = debug_mpi_log_call("MPI_Iallreduce", file, line);
    debug_mpi_log_arg_ptr("sendbuf", sendbuf);
    debug_mpi_log_arg_ptr("recvbuf", recvbuf);
    debug_mpi_log_arg_int("count", count);
    debug_mpi_log_arg_typ("datatype", datatype);
    debug_mpi_log_arg_ptr("op", op);
    debug_mpi_log_arg_com("comm", comm);
    debug_mpi_log_arg_ptr("request", request);

    char fname[256];
    int dtsize = 1;
    if (sendbuf != NULL) {
        sprintf(fname, "%ld-%d-Iallreduce-sendbuf-%d.bin", t, rank, counter);
        FILE *data = fopen(fname, "wb");
        MPI_Type_size(datatype, &dtsize);
        fwrite(sendbuf, dtsize, count, data); 
        fclose(data);
    }
    if (recvbuf != NULL) {
        sprintf(fname, "%ld-%d-Iallreduce-recvbuf-%d.bin", t, rank, counter);
        FILE *data = fopen(fname, "wb");
        MPI_Type_size(datatype, &dtsize);
        fwrite(recvbuf, dtsize, count, data); 
        fclose(data);
    }

    counter++;

    return ret;
}

int DEBUG_MPI_Allreduce(
      void *sendbuf,
      void *recvbuf,
      int count,
      MPI_Datatype datatype,
      MPI_Op op,
      MPI_Comm comm,
      const char *file, const int line) {
    static int counter = 0;
    int ret = MPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
    long int t = debug_mpi_log_call("MPI_Allreduce", file, line);
    debug_mpi_log_arg_ptr("sendbuf", sendbuf);
    debug_mpi_log_arg_ptr("recvbuf", recvbuf);
    debug_mpi_log_arg_int("count", count);
    debug_mpi_log_arg_typ("datatype", datatype);
    debug_mpi_log_arg_ptr("op", op);
    debug_mpi_log_arg_com("comm", comm);

    char fname[256];
    int dtsize = 1;
    if (sendbuf != NULL) {
        sprintf(fname, "%ld-%d-Allreduce-sendbuf-%d.bin", t, rank, counter);
        FILE *data = fopen(fname, "wb");
        MPI_Type_size(datatype, &dtsize);
        fwrite(sendbuf, dtsize, count, data); 
        fclose(data);
    }
    if (recvbuf != NULL) {
        sprintf(fname, "%ld-%d-Allreduce-recvbuf-%d.bin", t, rank, counter);
        FILE *data = fopen(fname, "wb");
        MPI_Type_size(datatype, &dtsize);
        fwrite(recvbuf, dtsize, count, data); 
        fclose(data);
    }

    counter++;
    return ret;
}

int DEBUG_MPI_Reduce(
      void *sendbuf,
      void *recvbuf,
      int count,
      MPI_Datatype datatype,
      MPI_Op op,
      int root,
      MPI_Comm comm,
      const char *file, const int line) {
    static int counter = 0;
    int ret = MPI_Reduce(sendbuf, recvbuf, count, datatype, op, root, comm);
    long int t = debug_mpi_log_call("MPI_Reduce", file, line);
    debug_mpi_log_arg_ptr("sendbuf", sendbuf);
    debug_mpi_log_arg_ptr("recvbuf", recvbuf);
    debug_mpi_log_arg_int("count", count);
    debug_mpi_log_arg_typ("datatype", datatype);
    debug_mpi_log_arg_ptr("op", op);
    debug_mpi_log_arg_int("root", (int)root);
    debug_mpi_log_arg_com("comm", comm);

    char fname[256];
    int dtsize = 1;
    if (sendbuf != NULL) {
        sprintf(fname, "%ld-%d-Reduce-sendbuf-%d.bin", t, rank, counter);
        FILE *data = fopen(fname, "wb");
        MPI_Type_size(datatype, &dtsize);
        fwrite(sendbuf, dtsize, count, data); 
        fclose(data);
    }
    if (recvbuf != NULL) {
        sprintf(fname, "%ld-%d-Reduce-recvbuf-%d.bin", t, rank, counter);
        FILE *data = fopen(fname, "wb");
        MPI_Type_size(datatype, &dtsize);
        fwrite(recvbuf, dtsize, count, data); 
        fclose(data);
    }

    counter++;
    return ret;
}

int DEBUG_MPI_Irecv(
      void *buf,
      int count,
      MPI_Datatype datatype,
      int source,
      int tag,
      MPI_Comm comm,
      MPI_Request *request,
      const char *file, const int line) {
    static int counter = 0;
    int ret = MPI_Irecv(buf, count, datatype, source, tag, comm, request);
    long int t = debug_mpi_log_call("MPI_Irecv", file, line);
    debug_mpi_log_arg_ptr("buf", buf);
    debug_mpi_log_arg_int("count", count);
    debug_mpi_log_arg_typ("datatype", datatype);
    debug_mpi_log_arg_int("source", source);
    debug_mpi_log_arg_int("tag", tag);
    debug_mpi_log_arg_com("comm", comm);
    debug_mpi_log_arg_ptr("request", request);

    char fname[256];
    int dtsize = 1;
    sprintf(fname, "%ld-%d-Irecv-%d.bin", t, rank, counter++);
    FILE *data = fopen(fname, "wb");
    MPI_Type_size(datatype, &dtsize);
    fwrite(buf, dtsize, count, data); 
    fclose(data);
    return ret;
}

int DEBUG_MPI_Recv(
      void *buf,
      int count,
      MPI_Datatype datatype,
      int source,
      int tag,
      MPI_Comm comm,
      MPI_Status *status,
      const char *file, const int line) {
    static int counter = 0;
    int ret = MPI_Recv(buf, count, datatype, source, tag, comm, status);
    long int t = debug_mpi_log_call("MPI_Recv", file, line);
    debug_mpi_log_arg_ptr("buf", buf);
    debug_mpi_log_arg_int("count", count);
    debug_mpi_log_arg_typ("datatype", datatype);
    debug_mpi_log_arg_int("source", source);
    debug_mpi_log_arg_int("tag", tag);
    debug_mpi_log_arg_com("comm", comm);
    debug_mpi_log_arg_ptr("status", status);

    char fname[256];
    int dtsize = 1;
    sprintf(fname, "%ld-%d-Recv-%d.bin", t, rank, counter++);
    FILE *data = fopen(fname, "wb");
    MPI_Type_size(datatype, &dtsize);
    fwrite(buf, dtsize, count, data); 
    fclose(data);
    return ret;
}

int DEBUG_MPI_Isend(
      void *buf,
      int count,
      MPI_Datatype datatype,
      int dest,
      int tag,
      MPI_Comm comm,
      MPI_Request *request,
      const char *file, const int line) {
    static int counter = 0;
    int ret = MPI_Isend(buf, count, datatype, dest, tag, comm, request);
    long int t = debug_mpi_log_call("MPI_Isend", file, line);
    debug_mpi_log_arg_ptr("buf", buf);
    debug_mpi_log_arg_int("count", count);
    debug_mpi_log_arg_typ("datatype", datatype);
    debug_mpi_log_arg_int("dest", dest);
    debug_mpi_log_arg_int("tag", tag);
    debug_mpi_log_arg_com("comm", comm);
    debug_mpi_log_arg_ptr("request", request);

    char fname[256];
    int dtsize = 1;
    sprintf(fname, "%ld-%d-Isend-%d.bin", t, rank, counter);
    FILE *data = fopen(fname, "wb");
    MPI_Type_size(datatype, &dtsize);
    fwrite(buf, dtsize, count, data); 
    fclose(data);

    return ret;
}

int DEBUG_MPI_Send(void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm,
      const char *file, const int line) {
    static int counter = 0;
    int ret = MPI_Send(buf, count, datatype, dest, tag, comm);
    long int t = debug_mpi_log_call("MPI_Send", file, line);
    debug_mpi_log_arg_ptr("buf", buf);
    debug_mpi_log_arg_int("count", count);
    debug_mpi_log_arg_typ("datatype", datatype);
    debug_mpi_log_arg_int("dest", dest);
    debug_mpi_log_arg_int("tag", tag);
    debug_mpi_log_arg_com("comm", comm);

    char fname[256];
    int dtsize = 1;
    sprintf(fname, "%ld-%d-Send-%d.bin", t, rank, counter);
    FILE *data = fopen(fname, "wb");
    MPI_Type_size(datatype, &dtsize);
    fwrite(buf, dtsize, count, data); 
    fclose(data);

    return ret;
}

int DEBUG_MPI_Testall(int count, MPI_Request *reqs, int *flag, MPI_Status *stats,
      const char *file, const int line) {
    int ret = MPI_Testall(count, reqs, flag, stats);
    debug_mpi_log_call("MPI_Testall", file, line);
    debug_mpi_log_arg_int("count", count);
    debug_mpi_log_arg_ptr("reqs", reqs);
    debug_mpi_log_arg_ptr("flag", flag);
    debug_mpi_log_arg_ptr("stats", stats);
    return ret;
}

int DEBUG_MPI_Waitall(int count, MPI_Request array_of_requests[], MPI_Status array_of_statuses[],
      const char *file, const int line) {
    int ret = MPI_Waitall(count, array_of_requests, array_of_statuses);
    debug_mpi_log_call("MPI_Waitall", file, line);
    debug_mpi_log_arg_int("count", count);
    debug_mpi_log_arg_ptr("array_of_requests", array_of_requests);
    debug_mpi_log_arg_ptr("array_of_statuses", array_of_statuses);
    return ret;
}


