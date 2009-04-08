/*
 * io.h
 *
 *  Created on: Oct 24, 2008
 *      Author: rasmussn
 */

#ifndef IO_H_
#define IO_H_

#include "../layers/PVLayer.h"
#include <mpi.h>

#ifdef __cplusplus
extern "C"
{
#endif

int readFile(const char * filename, float * buf, int * nx, int * ny);
int scatterReadFile(const char * filename, PVLayer * l, float * buf, MPI_Comm comm);
int gatherWriteFile(const char * filename, PVLayer * l, float * ibuf, MPI_Comm comm);

int pv_text_write_patch(const char * filename, PVPatch * patch);
int pv_tiff_write_patch(const char * filename, PVPatch * patch);
int pv_tiff_write_cube(const char * filename, PVLayerCube * cube, int nx, int ny, int nf);

int log_parameters(int n_time_steps, char * input_filename);

int printStats(pvdata_t * buf, int nItems, char * msg);

int pv_dump(char * filename, int append, pvdata_t * I, int nx, int ny, int nf);
int pv_dump_sparse(char* filename, int append, pvdata_t * I, int nx, int ny, int nf);

FILE * pv_open_binary(char * filename, int * nx, int * ny, int * nf);
int    pv_close_binary(FILE * fd);
int    pv_read_binary_record(FILE * fd, pvdata_t * buf, int nItems);

int parse_options(int argc, char * argv[], char ** input_file,
                  char ** param_file, int * n_time_steps, int *shmem_threads);

#ifdef __cplusplus
}
#endif

#endif /* IO_H_ */
