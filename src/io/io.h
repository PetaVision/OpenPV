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
int scatterReadBuf(PVLayer * l, float * globalBuf, float * localBuf, MPI_Comm comm);
int scatterReadFile(const char * filename, PVLayer * l, float * buf, MPI_Comm comm);
int gatherWriteFile(const char * filename, PVLayer * l, float * ibuf, MPI_Comm comm);

int pv_text_write_patch(FILE * fd, PVPatch * patch);
int pv_tiff_write_patch(FILE * fd, PVPatch * patch);
int pv_tiff_write_cube(const char * filename, PVLayerCube * cube, int nx, int ny, int nf);
int pv_center_image(float * V, int nx0, int ny0, int nx, int ny);

int log_parameters(int n_time_steps, char * input_filename);

int printStats(pvdata_t * buf, int nItems, char * msg);

int pv_dump(const char * filename, int append, pvdata_t * I, int nx, int ny, int nf);
int pv_dump_sparse(const char * filename, int append, pvdata_t * I, int nx, int ny, int nf);
int pv_write_patches(const char * filename, int append,
                     int nx, int ny, int nf, float minVal, float maxVal,
                     int numPatches, PVPatch ** patches);
int pv_read_patches(FILE *fp, int nf, float minVal, float maxVal,
                    PVPatch ** patches, int numPatches);

FILE * pv_open_binary(char * filename, int * numParams, int * nx, int * ny, int * nf);
int    pv_read_binary_params(FILE * fp, int numParams, int params[]);
int    pv_close_binary(FILE * fd);
size_t pv_read_binary_record(FILE * fd, pvdata_t * buf, int nItems);

int parse_options(int argc, char * argv[], char ** input_file,
                  char ** param_file, int * n_time_steps, int *shmem_threads);

#ifdef __cplusplus
}
#endif

#endif /* IO_H_ */
