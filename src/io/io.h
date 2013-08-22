/*
 * io.h
 *
 *  Created on: Oct 24, 2008
 *      Author: rasmussn
 */

#ifndef IO_H_
#define IO_H_

#include "../layers/PVLayer.h"

#ifdef PV_USE_MPI
#  include <mpi.h>
#else
#  include "../include/mpi_stubs.h"
#endif // PV_USE_MPI

#define MIN_BIN_PARAMS  6
#define NUM_BIN_PARAMS (18 + sizeof(double)/sizeof(int))

#define NUM_WGT_EXTRA_PARAMS  6
#define NUM_WGT_PARAMS (NUM_BIN_PARAMS + NUM_WGT_EXTRA_PARAMS)
#define MAX_BIN_PARAMS NUM_WGT_PARAMS

// deprecated, use writeWeights and NUM_WGT_PARAMS
#define NUM_WEIGHT_PARAMS  (MIN_BIN_PARAMS + 3)

#define NUM_PAR_BYTE_PARAMS (NUM_BIN_PARAMS)

#define PV_ERR_FILE_NOT_FOUND 1

#define PV_BYTE_TYPE       1
#define PV_INT_TYPE        2
#define PV_FLOAT_TYPE      3

#define PVP_FILE_TYPE      1 // File type of the *_V_last.pvp and *_A_last.pvp files
#define PVP_ACT_FILE_TYPE  2 // File type of the a%d.pvp for spiking layers
#define PVP_WGT_FILE_TYPE  3 // File type of the w%d.pvp, w%d_last.pvp, and checkpoint files for non-KernelConn connections
#define PVP_NONSPIKING_ACT_FILE_TYPE  4 // File type of the a%d.pvp files for spiking layers and checkpoint files for all layers
#define PVP_KERNEL_FILE_TYPE 5 // File type of the w%d.pvp, w%d_last.pvp, and checkpoint files for KernelConns

#define INDEX_HEADER_SIZE  0
#define INDEX_NUM_PARAMS   1
#define INDEX_FILE_TYPE    2
#define INDEX_NX           3
#define INDEX_NY           4
#define INDEX_NF           (MIN_BIN_PARAMS - 1)
#define INDEX_NUM_RECORDS  6
#define INDEX_RECORD_SIZE  7
#define INDEX_DATA_SIZE    8
#define INDEX_DATA_TYPE    9
#define INDEX_NX_PROCS    10
#define INDEX_NY_PROCS    11
#define INDEX_NX_GLOBAL   12
#define INDEX_NY_GLOBAL   13
#define INDEX_KX0         14
#define INDEX_KY0         15
#define INDEX_NB          16
#define INDEX_NBANDS      17
#define INDEX_TIME        18

// these are extra parameters used by weight files
//
#define INDEX_WGT_NXP        0
#define INDEX_WGT_NYP        1
#define INDEX_WGT_NFP        2
#define INDEX_WGT_MIN        3
#define INDEX_WGT_MAX        4
#define INDEX_WGT_NUMPATCHES 5

#ifdef __cplusplus
extern "C"
{
#endif // __cplusplus

int pv_getopt_int(int argc, char * argv[], const char * opt, int *   iVal);
int pv_getopt_str(int argc, char * argv[], const char * opt, char ** sVal);
int pv_getopt_long(int argc, char * argv[], const char * opt, long int * ulVal);
int pv_getopt_unsigned(int argc, char * argv[], const char * opt, unsigned int * uVal);

int readFile(const char * filename, float * buf, int * nx, int * ny);

int pv_text_write_patch(PV_Stream * pvstream, PVPatch * patch, pvdata_t * data, int nf, int sx, int sy, int sf);
int pv_center_image(float * V, int nx0, int ny0, int nx, int ny);

#ifdef OBSOLETE // Marked obsolete April 29, 2013.  Use fileio's pvp_open_read_file instead.
FILE * pv_open_binary(const char * filename, int * numParams, int * type, int * nx, int * ny, int * nf);
#endif // OBSOLETE
#ifdef OBSOLETE // Marked obsolete April 29, 2013.  Use fileio's pvp_read_header instead.
int    pv_read_binary_params(FILE * pvstream, int numParams, int params[]);
#endif // OBSOLETE
#ifdef OBSOLETE // Marked obsolete April 29, 2013.  Use fileio's PV_fclose instead.
int    pv_close_binary(FILE * fp);
#endif // OBSOLETE
// size_t pv_read_binary_record(FILE * pvstream, pvdata_t * buf, int nItems); // No function definition to go with this prototype

int parse_options(int argc, char * argv[], char ** output_path,
                  char ** param_file, long int * n_time_steps, int * opencl_device,
                  unsigned int * random_seed, char ** working_dir);

#ifdef __cplusplus
}
#endif // __cplusplus

#endif /* IO_H_ */
