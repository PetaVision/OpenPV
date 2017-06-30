/*
 * io.hpp
 *
 *  Created on: Oct 24, 2008
 *      Author: rasmussn
 */

#ifndef IO_HPP_
#define IO_HPP_

#include "../arch/mpi/mpi.h"
#include <stdbool.h>
#include <string>

#include "include/pv_types.h"

#define MIN_BIN_PARAMS 6
#define NUM_BIN_PARAMS (18 + sizeof(double) / sizeof(int))

#define NUM_WGT_EXTRA_PARAMS 6
#define NUM_WGT_PARAMS (NUM_BIN_PARAMS + NUM_WGT_EXTRA_PARAMS)
#define MAX_BIN_PARAMS NUM_WGT_PARAMS

#define NUM_PAR_BYTE_PARAMS (NUM_BIN_PARAMS)

#define PV_ERR_FILE_NOT_FOUND 1

#define PVP_FILE_TYPE                                                                              \
   1 // File type of activities where there are no timestamps in the individual frames.  No longer
// used.
#define PVP_ACT_FILE_TYPE                                                                          \
   2 // File type for binary layers (activity is sparse and values are only 1 or 0). No longer
// used.
#define PVP_WGT_FILE_TYPE                                                                          \
   3 // File type of the w%d.pvp, and checkpoint files for connections without shared weights
#define PVP_NONSPIKING_ACT_FILE_TYPE                                                               \
   4 // File type of the a%d.pvp files for nonspiking layers and checkpoint files for all layers
#define PVP_KERNEL_FILE_TYPE                                                                       \
   5 // File type of the w%d.pvp, and checkpoint files for connections with shared weights
#define PVP_ACT_SPARSEVALUES_FILE_TYPE                                                             \
   6 // File type for sparse layers. The locations and values of nonzero neurons are stored.

#define INDEX_HEADER_SIZE 0
#define INDEX_NUM_PARAMS 1
#define INDEX_FILE_TYPE 2
#define INDEX_NX 3
#define INDEX_NY 4
#define INDEX_NF (MIN_BIN_PARAMS - 1)
#define INDEX_NUM_RECORDS 6
#define INDEX_RECORD_SIZE 7
#define INDEX_DATA_SIZE 8
#define INDEX_DATA_TYPE 9
#define INDEX_NX_PROCS 10
#define INDEX_NY_PROCS 11
#define INDEX_NX_EXTENDED 12
#define INDEX_NY_EXTENDED 13
#define INDEX_KX0 14
#define INDEX_KY0 15
#define INDEX_NBATCH 16
#define INDEX_NBANDS 17
#define INDEX_TIME 18

// these are extra parameters used by weight files
//
#define INDEX_WGT_NXP 0
#define INDEX_WGT_NYP 1
#define INDEX_WGT_NFP 2
#define INDEX_WGT_MIN 3
#define INDEX_WGT_MAX 4
#define INDEX_WGT_NUMPATCHES 5

namespace PV {

enum ParamsIOFlag { PARAMS_IO_READ, PARAMS_IO_WRITE };

int pv_getopt(int argc, char const *const *argv, const char *opt, bool *paramusage);
int pv_getopt_int(int argc, char const *const *argv, const char *opt, int *iVal, bool *paramusage);
int pv_getoptionalopt_int(
      int argc,
      char const *const *argv,
      const char *opt,
      int *iVal,
      bool *defaultVal,
      bool *paramusage);
int pv_getopt_str(
      int argc,
      char const *const *argv,
      const char *opt,
      char **sVal,
      bool *paramusage);
int pv_getopt_long(
      int argc,
      char const *const *argv,
      const char *opt,
      long int *ulVal,
      bool *paramusage);
int pv_getopt_unsigned(
      int argc,
      char const *const *argv,
      const char *opt,
      unsigned int *uVal,
      bool *paramusage);

int readFile(const char *filename, float *buf, int *nx, int *ny);

int parse_options(
      int argc,
      char const *const *argv,
      bool *paramusage,
      bool *require_return,
      char **output_path,
      char **param_file,
      char **log_file,
      char **gpu_devices,
      unsigned int *random_seed,
      char **working_dir,
      int *restart,
      char **checkpointReadDir,
      bool *useDefaultNumThreads,
      int *numthreads,
      int *numRows,
      int *numColumns,
      int *batch_width,
      int *dryrun);

/** If a filename begins with "~/" or is "~", presume the user means the home directory.
 * The return value is the expanded path; e.g. if the home directory is /home/user1,
 * calling with the path "~/directory/file.name" returns "/home/user1/directory/file.name"
 * If the input filename is longer than "~" and doesn't start with "~/", the return value
 * has the same contents as input (but is a different block of memory).
 * The calling routine has the responsibility for freeing the return value, and
 * if the input string needs to be free()'ed, the calling routine has that responsibility
 * as well.
 */
std::string expandLeadingTilde(std::string const &path);

/** An overload of expandLeadingTilde to take a C-style string instead of a C++ std::string.
 */
std::string expandLeadingTilde(char const *path);

} // end namespace PV

#endif /* IO_HPP_ */
