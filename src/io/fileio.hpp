/*
 * fileio.hpp
 *
 *  Created on: Oct 21, 2009
 *      Author: rasmussn
 */

#ifndef FILEIO_HPP_
#define FILEIO_HPP_

#include "FileStream.hpp"
#include "arch/mpi/mpi.h"
#include "columns/Communicator.hpp"
#include "include/PVLayerLoc.h"
#include "include/pv_types.h"
#include "io.hpp"

#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace PV {

// index/value pairs used by writeActivitySparseNonspiking()
typedef struct indexvaluepair_ {
   uint32_t index;
   float value;
} indexvaluepair;

void timeToParams(double time, void *params);
double timeFromParams(void *params);

size_t pv_sizeof(int datatype);

PV_Stream *PV_fopen(const char *path, const char *mode, bool verifyWrites);
int PV_stat(const char *path, struct stat *buf);
long int getPV_StreamFilepos(PV_Stream *pvstream);
long int updatePV_StreamFilepos(PV_Stream *pvstream);
long int PV_ftell(PV_Stream *pvstream);
int PV_fseek(PV_Stream *pvstream, long int offset, int whence);
size_t
PV_fwrite(const void *RESTRICT ptr, size_t size, size_t nitems, PV_Stream *RESTRICT pvstream);
size_t PV_fread(void *RESTRICT ptr, size_t size, size_t nitems, PV_Stream *RESTRICT pvstream);
int PV_fclose(PV_Stream *pvstream);
int ensureDirExists(Communicator *comm, char const *dirname);

PV_Stream *pvp_open_read_file(const char *filename, Communicator *comm);

PV_Stream *pvp_open_write_file(const char *filename, Communicator *comm, bool append);

int pvp_close_file(PV_Stream *pvstream, Communicator *comm);

int pvp_read_header(PV_Stream *pvstream, Communicator *comm, int *params, int *numParams);
int pvp_read_header(
      const char *filename,
      Communicator *comm,
      double *time,
      int *filetype,
      int *datatype,
      int params[],
      int *numParams);
int pvp_read_header(
      PV_Stream *pvstream,
      double *time,
      int *filetype,
      int *datatype,
      int params[],
      int *numParams);

void read_header_err(
      const char *filename,
      Communicator *comm,
      int returned_num_params,
      int *params);
int pvp_write_header(PV_Stream *pvstream, Communicator *comm, int *params, int numParams);

// The pvp_write_header below will go away in favor of the pvp_write_header above.
int pvp_write_header(
      PV_Stream *pvstream,
      Communicator *comm,
      double time,
      const PVLayerLoc *loc,
      int filetype,
      int datatype,
      int numbands,
      bool extended,
      bool contiguous,
      unsigned int numParams,
      size_t recordSize);

// Oct 21, 2016. pvp_set_file_params removed, as filetype PVP_FILE_TYPE is obsolete.
int *pvp_set_activity_params(
      Communicator *comm,
      double timed,
      const PVLayerLoc *loc,
      int datatype,
      int numbands);
int *pvp_set_weight_params(
      Communicator *comm,
      double timed,
      const PVLayerLoc *loc,
      int datatype,
      int numbands,
      int nxp,
      int nyp,
      int nfp,
      float min,
      float max,
      int numPatches);
int *pvp_set_nonspiking_act_params(
      Communicator *comm,
      double timed,
      const PVLayerLoc *loc,
      int datatype,
      int numbands);
int *pvp_set_kernel_params(
      Communicator *comm,
      double timed,
      const PVLayerLoc *loc,
      int datatype,
      int numbands,
      int nxp,
      int nyp,
      int nfp,
      float min,
      float max,
      int numPatches);
int *alloc_params(int numParams);
int set_weight_params(int *params, int nxp, int nyp, int nfp, float min, float max, int numPatches);

int pvp_read_time(PV_Stream *pvstream, Communicator *comm, int root_process, double *timed);

int writeActivity(FileStream *fileStream, Communicator *comm, double timed, PVLayerCube *cube);

int writeActivitySparse(
      FileStream *fileStream,
      Communicator *comm,
      double timed,
      PVLayerCube *cube,
      bool includeValues);

int readWeights(
      PVPatch ***patches,
      float **dataStart,
      int numArbors,
      int numPatches,
      int nxp,
      int nyp,
      int nfp,
      const char *filename,
      Communicator *comm,
      double *timed,
      const PVLayerLoc *loc);

int pv_text_write_patch(
      PrintStream *pvstream,
      PVPatch *patch,
      float *data,
      int nf,
      int sx,
      int sy,
      int sf);

int writeWeights(
      const char *filename,
      Communicator *comm,
      double timed,
      bool append,
      const PVLayerLoc *preLoc,
      const PVLayerLoc *postLoc,
      int nxp,
      int nyp,
      int nfp,
      float minVal,
      float maxVal,
      PVPatch ***patches,
      float **dataStart,
      int numPatches,
      int numArbors,
      bool compress = true,
      int file_type = PVP_WGT_FILE_TYPE);

int pvp_check_file_header(Communicator *comm, const PVLayerLoc *loc, int params[], int numParams);

template <typename T>
int gatherActivity(
      PV_Stream *pvstream,
      Communicator *comm,
      int rootproc,
      T *buffer,
      const PVLayerLoc *layerLoc,
      bool extended);
template <typename T>
int scatterActivity(
      PV_Stream *pvstream,
      Communicator *comm,
      int rootproc,
      T *buffer,
      const PVLayerLoc *layerLoc,
      bool extended,
      const PVLayerLoc *fileLoc = NULL,
      int offsetX               = 0,
      int offsetY               = 0,
      int filetype              = PVP_NONSPIKING_ACT_FILE_TYPE,
      int numActive             = 0);
} // namespace PV

#endif /* FILEIO_HPP_ */
