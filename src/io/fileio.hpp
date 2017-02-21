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
#include "include/PVLayerLoc.h"
#include "include/pv_types.h"
#include "io.hpp"
#include "structures/MPIBlock.hpp"

#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace PV {

void timeToParams(double time, void *params);
double timeFromParams(void *params);

size_t pv_sizeof(int datatype);

PV_Stream *PV_fopen(const char *path, const char *mode, bool verifyWrites);
int PV_stat(const char *path, struct stat *buf);
long int getPV_StreamFilepos(PV_Stream *pvstream);
long int PV_ftell(PV_Stream *pvstream);
int PV_fseek(PV_Stream *pvstream, long int offset, int whence);
size_t
PV_fwrite(const void *RESTRICT ptr, size_t size, size_t nitems, PV_Stream *RESTRICT pvstream);
size_t PV_fread(void *RESTRICT ptr, size_t size, size_t nitems, PV_Stream *RESTRICT pvstream);
int PV_fclose(PV_Stream *pvstream);
int ensureDirExists(MPIBlock const *mpiBlock, char const *dirname);

PV_Stream *pvp_open_read_file(const char *filename, MPIBlock const *mpiBlock);

PV_Stream *pvp_open_write_file(const char *filename, MPIBlock const *mpiBlock, bool append);

int pvp_close_file(PV_Stream *pvstream, MPIBlock const *mpiBlock);

int pvp_read_header(PV_Stream *pvstream, MPIBlock const *mpiBlock, int *params, int *numParams);
int pvp_read_header(
      const char *filename,
      MPIBlock const *mpiBlock,
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

int pvp_write_header(PV_Stream *pvstream, MPIBlock const *mpiBlock, int *params, int numParams);

// The pvp_write_header below will go away in favor of the pvp_write_header above.
int pvp_write_header(
      PV_Stream *pvstream,
      MPIBlock const *mpiBlock,
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

// Unused function pvp_set_activity_params was removed Jan 26, 2017.
// Unused function alloc_params was removed Feb 21, 2017.

// writeActivity and writeActivitySparse removed Feb 17, 2017.
// Corresponding HyPerLayer methods now use BufferUtils routines
// gatherActivity and scatterActivity were also removed.
// Use BufferUtils::gather and BufferUtils::scatter instead.

int readWeights(
      PVPatch ***patches,
      float **dataStart,
      int numArbors,
      int numPatches,
      int nxp,
      int nyp,
      int nfp,
      const char *filename,
      MPIBlock const *mpiBlock,
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
      MPIBlock const *mpiBlock,
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

int pvp_check_file_header(
      MPIBlock const *mpiBlock,
      const PVLayerLoc *loc,
      int params[],
      int numParams);
} // namespace PV

#endif /* FILEIO_HPP_ */
