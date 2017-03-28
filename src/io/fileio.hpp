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
#include "utils/BufferUtilsPvp.hpp"

#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace PV {

struct PatchHeader {
   unsigned short int nx;
   unsigned short int ny;
   unsigned int offset;
};

// Unused function timeToParams was removed Mar 10, 2017.
// Unused function timeFromParams was removed Mar 15, 2017.
// Read/write pvp header using ActivityHeader in utils/BufferUtilsPvp, instead
// of as an array of ints. ActivityHeader declares timestamp as double.

// Unused function pv_sizeof was removed Mar 15, 2017.
// Unused function pv_sizeof_patch was removed Mar 15, 2017.
// Instead, use BufferUtils::weightPatchSize template in utils/BufferUtilsPvp.

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

// Unused function pvp_open_read_file was removed Mar 23, 2017. Instead, construct a FileStream.
// Unused function pvp_open_write_file was removed Mar 10, 2017. Instead, construct a FileStream.
// Unused function pvp_close_file was removed Mar 23, 2017.
// Unused functions pvp_read_header and pvp_write_header were removed Mar 15, 2017.
// Instead, use the WeightHeader-returning functions in BufferUtils
// together with FileStream::read and FileStream::write.

// Oct 21, 2016. pvp_set_file_params removed, as filetype PVP_FILE_TYPE is obsolete.

// Unused function pvp_set_activity_params was removed Jan 26, 2017.
// Unused function alloc_params was removed Feb 21, 2017.

// writeActivity and writeActivitySparse removed Feb 17, 2017.
// Corresponding HyPerLayer methods now use BufferUtils routines
// gatherActivity and scatterActivity were also removed.
// Use BufferUtils::gather and BufferUtils::scatter instead.

// readWeights was removed Mar 15, 2017. Use readSharedWeights or readNonsharedWeights instead.

double readSharedWeights(
      FileStream *fileStream,
      MPIBlock const *mpiBlock,
      PVLayerLoc const *preLoc,
      int nxp,
      int nyp,
      int nfp,
      int numArbors,
      float **dataStart,
      int numPatchesX,
      int numPatchesY,
      int numPatchesF);

double readNonsharedWeights(
      FileStream *fileStream,
      MPIBlock const *mpiBlock,
      const PVLayerLoc *preLoc,
      int nxp,
      int nyp,
      int nfp,
      int numArbors,
      float **dataStart,
      bool extended,
      const PVLayerLoc *postLoc,
      int offsetX,
      int offsetY);

bool isCompressedHeader(BufferUtils::WeightHeader const &header, std::string const &filename);

int pv_text_write_patch(
      PrintStream *pvstream,
      PVPatch *patch,
      float *data,
      int nf,
      int sx,
      int sy,
      int sf);

void writeSharedWeights(
      double timed,
      FileStream *fileStream,
      MPIBlock const *mpiBlock,
      PVLayerLoc const *preLoc,
      int nxp,
      int nyp,
      int nfp,
      int numArbors,
      float **dataStart,
      bool compress,
      float minVal,
      float maxVal,
      int numPatchesX,
      int numPatchesY,
      int numPatchesF);

void writeNonsharedWeights(
      double timed,
      FileStream *fileStream,
      MPIBlock const *mpiBlock,
      const PVLayerLoc *preLoc,
      int nxp,
      int nyp,
      int nfp,
      int numArbors,
      float **dataStart,
      bool compress,
      float minVal,
      float maxVal,
      bool extended,
      const PVLayerLoc *postLoc);

// Unused function pvp_check_file_header was removed Mar 15, 2017.
} // namespace PV

#endif /* FILEIO_HPP_ */
