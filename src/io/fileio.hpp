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
#include <vector>

namespace PV {

struct PatchHeader {
   unsigned short int nx;
   unsigned short int ny;
   unsigned int offset;
};

/**
 * PatchListDescription is a struct that packages the description of a list of
 * patch indices of interest. The standard use case is to generate the list of
 * data patches of a HyPerConn that need to be considered in I/O. This list
 * may differ from the list 0,1,...(numDataPatches-1) in the case of nonshared
 * weights where the pre-synaptic layer has a larger border than the connection
 * requires. In this case we want to look at only those patches the connection
 * does require.
 *
 * The assumptions is that the patch indices of interest form a regular block
 * within the complete list of patches, and that for any x-y location,
 * either all features are to be considered or none are.
 *
 * mStartIndex is the first patch index of interest.
 *
 * mLineStride is the number of patches needed to step from a patch at (x,y,f)
 *     to a patch at (x,y+1,f). It should also be a multiple of the number of
 *     features.
 *
 * mNumPatchesX is the width of the block of patch indices of interest in the
 *     x-direction.
 * mNumPatchesY is the height of the block of patch indices of interest in the
 *     y-direction.
 * mNumPatchesF is the number of (pre-synaptic) features.
 */
struct PatchListDescription {
   int mStartIndex;
   int mStrideY;
   int mNumPatchesX;
   int mNumPatchesY;
   int mNumPatchesF;
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

// Unused function pvp_set_activity_params was removed Jan 26, 2017.
// Unused function alloc_params was removed Feb 21, 2017.

// writeActivity and writeActivitySparse removed Feb 17, 2017.
// Corresponding HyPerLayer methods now use BufferUtils routines
// gatherActivity and scatterActivity were also removed.
// Use BufferUtils::gather and BufferUtils::scatter instead.

// readWeights was removed Mar 15, 2017. Use readSharedWeights or readNonsharedWeights instead.

PatchListDescription createPatchListDescription(
      PVLayerLoc const *preLoc,
      PVLayerLoc const *postLoc,
      int nxp,
      int nyp,
      bool shared);

/**
 * Calculates the minimum of the value of minWeight and all the weights in the
 * patch pointed to by patchData, and defined by nf, nx, ny, offset and syp.
 * nf, nx, ny define the size of the patch, and offset defines the patch's
 * start in patchData. syp is the stride between adjacent y-values.
 * The stride in f is assumed to be 1, and the stride in x is assumed to be nf.
 *
 * It also calculates the maximum of the value of maxWeight and all the
 * weights in the same patch.
 *
 * Note that calcMinMaxPatch does not initialize minWeight or maxWeight.
 * This way the min/maximum of several patches in sequence is readily computed.
 */
void calcMinMaxPatch(
      float &minWeight,
      float &maxWeight,
      float const *patchData,
      unsigned int nf,
      unsigned int nx,
      unsigned int ny,
      unsigned int offset,
      unsigned int syp);

void calcMinMaxNonsharedWeights(
      float &minWeight,
      float &maxWeight,
      float const *const *patchData,
      int numArbors,
      int nxp,
      int nyp,
      int nfp,
      PatchListDescription const &patchIndices,
      PVPatch const *const *const *patchGeometry);

void calcMinMaxSharedWeights(
      float &minWeight,
      float &maxWeight,
      float const *const *patchData,
      int numArbors,
      int nxp,
      int nyp,
      int nfp,
      PatchListDescription const &patchIndices);

double readSharedWeights(
      FileStream *fileStream,
      int frameNumber,
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
      int frameNumber,
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

/**
 * Positions a weight pvp file to the start of the data (i.e. just past the end of the header)
 * of the indicated frame. The header for that frame is read into the buffer pointed by
 * the first argument.
 */
void setInPosByFrame(BufferUtils::WeightHeader &header, FileStream *fileStream, int frameNumber);

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
      bool extended,
      const PVLayerLoc *postLoc,
      PVPatch const *const *const *patchGeometry);

// Unused function pvp_check_file_header was removed Mar 15, 2017.
} // namespace PV

#endif /* FILEIO_HPP_ */
