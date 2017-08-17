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
#include "components/Patch.hpp"
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

// readWeights, writeWeights, and functions used only by them were removed Mar 15, 2017.
// Use the WeightsFileIO class instead

// calcMinMaxPatch, calcMinMaxNonsharedWeights, calcMinMaxSharedWeights were removed Aug 16, 2017.
// Use Weights::calcMinWeights and Weights::calcMaxWeights instead.

int pv_text_write_patch(
      PrintStream *pvstream,
      Patch const *patch,
      float *data,
      int nf,
      int sx,
      int sy,
      int sf);

// Unused function pvp_check_file_header was removed Mar 15, 2017.
} // namespace PV

#endif /* FILEIO_HPP_ */
