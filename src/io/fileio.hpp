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

// The no-longer-used PatchHeader and PatchListDescription structs were removed Aug 23, 2017.
// PatchHeader has the same information as the Patch struct defined in components/Patch.hpp
// PatchListDescription was only used internally by readNonsharedWeights and writeNonsharedWeights,
// which is now handled by the WeightsFileIO class.

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
void ensureDirExists(MPIBlock const *mpiBlock, char const *dirname);

// Unused function pvp_open_read_file was removed Mar 23, 2017. Instead, construct a FileStream.
// Unused function pvp_open_write_file was removed Mar 10, 2017. Instead, construct a FileStream.
// Unused function pvp_close_file was removed Mar 23, 2017.
// Unused functions pvp_read_header and pvp_write_header were removed Mar 15, 2017.
// Instead, use the WeightHeader-returning functions in BufferUtils
// together with FileStream::read and FileStream::write.

// readWeights and writeWeights were separated into Shared and Nonshared versions Mar 15, 2017.
// These functions, and functions used only by them were removed Aug 17, 2017.
// Use the WeightsFileIO class instead.

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
