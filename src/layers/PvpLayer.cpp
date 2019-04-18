#include "PvpLayer.hpp"
#include "arch/mpi/mpi.h"
#include "io/FileStream.hpp"
#include "io/fileio.hpp"
#include "structures/SparseList.hpp"

#include <cstring>
#include <iostream>

namespace PV {

PvpLayer::PvpLayer(const char *name, HyPerCol *hc) { initialize(name, hc); }

PvpLayer::~PvpLayer() {}

Response::Status PvpLayer::allocateDataStructures() { return InputLayer::allocateDataStructures(); }

int PvpLayer::countInputImages() {
   FileStream headerStream(
         getInputPath().c_str(), std::ios_base::in | std::ios_base::binary, false);
   struct BufferUtils::ActivityHeader header = BufferUtils::readActivityHeader(headerStream);

   int pvpFrameCount = header.nBands;
   if (header.fileType == PVP_ACT_SPARSEVALUES_FILE_TYPE || header.fileType == PVP_ACT_FILE_TYPE) {
      sparseTable = BufferUtils::buildSparseFileTable(headerStream, pvpFrameCount - 1);
   }
   return header.nBands;
}

Buffer<float> PvpLayer::retrieveData(int inputIndex) {
   // If we're playing through the pvp file like a movie, use
   // BatchIndexer to get the frame number. Otherwise, just use
   // the start_frame_index value for this batch.
   Buffer<float> result;
   BufferUtils::readActivityFromPvp<float>(
         getInputPath().c_str(), &result, inputIndex, &sparseTable);

   return result;
}
} // end namespace PV
