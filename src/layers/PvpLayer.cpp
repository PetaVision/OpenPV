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

int PvpLayer::allocateDataStructures() {
   int status = InputLayer::allocateDataStructures();
   return status;
}

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

Buffer<float> PvpLayer::retrieveData(int inputIndex, int batchElement) {
   // If we're playing through the pvp file like a movie, use
   // BatchIndexer to get the frame number. Otherwise, just use
   // the start_frame_index value for this batch.
   int frameNumber = getDisplayPeriod() > 0 ? inputIndex : getStartIndex(batchElement);

   Buffer<float> result;
   BufferUtils::readActivityFromPvp<float>(
         getInputPath().c_str(), &result, frameNumber, &sparseTable);

   return result;
}

Buffer<float> PvpLayer::retrieveData(std::string filename, int batchIndex) {
   int frameNumber = 0;

   // If we're playing through the pvp file like a movie, use
   // BatchIndexer to get the frame number. Otherwise, just use
   // the start_frame_index value for this batch.
   if (getDisplayPeriod() > 0) {
      frameNumber = mBatchIndexer->nextIndex(batchIndex);
   }
   else {
      frameNumber = getStartIndex(batchIndex);
   }

   Buffer<float> result;
   BufferUtils::readActivityFromPvp<float>(filename.c_str(), &result, frameNumber, &sparseTable);

   return result;
}
}
