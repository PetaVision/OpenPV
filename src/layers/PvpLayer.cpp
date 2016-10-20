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
   if (status != PV_SUCCESS) {
      return status;
   }
   if (parent->columnId() == 0) {
      pvErrorIf(
            getUsingFileList(),
            "%s: PvpLayer does not support using a list of files.\n",
            getName());
      initializeBatchIndexer(mPvpFrameCount);
   }
   return status;
}

Buffer<float> PvpLayer::retrieveData(std::string filename, int batchIndex) {

   // This is present so that when nextInput() is called during
   // InputLayer::allocateDataStructures, we correctly load the
   // inital state of the layer. Then, after InputLayer::allocate
   // is finished, PvpLayer::allocate reinitializes the BatchIndexer
   // so that the first update state does not skip the first
   // frame in the batch.
   if (mPvpFrameCount == -1) {
      FileStream headerStream(filename.c_str(), std::ios_base::in | std::ios_base::binary, false);
      vector<int> header = BufferUtils::readHeader(headerStream);
      mInputNx           = header.at(INDEX_NX);
      mInputNy           = header.at(INDEX_NY);
      mInputNf           = header.at(INDEX_NF);
      mFileType          = header.at(INDEX_FILE_TYPE);
      mPvpFrameCount     = header.at(INDEX_NBANDS);
      initializeBatchIndexer(mPvpFrameCount);
      if (header.at(INDEX_FILE_TYPE) == PVP_ACT_SPARSEVALUES_FILE_TYPE
          || header.at(INDEX_FILE_TYPE) == PVP_ACT_FILE_TYPE) {
         sparseTable = BufferUtils::buildSparseFileTable(headerStream, mPvpFrameCount - 1);
      }
   }

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

   SparseList<float> list;
   Buffer<float> result(mInputNx, mInputNy, mInputNf);

   switch (mFileType) {
      case PVP_NONSPIKING_ACT_FILE_TYPE:
         BufferUtils::readFromPvp<float>(filename.c_str(), &result, frameNumber);
         break;
      case PVP_ACT_SPARSEVALUES_FILE_TYPE:
         BufferUtils::readSparseFromPvp<float>(filename.c_str(), &list, frameNumber, &sparseTable);
         // This is a hack. We should only ever be
         // calling this with T == float.
         list.toBuffer(result, {0});
         break;
      case PVP_ACT_FILE_TYPE:
         // The {1} and {0} are the same hack.
         BufferUtils::readSparseBinaryFromPvp<float>(
               filename.c_str(), &list, frameNumber, {1}, &sparseTable);
         list.toBuffer(result, {0});
         break;
   }

   return result;
}
}
