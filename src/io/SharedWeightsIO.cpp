#include "SharedWeightsIO.hpp"

#include "include/pv_common.h"
#include "utils/PVAssert.hpp"
#include "utils/PVLog.hpp"
#include "utils/weight_conversions.hpp"

#include <cfloat>
#include <cstdint>
#include <ios>

namespace PV {

SharedWeightsIO::SharedWeightsIO(
      std::shared_ptr<FileStream> fileStream,
      int patchSizeX,
      int patchSizeY,
      int patchSizeF,
      int numPatchesX,
      int numPatchesY,
      int numPatchesF,
      int numArbors,
      bool compressedFlag)
      : mFileStream(fileStream),
        mPatchSizeX(patchSizeX),
        mPatchSizeY(patchSizeY),
        mPatchSizeF(patchSizeF),
        mNumPatchesX(numPatchesX),
        mNumPatchesY(numPatchesY),
        mNumPatchesF(numPatchesF),
        mNumArbors(numArbors),
        mCompressedFlag(compressedFlag) {
   FatalIf(
         fileStream and !fileStream->readable(),
         "FileStream \"%s\" is not readable and can't be used in a SharedWeightsIO object.\n",
         fileStream->getFileName().c_str());
   mDataSize = static_cast<long>(mCompressedFlag ? sizeof(uint8_t) : sizeof(float));
   initializeFrameIndexer();

   if (!getFileStream()) {
      return;
   }

   // If writeable, initialize position at end of file.
   // If read-only, initialize position at beginning.
   // Users can call setFrameNumber() if something else is desired.
   if (getFileStream()->writeable()) {
      int numFrames = getNumFrames();
      setFrameNumber(numFrames);
   }
   else {
      setFrameNumber(0);
   }
}

long SharedWeightsIO::calcFilePositionFromFrameNumber(int frameNumber) const {
   return mFrameIndexer->calcFilePositionFromFrameNumber(frameNumber);
}

int SharedWeightsIO::calcFrameNumberFromFilePosition(long filePosition) const {
   return mFrameIndexer->calcFrameNumberFromFilePosition(filePosition);
}

void SharedWeightsIO::read(WeightData &weightData) {
   if (!mFileStream) {
      return;
   }
   double dummyTimestamp;
   read(weightData, dummyTimestamp);
}

void SharedWeightsIO::read(WeightData &weightData, double &timestamp) {
   if (!mFileStream) {
      return;
   }
   checkDimensions(weightData);
   BufferUtils::WeightHeader header;
   mFileStream->read(&header, mHeaderSize);
   checkHeader(header);

   Patch patchHeader;
   long numValuesInPatch = getPatchSizeOverall();
   for (int a = 0; a < mNumArbors; ++a) {
      for (int p = 0; p < getNumPatchesOverall(); ++p) {
         mFileStream->read(&patchHeader, mPatchHeaderSize);
         long sizeInFile  = numValuesInPatch * mDataSize;
         float *dataStart = weightData.getDataFromDataIndex(a, p);
         if (mCompressedFlag) {
            uint8_t compressedData[numValuesInPatch];
            mFileStream->read(compressedData, numValuesInPatch);
            for (long k = 0; k < numValuesInPatch; ++k) {
               dataStart[k] = uncompressWeight(compressedData[k], header.minVal, header.maxVal);
            }
         }
         else {
            mFileStream->read(dataStart, sizeInFile);
         }
      }
   }
   setFrameNumber(getFrameNumber() + 1);

   timestamp = header.baseHeader.timestamp;
}

void SharedWeightsIO::read(WeightData &weightData, double &timestamp, int frameNumber) {
   if (mFileStream) {
      setFrameNumber(frameNumber);
      read(weightData, timestamp);
   }
}

void SharedWeightsIO::write(WeightData const &weightData, double timestamp) {
   if (!mFileStream) {
      return;
   }

   checkDimensions(weightData);
   float minWeight, maxWeight;
   weightData.calcExtremeWeights(minWeight, maxWeight);
   auto header = BufferUtils::buildWeightHeader(
         true /*sharedFlag*/,
         getNumPatchesX(),
         getNumPatchesY(),
         getNumPatchesF(),
         getNumPatchesX(),
         getNumPatchesY(),
         getNumArbors(),
         timestamp,
         getPatchSizeX(),
         getPatchSizeY(),
         getPatchSizeF(),
         getCompressedFlag(),
         minWeight,
         maxWeight);
   mFileStream->write(&header, mHeaderSize);
   checkHeader(header);

   long numValuesInPatch = getPatchSizeOverall();
   for (int a = 0; a < mNumArbors; ++a) {
      Patch writePatch;
      writePatch.nx     = static_cast<uint16_t>(getPatchSizeX());
      writePatch.ny     = static_cast<uint16_t>(getPatchSizeY());
      writePatch.offset = static_cast<uint32_t>(0);
      for (int p = 0; p < getNumPatchesOverall(); ++p) {
         mFileStream->write(&writePatch, mPatchHeaderSize);
         long sizeInFile        = numValuesInPatch * mDataSize;
         float const *arbor     = weightData.getData(a);
         float const *dataStart = &arbor[p * numValuesInPatch];
         if (mCompressedFlag) {
            uint8_t compressedData[numValuesInPatch];
            for (long k = 0; k < numValuesInPatch; ++k) {
               compressedData[k] = compressWeight(dataStart[k], header.minVal, header.maxVal);
            }
            mFileStream->write(compressedData, sizeInFile);
         }
         else {
            mFileStream->write(dataStart, sizeInFile);
         }
      }
   }
   setFrameNumber(getFrameNumber() + 1);
}

void SharedWeightsIO::write(WeightData const &weightData, double timestamp, int frameNumber) {
   setFrameNumber(frameNumber);
   write(weightData, timestamp);
}

void SharedWeightsIO::open() {
   if (mFileStream) {
      mFrameIndexer = nullptr;
      mFileStream->open();
      initializeFrameIndexer();
   }
}

void SharedWeightsIO::close() {
   if (mFileStream) {
      mFrameIndexer = nullptr;
      mFileStream->close();
   }
}

void SharedWeightsIO::checkDimensions(WeightData const &weightData) {
   int status = PV_SUCCESS;
   if (weightData.getNumArbors() != mNumArbors) {
      ErrorLog().printf(
            "WeightData object passed to SharedWeightsIO has %d arbors, but it expects %d\n",
            weightData.getNumArbors(),
            getNumArbors());
      status = PV_FAILURE;
   }
   if (getNumPatchesX() != static_cast<long>(weightData.getNumDataPatchesX())) {
      ErrorLog().printf(
            "WeightData object has width %d, but SharedWeightsIO object expects %ld\n",
            weightData.getNumDataPatchesX(),
            getNumPatchesX());
      status = PV_FAILURE;
   }
   if (getNumPatchesY() != static_cast<long>(weightData.getNumDataPatchesY())) {
      ErrorLog().printf(
            "WeightData object has height %d, but SharedWeightsIO object expects %ld\n",
            weightData.getNumDataPatchesY(),
            getNumPatchesY());
      status = PV_FAILURE;
   }
   if (getNumPatchesF() != static_cast<long>(weightData.getNumDataPatchesF())) {
      ErrorLog().printf(
            "WeightData object has %d features, but SharedWeightsIO object expects %ld\n",
            weightData.getNumDataPatchesF(),
            getNumPatchesF());
      status = PV_FAILURE;
   }
   FatalIf(status != PV_SUCCESS, "checkArborListDimensions failed.\n");
}

void SharedWeightsIO::checkHeader(BufferUtils::WeightHeader const &header) const {
   int status = PV_SUCCESS;
   if (header.baseHeader.numRecords != mNumArbors) {
      ErrorLog().printf(
            "SharedWeightsIO object expects %d arbors, but file has %d.\n",
            mNumArbors,
            header.baseHeader.numRecords);
      status = PV_FAILURE;
   }
   if (header.nxp != mPatchSizeX) {
      ErrorLog().printf(
            "SharedWeightsIO object expects PatchSizeX=%d, but file has %d.\n",
            mPatchSizeX,
            header.nxp);
      status = PV_FAILURE;
   }
   if (header.nyp != mPatchSizeY) {
      ErrorLog().printf(
            "SharedWeightsIO object expects PatchSizeX=%d, but file has %d.\n",
            mPatchSizeY,
            header.nyp);
      status = PV_FAILURE;
   }
   if (header.nfp != mPatchSizeF) {
      ErrorLog().printf(
            "SharedWeightsIO object expects PatchSizeF=%d, but file has %d.\n",
            mPatchSizeF,
            header.nfp);
      status = PV_FAILURE;
   }
   if (static_cast<long>(header.numPatches) != getNumPatchesOverall()) {
      ErrorLog().printf(
            "SharedWeightsIO object expects %ld patches, but file has %d.\n",
            getNumPatchesOverall(),
            header.numPatches);
      status = PV_FAILURE;
   }
   FatalIf(status != PV_SUCCESS, "checkHeader failed.\n");
}

void SharedWeightsIO::initializeFrameIndexer() {
   long patchSizeBytes = mPatchHeaderSize + mDataSize * static_cast<long>(getPatchSizeOverall());
   long numPatches     = getNumPatchesOverall();
   long frameSize      = mHeaderSize + static_cast<long>(mNumArbors) * numPatches * patchSizeBytes;

   mFrameIndexer = std::make_shared<PVPFrameIndexer>(
         mFileStream, frameSize, 0L /*externalHeaderSize*/);
   // Weight PVP files have a header in each frame and no external header; hence
   // externalHeaderSize is 0 and not the weight header size of 104.
}

} // namespace PV
