/*
 * CheckpointEntryDataStore.cpp
 *
 *  Created on Sep 27, 2016
 *      Author: Pete Schultz
 */

#include "CheckpointEntryDataStore.hpp"
#include "structures/Buffer.hpp"
#include "utils/BufferUtilsMPI.hpp"
#include "utils/BufferUtilsPvp.hpp"
#include "utils/PVAssert.hpp"

namespace PV {

// TODO: many commonalities between CheckpointEntryPvp and CheckpointEntryDataStore.
// Refactor to eliminate code duplication

void CheckpointEntryDataStore::initialize(DataStore *dataStore, PVLayerLoc const *layerLoc) {
   mDataStore = dataStore;
   mLayerLoc  = layerLoc;
   mXMargins  = layerLoc->halo.lt + layerLoc->halo.rt;
   mYMargins  = layerLoc->halo.dn + layerLoc->halo.up;
}

void CheckpointEntryDataStore::write(
      std::string const &checkpointDirectory,
      double simTime,
      bool verifyWritesFlag) const {
   int const numFrames = getNumFrames();
   int const nxBlock   = mLayerLoc->nx * getMPIBlock()->getNumColumns();
   int const nyBlock   = mLayerLoc->ny * getMPIBlock()->getNumRows();

   FileStream *fileStream = nullptr;
   if (getMPIBlock()->getRank() == 0) {
      std::string path = generatePath(checkpointDirectory, "pvp");
      fileStream       = new FileStream(path.c_str(), std::ios_base::out, verifyWritesFlag);
      BufferUtils::ActivityHeader header =
            BufferUtils::buildActivityHeader<float>(nxBlock, nyBlock, mLayerLoc->nf, numFrames);
      BufferUtils::writeActivityHeader(*fileStream, header);
   }
   int const nxExtLocal = mLayerLoc->nx + mXMargins;
   int const nyExtLocal = mLayerLoc->ny + mYMargins;
   int const nf         = mLayerLoc->nf;

   for (int frame = 0; frame < numFrames; frame++) {
      float const *localData = calcBatchElementStart(frame);

      Buffer<float> pvpBuffer{localData, nxExtLocal, nyExtLocal, nf};
      pvpBuffer.crop(mLayerLoc->nx, mLayerLoc->ny, Buffer<float>::CENTER);

      // All ranks with BatchIndex==mpiBatchIndex must call gather; so must
      // the root process (which may or may not have BatchIndex==mpiBatchIndex).
      // Other ranks will return from gather() immediately.
      int const mpiBatchIndex       = calcMPIBatchIndex(frame);
      Buffer<float> globalPvpBuffer = BufferUtils::gather(
            getMPIBlock(), pvpBuffer, mLayerLoc->nx, mLayerLoc->ny, mpiBatchIndex, 0);

      if (getMPIBlock()->getRank() == 0) {
         pvAssert(fileStream);
         pvAssert(globalPvpBuffer.getWidth() == nxBlock);
         pvAssert(globalPvpBuffer.getHeight() == nyBlock);
         BufferUtils::writeFrame(*fileStream, &globalPvpBuffer, simTime);
      }
   }
   delete fileStream;
}

void CheckpointEntryDataStore::read(std::string const &checkpointDirectory, double *simTimePtr)
      const {
   int const numFrames = getNumFrames();
   int const nxBlock   = mLayerLoc->nx * getMPIBlock()->getNumColumns();
   int const nyBlock   = mLayerLoc->ny * getMPIBlock()->getNumRows();

   int const nxExtLocal  = mLayerLoc->nx + mXMargins;
   int const nyExtLocal  = mLayerLoc->ny + mYMargins;
   int const nxExtGlobal = nxBlock + mXMargins;
   int const nyExtGlobal = nyBlock + mYMargins;

   std::string path;
   if (getMPIBlock()->getRank() == 0) {
      path = generatePath(checkpointDirectory, "pvp");
      FileStream fileStream(path.c_str(), std::ios_base::in, false);
      struct BufferUtils::ActivityHeader header = BufferUtils::readActivityHeader(fileStream);
      FatalIf(
            header.nBands != numFrames,
            "CheckpointEntryDataStore::read error reading \"%s\": delays*batchwidth in file is %d, "
            "but delays*batchwidth in layer is %d\n",
            path.c_str(),
            header.nBands,
            numFrames);
   }
   Buffer<float> pvpBuffer;
   std::vector<double> frameTimestamps;
   frameTimestamps.resize(numFrames);
   for (int frame = 0; frame < numFrames; frame++) {
      int const mpiBatchIndex = calcMPIBatchIndex(frame);
      if (getMPIBlock()->getRank() == 0) {
         frameTimestamps.at(frame) =
               BufferUtils::readActivityFromPvp(path.c_str(), &pvpBuffer, frame);
         pvpBuffer.grow(nxExtGlobal, nyExtGlobal, Buffer<float>::CENTER);
      }
      else if (mpiBatchIndex == getMPIBlock()->getBatchIndex()) {
         pvpBuffer.resize(nxExtLocal, nyExtLocal, mLayerLoc->nf);
      }
      // All ranks with BatchIndex==mpiBatchIndex must call scatter; so must
      // the root process (which may or may not have BatchIndex==mpiBatchIndex).
      // Other ranks will return from scatter() immediately.
      BufferUtils::scatter(
            getMPIBlock(), pvpBuffer, mLayerLoc->nx, mLayerLoc->ny, mpiBatchIndex, 0);
      if (mpiBatchIndex == getMPIBlock()->getBatchIndex()) {
         std::vector<float> bufferData = pvpBuffer.asVector();
         float *localData              = calcBatchElementStart(frame);
         std::memcpy(
               localData,
               bufferData.data(),
               (std::size_t)pvpBuffer.getTotalElements() * sizeof(float));
      }
   }
   MPI_Bcast(
         frameTimestamps.data(), getNumFrames(), MPI_DOUBLE, 0 /*root*/, getMPIBlock()->getComm());
   setLastUpdateTimes(frameTimestamps);
}

int CheckpointEntryDataStore::getNumFrames() const {
   int const numBuffers  = mDataStore->getNumBuffers();
   int const numLevels   = mDataStore->getNumLevels();
   int const mpiBatchDim = getMPIBlock()->getBatchDimension();

   return numLevels * numBuffers * mpiBatchDim;
}

float *CheckpointEntryDataStore::calcBatchElementStart(int frame) const {
   int const numBuffers = mDataStore->getNumBuffers();
   int const numLevels  = mDataStore->getNumLevels();
   int const level      = frame % numLevels;
   int const buffer     = (frame / numLevels) % numBuffers; // Integer division
   return mDataStore->buffer(buffer, level);
}

int CheckpointEntryDataStore::calcMPIBatchIndex(int frame) const {
   return frame / (mDataStore->getNumLevels() * mDataStore->getNumBuffers());
}

void CheckpointEntryDataStore::setLastUpdateTimes(std::vector<double> const &timestamps) const {
   int const numBuffers                  = mDataStore->getNumBuffers();
   int const numLevels                   = mDataStore->getNumLevels();
   int const mpiBatchIndex               = getMPIBlock()->getBatchIndex();
   double const *updateTimesBatchElement = &timestamps[numBuffers * numLevels * mpiBatchIndex];
   for (int b = 0; b < numBuffers; b++) {
      for (int l = 0; l < numLevels; l++) {
         mDataStore->setLastUpdateTime(b, l, updateTimesBatchElement[b * numLevels + l]);
      }
   }
}

void CheckpointEntryDataStore::remove(std::string const &checkpointDirectory) const {
   deleteFile(checkpointDirectory, "pvp");
}

} // end namespace PV
