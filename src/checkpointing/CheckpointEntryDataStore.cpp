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
}

void CheckpointEntryDataStore::write(
      std::string const &checkpointDirectory,
      double simTime,
      bool verifyWritesFlag) const {
   int const numBuffers  = mDataStore->getNumBuffers();
   int const numLevels   = mDataStore->getNumLevels();
   int const mpiBatchDim = getMPIBlock()->getBatchDimension();
   int const numFrames   = numLevels * numBuffers * mpiBatchDim;
   int const nxBlock     = mLayerLoc->nx * getMPIBlock()->getNumColumns();
   int const nyBlock     = mLayerLoc->ny * getMPIBlock()->getNumRows();

   FileStream *fileStream = nullptr;
   if (getMPIBlock()->getRank() == 0) {
      std::string path = generatePath(checkpointDirectory, "pvp");
      fileStream       = new FileStream(path.c_str(), std::ios_base::out, verifyWritesFlag);
      BufferUtils::ActivityHeader header =
            BufferUtils::buildActivityHeader<float>(nxBlock, nyBlock, mLayerLoc->nf, numFrames);
      BufferUtils::writeActivityHeader(*fileStream, header);
   }
   PVHalo const &halo   = mLayerLoc->halo;
   int const nxExtLocal = mLayerLoc->nx + halo.lt + halo.rt;
   int const nyExtLocal = mLayerLoc->ny + halo.dn + halo.up;
   int const nf         = mLayerLoc->nf;
   for (int frame = 0; frame < numFrames; frame++) {
      int const level         = frame % numLevels;
      int const buffer        = (frame / numLevels) % numBuffers; // Integer division
      int const mpiBatchIndex = frame / (numLevels * numBuffers); // Integer division

      float const *localData = mDataStore->buffer(buffer, level);
      Buffer<float> pvpBuffer{localData, nxExtLocal, nyExtLocal, nf};
      pvpBuffer.crop(mLayerLoc->nx, mLayerLoc->ny, Buffer<float>::CENTER);

      // All ranks with BatchIndex==mpiBatchIndex must call gather; so must
      // the root process (which may or may not have BatchIndex==mpiBatchIndex).
      // Other ranks will return from gather() immediately.
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
   int const numBuffers  = mDataStore->getNumBuffers();
   int const numLevels   = mDataStore->getNumLevels();
   int const mpiBatchDim = getMPIBlock()->getBatchDimension();
   int const numFrames   = numLevels * numBuffers * mpiBatchDim;
   int const nxBlock     = mLayerLoc->nx * getMPIBlock()->getNumColumns();
   int const nyBlock     = mLayerLoc->ny * getMPIBlock()->getNumRows();

   PVHalo const &halo    = mLayerLoc->halo;
   int const nxExtLocal  = mLayerLoc->nx + halo.lt + halo.rt;
   int const nyExtLocal  = mLayerLoc->ny + halo.dn + halo.up;
   int const nxExtGlobal = nxBlock + halo.lt + halo.rt;
   int const nyExtGlobal = nyBlock + halo.dn + halo.up;

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
   std::vector<double> updateTimes;
   updateTimes.resize(numFrames);
   for (int frame = 0; frame < numFrames; frame++) {
      int const level         = frame % numLevels;
      int const buffer        = (frame / numLevels) % numBuffers; // Integer division
      int const mpiBatchIndex = frame / (numLevels * numBuffers); // Integer division
      if (getMPIBlock()->getRank() == 0) {
         updateTimes.at(frame) = BufferUtils::readActivityFromPvp(path.c_str(), &pvpBuffer, frame);
         pvpBuffer.grow(nxExtGlobal, nyExtGlobal, Buffer<float>::CENTER);
      }
      else {
         pvpBuffer.resize(nxExtLocal, nyExtLocal, mLayerLoc->nf);
      }
      // All ranks with BatchIndex==m must call scatter; so must the root
      // process (which may or may not have BatchIndex==m).
      // Other ranks will return from scatter() immediately.
      BufferUtils::scatter(
            getMPIBlock(), pvpBuffer, mLayerLoc->nx, mLayerLoc->ny, mpiBatchIndex, 0);
      if (mpiBatchIndex == getMPIBlock()->getBatchIndex()) {
         std::vector<float> bufferData = pvpBuffer.asVector();
         float *localData              = mDataStore->buffer(buffer, level);
         std::memcpy(
               localData,
               bufferData.data(),
               (std::size_t)pvpBuffer.getTotalElements() * sizeof(float));
      }
   }
   MPI_Bcast(updateTimes.data(), numFrames, MPI_DOUBLE, 0 /*root*/, getMPIBlock()->getComm());
   int const mpiBatchIndex         = getMPIBlock()->getBatchIndex();
   double *updateTimesBatchElement = &updateTimes[mpiBatchIndex * numLevels * numBuffers];
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
