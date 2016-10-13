/*
 * CheckpointEntry.cpp
 *
 *  Created on Sep 27, 2016
 *      Author: Pete Schultz
 */

#include "CheckpointEntryDataStore.hpp"
#include "structures/Buffer.hpp"
#include "utils/BufferUtilsMPI.hpp"
#include "utils/BufferUtilsPvp.hpp"

namespace PV {

void CheckpointEntryDataStore::initialize(RingBuffer<pvdata_t> *buffer, RingBuffer<double> *lastUpdateTimes, PVLayerLoc const *layerLoc) {
   mRingBuffer = buffer;
   mLastUpdateTimes = lastUpdateTimes;
   mLayerLoc = layerLoc;
}

void CheckpointEntryDataStore::write(std::string const &checkpointDirectory, double simTime, bool verifyWritesFlag) const {
   FileStream *fileStream = nullptr;
   if (getCommunicator()->commRank() == 0) {
      std::string path = generatePath(checkpointDirectory, "pvp");
      fileStream = new FileStream(path.c_str(), std::ios_base::out, verifyWritesFlag);
      int const numBands = mRingBuffer->getNumLevels() * mLayerLoc->nbatch;
      int *params      = pvp_set_nonspiking_act_params(
            getCommunicator(), simTime, mLayerLoc, PV_FLOAT_TYPE, numBands /*numbands*/);
      pvAssert(params && params[1] == NUM_BIN_PARAMS);
      fileStream->write(params, params[0]);
   }
   int const numBuffers = mLayerLoc->nbatch;
   int const numLevels = mRingBuffer->getNumLevels();
   int const nxExt = mLayerLoc->nx + mLayerLoc->halo.lt + mLayerLoc->halo.rt;
   int const nyExt = mLayerLoc->ny + mLayerLoc->halo.dn + mLayerLoc->halo.up;
   int const nf = mLayerLoc->nf;
   int const numElements = nxExt * nyExt * nf;
   for (int b = 0; b < numBuffers; b++) {
      for (int l = 0; l < numLevels; l++) {
         double lastUpdateTime = *mLastUpdateTimes->getBuffer(l, b);
         pvdata_t const *localData = mRingBuffer->getBuffer(l, b*numElements);
         Buffer<pvdata_t> localPvpBuffer = Buffer<pvdata_t>{localData, nxExt, nyExt, nf};
         Buffer<pvdata_t> globalPvpBuffer = BufferUtils::gather<pvdata_t>(getCommunicator(), localPvpBuffer, mLayerLoc->nx, mLayerLoc->ny);
         if (fileStream) { BufferUtils::writeFrame(*fileStream, &globalPvpBuffer, lastUpdateTime); }
      }
   }
   delete fileStream;
}

void CheckpointEntryDataStore::read(std::string const &checkpointDirectory, double *simTimePtr) const {
   
}

void CheckpointEntryDataStore::remove(std::string const &checkpointDirectory) const {
   deleteFile(checkpointDirectory, "pvp");
}

}  // end namespace PV
