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

// Constructors defined in .hpp file.
// Write and remove methods inherited from CheckpointEntryPvp.

void CheckpointEntryDataStore::read(std::string const &checkpointDirectory, double *simTimePtr)
      const {
   CheckpointEntryPvp::read(checkpointDirectory, simTimePtr);
   for (int bufferId = 0; bufferId < mDataStore->getNumBuffers(); bufferId++) {
      for (int levelId = 0; levelId < mDataStore->getNumLevels(); levelId++) {
         mDataStore->markActiveIndicesOutOfSync(bufferId, levelId);
      }
   }
}

int CheckpointEntryDataStore::getNumFrames() const {
   int const numBuffers  = getDataStore()->getNumBuffers();
   int const numLevels   = getDataStore()->getNumLevels();
   int const mpiBatchDim = getMPIBlock()->getBatchDimension();

   return numLevels * numBuffers * mpiBatchDim;
}

float *CheckpointEntryDataStore::calcBatchElementStart(int frame) const {
   int const numBuffers = getDataStore()->getNumBuffers();
   int const numLevels  = getDataStore()->getNumLevels();
   int const level      = frame % numLevels;
   int const buffer     = (frame / numLevels) % numBuffers; // Integer division
   return getDataStore()->buffer(buffer, level);
}

int CheckpointEntryDataStore::calcMPIBatchIndex(int frame) const {
   return frame / (getDataStore()->getNumLevels() * getDataStore()->getNumBuffers());
}

void CheckpointEntryDataStore::setLastUpdateTimes(std::vector<double> const &timestamps) const {
   int const numBuffers                  = getDataStore()->getNumBuffers();
   int const numLevels                   = getDataStore()->getNumLevels();
   int const mpiBatchIndex               = getMPIBlock()->getBatchIndex();
   double const *updateTimesBatchElement = &timestamps[numBuffers * numLevels * mpiBatchIndex];
   for (int b = 0; b < numBuffers; b++) {
      for (int l = 0; l < numLevels; l++) {
         getDataStore()->setLastUpdateTime(b, l, updateTimesBatchElement[b * numLevels + l]);
      }
   }
}

} // end namespace PV
