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

void CheckpointEntryDataStore::read(
      std::shared_ptr<FileManager const> fileManager, double *simTimePtr) const {
   CheckpointEntryPvp::read(fileManager, simTimePtr);
   for (int bufferId = 0; bufferId < mDataStore->getNumBuffers(); bufferId++) {
      for (int levelId = 0; levelId < mDataStore->getNumLevels(); levelId++) {
         mDataStore->markActiveIndicesOutOfSync(bufferId, levelId);
      }
   }
}

int CheckpointEntryDataStore::getNumIndices() const {
   return getDataStore()->getNumLevels();
}

float *CheckpointEntryDataStore::calcBatchElementStart(int batchElement, int index) const {
   return getDataStore()->buffer(batchElement, index);
}

void CheckpointEntryDataStore::setLastUpdateTimes(std::vector<double> const &timestamps) const {
   int const numBuffers                  = getDataStore()->getNumBuffers(); // HyPerCol's nbatch
   int const numLevels                   = getDataStore()->getNumLevels();  // No. of delay levels
   for (int b = 0; b < numBuffers; b++) {
      for (int l = 0; l < numLevels; l++) {
         getDataStore()->setLastUpdateTime(b, l, timestamps[l]);
      }
   }
}

} // end namespace PV
