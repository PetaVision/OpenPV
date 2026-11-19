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
// Write and remove methods inherited from CheckpointEntryLayerPvp.

int CheckpointEntryDataStore::getNumIndices() const {
   return getDataStore()->getNumLevels();
}

float *CheckpointEntryDataStore::calcBatchElementStart(int batchElement, int index) const {
   return getDataStore()->getBuffer(batchElement, index).data();
}

void CheckpointEntryDataStore::writeSparseBroadcast(
      std::shared_ptr<FileManager const> fileManager,
      std::string const &filename,
      double simTime,
      bool verifyWritesFlag) const {
   SparseBroadcastLayerFile layerFile(
         fileManager,
         filename,
         getLayerLoc()->nf,
         getLayerLoc()->nbatch,
         false /*readOnlyFlag*/,
         true /*clobberFlag*/,
         verifyWritesFlag);
   for (int i = 0; i < getNumIndices(); ++i) {
      for (int b = 0; b < getLayerLoc()->nbatch; ++b) {
         SparseList<float> &sparseList = mDataStore->getActiveIndices(b, i);
         layerFile.setListLocation(&sparseList, b);
      }
      layerFile.write(simTime);
   }
}

void CheckpointEntryDataStore::writeSparseFile(
      std::shared_ptr<FileManager const> fileManager,
      std::string const &filename,
      double simTime,
      bool verifyWritesFlag) const {
   SparseLayerFile layerFile(
         fileManager,
         filename,
         *getLayerLoc(),
         getExtendedFlag(),
         false /*fileExtendedFlag*/,
         false /*readOnlyFlag*/,
         true /*clobberFlag*/,
         verifyWritesFlag);
   for (int i = 0; i < getNumIndices(); ++i) {
      for (int b = 0; b < getLayerLoc()->nbatch; ++b) {
         SparseList<float> &sparseList = mDataStore->getActiveIndices(b, i);
         layerFile.setListLocation(&sparseList, b);
      }
      layerFile.write(simTime);
   }
}

void CheckpointEntryDataStore::loadBatchElement(
      SparseList<float> const &sparseList, int batchElement, int index) const {
   SparseList<float> &destSparseList = mDataStore->getActiveIndices(batchElement, index);
   destSparseList = sparseList;
}

void CheckpointEntryDataStore::setLastUpdateTimes(std::vector<double> const &timestamps) const {
   int const numBuffers = getDataStore()->getNumBuffers(); // HyPerCol's nbatch
   int const numLevels  = getDataStore()->getNumLevels();  // No. of delay levels
   for (int l = 0; l < numLevels; l++) {
      getDataStore()->setLastUpdateTime(l, timestamps[l]);
   }
}

} // end namespace PV
