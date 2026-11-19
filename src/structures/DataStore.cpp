/*
 * DataStore.cpp
 *
 *  Created on: Sep 10, 2008
 *      Author: Craig Rasmussen
 */

#include <limits> // numeric_limits::infinity()

#include "DataStore.hpp"
#include "utils/PVAssert.hpp"

namespace PV {

DataStore::DataStore(
      int numBuffers, int numLevels, int width, int height, int features, bool sparseFlag) {
   pvAssert(numLevels > 0 && numBuffers > 0);
   mWidth      = width;
   mHeight     = height;
   mFeatures   = features;
   mNumItems   = static_cast<long>(width) * static_cast<long>(height) * static_cast<long>(features);
   mNumLevels  = numLevels;
   mNumBuffers = numBuffers;

   double neginf    = -std::numeric_limits<double>::infinity(); // Initially has never been updated
   mLastUpdateTimes = std::make_shared<RingBuffer<double>>(
         numLevels, 1 /*numBuffers*/, neginf /*initial value*/);
   // All batch elements in a level have the same last-update time

   mSparseFlag = sparseFlag;
   if (mSparseFlag) {
      mActiveIndices = std::make_shared<RingBuffer<SparseList<float>>>(
            numLevels, numBuffers, SparseList<float>(width, height, features));
      mNumActive = std::make_shared<RingBuffer<long>>(numLevels, numBuffers);
   }
   else {
      mBuffer = std::make_shared<RingBuffer<std::vector<float>>>(
            numLevels, numBuffers, std::vector<float>(mNumItems));
   }
}

void DataStore::newLevelIndex() {
   mLastUpdateTimes->newLevel();

   if (isSparse()) {
      mNumActive->newLevel();
      mActiveIndices->newLevel();
      pvAssert(mNumActive->getCurrentLevel() == mLastUpdateTimes->getCurrentLevel());
      pvAssert(mActiveIndices->getCurrentLevel() == mLastUpdateTimes->getCurrentLevel());
   }
   else {
      mBuffer->newLevel();
      pvAssert(mBuffer->getCurrentLevel() == mLastUpdateTimes->getCurrentLevel());
   }
}

void DataStore::updateActiveIndices(int bufferId, int level) {
   if (!mSparseFlag) {
      return;
   }
   std::vector<float> &activity = getBuffer(bufferId, level);

   SparseList<float> &activeIndices = getActiveIndices(bufferId, level);
   activeIndices.reset(0, 0, 0);
   SparseEntry<float> entry;
   for (int kex = 0; kex < getNumItems(); kex++) {
      float a = activity[kex];
      if (a != 0.0f) {
         activeIndices.addEntry(kex, a);
      }
   }
}

PVLayerCube DataStore::createCube(PVLayerLoc const &loc, int delay) {
   PVLayerCube cube;
   cube.numItems = mNumItems;
   // All batch elements allocated contiguously, so the numItems and data fields cover all
   // batch elements.
   cube.loc      = loc;
   cube.isSparse = isSparse();
   cube.data.resize(mNumBuffers);
   if (isSparse()) {
      cube.activeIndices.resize(mNumBuffers);
      for (int b = 0; b < mNumBuffers; ++b) {
         cube.activeIndices[b] = &getActiveIndices(b, delay);
      }
   }
   else {
      for (int b = 0; b < mNumBuffers; ++b) {
         cube.data[b] = getBuffer(b, delay).data();
      }
   }
   return cube;
}

} // end namespace PV
