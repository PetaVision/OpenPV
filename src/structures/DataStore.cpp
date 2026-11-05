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

DataStore::DataStore(int numBuffers, long numItems, int numLevels, bool sparseFlag) {
   pvAssert(numLevels > 0 && numBuffers > 0);
   mNumItems   = numItems;
   mNumLevels  = numLevels;
   mNumBuffers = numBuffers;

   mBuffer = std::make_shared<RingBuffer<std::vector<float>>>(
         numLevels, numBuffers, std::vector<float>(numItems));
   double neginf    = -std::numeric_limits<double>::infinity(); // Initially has never been updated
   mLastUpdateTimes = std::make_shared<RingBuffer<double>>(
         numLevels, 1 /*numBuffers*/, neginf /*initial value*/);
   // All batch elements in a level have the same last-update time

   mSparseFlag = sparseFlag;
   if (mSparseFlag) {
      mActiveIndices = std::make_shared<RingBuffer<SparseList<float>>>(
            numLevels, numBuffers, SparseList<float>());
      mNumActive = std::make_shared<RingBuffer<long>>(numLevels, numBuffers);
   }
}

void DataStore::newLevelIndex() {
   mBuffer->newLevel();
   mLastUpdateTimes->newLevel();
   pvAssert(mBuffer->getCurrentLevel() == mLastUpdateTimes->getCurrentLevel());

   if (isSparse()) {
      mNumActive->newLevel();
      mActiveIndices->newLevel();
      pvAssert(mNumActive->getCurrentLevel() == mLastUpdateTimes->getCurrentLevel());
      pvAssert(mActiveIndices->getCurrentLevel() == mLastUpdateTimes->getCurrentLevel());
   }
}

void DataStore::markActiveIndicesOutOfSync(int bufferId, int level) {
   if (!mSparseFlag) {
      return;
   }
   long &numActiveBuf = numActiveBuffer(bufferId, level);
   numActiveBuf      = -1;
}

void DataStore::updateActiveIndices(int bufferId, int level) {
   if (!mSparseFlag) {
      return;
   }
   int numActive   = 0;
   std::vector<float> &activity = buffer(bufferId, level);

   SparseList<float> &activeIndices = activeIndicesBuffer(bufferId, level);
   activeIndices.reset(0, 0, 0); // DataStore knows # of items but not dimensions. Change this?
   SparseList<float>::Entry entry;
   for (int kex = 0; kex < getNumItems(); kex++) {
      float a = activity[kex];
      if (a != 0.0f) {
         activeIndices.addEntry(kex, a);
         numActive++;
      }
   }

   long &numActiveBuf = numActiveBuffer(bufferId, level);
   numActiveBuf       = numActive;
}

PVLayerCube DataStore::createCube(PVLayerLoc const &loc, int delay) {
   PVLayerCube cube;
   cube.numItems = mNumItems;
   // All batch elements allocated contiguously, so the numItems and data fields cover all
   // batch elements.
   cube.loc      = loc;
   cube.isSparse = isSparse();
   cube.data.resize(mNumBuffers);
   for (int b = 0; b < mNumBuffers; ++b) {
      cube.data[b] = buffer(b, delay).data();
   }
   if (isSparse()) {
      cube.numActive.resize(mNumBuffers);
      cube.activeIndices.resize(mNumBuffers);
      for (int b = 0; b < mNumBuffers; ++b) {
         cube.numActive[b]     = numActiveBuffer(b, delay);
         cube.activeIndices[b] = &activeIndicesBuffer(b, delay);
      }
   }
   return cube;
}

} // end namespace PV
