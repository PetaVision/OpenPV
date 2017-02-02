/*
 * DataStore.cpp
 *
 *  Created on: Sep 10, 2008
 *      Author: Craig Rasmussen
 */

#include "DataStore.hpp"
#include "include/pv_common.h"
#include "utils/PVAssert.hpp"
#include "utils/PVLog.hpp"

#include <limits>

namespace PV {

DataStore::DataStore(int numBuffers, int numItems, int numLevels, bool isSparse_flag) {
   assert(numLevels > 0 && numBuffers > 0);
   mCurrentLevel = 0; // Publisher::publish decrements levels when writing, so
   // first level written
   // to is numLevels - 1;
   mNumItems   = numItems;
   mNumLevels  = numLevels;
   mNumBuffers = numBuffers;

   mBuffer          = new RingBuffer<float>(numLevels, numBuffers * numItems);
   mLastUpdateTimes = new RingBuffer<double>(
         numLevels, numBuffers, -std::numeric_limits<double>::infinity() /*initial value*/);

   mSparseFlag = isSparse_flag;
   if (mSparseFlag) {
      mActiveIndices = new RingBuffer<SparseList<float>::Entry>(numLevels, numBuffers * numItems, {0,0});
      mNumActive     = new RingBuffer<long>(numLevels, numBuffers);
   }
}

void DataStore::updateActiveIndices(int bufferId, int level) {
   if (!mSparseFlag) { return; }
   int numActive   = 0;
   float *activity = buffer(bufferId, level);

   SparseList<float>::Entry *activeIndices = activeIndicesBuffer(bufferId, level);
//   unsigned int *activeIndices = activeIndicesBuffer(bufferId, level);
   for (int kex = 0; kex < getNumItems(); kex++) {
      float a = activity[kex];
      if (a != 0.0f) {
         activeIndices[numActive].index = kex;
         activeIndices[numActive].value = a;
         numActive++;
      }
   }

   long *numActiveBuf = numActiveBuffer(bufferId, level);
   *numActiveBuf = numActive;
}

} // end namespace PV
