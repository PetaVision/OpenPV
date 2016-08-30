/*
 * DataStore.cpp
 *
 *  Created on: Sep 10, 2008
 *      Author: Craig Rasmussen
 */

#include "DataStore.hpp"
#include "include/pv_common.h"
#include "utils/PVLog.hpp"
#include "utils/PVAssert.hpp"

#include <limits>

namespace PV
{

DataStore::DataStore(int numBuffers, int numItems, int numLevels, bool isSparse_flag)
{
   assert(numLevels > 0 && numBuffers > 0);
   mCurrentLevel = 0; // Publisher::publish decrements levels when writing, so first level written to is numLevels - 1;
   mNumItems = numItems;
   mNumLevels = numLevels;
   mNumBuffers = numBuffers;

   mBuffer = new RingBuffer<pvdata_t>(numLevels, numBuffers*numItems);
   mLastUpdateTimes = new RingBuffer<double>(numLevels, numBuffers, -std::numeric_limits<double>::infinity()/*initial value*/);

   mSparseFlag = isSparse_flag;
   if(mSparseFlag) {
      mActiveIndices = new RingBuffer<unsigned int>(numLevels, numBuffers*numItems);
      mNumActive = new RingBuffer<long>(numLevels, numBuffers);
   }
}

}
