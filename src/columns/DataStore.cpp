/*
 * DataStore.cpp
 *
 *  Created on: Sep 10, 2008
 *      Author: Craig Rasmussen
 */

#include "DataStore.hpp"
#include "include/pv_common.h"
#include "utils/PVLog.hpp"

#include <assert.h>
#include <stdlib.h>
#include <limits>

namespace PV
{

DataStore::DataStore(int numBuffers, int numItems, int numLevels, bool isSparse_flag)
{
   assert(numLevels > 0 && numBuffers > 0);
   this->curLevel = 0; // Publisher::publish decrements levels when writing, so first level written to is numLevels - 1;
   this->numItems = numItems;
   this->numLevels = numLevels;
   this->numBuffers = numBuffers;

   //Level (delay) spins slower than bufferId (batch element)
   mBuffer.resize(numLevels);
   for(auto& v : mBuffer) {
      v.resize(numBuffers*numItems);
   }
   mLastUpdateTimes.resize(numLevels);
   for(auto& v : mLastUpdateTimes) {
      v.resize(numBuffers, -std::numeric_limits<double>::infinity());
   }
   this->isSparse_flag = isSparse_flag;
   if(this->isSparse_flag) {
      mActiveIndices.resize(numLevels);
      for(auto& v : mActiveIndices) {
         v.resize(numBuffers*numItems);
      }
      mNumActive.resize(numLevels);
      for(auto& v : mNumActive) {
         v.resize(numBuffers);
      }
   }
   else {
      mActiveIndices.clear();
      mNumActive.clear();
   }
}

}
