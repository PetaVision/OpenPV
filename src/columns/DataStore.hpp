/*
 * DataStore.hpp
 *
 *  Created on: Sep 10, 2008
 *      Author: Craig Rasmussen
 */

#ifndef DATASTORE_HPP_
#define DATASTORE_HPP_

#include "include/pv_arch.h"
#include "include/pv_types.h"
#include <cstdlib>
#include <cstring>
#include <vector>

namespace PV
{

class DataStore
{
public:
   DataStore(int numBuffers, int numItems, int numLevels, bool isSparse);

   virtual ~DataStore() {}

   int getNumLevels() const {return mNumLevels;}
   int getNumBuffers() const {return mNumBuffers;}
   int newLevelIndex() {
      return (mCurrentLevel = (mNumLevels + mCurrentLevel - 1) % mNumLevels);
   }

   //Level (delay) spins slower than bufferId (batch element)

   pvdata_t * buffer(int bufferId, int level) {
      return &mBuffer[levelIndex(level)].at(bufferId*mNumItems);
   }

   pvdata_t * buffer(int bufferId) {
      return &mBuffer[mCurrentLevel].at(bufferId*mNumItems);
   }

   double getLastUpdateTime(int bufferId, int level) const {
      return mLastUpdateTimes[levelIndex(level)].at(bufferId);
   }

   double getLastUpdateTime(int bufferId) const {
      return mLastUpdateTimes[levelIndex(0)].at(bufferId);
   }

   void setLastUpdateTime(int bufferId, int level, double t) {
      mLastUpdateTimes[levelIndex(level)].at(bufferId) = t;
   }

   void setLastUpdateTime(int bufferId, double t) {
      mLastUpdateTimes[mCurrentLevel].at(bufferId) = t;
   }

   bool isSparse() const {return mSparseFlag;}

   unsigned int* activeIndicesBuffer(int bufferId, int level) {
      return &mActiveIndices[levelIndex(level)].at(bufferId*mNumItems);
   }

   unsigned int* activeIndicesBuffer(int bufferId) {
      return &mActiveIndices[mCurrentLevel].at(bufferId*mNumItems);
   }

   void setNumActive(int bufferId, long numActive) {
      mNumActive[mCurrentLevel].at(bufferId) = numActive;
   }

   long * numActiveBuffer(int bufferId, int level) {
      return &mNumActive[levelIndex(level)].at(bufferId);
   }

   long * numActiveBuffer(int bufferId) {
      return &mNumActive[mCurrentLevel].at(bufferId);
   }

   int getNumItems() const { return mNumItems;}

private:
   int levelIndex(int level) const { return ((level + mCurrentLevel) % mNumLevels); }

private:
   int   mNumItems;
   int   mCurrentLevel;
   int   mNumLevels;
   int   mNumBuffers;
   bool  mSparseFlag;

   std::vector<std::vector<pvdata_t> >     mBuffer;
   std::vector<std::vector<long> >         mNumActive;
   std::vector<std::vector<unsigned int> > mActiveIndices;
   std::vector<std::vector<double> >       mLastUpdateTimes;
};

} // NAMESPACE

#endif /* DATASTORE_HPP_ */
