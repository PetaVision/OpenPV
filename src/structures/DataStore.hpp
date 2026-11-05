/*
 * DataStore.hpp
 *
 *  Created on: Sep 10, 2008
 *      Author: Craig Rasmussen
 */

#ifndef DATASTORE_HPP_
#define DATASTORE_HPP_

#include <cstdlib>
#include <cstring>
#include <memory>

#include "structures/PVLayerCube.hpp"
#include "structures/RingBuffer.hpp"
#include "structures/SparseList.hpp"

namespace PV {

class DataStore {
  public:
   DataStore(int numBuffers, long numItems, int numLevels, bool sparseFlag);

   virtual ~DataStore() {}

   void newLevelIndex();

   // Level (delay) spins slower than bufferId (batch element)

   std::vector<float> &buffer(int bufferId, int level) {
      return mBuffer->getBuffer(level, bufferId);
   }

   std::vector<float> &buffer(int bufferId) { return mBuffer->getBuffer(bufferId); }

   double getLastUpdateTime(int level) const {
      return mLastUpdateTimes->getBuffer(level, 0);
   }

   double getLastUpdateTime() const { return mLastUpdateTimes->getBuffer(0); }

   void setLastUpdateTime(int level, double t) {
      mLastUpdateTimes->getBuffer(level, 0) = t;
   }

   void setLastUpdateTime(double t) { mLastUpdateTimes->getBuffer(0) = t; }

   bool isSparse() const { return mSparseFlag; }

   SparseList<float> &activeIndicesBuffer(int bufferId, int level) {
      return mActiveIndices->getBuffer(level, bufferId);
   }

   SparseList<float> &activeIndicesBuffer(int bufferId) {
      return mActiveIndices->getBuffer(bufferId);
   }

   void setNumActive(int bufferId, long numActive) { mNumActive->getBuffer(bufferId) = numActive; }

   long &numActiveBuffer(int bufferId, int level) { return mNumActive->getBuffer(level, bufferId); }

   long &numActiveBuffer(int bufferId) { return mNumActive->getBuffer(bufferId); }

   void markActiveIndicesOutOfSync(int bufferId, int level);

   void updateActiveIndices(int bufferId, int level);

   int getNumLevels() const { return mNumLevels; }
   int getNumBuffers() const { return mNumBuffers; }
   long getNumItems() const { return mNumItems; }

   /**
    * Returns a PVLayerCube pointing to the data at the given delay.
    * It does not check whether the PVLayerLoc is consistent with the
    * DataStore's numItems or numBuffers.
    */
   PVLayerCube createCube(PVLayerLoc const &loc, int delay);

  private:
   long mNumItems;
   int mNumLevels;
   int mNumBuffers;
   bool mSparseFlag;

   std::shared_ptr<RingBuffer<std::vector<float>>> mBuffer       = nullptr;
   std::shared_ptr<RingBuffer<double>> mLastUpdateTimes          = nullptr;
   std::shared_ptr<RingBuffer<long>> mNumActive                  = nullptr;
   std::shared_ptr<RingBuffer<SparseList<float>>> mActiveIndices = nullptr;
};

} // namespace PV

#endif /* DATASTORE_HPP_ */
