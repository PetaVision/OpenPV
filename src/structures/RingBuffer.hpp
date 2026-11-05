/*
 * RingBuffer.hpp
 *
 *  Created on: Aug 23, 2016
 *      Author: pschultz
 */

#ifndef RINGBUFFER_HPP_
#define RINGBUFFER_HPP_

#include <vector>

namespace PV {

/**
 * A class template for a ring buffer, each of whose levels is a vector of some number of items.
 * Each level's vector has the same length and the same datatype.
 */
template <typename T>
class RingBuffer {
  public:
   /**
    * RingBuffer constructor:
    *   numLevels is the number of levels in the ring buffer; numItems is how many items are
    *   in each level, and initialValue is the value of each item of each level initially.
    *   The various overloads of getBuffer() return references to one level; values can be
    *   retrieved and/or set using getBuffer().
    */
   RingBuffer(int numLevels, long numItems, T initialValue = (T)0) {
      mCurrentLevel = 0; // first level written to is numLevels - 1;
      mNumLevels    = numLevels;
      mNumItems     = numItems;
      mBuffer.resize(numLevels);
      for (auto &b : mBuffer) {
         b.resize(numItems, initialValue);
      }
   }
   virtual ~RingBuffer() {}

   int getCurrentLevel() const { return mCurrentLevel; }

   int getNumLevels() const { return mNumLevels; }

   long getNumItems() const { return mNumItems; }

   void newLevel() { mCurrentLevel = (mNumLevels + mCurrentLevel - 1) % mNumLevels; }

   T &getBuffer(int level, long offset) { return mBuffer[levelIndex(level)].at(offset); }
   T const &getBuffer(int level, long offset) const {
      return mBuffer[levelIndex(level)].at(offset); }

   T &getBuffer(long offset) { return mBuffer[mCurrentLevel].at(offset); }
   T const &getBuffer(long offset) const { return mBuffer[mCurrentLevel].at(offset); }

   T &getBuffer() { return mBuffer[mCurrentLevel].at(0); }
   T const &getBuffer() const { return mBuffer[mCurrentLevel].at(0); }

  private:
   int levelIndex(int level) const { return ((level + mCurrentLevel) % mNumLevels); }

  private:
   int mCurrentLevel;
   int mNumLevels;
   long mNumItems;
   std::vector<std::vector<T>> mBuffer;
};

} /* namespace PV */

#endif /* RINGBUFFER_HPP_ */
