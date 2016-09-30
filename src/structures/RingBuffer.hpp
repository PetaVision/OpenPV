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

template <typename T>
class RingBuffer {
public:
   RingBuffer(int numLevels, int numItems, T initialValue=(T) 0) {
      mCurrentLevel = 0;
      mNumLevels = numLevels;
      mNumItems = numItems;
      mBuffer.resize(numLevels);
      for (auto& b : mBuffer) {
         b.resize(numItems, initialValue);
      }
   }
   virtual ~RingBuffer() {}

   int getNumLevels() { return mNumLevels; }

   int getNumItems() { return mNumItems; }

   void newLevel() {
      mCurrentLevel = (mNumLevels + mCurrentLevel - 1) % mNumLevels;
   }

   T * getBuffer(int level, int offset) {
      return &mBuffer[levelIndex(level)].at(offset);
   }

   T * getBuffer(int offset) {
      return &mBuffer[mCurrentLevel].at(offset);
   }

   T * getBuffer() {
      return mBuffer[mCurrentLevel].data();
   }

private:
   int levelIndex(int level) const { return ((level + mCurrentLevel) % mNumLevels); }

private:
   int mCurrentLevel;
   int mNumLevels;
   int mNumItems;
   std::vector<std::vector<T> > mBuffer;
};

} /* namespace PV */

#endif /* RINGBUFFER_HPP_ */
