/*
 * RandomSeed.cpp
 *
 *  Created on: Jul 26, 2016
 *      Author: pschultz
 */

#include "columns/RandomSeed.hpp"
#include "utils/PVLog.hpp"

namespace PV {

RandomSeed * RandomSeed::instance() {
   static RandomSeed * singleton = new RandomSeed();
   return singleton;
}

RandomSeed::RandomSeed() {
   if (sizeof(unsigned int) < (size_t) 4) {
      pvError() << "Unsigned int must have a size of at least 4 bytes.\n";
   }
}

void RandomSeed::initialize(unsigned int initialSeed) {
   if (initialSeed < RandomSeed::minSeed) {
      pvError() << "random seed " << initialSeed << ""
            " is too small. Use a seed of at least " <<  minSeed << ".\n";
   }

   mInitialized = true;
   mNextSeed = initialSeed;
   if (mInitialized) {
      pvInfo() << "RandomSeed initialized to " << mNextSeed << ".\n";
   }
}

unsigned int RandomSeed::allocate(unsigned int numRequested) {
   if (!mInitialized) {
      pvError() << "RandomSeed has not been initialized.\n";
   }
   unsigned int allocation = mNextSeed;
   mNextSeed += numRequested;
   if (mNextSeed < minSeed) { mNextSeed += minSeed; }
   return allocation;
}

unsigned int constexpr RandomSeed::minSeed;
} /* namespace PV */
