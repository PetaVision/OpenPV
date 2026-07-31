/*
 * RandomSeed.cpp
 *
 *  Created on: Jul 26, 2016
 *      Author: pschultz
 */

#include "columns/RandomSeed.hpp"
#include "utils/PVLog.hpp"

namespace PV {

RandomSeed *RandomSeed::instance() {
   static RandomSeed *singleton = new RandomSeed();
   return singleton;
}

RandomSeed::RandomSeed() {
   if (sizeof(unsigned int) < (size_t)4) {
      Fatal() << "Unsigned int must have a size of at least 4 bytes.\n";
   }
}

void RandomSeed::initialize(unsigned long initialSeed) {
   if (initialSeed < RandomSeed::mMinSeed) {
      Fatal() << "random seed " << initialSeed << " is too small. Use a seed of at least "
              << mMinSeed << ".\n";
   }
   mInitialized = true;
   mInitialSeed = initialSeed;
   mNextSeed    = initialSeed;
}

unsigned long RandomSeed::allocate(unsigned long numRequested) {
   if (!mInitialized) {
      Fatal() << "RandomSeed has not been initialized.\n";
   }
   unsigned long allocation = mNextSeed;
   mNextSeed += numRequested;
   if (mNextSeed < mMinSeed) {
      mNextSeed += mMinSeed;
   }
   return allocation;
}

unsigned long constexpr RandomSeed::mMinSeed;
} /* namespace PV */
