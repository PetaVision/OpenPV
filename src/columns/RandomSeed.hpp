/*
 * RandomSeed.hpp
 *
 *  Created on: Jul 26, 2016
 *      Author: pschultz
 */

#ifndef RANDOMSEED_HPP_
#define RANDOMSEED_HPP_

namespace PV {

class RandomSeed {
  public:
   static RandomSeed *instance();
   void initialize(unsigned long initialSeed);
   unsigned long allocate(unsigned long numRequested);
   unsigned long getInitialSeed() { return mInitialSeed; }

  private:
   RandomSeed();
   virtual ~RandomSeed() {}

  public:
   static unsigned long constexpr mMinSeed = 10000000UL;
   // mMinSeed needs to be high enough that for the pseudorandom sequence to be
   // good, but must be less than (and should be much less than) ULONG_MAX/2

  private:
   unsigned long mNextSeed    = 0UL;
   unsigned long mInitialSeed = 0UL;
   bool mInitialized          = false;
};

} /* namespace PV */

#endif /* RANDOMSEED_HPP_ */
