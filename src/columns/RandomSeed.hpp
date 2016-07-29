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
   static RandomSeed * instance();
   void initialize(unsigned int initialSeed);
   unsigned int allocate(unsigned int numRequested);

private:
   RandomSeed();
   virtual ~RandomSeed() {}

public:
   static unsigned int constexpr minSeed = 10000000U;
private:
   unsigned int mNextSeed = 0U;
   bool mInitialized = false;
   // minSeed needs to be high enough that for the pseudorandom sequence to be good,
   // but must be less than (and should be much less than) ULONG_MAX/2
};

} /* namespace PV */

#endif /* RANDOMSEED_HPP_ */
