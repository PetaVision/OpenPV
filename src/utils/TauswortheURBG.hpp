#include "utils/cl_random.h"

#include <limits>
#include <cstdlib>

namespace PV {

/**
 * A class that uses the Tausworthe RNG in cl_random.c for random number generation and also
 * serves as a C++ Uniform Random Bit Generator
 */
class TauswortheURBG {
  public:
   typedef unsigned int result_type;
   TauswortheURBG(unsigned int seed = 123456789U);
   TauswortheURBG(taus_uint4 const &state);
   result_type operator()();
   taus_uint4 const &getState() const { return mTausUint4; }
   taus_uint4 &getState() { return mTausUint4; }
   void setState(taus_uint4 const &state) { mTausUint4 = state; }

   static constexpr result_type min() { return 0U; }
   static constexpr result_type max() { return std::numeric_limits<unsigned int>::max(); }

  private:
   taus_uint4 mTausUint4;
   
}; // TauswortheURGB

} // namespace PV
