#include "TauswortheURBG.hpp"

namespace PV {

TauswortheURBG::TauswortheURBG(unsigned int seed) {
   cl_random_init(&mTausUint4, 1UL, seed);
}

TauswortheURBG::TauswortheURBG(taus_uint4 const &state) {
   mTausUint4 = state;
}

TauswortheURBG::result_type TauswortheURBG::operator()() {
   mTausUint4 = cl_random_get(mTausUint4);
   return mTausUint4.s0;
}

} // namespace PV
