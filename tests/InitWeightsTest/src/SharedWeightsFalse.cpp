/*
 * SharedWeightsFalse.cpp
 *
 *  Created on: Jan 8, 2018
 *      Author: Pete Schultz
 */

#include "SharedWeightsFalse.hpp"
#include "columns/HyPerCol.hpp"
#include "components/ConnectionData.hpp"
#include "utils/MapLookupByType.hpp"

namespace PV {

SharedWeightsFalse::SharedWeightsFalse(char const *name, HyPerCol *hc) { initialize(name, hc); }

SharedWeightsFalse::~SharedWeightsFalse() {}

int SharedWeightsFalse::initialize(char const *name, HyPerCol *hc) {
   return SharedWeights::initialize(name, hc);
}

void SharedWeightsFalse::setObjectType() { mObjectType = "SharedWeightsFalse"; }

int SharedWeightsFalse::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   return SharedWeights::ioParamsFillGroup(ioFlag);
}
void SharedWeightsFalse::ioParam_sharedWeights(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      mSharedWeights = false;
      parent->parameters()->handleUnnecessaryParameter(name, "sharedWeights", mSharedWeights);
   }
}

} // namespace PV
