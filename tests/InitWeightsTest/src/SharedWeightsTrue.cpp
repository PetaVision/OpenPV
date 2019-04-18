/*
 * SharedWeightsTrue.cpp
 *
 *  Created on: Jan 8, 2018
 *      Author: Pete Schultz
 */

#include "SharedWeightsTrue.hpp"
#include "columns/HyPerCol.hpp"
#include "components/ConnectionData.hpp"
#include "utils/MapLookupByType.hpp"

namespace PV {

SharedWeightsTrue::SharedWeightsTrue(char const *name, HyPerCol *hc) { initialize(name, hc); }

SharedWeightsTrue::~SharedWeightsTrue() {}

int SharedWeightsTrue::initialize(char const *name, HyPerCol *hc) {
   return SharedWeights::initialize(name, hc);
}

void SharedWeightsTrue::setObjectType() { mObjectType = "SharedWeightsTrue"; }

int SharedWeightsTrue::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   return SharedWeights::ioParamsFillGroup(ioFlag);
}
void SharedWeightsTrue::ioParam_sharedWeights(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      mSharedWeights = true;
      parent->parameters()->handleUnnecessaryParameter(name, "sharedWeights", mSharedWeights);
   }
}

} // namespace PV
