/*
 * SharedWeightsParam.cpp
 *
 *  Created on: Jan 5, 2018
 *      Author: Pete Schultz
 */

#include "SharedWeightsParam.hpp"
#include "columns/HyPerCol.hpp"
#include "components/ConnectionData.hpp"
#include "utils/MapLookupByType.hpp"

namespace PV {

SharedWeightsParam::SharedWeightsParam(char const *name, HyPerCol *hc) { initialize(name, hc); }

SharedWeightsParam::~SharedWeightsParam() {}

int SharedWeightsParam::initialize(char const *name, HyPerCol *hc) {
   return BaseObject::initialize(name, hc);
}

int SharedWeightsParam::setDescription() {
   description.clear();
   description.append("SharedWeightsParam").append(" \"").append(getName()).append("\"");
   return PV_SUCCESS;
}

int SharedWeightsParam::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   ioParam_sharedWeights(ioFlag);
   return PV_SUCCESS;
}

void SharedWeightsParam::ioParam_sharedWeights(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(
         ioFlag, name, "sharedWeights", &mSharedWeights, mSharedWeights);
}

} // namespace PV
