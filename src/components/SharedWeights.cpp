/*
 * SharedWeights.cpp
 *
 *  Created on: Jan 5, 2018
 *      Author: Pete Schultz
 */

#include "SharedWeights.hpp"
#include "columns/HyPerCol.hpp"

namespace PV {

SharedWeights::SharedWeights(char const *name, HyPerCol *hc) { initialize(name, hc); }

SharedWeights::~SharedWeights() {}

int SharedWeights::initialize(char const *name, HyPerCol *hc) {
   return BaseObject::initialize(name, hc);
}

int SharedWeights::setDescription() {
   description.clear();
   description.append("SharedWeights").append(" \"").append(getName()).append("\"");
   return PV_SUCCESS;
}

int SharedWeights::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   ioParam_sharedWeights(ioFlag);
   return PV_SUCCESS;
}

void SharedWeights::ioParam_sharedWeights(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(
         ioFlag, name, "sharedWeights", &mSharedWeights, mSharedWeights);
}

} // namespace PV
