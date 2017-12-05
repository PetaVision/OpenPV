/*
 * SharedWeightsPair.cpp
 *
 *  Created on: Dec 4, 2017
 *      Author: Pete Schultz
 */

#include "SharedWeightsPair.hpp"
#include <columns/HyPerCol.hpp>

namespace PV {

SharedWeightsPair::SharedWeightsPair(char const *name, HyPerCol *hc) { initialize(name, hc); }

int SharedWeightsPair::initialize(char const *name, HyPerCol *hc) {
   return WeightsPair::initialize(name, hc);
}

int SharedWeightsPair::setDescription() {
   description.clear();
   description.append("SharedWeightsPair").append(" \"").append(getName()).append("\"");
   return PV_SUCCESS;
}

void SharedWeightsPair::ioParam_sharedWeights(enum ParamsIOFlag ioFlag) {
   mSharedWeights = true;
   parent->parameters()->handleUnnecessaryParameter(name, "sharedWeights", mSharedWeights);
}

} // namespace PV
