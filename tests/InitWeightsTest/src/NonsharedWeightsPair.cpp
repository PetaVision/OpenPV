/*
 * NonsharedWeightsPair.cpp
 *
 *  Created on: Dec 4, 2017
 *      Author: Pete Schultz
 */

#include "NonsharedWeightsPair.hpp"
#include <columns/HyPerCol.hpp>

namespace PV {

NonsharedWeightsPair::NonsharedWeightsPair(char const *name, HyPerCol *hc) { initialize(name, hc); }

int NonsharedWeightsPair::initialize(char const *name, HyPerCol *hc) {
   return WeightsPair::initialize(name, hc);
}

int NonsharedWeightsPair::setDescription() {
   description.clear();
   description.append("NonsharedWeightsPair").append(" \"").append(getName()).append("\"");
   return PV_SUCCESS;
}

void NonsharedWeightsPair::ioParam_sharedWeights(enum ParamsIOFlag ioFlag) {
   mSharedWeights = true;
   parent->parameters()->handleUnnecessaryParameter(name, "sharedWeights", mSharedWeights);
}

} // namespace PV
