/*
 * SharedWeights.cpp
 *
 *  Created on: Jan 5, 2018
 *      Author: Pete Schultz
 */

#include "SharedWeights.hpp"
#include "columns/HyPerCol.hpp"

namespace PV {

SharedWeights::SharedWeights(char const *name, PVParams *params, Communicator *comm) {
   initialize(name, params, comm);
}

SharedWeights::~SharedWeights() {}

void SharedWeights::initialize(char const *name, PVParams *params, Communicator *comm) {
   BaseObject::initialize(name, params, comm);
}

void SharedWeights::setObjectType() { mObjectType = "SharedWeights"; }

int SharedWeights::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   ioParam_sharedWeights(ioFlag);
   return PV_SUCCESS;
}

void SharedWeights::ioParam_sharedWeights(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, name, "sharedWeights", &mSharedWeights, mSharedWeights);
}

} // namespace PV
