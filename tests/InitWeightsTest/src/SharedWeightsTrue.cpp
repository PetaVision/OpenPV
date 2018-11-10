/*
 * SharedWeightsTrue.cpp
 *
 *  Created on: Jan 8, 2018
 *      Author: Pete Schultz
 */

#include "SharedWeightsTrue.hpp"
#include "components/ConnectionData.hpp"

namespace PV {

SharedWeightsTrue::SharedWeightsTrue(char const *name, PVParams *params, Communicator *comm) {
   initialize(name, params, comm);
}

SharedWeightsTrue::~SharedWeightsTrue() {}

void SharedWeightsTrue::initialize(char const *name, PVParams *params, Communicator *comm) {
   SharedWeights::initialize(name, params, comm);
}

void SharedWeightsTrue::setObjectType() { mObjectType = "SharedWeightsTrue"; }

int SharedWeightsTrue::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   return SharedWeights::ioParamsFillGroup(ioFlag);
}
void SharedWeightsTrue::ioParam_sharedWeights(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      mSharedWeights = true;
      parameters()->handleUnnecessaryParameter(name, "sharedWeights", mSharedWeights);
   }
}

} // namespace PV
