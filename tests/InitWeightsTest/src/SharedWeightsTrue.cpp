/*
 * SharedWeightsTrue.cpp
 *
 *  Created on: Jan 8, 2018
 *      Author: Pete Schultz
 */

#include "SharedWeightsTrue.hpp"
#include "components/ConnectionData.hpp"

namespace PV {

SharedWeightsTrue::SharedWeightsTrue(char const *name, PVParams *params, Communicator const *comm) {
   initialize(name, params, comm);
}

SharedWeightsTrue::~SharedWeightsTrue() {}

void SharedWeightsTrue::initialize(char const *name, PVParams *params, Communicator const *comm) {
   SharedWeights::initialize(name, params, comm);
}

void SharedWeightsTrue::setObjectType() { mObjectType = "SharedWeightsTrue"; }

int SharedWeightsTrue::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   return SharedWeights::ioParamsFillGroup(ioFlag);
}
void SharedWeightsTrue::ioParam_sharedWeights(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      mSharedWeightsFlag = true;
      parameters()->handleUnnecessaryParameter(getName(), "sharedWeights", mSharedWeightsFlag);
   }
}

} // namespace PV
