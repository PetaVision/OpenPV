/*
 * SharedWeights.cpp
 *
 *  Created on: Jan 5, 2018
 *      Author: Pete Schultz
 */

#include "SharedWeights.hpp"

namespace PV {

SharedWeights::SharedWeights(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   initialize(paramsIO, comm);
}

SharedWeights::~SharedWeights() {}

void SharedWeights::initialize(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   BaseObject::initialize(paramsIO, comm);
}

void SharedWeights::setObjectType() { mObjectType = "SharedWeights"; }

int SharedWeights::ioParamsFillGroup(ParamsIOSwitch ioSwitch) {
   ioParam_sharedWeights(ioSwitch);
   return PV_SUCCESS;
}

void SharedWeights::ioParam_sharedWeights(ParamsIOSwitch ioSwitch) {
   mParamsIO->ioParam(ioSwitch, "sharedWeights", &mSharedWeightsFlag);
}

} // namespace PV
