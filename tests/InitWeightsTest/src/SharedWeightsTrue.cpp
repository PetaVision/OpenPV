/*
 * SharedWeightsTrue.cpp
 *
 *  Created on: Jan 8, 2018
 *      Author: Pete Schultz
 */

#include "SharedWeightsTrue.hpp"
#include "components/ConnectionData.hpp"

namespace PV {

SharedWeightsTrue::SharedWeightsTrue(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   initialize(paramsIO, comm);
}

SharedWeightsTrue::~SharedWeightsTrue() {}

void SharedWeightsTrue::initialize(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   SharedWeights::initialize(paramsIO, comm);
}

void SharedWeightsTrue::setObjectType() { mObjectType = "SharedWeightsTrue"; }

int SharedWeightsTrue::ioParamsFillGroup(ParamsIOSwitch ioSwitch) {
   return SharedWeights::ioParamsFillGroup(ioSwitch);
}
void SharedWeightsTrue::ioParam_sharedWeights(ParamsIOSwitch ioSwitch) {
   if (ioSwitch == ParamsIOSwitch::Read) {
      mSharedWeightsFlag = true;
      mParamsIO->handleUnnecessaryParameter("sharedWeights", mSharedWeightsFlag);
   }
}

} // namespace PV
