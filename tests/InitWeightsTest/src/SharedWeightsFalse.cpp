/*
 * SharedWeightsFalse.cpp
 *
 *  Created on: Jan 8, 2018
 *      Author: Pete Schultz
 */

#include "SharedWeightsFalse.hpp"
#include "components/ConnectionData.hpp"

namespace PV {

SharedWeightsFalse::SharedWeightsFalse(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   initialize(paramsIO, comm);
}

SharedWeightsFalse::~SharedWeightsFalse() {}

void SharedWeightsFalse::initialize(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   SharedWeights::initialize(paramsIO, comm);
}

void SharedWeightsFalse::setObjectType() { mObjectType = "SharedWeightsFalse"; }

int SharedWeightsFalse::ioParamsFillGroup(ParamsIOSwitch ioSwitch) {
   return SharedWeights::ioParamsFillGroup(ioSwitch);
}
void SharedWeightsFalse::ioParam_sharedWeights(ParamsIOSwitch ioSwitch) {
   if (ioSwitch == ParamsIOSwitch::Read) {
      mSharedWeightsFlag = false;
      mParamsIO->handleUnnecessaryParameter("sharedWeights", mSharedWeightsFlag);
   }
}

} // namespace PV
