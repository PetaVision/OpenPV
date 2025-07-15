/*
 * FirmThresholdCostLayer.cpp
 *
 *  Created on: Dec 21, 2010
 *      Author: pschultz
 */

#include "FirmThresholdCostLayer.hpp"
#include "components/FirmThresholdCostActivityBuffer.hpp"
#include "components/GSynAccumulator.hpp"
#include "components/HyPerActivityComponent.hpp"
#include "components/HyPerInternalStateBuffer.hpp"

namespace PV {

FirmThresholdCostLayer::FirmThresholdCostLayer(std::shared_ptr<ParamsIO> paramsIO,
      Communicator const *comm) {
   initialize(paramsIO, comm);
}

FirmThresholdCostLayer::~FirmThresholdCostLayer() {}

void FirmThresholdCostLayer::initialize(std::shared_ptr<ParamsIO> paramsIO,
      Communicator const *comm) {
   HyPerLayer::initialize(paramsIO, comm);
}

ActivityComponent *FirmThresholdCostLayer::createActivityComponent() {
   return new HyPerActivityComponent<GSynAccumulator,
                                     HyPerInternalStateBuffer,
                                     FirmThresholdCostActivityBuffer>(
         mParamsIO, mCommunicator);
}

} // end namespace PV
