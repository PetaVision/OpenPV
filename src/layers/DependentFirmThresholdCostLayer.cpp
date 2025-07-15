/*
 * DependentFirmThresholdCostLayer.cpp
 *
 *  Created on: Dec 21, 2010
 *      Author: pschultz
 */

#include "DependentFirmThresholdCostLayer.hpp"
#include "components/DependentFirmThresholdCostActivityBuffer.hpp"
#include "components/GSynAccumulator.hpp"
#include "components/HyPerActivityComponent.hpp"
#include "components/HyPerInternalStateBuffer.hpp"

namespace PV {

DependentFirmThresholdCostLayer::DependentFirmThresholdCostLayer(std::shared_ptr<ParamsIO> paramsIO,
      Communicator const *comm) {
   initialize(paramsIO, comm);
}

DependentFirmThresholdCostLayer::~DependentFirmThresholdCostLayer() {}

void DependentFirmThresholdCostLayer::initialize(std::shared_ptr<ParamsIO> paramsIO,
      Communicator const *comm) {
   FirmThresholdCostLayer::initialize(paramsIO, comm);
}

void DependentFirmThresholdCostLayer::fillComponentTable() {
   FirmThresholdCostLayer::fillComponentTable();
   auto *originalLayerNameParam = createOriginalLayerNameParam();
   if (originalLayerNameParam) {
      addUniqueComponent(originalLayerNameParam);
   }
}

OriginalLayerNameParam *DependentFirmThresholdCostLayer::createOriginalLayerNameParam() {
   return new OriginalLayerNameParam(mParamsIO, mCommunicator);
}

ActivityComponent *DependentFirmThresholdCostLayer::createActivityComponent() {
   return new HyPerActivityComponent<GSynAccumulator,
                                     HyPerInternalStateBuffer,
                                     DependentFirmThresholdCostActivityBuffer>(
         mParamsIO, mCommunicator);
}

} // end namespace PV
