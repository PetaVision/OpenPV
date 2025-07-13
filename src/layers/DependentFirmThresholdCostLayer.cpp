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

DependentFirmThresholdCostLayer::DependentFirmThresholdCostLayer(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   initialize(params, defaults, comm);
}

DependentFirmThresholdCostLayer::~DependentFirmThresholdCostLayer() {}

void DependentFirmThresholdCostLayer::initialize(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   FirmThresholdCostLayer::initialize(params, defaults, comm);
}

void DependentFirmThresholdCostLayer::fillComponentTable() {
   FirmThresholdCostLayer::fillComponentTable();
   auto *originalLayerNameParam = createOriginalLayerNameParam();
   if (originalLayerNameParam) {
      addUniqueComponent(originalLayerNameParam);
   }
}

OriginalLayerNameParam *DependentFirmThresholdCostLayer::createOriginalLayerNameParam() {
   return new OriginalLayerNameParam(mParamsIO->getParams(), mParamsIO->getDefaults(), mCommunicator);
}

ActivityComponent *DependentFirmThresholdCostLayer::createActivityComponent() {
   return new HyPerActivityComponent<GSynAccumulator,
                                     HyPerInternalStateBuffer,
                                     DependentFirmThresholdCostActivityBuffer>(
         mParamsIO->getParams(), mParamsIO->getDefaults(), mCommunicator);
}

} // end namespace PV
