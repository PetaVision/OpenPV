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
      const char *name,
      PVParams *params,
      Communicator const *comm) {
   initialize(name, params, comm);
}

DependentFirmThresholdCostLayer::~DependentFirmThresholdCostLayer() {}

void DependentFirmThresholdCostLayer::initialize(
      const char *name,
      PVParams *params,
      Communicator const *comm) {
   FirmThresholdCostLayer::initialize(name, params, comm);
}

void DependentFirmThresholdCostLayer::fillComponentTable() {
   FirmThresholdCostLayer::fillComponentTable();
   auto *originalLayerNameParam = createOriginalLayerNameParam();
   if (originalLayerNameParam) {
      addUniqueComponent(originalLayerNameParam);
   }
}

OriginalLayerNameParam *DependentFirmThresholdCostLayer::createOriginalLayerNameParam() {
   return new OriginalLayerNameParam(name, parameters(), mCommunicator);
}

ActivityComponent *DependentFirmThresholdCostLayer::createActivityComponent() {
   return new HyPerActivityComponent<GSynAccumulator,
                                     HyPerInternalStateBuffer,
                                     DependentFirmThresholdCostActivityBuffer>(
         getName(), parameters(), mCommunicator);
}

} // end namespace PV
