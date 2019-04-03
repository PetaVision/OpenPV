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

FirmThresholdCostLayer::FirmThresholdCostLayer(
      const char *name,
      PVParams *params,
      Communicator const *comm) {
   initialize(name, params, comm);
}

FirmThresholdCostLayer::~FirmThresholdCostLayer() {}

void FirmThresholdCostLayer::initialize(
      const char *name,
      PVParams *params,
      Communicator const *comm) {
   HyPerLayer::initialize(name, params, comm);
}

ActivityComponent *FirmThresholdCostLayer::createActivityComponent() {
   return new HyPerActivityComponent<GSynAccumulator,
                                     HyPerInternalStateBuffer,
                                     FirmThresholdCostActivityBuffer>(
         getName(), parameters(), mCommunicator);
}

} // end namespace PV
