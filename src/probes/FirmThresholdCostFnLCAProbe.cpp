/*
 * FirmThresholdCostFnLCAProbe.cpp
 *
 *  Created on: Oct 9, 2015
 *      Author: pschultz
 */

#include "FirmThresholdCostFnLCAProbe.hpp"
#include "components/ANNActivityBuffer.hpp"
#include "components/ActivityComponent.hpp"
#include "probes/FirmThresholdCostFnLCAProbeLocal.hpp"
#include "probes/VThreshEnergyProbeComponent.hpp"

namespace PV {

FirmThresholdCostFnLCAProbe::FirmThresholdCostFnLCAProbe(
      const char *name,
      PVParams *params,
      Communicator const *comm) {
   initialize(name, params, comm);
}

Response::Status FirmThresholdCostFnLCAProbe::allocateDataStructures() {
   auto status = FirmThresholdCostFnProbe::allocateDataStructures();
   if (!Response::completed(status)) {
      return status;
   }

   auto probeLocal = std::dynamic_pointer_cast<FirmThresholdCostFnLCAProbeLocal>(mProbeLocal);
   pvAssert(probeLocal);

   auto targetLayer = mProbeTargetLayer->getTargetLayer();
   pvAssert(targetLayer);

   auto *activityComponent = targetLayer->getComponentByType<ActivityComponent>();
   FatalIf(
         activityComponent == nullptr,
         "%s: targetLayer \"%s\" does not have an activity component.\n",
         getDescription_c(),
         targetLayer->getName());
   ANNActivityBuffer *activityBuffer = activityComponent->getComponentByType<ANNActivityBuffer>();
   FatalIf(
         activityBuffer == nullptr,
         "%s: targetLayer \"%s\" does not have an ANNActivityBuffer component.\n",
         getDescription_c(),
         targetLayer->getName());
   FatalIf(
         activityBuffer->usingVerticesListInParams() == true,
         "%s: LCAProbes require target layer \"%s\" to use VThresh etc. "
         "instead of verticesV/verticesA.\n",
         getDescription_c(),
         targetLayer->getName());

   probeLocal->setFirmThresholdParams(activityBuffer->getVThresh(), activityBuffer->getVWidth());
   setCoefficient(activityBuffer->getVThresh());

   return Response::SUCCESS;
}

void FirmThresholdCostFnLCAProbe::createProbeLocal(char const *name, PVParams *params) {
   mProbeLocal = std::make_shared<FirmThresholdCostFnLCAProbeLocal>(name, params);
}

void FirmThresholdCostFnLCAProbe::createEnergyProbeComponent(char const *name, PVParams *params) {
   mEnergyProbeComponent = std::make_shared<VThreshEnergyProbeComponent>(name, params);
}

void FirmThresholdCostFnLCAProbe::initialize(
      const char *name,
      PVParams *params,
      Communicator const *comm) {
   FirmThresholdCostFnProbe::initialize(name, params, comm);
}

} /* namespace PV */
