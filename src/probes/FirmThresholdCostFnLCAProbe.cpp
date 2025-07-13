/*
 * FirmThresholdCostFnLCAProbe.cpp
 *
 *  Created on: Oct 9, 2015
 *      Author: pschultz
 */

#include "FirmThresholdCostFnLCAProbe.hpp"
#include "components/ANNActivityBuffer.hpp"
#include "components/ActivityComponent.hpp"
#include "probes/ANNLayerLocator.hpp"
#include "probes/FirmThresholdCostFnLCAProbeLocal.hpp"
#include "probes/VThreshEnergyProbeComponent.hpp"

namespace PV {

FirmThresholdCostFnLCAProbe::FirmThresholdCostFnLCAProbe(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   initialize(params, defaults, comm);
}

Response::Status FirmThresholdCostFnLCAProbe::allocateDataStructures() {
   auto status = FirmThresholdCostFnProbe::allocateDataStructures();
   if (!Response::completed(status)) {
      return status;
   }

   auto probeLocal = std::dynamic_pointer_cast<FirmThresholdCostFnLCAProbeLocal>(mProbeLocal);
   pvAssert(probeLocal);

   pvAssert(mProbeTargetLayer);
   auto const *activityBuffer = locateANNActivityBuffer(mProbeTargetLayer);
   FatalIf(
         activityBuffer == nullptr,
         "%s: TargetLayerComponent \"%s\" was unable to find the needed activity buffer.\n",
         getDescription_c(),
         mProbeTargetLayer->getName_c());

   probeLocal->setFirmThresholdParams(activityBuffer->getVThresh(), activityBuffer->getVWidth());
   setCoefficient(activityBuffer->getVThresh());

   return Response::SUCCESS;
}

void FirmThresholdCostFnLCAProbe::createProbeLocal(
      std::shared_ptr<ParamGroup> params, std::shared_ptr<ParamGroup> defaults) {
   mProbeLocal = std::make_shared<FirmThresholdCostFnLCAProbeLocal>(params, defaults);
}

void FirmThresholdCostFnLCAProbe::createEnergyProbeComponent(
      std::shared_ptr<ParamGroup> params, std::shared_ptr<ParamGroup> defaults) {
   mEnergyProbeComponent = std::make_shared<VThreshEnergyProbeComponent>(params, defaults);
}

void FirmThresholdCostFnLCAProbe::initialize(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   FirmThresholdCostFnProbe::initialize(params, defaults, comm);
}

} /* namespace PV */
