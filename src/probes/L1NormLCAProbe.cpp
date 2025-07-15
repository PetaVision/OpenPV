/*
 * L1NormLCAProbe.cpp
 *
 *  Created on: Oct 9, 2015
 *      Author: pschultz
 */

#include "L1NormLCAProbe.hpp"
#include "probes/ANNLayerLocator.hpp"
#include "probes/L1NormLCAProbeLocal.hpp"
#include "probes/VThreshEnergyProbeComponent.hpp"

namespace PV {

L1NormLCAProbe::L1NormLCAProbe(std::shared_ptr<ParamsIO> paramsIO,
      Communicator const *comm) {
   initialize(paramsIO, comm);
}

Response::Status L1NormLCAProbe::allocateDataStructures() {
   auto status = L1NormProbe::allocateDataStructures();
   if (!Response::completed(status)) {
      return status;
   }

   auto probeLocal = std::dynamic_pointer_cast<L1NormLCAProbeLocal>(mProbeLocal);
   pvAssert(probeLocal);

   pvAssert(mProbeTargetLayer);
   auto const *activityBuffer = locateANNActivityBuffer(mProbeTargetLayer);
   FatalIf(
         activityBuffer == nullptr,
         "%s: TargetLayerComponent \"%s\" was unable to find the needed activity buffer.\n",
         getDescription_c(),
         mProbeTargetLayer->getName_c());

   setCoefficient(activityBuffer->getVThresh());

   return Response::SUCCESS;
}

void L1NormLCAProbe::createEnergyProbeComponent(std::shared_ptr<ParamsIO> paramsIO) {
   mEnergyProbeComponent = std::make_shared<VThreshEnergyProbeComponent>(paramsIO);
}

void L1NormLCAProbe::createProbeLocal(std::shared_ptr<ParamsIO> paramsIO) {
   mProbeLocal = std::make_shared<L1NormLCAProbeLocal>(paramsIO);
}

void L1NormLCAProbe::initialize(std::shared_ptr<ParamsIO> paramsIO,
      Communicator const *comm) {
   L1NormProbe::initialize(paramsIO, comm);
}

} /* namespace PV */
