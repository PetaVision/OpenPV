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

L1NormLCAProbe::L1NormLCAProbe(const char *name, PVParams *params, Communicator const *comm) {
   initialize(name, params, comm);
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

void L1NormLCAProbe::createEnergyProbeComponent(char const *name, PVParams *params) {
   mEnergyProbeComponent = std::make_shared<VThreshEnergyProbeComponent>(name, params);
}

void L1NormLCAProbe::createProbeLocal(char const *name, PVParams *params) {
   mProbeLocal = std::make_shared<L1NormLCAProbeLocal>(name, params);
}

void L1NormLCAProbe::initialize(const char *name, PVParams *params, Communicator const *comm) {
   L1NormProbe::initialize(name, params, comm);
}

} /* namespace PV */
