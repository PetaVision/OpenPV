/*
 * L0NormLCAProbe.cpp
 *
 *  Created on: Oct 9, 2015
 *      Author: pschultz
 */

#include "L0NormLCAProbe.hpp"
#include "probes/ANNLayerLocator.hpp"
#include "probes/L0NormLCAEnergyProbeComponent.hpp"
#include "probes/L0NormLCAProbeLocal.hpp"

namespace PV {

L0NormLCAProbe::L0NormLCAProbe(const char *name, PVParams *params, Communicator const *comm) {
   initialize(name, params, comm);
}

Response::Status L0NormLCAProbe::allocateDataStructures() {
   auto status = L0NormProbe::allocateDataStructures();
   if (!Response::completed(status)) {
      return status;
   }

   auto probeLocal = std::dynamic_pointer_cast<L0NormLCAProbeLocal>(mProbeLocal);
   pvAssert(probeLocal);

   pvAssert(mProbeTargetLayer);
   auto const *activityBuffer = locateANNActivityBuffer(mProbeTargetLayer);
   FatalIf(
         activityBuffer == nullptr,
         "%s: TargetLayerComponent \"%s\" was unable to find the needed activity buffer.\n",
         getDescription_c(),
         mProbeTargetLayer->getName_c());

   double vThresh = activityBuffer->getVThresh();
   probeLocal->setNnzThreshold(vThresh);
   setCoefficient(0.5 * vThresh * vThresh);

   return Response::SUCCESS;
}

void L0NormLCAProbe::createEnergyProbeComponent(char const *name, PVParams *params) {
   mEnergyProbeComponent = std::make_shared<L0NormLCAEnergyProbeComponent>(name, params);
}

void L0NormLCAProbe::createProbeLocal(char const *name, PVParams *params) {
   mProbeLocal = std::make_shared<L0NormLCAProbeLocal>(name, params);
}

void L0NormLCAProbe::initialize(const char *name, PVParams *params, Communicator const *comm) {
   L0NormProbe::initialize(name, params, comm);
}

} /* namespace PV */
