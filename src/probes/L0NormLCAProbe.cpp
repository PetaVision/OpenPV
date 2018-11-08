/*
 * L0NormLCAProbe.cpp
 *
 *  Created on: Oct 9, 2015
 *      Author: pschultz
 */

#include "L0NormLCAProbe.hpp"
#include "components/ANNActivityBuffer.hpp"
#include "layers/HyPerLCALayer.hpp"

namespace PV {

L0NormLCAProbe::L0NormLCAProbe(const char *name, PVParams *params, Communicator *comm) {
   initialize_base();
   initialize(name, params, comm);
}

L0NormLCAProbe::L0NormLCAProbe() { initialize_base(); }

Response::Status
L0NormLCAProbe::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto status = L0NormProbe::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }
   assert(targetLayer);
   auto *activityComponent = targetLayer->getComponentByType<ActivityComponent>();
   FatalIf(
         activityComponent == nullptr,
         "%s: targetLayer \"%s\" does not have an activity component.\n",
         getDescription_c(),
         getTargetName());
   ANNActivityBuffer *activityBuffer = activityComponent->getComponentByType<ANNActivityBuffer>();
   FatalIf(
         activityBuffer == nullptr,
         "%s: targetLayer \"%s\" does not have an ANNActivityBuffer component.\n",
         getDescription_c(),
         getTargetName());

   FatalIf(
         activityBuffer->usingVerticesListInParams() == true,
         "%s: LCAProbes require targetLayer \"%s\" to use VThresh etc. "
         "instead of verticesV/verticesV.\n",
         getDescription_c(),
         getTargetName());
   float vThresh = activityBuffer->getVThresh();
   coefficient   = vThresh * vThresh / 2.0f;
   return Response::SUCCESS;
}

} /* namespace PV */
