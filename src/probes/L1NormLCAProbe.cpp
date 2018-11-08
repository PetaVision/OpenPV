/*
 * L1NormLCAProbe.cpp
 *
 *  Created on: Oct 9, 2015
 *      Author: pschultz
 */

#include "L1NormLCAProbe.hpp"
#include "components/ANNActivityBuffer.hpp"
#include "layers/HyPerLCALayer.hpp"

namespace PV {

L1NormLCAProbe::L1NormLCAProbe(const char *name, PVParams *params, Communicator *comm) {
   initialize_base();
   initialize(name, params, comm);
}

L1NormLCAProbe::L1NormLCAProbe() { initialize_base(); }

Response::Status
L1NormLCAProbe::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto status = L1NormProbe::communicateInitInfo(message);
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
   coefficient = activityBuffer->getVThresh();
   return Response::SUCCESS;
}

} /* namespace PV */
