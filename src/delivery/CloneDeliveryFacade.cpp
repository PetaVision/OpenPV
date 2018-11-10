/*
 * CloneDeliveryFacade.cpp
 *
 *  Created on: Dec 18, 2017
 *      Author: Pete Schultz
 */

#include "CloneDeliveryFacade.hpp"
#include "components/CloneWeightsPair.hpp"

namespace PV {

CloneDeliveryFacade::CloneDeliveryFacade(char const *name, PVParams *params, Communicator *comm) {
   initialize(name, params, comm);
}

CloneDeliveryFacade::CloneDeliveryFacade() {}

CloneDeliveryFacade::~CloneDeliveryFacade() {}

void CloneDeliveryFacade::initialize(char const *name, PVParams *params, Communicator *comm) {
   HyPerDeliveryFacade::initialize(name, params, comm);
}

void CloneDeliveryFacade::setObjectType() { mObjectType = "CloneDeliveryFacade"; }

Response::Status CloneDeliveryFacade::communicateInitInfo(
      std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto status = HyPerDeliveryFacade::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }
   if (mUpdateGSynFromPostPerspective) {
      auto *cloneWeightsPair = message->mHierarchy->lookupByType<CloneWeightsPair>();
      pvAssert(cloneWeightsPair);
      cloneWeightsPair->synchronizeMarginsPost();
   }
   return Response::SUCCESS;
}

} // end namespace PV
