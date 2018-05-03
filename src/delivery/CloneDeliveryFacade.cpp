/*
 * CloneDeliveryFacade.cpp
 *
 *  Created on: Dec 18, 2017
 *      Author: Pete Schultz
 */

#include "CloneDeliveryFacade.hpp"
#include "columns/HyPerCol.hpp"
#include "components/CloneWeightsPair.hpp"
#include "utils/MapLookupByType.hpp"

namespace PV {

CloneDeliveryFacade::CloneDeliveryFacade(char const *name, HyPerCol *hc) { initialize(name, hc); }

CloneDeliveryFacade::CloneDeliveryFacade() {}

CloneDeliveryFacade::~CloneDeliveryFacade() {}

int CloneDeliveryFacade::initialize(char const *name, HyPerCol *hc) {
   return HyPerDeliveryFacade::initialize(name, hc);
}

void CloneDeliveryFacade::setObjectType() { mObjectType = "CloneDeliveryFacade"; }

Response::Status CloneDeliveryFacade::communicateInitInfo(
      std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto status = HyPerDeliveryFacade::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }
   if (mUpdateGSynFromPostPerspective) {
      auto *cloneWeightsPair =
            mapLookupByType<CloneWeightsPair>(message->mHierarchy, getDescription());
      pvAssert(cloneWeightsPair);
      cloneWeightsPair->synchronizeMarginsPost();
   }
   return Response::SUCCESS;
}

} // end namespace PV
