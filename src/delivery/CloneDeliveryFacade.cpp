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

int CloneDeliveryFacade::communicateInitInfo(
      std::shared_ptr<CommunicateInitInfoMessage const> message) {
   int status = HyPerDeliveryFacade::communicateInitInfo(message);
   if (status != PV_SUCCESS) {
      return status;
   }
   if (mUpdateGSynFromPostPerspective) {
      auto *cloneWeightsPair =
            mapLookupByType<CloneWeightsPair>(message->mHierarchy, getDescription());
      pvAssert(cloneWeightsPair);
      cloneWeightsPair->synchronizeMarginsPost();
   }
   return PV_SUCCESS;
}

} // end namespace PV
