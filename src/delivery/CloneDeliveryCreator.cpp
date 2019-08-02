/*
 * CloneDeliveryCreator.cpp
 *
 *  Created on: Dec 18, 2017
 *      Author: Pete Schultz
 */

#include "CloneDeliveryCreator.hpp"
#include "components/CloneWeightsPair.hpp"

namespace PV {

CloneDeliveryCreator::CloneDeliveryCreator(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   initialize(name, params, comm);
}

CloneDeliveryCreator::CloneDeliveryCreator() {}

CloneDeliveryCreator::~CloneDeliveryCreator() {}

void CloneDeliveryCreator::initialize(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   HyPerDeliveryCreator::initialize(name, params, comm);
}

void CloneDeliveryCreator::setObjectType() { mObjectType = "CloneDeliveryCreator"; }

Response::Status CloneDeliveryCreator::communicateInitInfo(
      std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto status = HyPerDeliveryCreator::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }
   if (mUpdateGSynFromPostPerspective) {
      auto *cloneWeightsPair = message->mObjectTable->findObject<CloneWeightsPair>(getName());
      if (!cloneWeightsPair->getInitInfoCommunicatedFlag()) {
         return Response::POSTPONE;
      }
      pvAssert(cloneWeightsPair);
      cloneWeightsPair->synchronizeMarginsPost();
   }
   return Response::SUCCESS;
}

} // end namespace PV
