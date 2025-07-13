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
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   initialize(params, defaults, comm);
}

CloneDeliveryCreator::CloneDeliveryCreator() {}

CloneDeliveryCreator::~CloneDeliveryCreator() {}

void CloneDeliveryCreator::initialize(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   HyPerDeliveryCreator::initialize(params, defaults, comm);
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
