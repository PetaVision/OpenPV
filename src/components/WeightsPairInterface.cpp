/*
 * WeightsPairInterface.cpp
 *
 *  Created on: Jan 8, 2018
 *      Author: Pete Schultz
 */

#include "WeightsPairInterface.hpp"
#include "columns/HyPerCol.hpp"
#include "utils/MapLookupByType.hpp"

namespace PV {

WeightsPairInterface::WeightsPairInterface(char const *name, HyPerCol *hc) { initialize(name, hc); }

WeightsPairInterface::~WeightsPairInterface() {
   delete mPreWeights;
   delete mPostWeights;
}

int WeightsPairInterface::initialize(char const *name, HyPerCol *hc) {
   return BaseObject::initialize(name, hc);
}

void WeightsPairInterface::setObjectType() { mObjectType = "WeightsPairInterface"; }

Response::Status WeightsPairInterface::communicateInitInfo(
      std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto status = BaseObject::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }
   if (mConnectionData == nullptr) {
      mConnectionData = mapLookupByType<ConnectionData>(message->mHierarchy);
      pvAssert(mConnectionData);
   }

   if (!mConnectionData->getInitInfoCommunicatedFlag()) {
      if (parent->getCommunicator()->globalCommRank() == 0) {
         InfoLog().printf(
               "%s must wait until the ConnectionData component has finished its "
               "communicateInitInfo stage.\n",
               getDescription_c());
      }
      return Response::POSTPONE;
   }

   if (mPatchSize == nullptr) {
      mPatchSize = nullptr;
      try {
         mPatchSize = mapLookupByType<PatchSize>(message->mHierarchy);
      } catch (const std::invalid_argument &e) {
         pvAssertMessage(
               0,
               "Communicate message to %s has %s PatchSize component.\n",
               getDescription_c(),
               e.what() /*either "more than one" or "no"*/);
      }
      pvAssert(mPatchSize);
   }

   if (!mPatchSize->getInitInfoCommunicatedFlag()) {
      if (parent->getCommunicator()->globalCommRank() == 0) {
         InfoLog().printf(
               "%s must wait until the PatchSize component has finished its "
               "communicateInitInfo stage.\n",
               getDescription_c());
      }
      return Response::POSTPONE;
   }

   HyPerLayer *pre           = mConnectionData->getPre();
   HyPerLayer *post          = mConnectionData->getPost();
   PVLayerLoc const *preLoc  = pre->getLayerLoc();
   PVLayerLoc const *postLoc = post->getLayerLoc();

   // Margins
   int xmargin = requiredConvolveMargin(preLoc->nx, postLoc->nx, mPatchSize->getPatchSizeX());
   int receivedxmargin = 0;
   pre->requireMarginWidth(xmargin, &receivedxmargin, 'x');
   int ymargin = requiredConvolveMargin(preLoc->ny, postLoc->ny, mPatchSize->getPatchSizeY());
   int receivedymargin = 0;
   pre->requireMarginWidth(ymargin, &receivedymargin, 'y');

   return Response::SUCCESS;
}

void WeightsPairInterface::needPre() {
   FatalIf(
         !mInitInfoCommunicatedFlag,
         "%s must finish CommunicateInitInfo before needPre can be called.\n",
         getDescription_c());
   if (mPreWeights == nullptr) {
      createPreWeights(std::string(name));
   }
}

void WeightsPairInterface::needPost() {
   FatalIf(
         !mInitInfoCommunicatedFlag,
         "%s must finish CommunicateInitInfo before needPost can be called.\n",
         getDescription_c());
   if (mPostWeights == nullptr) {
      std::string weightsName(std::string(name) + " post-perspective");
      createPostWeights(weightsName);
   }
}

Response::Status WeightsPairInterface::allocateDataStructures() {
   if (mPreWeights) {
      allocatePreWeights();
   }
   if (mPostWeights) {
      allocatePostWeights();
   }
   return Response::SUCCESS;
}

void WeightsPairInterface::allocatePreWeights() {
   mPreWeights->setMargins(
         mConnectionData->getPre()->getLayerLoc()->halo,
         mConnectionData->getPost()->getLayerLoc()->halo);
   mPreWeights->allocateDataStructures();
}

void WeightsPairInterface::allocatePostWeights() {
   mPostWeights->setMargins(
         mConnectionData->getPost()->getLayerLoc()->halo,
         mConnectionData->getPre()->getLayerLoc()->halo);
   mPostWeights->allocateDataStructures();
}

} // namespace PV
