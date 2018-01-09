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

int WeightsPairInterface::setDescription() {
   description.clear();
   description.append("WeightsPairInterface").append(" \"").append(getName()).append("\"");
   return PV_SUCCESS;
}

int WeightsPairInterface::communicateInitInfo(
      std::shared_ptr<CommunicateInitInfoMessage const> message) {
   int status = BaseObject::communicateInitInfo(message);
   if (status != PV_SUCCESS) {
      return status;
   }
   if (mConnectionData == nullptr) {
      mConnectionData = mapLookupByType<ConnectionData>(message->mHierarchy, getDescription());
   }
   FatalIf(
         mConnectionData == nullptr,
         "%s requires a ConnectionData component.\n",
         getDescription_c());

   if (!mConnectionData->getInitInfoCommunicatedFlag()) {
      if (parent->getCommunicator()->globalCommRank() == 0) {
         InfoLog().printf(
               "%s must wait until the ConnectionData component has finished its "
               "communicateInitInfo stage.\n",
               getDescription_c());
      }
      return PV_POSTPONE;
   }

   if (mArborList == nullptr) {
      mArborList = mapLookupByType<ArborList>(message->mHierarchy, getDescription());
   }
   FatalIf(mArborList == nullptr, "%s requires an ArborList component.\n", getDescription_c());

   if (!mArborList->getInitInfoCommunicatedFlag()) {
      if (parent->getCommunicator()->globalCommRank() == 0) {
         InfoLog().printf(
               "%s must wait until the ArborList component has finished its "
               "communicateInitInfo stage.\n",
               getDescription_c());
      }
      return PV_POSTPONE;
   }

   if (mPatchSize == nullptr) {
      mPatchSize = mapLookupByType<PatchSize>(message->mHierarchy, getDescription());
   }
   FatalIf(mPatchSize == nullptr, "%s requires a PatchSize component.\n", getDescription_c());

   if (!mPatchSize->getInitInfoCommunicatedFlag()) {
      if (parent->getCommunicator()->globalCommRank() == 0) {
         InfoLog().printf(
               "%s must wait until the PatchSize component has finished its "
               "communicateInitInfo stage.\n",
               getDescription_c());
      }
      return PV_POSTPONE;
   }

   HyPerLayer *pre           = mConnectionData->getPre();
   HyPerLayer *post          = mConnectionData->getPost();
   PVLayerLoc const *preLoc  = pre->getLayerLoc();
   PVLayerLoc const *postLoc = post->getLayerLoc();

   // Margins
   int xmargin = requiredConvolveMargin(preLoc->nx, postLoc->nx, mPatchSize->getPatchSizeX());
   int receivedxmargin = 0;
   int statusx         = pre->requireMarginWidth(xmargin, &receivedxmargin, 'x');
   if (statusx != PV_SUCCESS) {
      ErrorLog().printf(
            "Margin Failure for layer %s.  Received x-margin is %d, but %s requires margin of at "
            "least %d\n",
            pre->getDescription_c(),
            receivedxmargin,
            name,
            xmargin);
      status = PV_MARGINWIDTH_FAILURE;
   }
   int ymargin = requiredConvolveMargin(preLoc->ny, postLoc->ny, mPatchSize->getPatchSizeY());
   int receivedymargin = 0;
   int statusy         = pre->requireMarginWidth(ymargin, &receivedymargin, 'y');
   if (statusy != PV_SUCCESS) {
      ErrorLog().printf(
            "Margin Failure for layer %s.  Received y-margin is %d, but %s requires margin of at "
            "least %d\n",
            pre->getDescription_c(),
            receivedymargin,
            name,
            ymargin);
      status = PV_MARGINWIDTH_FAILURE;
   }

   return status;
}

void WeightsPairInterface::needPre() {
   FatalIf(
         !mInitInfoCommunicatedFlag,
         "%s must finish CommunicateInitInfo before needPre can be called.\n",
         getDescription_c());
   if (mPreWeights == nullptr) {
      createPreWeights();
   }
}

void WeightsPairInterface::needPost() {
   FatalIf(
         !mInitInfoCommunicatedFlag,
         "%s must finish CommunicateInitInfo before needPost can be called.\n",
         getDescription_c());
   if (mPostWeights == nullptr) {
      createPostWeights();
   }
}

int WeightsPairInterface::allocateDataStructures() {
   if (mPreWeights) {
      allocatePreWeights();
   }
   if (mPostWeights) {
      allocatePostWeights();
   }
   return PV_SUCCESS;
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
