/*
 * WeightsPairInterface.cpp
 *
 *  Created on: Jan 8, 2018
 *      Author: Pete Schultz
 */

#include "WeightsPairInterface.hpp"

namespace PV {

WeightsPairInterface::WeightsPairInterface(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   initialize(name, params, comm);
}

WeightsPairInterface::~WeightsPairInterface() {
   delete mPreWeights;
   delete mPostWeights;
}

void WeightsPairInterface::initialize(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   BaseObject::initialize(name, params, comm);
}

void WeightsPairInterface::setObjectType() { mObjectType = "WeightsPairInterface"; }

Response::Status WeightsPairInterface::communicateInitInfo(
      std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto status = BaseObject::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }
   if (mConnectionData == nullptr) {
      mConnectionData = message->mObjectTable->findObject<ConnectionData>(getName());
      pvAssert(mConnectionData);
   }

   if (!mConnectionData->getInitInfoCommunicatedFlag()) {
      if (mCommunicator->globalCommRank() == 0) {
         InfoLog().printf(
               "%s must wait until the ConnectionData component has finished its "
               "communicateInitInfo stage.\n",
               getDescription_c());
      }
      return Response::POSTPONE;
   }

   if (mPatchSize == nullptr) {
      mPatchSize = message->mObjectTable->findObject<PatchSize>(getName());
      FatalIf(
            mPatchSize == nullptr,
            "Communicate message to %s has no PatchSize component.\n",
            getDescription_c());
   }
   pvAssert(mPatchSize);

   if (!mPatchSize->getInitInfoCommunicatedFlag()) {
      if (mCommunicator->globalCommRank() == 0) {
         InfoLog().printf(
               "%s must wait until the PatchSize component has finished its "
               "communicateInitInfo stage.\n",
               getDescription_c());
      }
      return Response::POSTPONE;
   }

   HyPerLayer *pre  = mConnectionData->getPre();
   HyPerLayer *post = mConnectionData->getPost();
   pvAssert(pre and post);
   LayerGeometry *preGeom  = pre->getComponentByType<LayerGeometry>();
   LayerGeometry *postGeom = post->getComponentByType<LayerGeometry>();
   pvAssert(preGeom and postGeom);
   PVLayerLoc const *preLoc  = preGeom->getLayerLoc();
   PVLayerLoc const *postLoc = postGeom->getLayerLoc();

   // Margins
   int xmargin = requiredConvolveMargin(preLoc->nx, postLoc->nx, mPatchSize->getPatchSizeX());
   preGeom->requireMarginWidth(xmargin, 'x');
   int ymargin = requiredConvolveMargin(preLoc->ny, postLoc->ny, mPatchSize->getPatchSizeY());
   preGeom->requireMarginWidth(ymargin, 'y');

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
