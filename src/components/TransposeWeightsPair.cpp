/*
 * TransposeWeightsPair.cpp
 *
 *  Created on: Dec 8, 2017
 *      Author: Pete Schultz
 */

#include "TransposeWeightsPair.hpp"
#include "columns/ComponentBasedObject.hpp"
#include "components/OriginalConnNameParam.hpp"
#include "observerpattern/ObserverTable.hpp"

namespace PV {

TransposeWeightsPair::TransposeWeightsPair(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   initialize(params, defaults, comm);
}

TransposeWeightsPair::~TransposeWeightsPair() {
   mPreWeights  = nullptr;
   mPostWeights = nullptr;
}

void TransposeWeightsPair::initialize(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   WeightsPair::initialize(params, defaults, comm);
}

void TransposeWeightsPair::setObjectType() { mObjectType = "TransposeWeightsPair"; }

int TransposeWeightsPair::ioParamsFillGroup(ParamsIOSwitch ioSwitch) {
   int status = WeightsPair::ioParamsFillGroup(ioSwitch);
   return status;
}

void TransposeWeightsPair::ioParam_writeCompressedCheckpoints(ParamsIOSwitch ioSwitch) {
   if (ioSwitch == ParamsIOSwitch::Read) {
      mWriteCompressedCheckpoints = false;
      mParamsIO->handleUnnecessaryParameter("writeCompressedCheckpoints");
   }
   // TransposeWeightsPair never checkpoints, so we always set writeCompressedCheckpoints to false.
}

Response::Status TransposeWeightsPair::communicateInitInfo(
      std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto status                 = Response::NO_ACTION;
   auto *objectTable           = message->mObjectTable;
   auto *originalConnNameParam = objectTable->findObject<OriginalConnNameParam>(getName());
   FatalIf(
         originalConnNameParam == nullptr,
         "%s could not find an OriginalConnNameParam.\n",
         getDescription_c());

   if (!originalConnNameParam->getInitInfoCommunicatedFlag()) {
      if (mCommunicator->globalCommRank() == 0) {
         InfoLog().printf(
               "%s must wait until the OriginalConnNameParam component has finished its "
               "communicateInitInfo stage.\n",
               getDescription_c());
      }
      return Response::POSTPONE;
   }
   std::string const &originalConnName = originalConnNameParam->getLinkedObjectName();

   if (mOriginalWeightsPair == nullptr) {
      mOriginalWeightsPair = objectTable->findObject<WeightsPair>(originalConnName);
      FatalIf(
            mOriginalWeightsPair == nullptr,
            "%s could not find a WeightsPair in \"%s\".\n",
            getDescription_c(),
            originalConnName.c_str());
      status = status + Response::SUCCESS;
   }

   ConnectionData *originalConnData = objectTable->findObject<ConnectionData>(originalConnName);
   FatalIf(
         originalConnData == nullptr,
         "%s could not find a ConnectionData component in \"%s\".\n",
         getDescription_c(),
         originalConnName.c_str());
   if (!originalConnData->getInitInfoCommunicatedFlag()) {
      return status + Response::POSTPONE;
   }

   status = status + WeightsPair::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }

   int numArbors     = getArborList()->getNumAxonalArbors();
   int origNumArbors = mOriginalWeightsPair->getArborList()->getNumAxonalArbors();
   FatalIf(
         numArbors != origNumArbors,
         "%s has %d arbors but original connection %s has %d arbors.\n",
         mConnectionData->getDescription_c(),
         numArbors,
         mOriginalWeightsPair->getConnectionData()->getDescription_c(),
         origNumArbors);

   const PVLayerLoc *preLoc      = mConnectionData->getPre()->getLayerLoc();
   const PVLayerLoc *origPostLoc = originalConnData->getPost()->getLayerLoc();
   FatalIf(
         preLoc->nx != origPostLoc->nx || preLoc->ny != origPostLoc->ny
               || preLoc->nf != origPostLoc->nf,
         "%s: transpose's pre and original connection's post must have the same dimensions.\n"
         "    (x=%d, y=%d, f=%d) versus (x=%d, y=%d, f=%d).\n",
         getDescription_c(),
         preLoc->nx,
         preLoc->ny,
         preLoc->nf,
         origPostLoc->nx,
         origPostLoc->ny,
         origPostLoc->nf);
   originalConnData->getPre()->synchronizeMarginWidth(mConnectionData->getPost());
   mConnectionData->getPost()->synchronizeMarginWidth(originalConnData->getPre());

   const PVLayerLoc *postLoc    = mConnectionData->getPost()->getLayerLoc();
   const PVLayerLoc *origPreLoc = originalConnData->getPre()->getLayerLoc();
   FatalIf(
         postLoc->nx != origPreLoc->nx || postLoc->ny != origPreLoc->ny
               || postLoc->nf != origPreLoc->nf,
         "%s: transpose's post and original connection's pre must have the same dimensions.\n"
         "    (x=%d, y=%d, f=%d) versus (x=%d, y=%d, f=%d).\n",
         getDescription_c(),
         postLoc->nx,
         postLoc->ny,
         postLoc->nf,
         origPreLoc->nx,
         origPreLoc->ny,
         origPreLoc->nf);
   originalConnData->getPost()->synchronizeMarginWidth(mConnectionData->getPre());
   mConnectionData->getPre()->synchronizeMarginWidth(originalConnData->getPost());

   return Response::SUCCESS;
}

void TransposeWeightsPair::createPreWeights(std::string const &weightsName) {
   mOriginalWeightsPair->needPost();
   mPreWeights = mOriginalWeightsPair->getPostWeights();
}

void TransposeWeightsPair::createPostWeights(std::string const &weightsName) {
   mOriginalWeightsPair->needPre();
   mPostWeights = mOriginalWeightsPair->getPreWeights();
}

Response::Status TransposeWeightsPair::allocateDataStructures() { return Response::SUCCESS; }

Response::Status TransposeWeightsPair::registerData(
      std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) {
   if (mWriteStep >= 0) {
      return WeightsPair::registerData(message);
   }
   else {
      return Response::NO_ACTION;
   }
}

void TransposeWeightsPair::finalizeUpdate(double timestamp, double deltaTime) {}

} // namespace PV
