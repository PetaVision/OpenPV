/*
 * CloneWeightsPair.cpp
 *
 *  Created on: Dec 3, 2017
 *      Author: Pete Schultz
 */

#include "CloneWeightsPair.hpp"
#include "columns/ComponentBasedObject.hpp"
#include "components/OriginalConnNameParam.hpp"
#include "observerpattern/ObserverTable.hpp"

namespace PV {

CloneWeightsPair::CloneWeightsPair(char const *name, PVParams *params, Communicator const *comm) {
   initialize(name, params, comm);
}

CloneWeightsPair::~CloneWeightsPair() {
   mPreWeights  = nullptr;
   mPostWeights = nullptr;
}

void CloneWeightsPair::initialize(char const *name, PVParams *params, Communicator const *comm) {
   WeightsPair::initialize(name, params, comm);
}

void CloneWeightsPair::setObjectType() { mObjectType = "CloneWeightsPair"; }

int CloneWeightsPair::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = WeightsPair::ioParamsFillGroup(ioFlag);
   return status;
}

void CloneWeightsPair::ioParam_writeStep(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      parameters()->handleUnnecessaryParameter(name, "writeStep");
      mWriteStep = -1;
   }
   // CloneWeightsPair never writes output: set writeStep to -1.
}

void CloneWeightsPair::ioParam_writeCompressedCheckpoints(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      mWriteCompressedCheckpoints = false;
      parameters()->handleUnnecessaryParameter(name, "writeCompressedCheckpoints");
   }
   // CloneConn never writes checkpoints: set writeCompressedCheckpoints to false.
}

Response::Status
CloneWeightsPair::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   if (mOriginalWeightsPair == nullptr) {
      auto *objectTable = message->mObjectTable;
      pvAssert(mOriginalConnData == nullptr);
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
      char const *linkedObjectName = originalConnNameParam->getLinkedObjectName();

      mOriginalConnData = objectTable->findObject<ConnectionData>(linkedObjectName);
      FatalIf(
            mOriginalConnData == nullptr,
            "%s could not find a ConnectionData component in connection \"%s\"\n",
            getDescription_c());

      mOriginalWeightsPair = objectTable->findObject<WeightsPair>(linkedObjectName);
      FatalIf(
            mOriginalConnData == nullptr,
            "%s could not find a WeightsPair component in connection \"%s\"\n",
            getDescription_c());
   }

   Response::Status status = WeightsPair::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }

   // Presynaptic layers of the Clone and its original conn must have the same size, or the
   // patches won't line up with each other.
   synchronizeMarginsPre();

   return Response::SUCCESS;
}

void CloneWeightsPair::synchronizeMarginsPre() {
   int status = PV_SUCCESS;

   pvAssert(mConnectionData);
   auto *thisPre = mConnectionData->getPre();
   if (thisPre == nullptr) {
      ErrorLog().printf(
            "synchronzedMarginsPre called for %s, but this connection has not set its "
            "presynaptic layer yet.\n",
            getDescription_c());
      status = PV_FAILURE;
   }

   HyPerLayer *origPre = nullptr;
   if (mOriginalWeightsPair == nullptr) {
      ErrorLog().printf(
            "synchronzedMarginsPre called for %s, but this connection has not set its "
            "original connection yet.\n",
            getDescription_c());
      status = PV_FAILURE;
   }
   else {
      origPre = mOriginalConnData->getPre();
      if (origPre == nullptr) {
         ErrorLog().printf(
               "synchronzedMarginsPre called for %s, but the original connection has not set its "
               "presynaptic layer yet.\n",
               getDescription_c());
         status = PV_FAILURE;
      }
   }
   if (status != PV_SUCCESS) {
      exit(PV_FAILURE);
   }
   thisPre->synchronizeMarginWidth(origPre);
   origPre->synchronizeMarginWidth(thisPre);
}

void CloneWeightsPair::synchronizeMarginsPost() {
   int status = PV_SUCCESS;

   pvAssert(mConnectionData);
   auto *thisPost = mConnectionData->getPost();
   if (thisPost == nullptr) {
      ErrorLog().printf(
            "synchronzedMarginsPost called for %s, but this connection has not set its "
            "postsynaptic layer yet.\n",
            getDescription_c());
      status = PV_FAILURE;
   }

   HyPerLayer *origPost = nullptr;
   if (mOriginalWeightsPair == nullptr) {
      ErrorLog().printf(
            "synchronzedMarginsPre called for %s, but this connection has not set its "
            "original connection yet.\n",
            getDescription_c());
      status = PV_FAILURE;
   }
   else {
      origPost = mOriginalConnData->getPost();
      if (origPost == nullptr) {
         ErrorLog().printf(
               "synchronzedMarginsPost called for %s, but the original connection has not set its "
               "postsynaptic layer yet.\n",
               getDescription_c());
         status = PV_FAILURE;
      }
   }
   if (status != PV_SUCCESS) {
      exit(PV_FAILURE);
   }
   thisPost->synchronizeMarginWidth(origPost);
   origPost->synchronizeMarginWidth(thisPost);
}

void CloneWeightsPair::createPreWeights(std::string const &weightsName) {
   mOriginalWeightsPair->needPre();
   mPreWeights = mOriginalWeightsPair->getPreWeights();
}

void CloneWeightsPair::createPostWeights(std::string const &weightsName) {
   mOriginalWeightsPair->needPost();
   mPostWeights = mOriginalWeightsPair->getPostWeights();
}

void CloneWeightsPair::setDefaultWriteStep(
      std::shared_ptr<CommunicateInitInfoMessage const> message) {
   pvAssert(mWriteStep < 0); // CloneWeightsPair doesn't use WriteStep.
}

Response::Status CloneWeightsPair::allocateDataStructures() { return Response::SUCCESS; }

Response::Status
CloneWeightsPair::registerData(std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) {
   return Response::NO_ACTION;
}

void CloneWeightsPair::finalizeUpdate(double timestamp, double deltaTime) {}

void CloneWeightsPair::outputState(double timestamp) { return; }

} // namespace PV
