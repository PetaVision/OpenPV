/*
 * CloneWeightsPair.cpp
 *
 *  Created on: Dec 3, 2017
 *      Author: Pete Schultz
 */

#include "CloneWeightsPair.hpp"
#include "columns/ComponentBasedObject.hpp"
#include "columns/HyPerCol.hpp"
#include "columns/ObserverTableComponent.hpp"
#include "components/OriginalConnNameParam.hpp"

namespace PV {

CloneWeightsPair::CloneWeightsPair(char const *name, HyPerCol *hc) { initialize(name, hc); }

CloneWeightsPair::~CloneWeightsPair() {
   mPreWeights  = nullptr;
   mPostWeights = nullptr;
}

int CloneWeightsPair::initialize(char const *name, HyPerCol *hc) {
   return WeightsPair::initialize(name, hc);
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
      pvAssert(mOriginalConnData == nullptr);
      auto *originalConnNameParam = message->mHierarchy.lookupByType<OriginalConnNameParam>();
      pvAssert(originalConnNameParam);

      if (!originalConnNameParam->getInitInfoCommunicatedFlag()) {
         if (parent->getCommunicator()->globalCommRank() == 0) {
            InfoLog().printf(
                  "%s must wait until the OriginalConnNameParam component has finished its "
                  "communicateInitInfo stage.\n",
                  getDescription_c());
         }
         return Response::POSTPONE;
      }

      ComponentBasedObject *originalConn = nullptr;
      try {
         originalConn = originalConnNameParam->findLinkedObject(message->mHierarchy);
      } catch (std::invalid_argument &e) {
         Fatal().printf("%s: %s\n", getDescription_c(), e.what());
      }
      pvAssert(originalConn); // findLinkedObject() throws instead of returns nullptr

      if (!originalConn->getInitInfoCommunicatedFlag()) {
         if (parent->getCommunicator()->globalCommRank() == 0) {
            InfoLog().printf(
                  "%s must wait until original connection \"%s\" has finished its "
                  "communicateInitInfo stage.\n",
                  getDescription_c(),
                  mOriginalWeightsPair->getName());
         }
         return Response::POSTPONE;
      }

      mOriginalConnData = originalConn->getComponentByType<ConnectionData>();
      pvAssert(mOriginalConnData);
      pvAssert(mOriginalConnData->getInitInfoCommunicatedFlag());
      mOriginalWeightsPair = originalConn->getComponentByType<WeightsPair>();
      pvAssert(mOriginalWeightsPair);
      pvAssert(mOriginalWeightsPair->getInitInfoCommunicatedFlag());
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

Response::Status CloneWeightsPair::allocateDataStructures() { return Response::SUCCESS; }

Response::Status
CloneWeightsPair::registerData(std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) {
   return Response::NO_ACTION;
}

void CloneWeightsPair::finalizeUpdate(double timestamp, double deltaTime) {}

void CloneWeightsPair::outputState(double timestamp) { return; }

} // namespace PV
