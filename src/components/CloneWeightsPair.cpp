/*
 * CloneWeightsPair.cpp
 *
 *  Created on: Dec 3, 2017
 *      Author: Pete Schultz
 */

#include "CloneWeightsPair.hpp"
#include "columns/HyPerCol.hpp"
#include "columns/ObjectMapComponent.hpp"
#include "components/OriginalConnNameParam.hpp"
#include "utils/MapLookupByType.hpp"

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
      parent->parameters()->handleUnnecessaryParameter(name, "writeStep");
      mWriteStep = -1;
   }
   // CloneWeightsPair never writes output: set writeStep to -1.
}

void CloneWeightsPair::ioParam_writeCompressedCheckpoints(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      mWriteCompressedCheckpoints = false;
      parent->parameters()->handleUnnecessaryParameter(name, "writeCompressedCheckpoints");
   }
   // CloneConn never writes checkpoints: set writeCompressedCheckpoints to false.
}

Response::Status
CloneWeightsPair::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   if (mOriginalConn == nullptr) {
      OriginalConnNameParam *originalConnNameParam =
            mapLookupByType<OriginalConnNameParam>(message->mHierarchy, getDescription());
      FatalIf(
            originalConnNameParam == nullptr,
            "%s requires an OriginalConnNameParam component.\n",
            getDescription_c());

      if (!originalConnNameParam->getInitInfoCommunicatedFlag()) {
         if (parent->getCommunicator()->globalCommRank() == 0) {
            InfoLog().printf(
                  "%s must wait until the OriginalConnNameParam component has finished its "
                  "communicateInitInfo stage.\n",
                  getDescription_c());
         }
         return Response::POSTPONE;
      }
      char const *originalConnName = originalConnNameParam->getOriginalConnName();

      auto hierarchy = message->mHierarchy;
      ObjectMapComponent *objectMapComponent =
            mapLookupByType<ObjectMapComponent>(hierarchy, getDescription());
      pvAssert(objectMapComponent);
      mOriginalConn = objectMapComponent->lookup<HyPerConn>(std::string(originalConnName));
      if (mOriginalConn == nullptr) {
         if (parent->getCommunicator()->globalCommRank() == 0) {
            ErrorLog().printf(
                  "%s: originalConnName \"%s\" does not correspond to a HyPerConn in the column.\n",
                  getDescription_c(),
                  originalConnName);
         }
         MPI_Barrier(parent->getCommunicator()->globalCommunicator());
         exit(PV_FAILURE);
      }
   }
   mOriginalWeightsPair = mOriginalConn->getComponentByType<WeightsPair>();
   pvAssert(mOriginalWeightsPair);

   if (!mOriginalWeightsPair->getInitInfoCommunicatedFlag()) {
      if (parent->getCommunicator()->globalCommRank() == 0) {
         InfoLog().printf(
               "%s must wait until original connection \"%s\" has finished its communicateInitInfo "
               "stage.\n",
               getDescription_c(),
               mOriginalWeightsPair->getName());
      }
      return Response::POSTPONE;
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
   if (mOriginalConn == nullptr) {
      ErrorLog().printf(
            "synchronzedMarginsPre called for %s, but this connection has not set its "
            "original connection yet.\n",
            getDescription_c());
      status = PV_FAILURE;
   }
   else {
      origPre = mOriginalConn->getPre();
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
   if (mOriginalConn == nullptr) {
      ErrorLog().printf(
            "synchronzedMarginsPre called for %s, but this connection has not set its "
            "original connection yet.\n",
            getDescription_c());
      status = PV_FAILURE;
   }
   else {
      origPost = mOriginalConn->getPost();
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

Response::Status CloneWeightsPair::registerData(Checkpointer *checkpointer) {
   return Response::NO_ACTION;
}

void CloneWeightsPair::finalizeUpdate(double timestamp, double deltaTime) {}

void CloneWeightsPair::outputState(double timestamp) { return; }

} // namespace PV
