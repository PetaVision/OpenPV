/*
 * TransposeWeightsPair.cpp
 *
 *  Created on: Dec 8, 2017
 *      Author: Pete Schultz
 */

#include "TransposeWeightsPair.hpp"
#include "columns/ComponentBasedObject.hpp"
#include "columns/HyPerCol.hpp"
#include "components/OriginalConnNameParam.hpp"
#include "observerpattern/ObserverTable.hpp"

namespace PV {

TransposeWeightsPair::TransposeWeightsPair(char const *name, HyPerCol *hc) { initialize(name, hc); }

TransposeWeightsPair::~TransposeWeightsPair() {
   mPreWeights  = nullptr;
   mPostWeights = nullptr;
}

int TransposeWeightsPair::initialize(char const *name, HyPerCol *hc) {
   return WeightsPair::initialize(name, hc);
}

void TransposeWeightsPair::setObjectType() { mObjectType = "TransposeWeightsPair"; }

int TransposeWeightsPair::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = WeightsPair::ioParamsFillGroup(ioFlag);
   return status;
}

void TransposeWeightsPair::ioParam_writeCompressedCheckpoints(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      mWriteCompressedCheckpoints = false;
      parameters()->handleUnnecessaryParameter(name, "writeCompressedCheckpoints");
   }
   // TransposeWeightsPair never checkpoints, so we always set writeCompressedCheckpoints to false.
}

Response::Status TransposeWeightsPair::communicateInitInfo(
      std::shared_ptr<CommunicateInitInfoMessage const> message) {
   ConnectionData *originalConnData = nullptr;
   if (mOriginalWeightsPair == nullptr) {
      auto *originalConnNameParam = message->mHierarchy->lookupByType<OriginalConnNameParam>();
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
                  originalConn->getName());
         }
         return Response::POSTPONE;
      }

      mOriginalWeightsPair = originalConn->getComponentByType<WeightsPair>();
      pvAssert(mOriginalWeightsPair);
      pvAssert(mOriginalWeightsPair->getInitInfoCommunicatedFlag());
      originalConnData = originalConn->getComponentByType<ConnectionData>();
      pvAssert(originalConnData);
      pvAssert(originalConnData->getInitInfoCommunicatedFlag());
   }

   auto status = WeightsPair::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }

   int numArbors     = getArborList()->getNumAxonalArbors();
   int origNumArbors = mOriginalWeightsPair->getArborList()->getNumAxonalArbors();
   if (numArbors != origNumArbors) {
      if (parent->getCommunicator()->globalCommRank() == 0) {
         Fatal().printf(
               "%s has %d arbors but original connection %s has %d arbors.\n",
               mConnectionData->getDescription_c(),
               numArbors,
               mOriginalWeightsPair->getConnectionData()->getDescription_c(),
               origNumArbors);
      }
      MPI_Barrier(parent->getCommunicator()->globalCommunicator());
      exit(EXIT_FAILURE);
   }

   const PVLayerLoc *preLoc      = mConnectionData->getPre()->getLayerLoc();
   const PVLayerLoc *origPostLoc = originalConnData->getPost()->getLayerLoc();
   if (preLoc->nx != origPostLoc->nx || preLoc->ny != origPostLoc->ny
       || preLoc->nf != origPostLoc->nf) {
      if (parent->getCommunicator()->globalCommRank() == 0) {
         ErrorLog(errorMessage);
         errorMessage.printf(
               "%s: transpose's pre layer and original connection's post layer must have the same "
               "dimensions.\n",
               getDescription_c());
         errorMessage.printf(
               "    (x=%d, y=%d, f=%d) versus (x=%d, y=%d, f=%d).\n",
               preLoc->nx,
               preLoc->ny,
               preLoc->nf,
               origPostLoc->nx,
               origPostLoc->ny,
               origPostLoc->nf);
      }
      MPI_Barrier(parent->getCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
   originalConnData->getPre()->synchronizeMarginWidth(mConnectionData->getPost());
   mConnectionData->getPost()->synchronizeMarginWidth(originalConnData->getPre());

   const PVLayerLoc *postLoc    = mConnectionData->getPost()->getLayerLoc();
   const PVLayerLoc *origPreLoc = originalConnData->getPre()->getLayerLoc();
   if (postLoc->nx != origPreLoc->nx || postLoc->ny != origPreLoc->ny
       || postLoc->nf != origPreLoc->nf) {
      if (parent->getCommunicator()->globalCommRank() == 0) {
         ErrorLog(errorMessage);
         errorMessage.printf(
               "%s: transpose's post layer and original connection's pre layer must have the same "
               "dimensions.\n",
               getDescription_c());
         errorMessage.printf(
               "    (x=%d, y=%d, f=%d) versus (x=%d, y=%d, f=%d).\n",
               postLoc->nx,
               postLoc->ny,
               postLoc->nf,
               origPreLoc->nx,
               origPreLoc->ny,
               origPreLoc->nf);
      }
      MPI_Barrier(parent->getCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
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
