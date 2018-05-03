/*
 * TransposeWeightsPair.cpp
 *
 *  Created on: Dec 8, 2017
 *      Author: Pete Schultz
 */

#include "TransposeWeightsPair.hpp"
#include "columns/HyPerCol.hpp"
#include "columns/ObjectMapComponent.hpp"
#include "components/OriginalConnNameParam.hpp"
#include "utils/MapLookupByType.hpp"

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
      parent->parameters()->handleUnnecessaryParameter(name, "writeCompressedCheckpoints");
   }
   // TransposeWeightsPair never checkpoints, so we always set writeCompressedCheckpoints to false.
}

Response::Status TransposeWeightsPair::communicateInitInfo(
      std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto hierarchy = message->mHierarchy;
   if (mOriginalConn == nullptr) {
      OriginalConnNameParam *originalConnNameParam =
            mapLookupByType<OriginalConnNameParam>(hierarchy, getDescription());
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
   const PVLayerLoc *origPostLoc = mOriginalConn->getPost()->getLayerLoc();
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
   mOriginalConn->getPre()->synchronizeMarginWidth(mConnectionData->getPost());
   mConnectionData->getPost()->synchronizeMarginWidth(mOriginalConn->getPre());

   const PVLayerLoc *postLoc    = mConnectionData->getPost()->getLayerLoc();
   const PVLayerLoc *origPreLoc = mOriginalConn->getPre()->getLayerLoc();
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
   mOriginalConn->getPost()->synchronizeMarginWidth(mConnectionData->getPre());
   mConnectionData->getPre()->synchronizeMarginWidth(mOriginalConn->getPost());

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

Response::Status TransposeWeightsPair::registerData(Checkpointer *checkpointer) {
   return Response::NO_ACTION;
}

void TransposeWeightsPair::finalizeUpdate(double timestamp, double deltaTime) {}

void TransposeWeightsPair::outputState(double timestamp) { return; }

} // namespace PV
