/*
 * TransposeWeightsPair.cpp
 *
 *  Created on: Dec 8, 2017
 *      Author: Pete Schultz
 */

#include "TransposeWeightsPair.hpp"
#include "columns/HyPerCol.hpp"
#include "columns/ObjectMapComponent.hpp"
#include "connections/HyPerConn.hpp"
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

int TransposeWeightsPair::setDescription() {
   description.clear();
   description.append("TransposeWeightsPair").append(" \"").append(getName()).append("\"");
   return PV_SUCCESS;
}

int TransposeWeightsPair::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = WeightsPair::ioParamsFillGroup(ioFlag);
   ioParam_originalConnName(ioFlag);
   return status;
}

void TransposeWeightsPair::ioParam_nxp(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      parent->parameters()->handleUnnecessaryParameter(name, "nxp");
   }
   // During the communication phase, nxp will be computed from originalConn
}

void TransposeWeightsPair::ioParam_nyp(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      parent->parameters()->handleUnnecessaryParameter(name, "nyp");
   }
   // During the communication phase, nyp will be computed from originalConn
}

void TransposeWeightsPair::ioParam_nfp(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      parent->parameters()->handleUnnecessaryParameter(name, "nfp");
   }
   // During the communication phase, nfp will be computed from originalConn
}

void TransposeWeightsPair::ioParam_sharedWeights(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      parent->parameters()->handleUnnecessaryParameter(name, "sharedWeights");
   }
   // During the communication phase, sharedWeights will be copied from originalConn
}

void TransposeWeightsPair::ioParam_writeCompressedCheckpoints(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      mWriteCompressedCheckpoints = false;
      parent->parameters()->handleUnnecessaryParameter(name, "writeCompressedCheckpoints");
   }
   // CloneConn never writes checkpoints: set writeCompressedCheckpoints to false.
}

void TransposeWeightsPair::ioParam_originalConnName(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamStringRequired(
         ioFlag, name, "originalConnName", &mOriginalConnName);
}

int TransposeWeightsPair::communicateInitInfo(
      std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto hierarchy = message->mHierarchy;
   ObjectMapComponent *objectMapComponent =
         mapLookupByType<ObjectMapComponent>(hierarchy, getDescription());
   pvAssert(objectMapComponent);
   HyPerConn *originalConn = objectMapComponent->lookup<HyPerConn>(std::string(mOriginalConnName));
   if (originalConn == nullptr) {
      if (parent->getCommunicator()->globalCommRank() == 0) {
         ErrorLog().printf(
               "%s: originalConnName \"%s\" does not correspond to a HyPerConn in the column.\n",
               getDescription_c(),
               mOriginalConnName);
      }
      MPI_Barrier(parent->getCommunicator()->globalCommunicator());
      exit(PV_FAILURE);
   }

   mOriginalWeightsPair = originalConn->getComponentByType<WeightsPair>();
   pvAssert(mOriginalWeightsPair);
   if (!mOriginalWeightsPair->getInitInfoCommunicatedFlag()) {
      if (parent->getCommunicator()->globalCommRank() == 0) {
         InfoLog().printf(
               "%s must wait until original connection \"%s\" has finished its communicateInitInfo "
               "stage.\n",
               getDescription_c(),
               originalConn->getName());
      }
      return PV_POSTPONE;
   }

   // Get some parameters from originalConn.  Check if parameters exist in
   // the transpose's param group, and issue a warning (if the param has the right
   // value) or an error (if it has the wrong value). Note that
   // nxp, nyp, and nfp are not necessarily the same as the original conn,
   // but are determined by the original conn's nxp,nyp,nfp, and the
   // relative sizes of the original conn's pre and post layers.
   inferParameters();

   int status = WeightsPair::communicateInitInfo(message);
   if (status != PV_SUCCESS) {
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
   const PVLayerLoc *origPostLoc = originalConn->getPost()->getLayerLoc();
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
   originalConn->getPre()->synchronizeMarginWidth(mConnectionData->getPost());
   mConnectionData->getPost()->synchronizeMarginWidth(originalConn->getPre());

   const PVLayerLoc *postLoc    = mConnectionData->getPost()->getLayerLoc();
   const PVLayerLoc *origPreLoc = originalConn->getPre()->getLayerLoc();
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
   originalConn->getPost()->synchronizeMarginWidth(mConnectionData->getPre());
   mConnectionData->getPre()->synchronizeMarginWidth(originalConn->getPost());

   return status;
}

void TransposeWeightsPair::inferParameters() {
   mOriginalWeightsPair->needPost();
   mPreWeights = mOriginalWeightsPair->getPostWeights();
   mPatchSizeX = mPreWeights->getPatchSizeX();
   parent->parameters()->handleUnnecessaryParameter(name, "nxp", mPatchSizeX);

   mPatchSizeY = mPreWeights->getPatchSizeY();
   parent->parameters()->handleUnnecessaryParameter(name, "nyp", mPatchSizeY);

   mPatchSizeF = mPreWeights->getPatchSizeF();
   parent->parameters()->handleUnnecessaryParameter(name, "nfp", mPatchSizeF);

   mSharedWeights = mPreWeights->getSharedFlag();
   parent->parameters()->handleUnnecessaryParameter(name, "sharedWeights", mSharedWeights);
}

void TransposeWeightsPair::needPre() {
   if (mPreWeights == nullptr) {
      mOriginalWeightsPair->needPost();
      mPreWeights = mOriginalWeightsPair->getPostWeights();
   }
}

void TransposeWeightsPair::needPost() {
   if (mPostWeights == nullptr) {
      mOriginalWeightsPair->needPre();
      mPostWeights = mOriginalWeightsPair->getPreWeights();
   }
}

int TransposeWeightsPair::allocateDataStructures() { return PV_SUCCESS; }

int TransposeWeightsPair::registerData(Checkpointer *checkpointer) { return PV_SUCCESS; }

void TransposeWeightsPair::finalizeUpdate(double timestamp, double deltaTime) {}

void TransposeWeightsPair::outputState(double timestamp) { return; }

} // namespace PV
