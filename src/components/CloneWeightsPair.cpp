/*
 * CloneWeightsPair.cpp
 *
 *  Created on: Dec 3, 2017
 *      Author: Pete Schultz
 */

#include "CloneWeightsPair.hpp"
#include "columns/HyPerCol.hpp"
#include "columns/ObjectMapComponent.hpp"
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

int CloneWeightsPair::setDescription() {
   description.clear();
   description.append("CloneWeightsPair").append(" \"").append(getName()).append("\"");
   return PV_SUCCESS;
}

int CloneWeightsPair::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = WeightsPair::ioParamsFillGroup(ioFlag);
   ioParam_originalConnName(ioFlag);
   return status;
}

void CloneWeightsPair::ioParam_nxp(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      parent->parameters()->handleUnnecessaryParameter(name, "nxp");
   }
   // During the communication phase, nxp will be copied from originalConn
}

void CloneWeightsPair::ioParam_nyp(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      parent->parameters()->handleUnnecessaryParameter(name, "nyp");
   }
   // During the communication phase, nyp will be copied from originalConn
}

void CloneWeightsPair::ioParam_nfp(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      parent->parameters()->handleUnnecessaryParameter(name, "nfp");
   }
   // During the communication phase, nfp will be copied from originalConn
}

void CloneWeightsPair::ioParam_sharedWeights(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      parent->parameters()->handleUnnecessaryParameter(name, "sharedWeights");
   }
   // During the communication phase, sharedWeights will be copied from originalConn
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

void CloneWeightsPair::ioParam_originalConnName(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamStringRequired(
         ioFlag, name, "originalConnName", &mOriginalConnName);
}

int CloneWeightsPair::communicateInitInfo(
      std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto hierarchy = message->mHierarchy;
   ObjectMapComponent *objectMapComponent =
         mapLookupByType<ObjectMapComponent>(hierarchy, getDescription());
   pvAssert(objectMapComponent);
   mOriginalConn = objectMapComponent->lookup<HyPerConn>(std::string(mOriginalConnName));
   if (mOriginalConn == nullptr) {
      if (parent->getCommunicator()->globalCommRank() == 0) {
         ErrorLog().printf(
               "%s: originalConnName \"%s\" does not correspond to a HyPerConn in the column.\n",
               getDescription_c(),
               mOriginalConnName);
      }
      MPI_Barrier(parent->getCommunicator()->globalCommunicator());
      exit(PV_FAILURE);
   }
   mOriginalWeightsPair = mOriginalConn->getComponentByType<WeightsPair>();
   pvAssert(mOriginalWeightsPair);

   if (!mOriginalWeightsPair->getInitInfoCommunicatedFlag()) {
      if (parent->getCommunicator()->globalCommRank() == 0) {
         InfoLog().printf(
               "%s must wait until original connection \"%s\" has finished its communicateInitInfo "
               "stage.\n",
               getDescription_c(),
               mOriginalConn->getName());
      }
      return PV_POSTPONE;
   }

   // Copy some parameters from originalConn.  Check if parameters exist is
   // the clone's param group, and issue a warning (if the param has the right
   // value) or an error (if it has the wrong value).
   copyParameters();

   int status = WeightsPair::communicateInitInfo(message);
   if (status != PV_SUCCESS) {
      return status;
   }

   // Presynaptic layers of the Clone and its original conn must have the same size, or the
   // patches won't line up with each other.
   synchronizeMarginsPre();

   return status;
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

void CloneWeightsPair::copyParameters() {
   mPatchSizeX = mOriginalWeightsPair->getPatchSizeX();
   parent->parameters()->handleUnnecessaryParameter(name, "nxp", mPatchSizeX);

   mPatchSizeY = mOriginalWeightsPair->getPatchSizeY();
   parent->parameters()->handleUnnecessaryParameter(name, "nyp", mPatchSizeY);

   mPatchSizeF = mOriginalWeightsPair->getPatchSizeF();
   parent->parameters()->handleUnnecessaryParameter(name, "nfp", mPatchSizeF);

   mSharedWeights = mOriginalWeightsPair->getSharedWeights();
   parent->parameters()->handleUnnecessaryParameter(name, "sharedWeights", mSharedWeights);
}

void CloneWeightsPair::needPre() {
   mOriginalWeightsPair->needPre();
   mPreWeights = mOriginalWeightsPair->getPreWeights();
}

void CloneWeightsPair::needPost() {
   mOriginalWeightsPair->needPost();
   mPostWeights = mOriginalWeightsPair->getPostWeights();
}

int CloneWeightsPair::allocateDataStructures() { return PV_SUCCESS; }

int CloneWeightsPair::registerData(Checkpointer *checkpointer) { return PV_SUCCESS; }

void CloneWeightsPair::finalizeUpdate(double timestamp, double deltaTime) {}

void CloneWeightsPair::outputState(double timestamp) { return; }

} // namespace PV
