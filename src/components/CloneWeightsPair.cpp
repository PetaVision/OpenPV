/*
 * CloneWeightsPair.cpp
 *
 *  Created on: Dec 3, 2017
 *      Author: Pete Schultz
 */

#include "CloneWeightsPair.hpp"
#include "columns/HyPerCol.hpp"
#include "columns/ObjectMapComponent.hpp"
#include "connections/HyPerConn.hpp"
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
   mPreWeights  = mOriginalWeightsPair->getPreWeights();
   mPostWeights = mOriginalWeightsPair->getPostWeights();

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

   // Copy some parameters from originalConn.  Check if parameters exist is
   // the clone's param group, and issue a warning (if the param has the right
   // value) or an error (if it has the wrong value).
   copyParameters();

   int status = WeightsPair::communicateInitInfo(message);
   return status;
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

void CloneWeightsPair::needPre() { mOriginalWeightsPair->needPre(); }

void CloneWeightsPair::needPost() { mOriginalWeightsPair->needPost(); }

int CloneWeightsPair::allocateDataStructures() { return PV_SUCCESS; }

int CloneWeightsPair::registerData(Checkpointer *checkpointer) { return PV_SUCCESS; }

void CloneWeightsPair::outputState(double timestamp) { return; }

} // namespace PV
