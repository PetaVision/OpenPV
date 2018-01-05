/*
 * CopyWeightsPair.cpp
 *
 *  Created on: Dec 15, 2017
 *      Author: Pete Schultz
 */

#include "CopyWeightsPair.hpp"
#include "columns/HyPerCol.hpp"
#include "columns/ObjectMapComponent.hpp"
#include "connections/HyPerConn.hpp"
#include "utils/MapLookupByType.hpp"

namespace PV {

CopyWeightsPair::CopyWeightsPair(char const *name, HyPerCol *hc) { initialize(name, hc); }

CopyWeightsPair::~CopyWeightsPair() {}

int CopyWeightsPair::initialize(char const *name, HyPerCol *hc) {
   return WeightsPair::initialize(name, hc);
}

int CopyWeightsPair::setDescription() {
   description.clear();
   description.append("CopyWeightsPair").append(" \"").append(getName()).append("\"");
   return PV_SUCCESS;
}

int CopyWeightsPair::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = WeightsPair::ioParamsFillGroup(ioFlag);
   ioParam_originalConnName(ioFlag);
   return status;
}

void CopyWeightsPair::ioParam_nxp(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      parent->parameters()->handleUnnecessaryParameter(name, "nxp");
   }
   // During the communication phase, nxp will be copied from originalConn
}

void CopyWeightsPair::ioParam_nyp(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      parent->parameters()->handleUnnecessaryParameter(name, "nyp");
   }
   // During the communication phase, nyp will be copied from originalConn
}

void CopyWeightsPair::ioParam_nfp(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      parent->parameters()->handleUnnecessaryParameter(name, "nfp");
   }
   // During the communication phase, nfp will be copied from originalConn
}

void CopyWeightsPair::ioParam_sharedWeights(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      parent->parameters()->handleUnnecessaryParameter(name, "sharedWeights");
   }
   // During the communication phase, sharedWeights will be copied from originalConn
}

void CopyWeightsPair::ioParam_originalConnName(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamStringRequired(
         ioFlag, name, "originalConnName", &mOriginalConnName);
}

int CopyWeightsPair::communicateInitInfo(
      std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto hierarchy           = message->mHierarchy;
   auto *objectMapComponent = mapLookupByType<ObjectMapComponent>(hierarchy, getDescription());
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

   // Copy some parameters from originalConn.  Check if parameters exist is
   // the clone's param group, and issue a warning (if the param has the right
   // value) or an error (if it has the wrong value).
   copyParameters();

   int status = WeightsPair::communicateInitInfo(message);
   return status;
}

void CopyWeightsPair::copyParameters() {
   mPatchSizeX = mOriginalWeightsPair->getPatchSizeX();
   parent->parameters()->handleUnnecessaryParameter(name, "nxp", mPatchSizeX);

   mPatchSizeY = mOriginalWeightsPair->getPatchSizeY();
   parent->parameters()->handleUnnecessaryParameter(name, "nyp", mPatchSizeY);

   mPatchSizeF = mOriginalWeightsPair->getPatchSizeF();
   parent->parameters()->handleUnnecessaryParameter(name, "nfp", mPatchSizeF);

   mSharedWeights = mOriginalWeightsPair->getSharedWeights();
   parent->parameters()->handleUnnecessaryParameter(name, "sharedWeights", mSharedWeights);
}

void CopyWeightsPair::needPre() {
   WeightsPair::needPre();
   pvAssert(mOriginalWeightsPair);
   mOriginalWeightsPair->needPre();
}

void CopyWeightsPair::needPost() {
   WeightsPair::needPost();
   pvAssert(mOriginalWeightsPair);
   mOriginalWeightsPair->needPost();
}

void CopyWeightsPair::copy() {
   if (mPreWeights) {
      auto *originalPreWeights = mOriginalWeightsPair->getPreWeights();
      pvAssert(originalPreWeights);

      int const numArbors        = mPreWeights->getNumArbors();
      int const patchSizeOverall = mPreWeights->getPatchSizeOverall();
      int const numDataPatches   = mPreWeights->getNumDataPatches();
      pvAssert(numArbors == originalPreWeights->getNumArbors());
      pvAssert(patchSizeOverall == originalPreWeights->getPatchSizeOverall());
      pvAssert(numDataPatches == originalPreWeights->getNumDataPatches());

      auto arborSize = (std::size_t)(patchSizeOverall * numDataPatches) * sizeof(float);
      for (int arbor = 0; arbor < numArbors; arbor++) {
         float const *sourceArbor = originalPreWeights->getDataReadOnly(arbor);
         std::memcpy(mPreWeights->getData(arbor), sourceArbor, arborSize);
      }
   }
}

} // namespace PV
