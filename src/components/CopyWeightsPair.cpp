/*
 * CopyWeightsPair.cpp
 *
 *  Created on: Dec 15, 2017
 *      Author: Pete Schultz
 */

#include "CopyWeightsPair.hpp"
#include "columns/HyPerCol.hpp"
#include "columns/ObjectMapComponent.hpp"
#include "components/OriginalConnNameParam.hpp"
#include "connections/HyPerConn.hpp"
#include "utils/MapLookupByType.hpp"

namespace PV {

CopyWeightsPair::CopyWeightsPair(char const *name, HyPerCol *hc) { initialize(name, hc); }

CopyWeightsPair::~CopyWeightsPair() {}

int CopyWeightsPair::initialize(char const *name, HyPerCol *hc) {
   return WeightsPair::initialize(name, hc);
}

void CopyWeightsPair::setObjectType() { mObjectType = "CopyWeightsPair"; }

int CopyWeightsPair::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = WeightsPair::ioParamsFillGroup(ioFlag);
   return status;
}

Response::Status
CopyWeightsPair::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
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

   auto status = WeightsPair::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }

   // Presynaptic layers of the copy and its original conn must have the same size, or the
   // patches won't line up with each other.
   synchronizeMarginsPre();

   return Response::SUCCESS;
}

void CopyWeightsPair::synchronizeMarginsPre() {
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

void CopyWeightsPair::synchronizeMarginsPost() {
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

void CopyWeightsPair::createPreWeights(std::string const &weightsName) {
   WeightsPair::createPreWeights(weightsName);
   pvAssert(mOriginalWeightsPair);
   mOriginalWeightsPair->needPre();
}

void CopyWeightsPair::createPostWeights(std::string const &weightsName) {
   WeightsPair::createPostWeights(weightsName);
   pvAssert(mOriginalWeightsPair);
   mOriginalWeightsPair->needPost();
}

void CopyWeightsPair::copy() {
   // Called by CopyUpdater to update the weights when the original weights change,
   // and by CopyConn::initializeState to initialize from the original weights.
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
   if (mPostWeights) {
      auto *originalPostWeights = mOriginalWeightsPair->getPostWeights();
      pvAssert(originalPostWeights);

      int const numArbors        = mPostWeights->getNumArbors();
      int const patchSizeOverall = mPostWeights->getPatchSizeOverall();
      int const numDataPatches   = mPostWeights->getNumDataPatches();
      pvAssert(numArbors == originalPostWeights->getNumArbors());
      pvAssert(patchSizeOverall == originalPostWeights->getPatchSizeOverall());
      pvAssert(numDataPatches == originalPostWeights->getNumDataPatches());

      auto arborSize = (std::size_t)(patchSizeOverall * numDataPatches) * sizeof(float);
      for (int arbor = 0; arbor < numArbors; arbor++) {
         float const *sourceArbor = originalPostWeights->getDataReadOnly(arbor);
         std::memcpy(mPostWeights->getData(arbor), sourceArbor, arborSize);
      }
   }
}

} // namespace PV
