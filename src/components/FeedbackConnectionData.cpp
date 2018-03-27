/*
 * FeedbackConnectionData.cpp
 *
 *  Created on: Jan 9, 2017
 *      Author: pschultz
 */

#include "FeedbackConnectionData.hpp"
#include "columns/HyPerCol.hpp"
#include "columns/ObjectMapComponent.hpp"
#include "components/OriginalConnNameParam.hpp"
#include "connections/HyPerConn.hpp"
#include "utils/MapLookupByType.hpp"

namespace PV {

FeedbackConnectionData::FeedbackConnectionData(char const *name, HyPerCol *hc) {
   initialize(name, hc);
}

FeedbackConnectionData::FeedbackConnectionData() {}

FeedbackConnectionData::~FeedbackConnectionData() {}

int FeedbackConnectionData::initialize(char const *name, HyPerCol *hc) {
   return ConnectionData::initialize(name, hc);
}

void FeedbackConnectionData::setObjectType() { mObjectType = "FeedbackConnectionData"; }

int FeedbackConnectionData::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   return ConnectionData::ioParamsFillGroup(ioFlag);
}

// FeedbackConn doesn't use preLayerName or postLayerName
// If they're present, errors are handled byy setPreAndPostLayerNames
void FeedbackConnectionData::ioParam_preLayerName(enum ParamsIOFlag ioFlag) {}
void FeedbackConnectionData::ioParam_postLayerName(enum ParamsIOFlag ioFlag) {}

Response::Status FeedbackConnectionData::communicateInitInfo(
      std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto hierarchy = message->mHierarchy;
   auto *originalConnNameParam =
         mapLookupByType<OriginalConnNameParam>(hierarchy, getDescription());
   FatalIf(
         originalConnNameParam == nullptr,
         "%s requires an OriginalConnNameParam component.\n",
         getDescription_c());
   if (!originalConnNameParam->getInitInfoCommunicatedFlag()) {
      return Response::POSTPONE;
   }
   char const *originalConnName = originalConnNameParam->getOriginalConnName();
   pvAssert(originalConnName != nullptr);

   ObjectMapComponent *objectMapComponent =
         mapLookupByType<ObjectMapComponent>(hierarchy, getDescription());
   pvAssert(objectMapComponent);
   HyPerConn *originalConn = objectMapComponent->lookup<HyPerConn>(std::string(originalConnName));
   if (originalConn == nullptr) {
      if (parent->getCommunicator()->globalCommRank() == 0) {
         ErrorLog().printf(
               "%s: originalConnName \"%s\" does not correspond to a HyPerConn in the column.\n",
               getDescription_c(),
               originalConnName);
      }
      MPI_Barrier(parent->getCommunicator()->globalCommunicator());
      exit(PV_FAILURE);
   }
   auto *originalConnectionData = originalConn->getComponentByType<ConnectionData>();
   FatalIf(
         originalConnectionData == nullptr,
         "%s has original connection \"%s\", which does not have a ConnectionData component.\n",
         getDescription_c(),
         originalConn->getName());
   if (!originalConnectionData->getInitInfoCommunicatedFlag()) {
      return Response::POSTPONE;
   }
   free(mPreLayerName);
   mPreLayerName = strdup(originalConnectionData->getPostLayerName());
   free(mPostLayerName);
   mPostLayerName = strdup(originalConnectionData->getPreLayerName());

   return ConnectionData::communicateInitInfo(message);
}

} // namespace PV
