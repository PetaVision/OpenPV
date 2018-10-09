/*
 * FeedbackConnectionData.cpp
 *
 *  Created on: Jan 9, 2017
 *      Author: pschultz
 */

#include "FeedbackConnectionData.hpp"
#include "columns/ComponentBasedObject.hpp"
#include "columns/HyPerCol.hpp"
#include "components/OriginalConnNameParam.hpp"
#include "observerpattern/ObserverTable.hpp"

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
   auto hierarchy              = message->mHierarchy;
   auto *originalConnNameParam = hierarchy->lookupByType<OriginalConnNameParam>();
   pvAssert(originalConnNameParam);
   char const *originalConnName = originalConnNameParam->getLinkedObjectName();

   ObserverTable *tableComponent = hierarchy->lookupByType<ObserverTable>();
   pvAssert(tableComponent);
   ComponentBasedObject *originalConn =
         tableComponent->lookupByName<ComponentBasedObject>(std::string(originalConnName));
   if (originalConn == nullptr) {
      if (parent->getCommunicator()->globalCommRank() == 0) {
         ErrorLog().printf(
               "%s: originalConnName \"%s\" does not correspond to an object in the column.\n",
               getDescription_c(),
               originalConnName);
      }
      MPI_Barrier(parent->getCommunicator()->globalCommunicator());
      exit(PV_FAILURE);
   }
   auto *originalConnectionData = originalConn->getComponentByType<ConnectionData>();
   FatalIf(
         originalConnectionData == nullptr,
         "%s set original connection to \"%s\", which does not have a ConnectionData component.\n",
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
