/*
 * FeedbackConnectionData.cpp
 *
 *  Created on: Jan 9, 2017
 *      Author: pschultz
 */

#include "FeedbackConnectionData.hpp"
#include "columns/ComponentBasedObject.hpp"
#include "components/OriginalConnNameParam.hpp"
#include "observerpattern/ObserverTable.hpp"

namespace PV {

FeedbackConnectionData::FeedbackConnectionData(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   initialize(name, params, comm);
}

FeedbackConnectionData::FeedbackConnectionData() {}

FeedbackConnectionData::~FeedbackConnectionData() {}

void FeedbackConnectionData::initialize(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   ConnectionData::initialize(name, params, comm);
}

void FeedbackConnectionData::setObjectType() { mObjectType = "FeedbackConnectionData"; }

int FeedbackConnectionData::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   return ConnectionData::ioParamsFillGroup(ioFlag);
}

// FeedbackConn doesn't use preLayerName or postLayerName
// If they're present, errors are handled by setPreAndPostLayerNames
void FeedbackConnectionData::ioParam_preLayerName(enum ParamsIOFlag ioFlag) {}
void FeedbackConnectionData::ioParam_postLayerName(enum ParamsIOFlag ioFlag) {}

Response::Status FeedbackConnectionData::communicateInitInfo(
      std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto *objectTable           = message->mObjectTable;
   auto *originalConnNameParam = objectTable->findObject<OriginalConnNameParam>(getName());
   FatalIf(
         originalConnNameParam == nullptr,
         "%s could not find an OriginalConnNameParam.\n",
         getDescription_c());
   char const *originalConnName = originalConnNameParam->getLinkedObjectName();

   auto *originalConnectionData = objectTable->findObject<ConnectionData>(originalConnName);
   FatalIf(
         originalConnectionData == nullptr,
         "%s set original connection to \"%s\", which does not have a ConnectionData component.\n",
         getDescription_c(),
         originalConnName);
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
