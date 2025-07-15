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

FeedbackConnectionData::FeedbackConnectionData(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   initialize(paramsIO, comm);
}

FeedbackConnectionData::FeedbackConnectionData() {}

FeedbackConnectionData::~FeedbackConnectionData() {}

void FeedbackConnectionData::initialize(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   ConnectionData::initialize(paramsIO, comm);
}

void FeedbackConnectionData::setObjectType() { mObjectType = "FeedbackConnectionData"; }

int FeedbackConnectionData::ioParamsFillGroup(ParamsIOSwitch ioSwitch) {
   return ConnectionData::ioParamsFillGroup(ioSwitch);
}

// FeedbackConn doesn't use preLayerName or postLayerName
// If they're present, errors are handled by setPreAndPostLayerNames
void FeedbackConnectionData::ioParam_preLayerName(ParamsIOSwitch ioSwitch) {}
void FeedbackConnectionData::ioParam_postLayerName(ParamsIOSwitch ioSwitch) {}

Response::Status FeedbackConnectionData::communicateInitInfo(
      std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto *objectTable           = message->mObjectTable;
   auto *originalConnNameParam = objectTable->findObject<OriginalConnNameParam>(getName());
   FatalIf(
         originalConnNameParam == nullptr,
         "%s could not find an OriginalConnNameParam.\n",
         getDescription_c());
   std::string const &originalConnName = originalConnNameParam->getLinkedObjectName();

   auto *originalConnectionData = objectTable->findObject<ConnectionData>(originalConnName);
   FatalIf(
         originalConnectionData == nullptr,
         "%s set original connection to \"%s\", which does not have a ConnectionData component.\n",
         getDescription_c(),
         originalConnName.c_str());
   if (!originalConnectionData->getInitInfoCommunicatedFlag()) {
      return Response::POSTPONE;
   }
   mPreLayerName = originalConnectionData->getPostLayerName();
   mPostLayerName = originalConnectionData->getPreLayerName();

   return ConnectionData::communicateInitInfo(message);
}

} // namespace PV
