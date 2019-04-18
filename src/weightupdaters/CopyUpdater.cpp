/*
 * CopyUpdater.cpp
 *
 *  Created on: Dec 15, 2017
 *      Author: Pete Schultz
 */

#include "CopyUpdater.hpp"
#include "columns/HyPerCol.hpp"
#include "columns/ObjectMapComponent.hpp"
#include "components/OriginalConnNameParam.hpp"
#include "connections/HyPerConn.hpp"
#include "utils/MapLookupByType.hpp"
#include "utils/TransposeWeights.hpp"

namespace PV {

CopyUpdater::CopyUpdater(char const *name, HyPerCol *hc) { initialize(name, hc); }

int CopyUpdater::initialize(char const *name, HyPerCol *hc) {
   return BaseWeightUpdater::initialize(name, hc);
}

void CopyUpdater::setObjectType() { mObjectType = "CopyUpdater"; }

void CopyUpdater::ioParam_plasticityFlag(enum ParamsIOFlag ioFlag) {
   // During the CommunicateInitInfo stage, plasticityFlag will be copied from
   // the original connection's updater.
}

Response::Status
CopyUpdater::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto componentMap = message->mHierarchy;

   mCopyWeightsPair = mapLookupByType<CopyWeightsPair>(componentMap, getDescription());
   FatalIf(
         mCopyWeightsPair == nullptr,
         "%s requires a CopyWeightsPair component.\n",
         getDescription_c());
   if (!mCopyWeightsPair->getInitInfoCommunicatedFlag()) {
      return Response::POSTPONE;
   }
   mCopyWeightsPair->needPre();

   auto *originalConnNameParam =
         mapLookupByType<OriginalConnNameParam>(componentMap, getDescription());
   FatalIf(
         originalConnNameParam == nullptr,
         "%s requires a OriginalConnNameParam component.\n",
         getDescription_c());
   if (!originalConnNameParam->getInitInfoCommunicatedFlag()) {
      return Response::POSTPONE;
   }

   char const *originalConnName = originalConnNameParam->getOriginalConnName();
   pvAssert(originalConnName != nullptr and originalConnName[0] != '\0');

   auto hierarchy           = message->mHierarchy;
   auto *objectMapComponent = mapLookupByType<ObjectMapComponent>(hierarchy, getDescription());
   pvAssert(objectMapComponent);
   HyPerConn *originalConn = objectMapComponent->lookup<HyPerConn>(std::string(originalConnName));
   pvAssert(originalConn);
   auto *originalWeightUpdater = originalConn->getComponentByType<BaseWeightUpdater>();
   if (originalWeightUpdater and !originalWeightUpdater->getInitInfoCommunicatedFlag()) {
      return Response::POSTPONE;
   }
   mPlasticityFlag = originalWeightUpdater ? originalWeightUpdater->getPlasticityFlag() : false;

   auto *originalWeightsPair = originalConn->getComponentByType<WeightsPair>();
   pvAssert(originalWeightsPair);
   if (!originalWeightsPair->getInitInfoCommunicatedFlag()) {
      return Response::POSTPONE;
   }
   originalWeightsPair->needPre();
   mOriginalWeights = originalWeightsPair->getPreWeights();
   pvAssert(mOriginalWeights);

   auto status = BaseWeightUpdater::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }

   if (mPlasticityFlag) {
      mCopyWeightsPair->getPreWeights()->setWeightsArePlastic();
   }
   mWriteCompressedCheckpoints = mCopyWeightsPair->getWriteCompressedCheckpoints();

   return Response::SUCCESS;
}

Response::Status CopyUpdater::registerData(Checkpointer *checkpointer) {
   auto status = BaseWeightUpdater::registerData(checkpointer);
   if (!Response::completed(status)) {
      return status;
   }
   std::string nameString = std::string(name);
   checkpointer->registerCheckpointData(
         nameString,
         "lastUpdateTime",
         &mLastUpdateTime,
         (std::size_t)1,
         true /*broadcast*/,
         false /*not constant*/);
   return Response::SUCCESS;
}

void CopyUpdater::updateState(double simTime, double dt) {
   pvAssert(mCopyWeightsPair and mCopyWeightsPair->getPreWeights());
   if (mOriginalWeights->getTimestamp() > mCopyWeightsPair->getPreWeights()->getTimestamp()) {
      mCopyWeightsPair->copy();
      mCopyWeightsPair->getPreWeights()->setTimestamp(simTime);
      mLastUpdateTime = simTime;
   }
}

} // namespace PV
