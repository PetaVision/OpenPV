/*
 * AffineCopyUpdater.cpp
 *
 *  Created on: July 12, 2019
 *      Author: Xinhua Zhang
 */

#include "AffineCopyUpdater.hpp"
#include "components/OriginalConnNameParam.hpp"
#include "connections/HyPerConn.hpp"
#include "observerpattern/ObserverTable.hpp"
#include "utils/TransposeWeights.hpp"

namespace PV {

AffineCopyUpdater::AffineCopyUpdater(char const *name, PVParams *params, Communicator const *comm) {
   initialize(name, params, comm);
}

void AffineCopyUpdater::initialize(char const *name, PVParams *params, Communicator const *comm) {
   BaseWeightUpdater::initialize(name, params, comm);
}

void AffineCopyUpdater::setObjectType() { mObjectType = "AffineCopyUpdater"; }

Response::Status
AffineCopyUpdater::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto *objectTable = message->mObjectTable;

   mCopyWeightsPair = objectTable->findObject<AffineCopyWeightsPair>(getName());
   FatalIf(
         mCopyWeightsPair == nullptr,
         "%s requires a CopyWeightsPair component.\n",
         getDescription_c());
   if (!mCopyWeightsPair->getInitInfoCommunicatedFlag()) {
      return Response::POSTPONE;
   }
   mCopyWeightsPair->needPre();

   auto *originalConnNameParam = objectTable->findObject<OriginalConnNameParam>(getName());
   FatalIf(
         originalConnNameParam == nullptr,
         "%s requires a OriginalConnNameParam component.\n",
         getDescription_c());
   if (!originalConnNameParam->getInitInfoCommunicatedFlag()) {
      return Response::POSTPONE;
   }

   char const *originalConnName = originalConnNameParam->getLinkedObjectName();
   pvAssert(originalConnName != nullptr and originalConnName[0] != '\0');

   auto *originalWeightUpdater = objectTable->findObject<BaseWeightUpdater>(originalConnName);
   if (originalWeightUpdater and !originalWeightUpdater->getInitInfoCommunicatedFlag()) {
      return Response::POSTPONE;
   }
   mPlasticityFlag = originalWeightUpdater ? originalWeightUpdater->getPlasticityFlag() : false;

   auto *originalWeightsPair = objectTable->findObject<WeightsPair>(originalConnName);
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

Response::Status
AffineCopyUpdater::registerData(std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) {
   auto status = BaseWeightUpdater::registerData(message);
   if (!Response::completed(status)) {
      return status;
   }
   std::string nameString = std::string(name);
   auto *checkpointer     = message->mDataRegistry;
   checkpointer->registerCheckpointData(
         nameString,
         "lastUpdateTime",
         &mLastUpdateTime,
         (std::size_t)1,
         true /*broadcast*/,
         false /*not constant*/);
   return Response::SUCCESS;
}

void AffineCopyUpdater::updateState(double simTime, double dt) {
   pvAssert(mCopyWeightsPair and mCopyWeightsPair->getPreWeights());
   if (mOriginalWeights->getTimestamp() > mCopyWeightsPair->getPreWeights()->getTimestamp()) {
      mCopyWeightsPair->copy();
      mCopyWeightsPair->getPreWeights()->setTimestamp(simTime);
      mLastUpdateTime = simTime;
   }
}

} // namespace PV
