/*
 * NormalizeGroup.cpp
 *
 *  Created on: Jun 22, 2016
 *      Author: pschultz
 */

#include "normalizers/NormalizeGroup.hpp"
#include "components/WeightsPair.hpp"
#include "connections/HyPerConn.hpp"
#include "observerpattern/ObserverTable.hpp"
#include "structures/Weights.hpp"

namespace PV {

NormalizeGroup::NormalizeGroup(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   initialize(params, defaults, comm);
}

NormalizeGroup::NormalizeGroup() {}

NormalizeGroup::~NormalizeGroup() {}

void NormalizeGroup::initialize(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   NormalizeBase::initialize(params, defaults, comm);
}

int NormalizeGroup::ioParamsFillGroup(ParamsIOSwitch ioSwitch) {
   int status = NormalizeBase::ioParamsFillGroup(ioSwitch);
   ioParam_normalizeGroupName(ioSwitch);
   return status;
}

// The NormalizeBase parameters are overridden to do nothing in NormalizeGroup.
void NormalizeGroup::ioParam_normalizeArborsIndividually(ParamsIOSwitch ioSwitch) {}
void NormalizeGroup::ioParam_normalizeOnInitialize(ParamsIOSwitch ioSwitch) {}
void NormalizeGroup::ioParam_normalizeOnWeightUpdate(ParamsIOSwitch ioSwitch) {}

void NormalizeGroup::ioParam_normalizeGroupName(ParamsIOSwitch ioSwitch) {
   mParamsIO->ioParam(ioSwitch, "normalizeGroupName", &mNormalizeGroupName);
}

Response::Status
NormalizeGroup::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto status = NormalizeBase::communicateInitInfo(message);
   if (status != Response::SUCCESS) {
      return status;
   }

   auto *objectTable = message->mObjectTable;
   mGroupHead        = objectTable->findObject<NormalizeBase>(mNormalizeGroupName);
   FatalIf(
         mGroupHead == nullptr,
         "%s: normalizeGroupName \"%s\" is not a recognized normalizer.\n",
         getDescription_c(),
         mNormalizeGroupName.c_str());
   FatalIf(
         !strcmp(mGroupHead->getName(), getName()),
         "%s: normalizeGroupName must point to a connection other than itself.\n",
         getDescription_c());
   FatalIf(
         !strcmp(mGroupHead->getObjectType().c_str(), "normalizeGroup"),
         "%s: normalizeGroupName points to \"%s\", but that connection itself has "
         "normalizeMethod set to normalizeGroup.\n",
         getDescription_c(),
         mGroupHead->getName());

   if (mGroupHead == nullptr) {
      if (mCommunicator->globalCommRank() == 0) {
         ErrorLog().printf(
               "%s: normalizeGroupName \"%s\" is not a recognized normalizer.\n",
               getDescription_c(),
               mNormalizeGroupName.c_str());
      }
      MPI_Barrier(mCommunicator->globalCommunicator());
      exit(EXIT_FAILURE);
   }

   WeightsPair *weightsPair = objectTable->findObject<WeightsPair>(getName());
   pvAssert(weightsPair); // NormalizeBase::communicateInitInfo should have checked for this.
   Weights *preWeights = weightsPair->getPreWeights();
   pvAssert(preWeights); // NormalizeBase::communicateInitInfo should have called needPre.

   auto *thisConnectionData = objectTable->findObject<ConnectionData>(getName());
   if (!thisConnectionData->getInitInfoCommunicatedFlag()) {
      return Response::POSTPONE;
   }
   auto *headConnectionData = objectTable->findObject<ConnectionData>(mNormalizeGroupName);
   if (!headConnectionData->getInitInfoCommunicatedFlag()) {
      return Response::POSTPONE;
   }
   if (thisConnectionData->getPreIsBroadcast() != headConnectionData->getPreIsBroadcast()) {
      ErrorLog().printf(
            "broadcast flag for %s does not match that of normalizeGroupName \"%s\".\n",
            getDescription_c(), mNormalizeGroupName.c_str());
      MPI_Barrier(mCommunicator->globalCommunicator());
      exit(EXIT_FAILURE);
   }

   mGroupHead->addWeightsToList(preWeights);
   return Response::SUCCESS;
}

int NormalizeGroup::normalizeWeights() { return PV_SUCCESS; }

} /* namespace PV */
