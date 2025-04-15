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

NormalizeGroup::NormalizeGroup(char const *name, PVParams *params, Communicator const *comm) {
   initialize(name, params, comm);
}

NormalizeGroup::NormalizeGroup() {}

NormalizeGroup::~NormalizeGroup() { free(mNormalizeGroupName); }

void NormalizeGroup::initialize(char const *name, PVParams *params, Communicator const *comm) {
   NormalizeBase::initialize(name, params, comm);
}

int NormalizeGroup::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = NormalizeBase::ioParamsFillGroup(ioFlag);
   ioParam_normalizeGroupName(ioFlag);
   return status;
}

// The NormalizeBase parameters are overridden to do nothing in NormalizeGroup.
void NormalizeGroup::ioParam_normalizeArborsIndividually(enum ParamsIOFlag ioFlag) {}
void NormalizeGroup::ioParam_normalizeOnInitialize(enum ParamsIOFlag ioFlag) {}
void NormalizeGroup::ioParam_normalizeOnWeightUpdate(enum ParamsIOFlag ioFlag) {}

void NormalizeGroup::ioParam_normalizeGroupName(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamStringRequired(ioFlag, getName(), "normalizeGroupName", &mNormalizeGroupName);
}

Response::Status
NormalizeGroup::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto status = NormalizeBase::communicateInitInfo(message);
   if (status != Response::SUCCESS) {
      return status;
   }

   auto *objectTable        = message->mObjectTable;
   mGroupHead               = objectTable->findObject<NormalizeBase>(mNormalizeGroupName);
   FatalIf(
         mGroupHead == nullptr,
         "%s: normalizeGroupName \"%s\" is not a recognized normalizer.\n",
         getDescription_c(),
         mNormalizeGroupName);
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
               mNormalizeGroupName);
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
            getDescription_c(), mNormalizeGroupName);
      MPI_Barrier(mCommunicator->globalCommunicator());
      exit(EXIT_FAILURE);
   }

   mGroupHead->addWeightsToList(preWeights);
   return Response::SUCCESS;
}

int NormalizeGroup::normalizeWeights() { return PV_SUCCESS; }

} /* namespace PV */
